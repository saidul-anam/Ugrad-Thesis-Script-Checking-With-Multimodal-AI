"""Clean exam-script PDFs: remove examiner red ink AND pen ink soaked through from the back
of each sheet, without touching the student's answers.

  python clean_pdf.py input.pdf                     -> input_cleaned.pdf (next to input)
  python clean_pdf.py input.pdf -o out.pdf
  python clean_pdf.py folder_of_pdfs/ -o out_folder/
  add --report report.csv for per-page numbers, --qa qa_folder/ for before|after sheets

Requirements: numpy, opencv-python(-headless), pymupdf, pillow.

Input: CamScanner-style PDFs, one scanned JPEG per page (the small logo image is ignored).

Per PDF, the examiner's pen colour is measured first; detection is re-centred only if it
clearly differs from the usual crimson (pink, magenta, orange pens). Blue ink is never red.

Per page:
 1. RED INK. Colour mask with hysteresis (confident red seeds grow into connected weaker
    red edge pixels, plus a 1-2 px halo over light/reddish pixels only, plus the small dark
    specks darker pens leave along their edges); masked pixels are inpainted. Faint pink
    (red marks soaked through from the reverse) is set to paper colour.
 2. SOAKED-THROUGH INK. Neighbouring pages are front and back of one sheet. Each page is
    aligned to the mirrored next page (coarse rotation/scale/shift search on "any mark"
    maps, then ECC affine -> homography; reverse direction if one fails), and pages are
    paired into sheets by alignment score. The back side's own writing, warped onto the
    front (with optional optical-flow refinement), gives a footprint; only light marks
    inside it are set to paper colour. Protected: dark ink (unless it lies exactly on the
    back's writing), blue ink, ruled lines, anything outside the footprint. Pages without a
    confident partner get only step 1.
Everything outside the masks is untouched; the page JPEG is re-encoded once with the scan's
own quantization tables, so quality and size stay the same.

At the end it lists PDFs worth a manual look (pen colour unusual or not found, pages not in
the expected format, few pages matched to a back side, red still visible). The report's
"note" column marks pages whose soak-through stains were kept.
"""
import os, io, glob, csv, argparse
import numpy as np, cv2, pymupdf
from PIL import Image, JpegImagePlugin

MIN_IMG_SIDE = 1000      # ignore the small CamScanner logo image on each page
JPEG_Q = 90              # fallback only; JPEG pages reuse the scan's own tables
SC = 0.25                # fine registration scale
CS = 0.125               # coarse-search scale
MIN_ECC = 0.30           # pairs scoring this or more are accepted (sanity: footprint >= MIN_LIFT)
LOW_ECC = 0.15           # pairs LOW_ECC..MIN_ECC need the stains to confirm them:
STRONG_LIFT = 2.0        #   one of the two pages must fit its partner's footprint this well
MIN_LIFT = 1.7
RED_HUE_DEFAULT = 248    # OpenCV HSV_FULL hue (0-255) of the crimson/maroon pens seen so far
RED_HUE_TOLERANCE = 10   # a PDF's measured pen hue within this of the default keeps the default
RED_HUE_RANGE = (215, 25)  # pen hues accepted by calibration: magenta/pink .. red .. orange
RED_MIN_PX = 1500        # fewer confident red pixels than this: no examiner pen found
RED_HUE = RED_HUE_DEFAULT  # set per PDF by calibrate_red()


# =====================================================================  1. red ink
def _red_hue_dist(H):
    d = np.abs(H.astype(np.int16) - RED_HUE)
    return np.minimum(d, 256 - d)


def calibrate_red(pages):
    """measure the examiner's pen hue over a PDF's pages (strongly saturated pixels in the
    pink..orange range). Returns (hue to use, confident red pixel count). Detection is only
    re-centred when the pen clearly differs from the default, so known batches are unchanged."""
    hues = []
    for a in pages:
        if a is None: continue
        hsv = cv2.cvtColor(np.ascontiguousarray(a[::2, ::2]), cv2.COLOR_RGB2HSV_FULL).reshape(-1, 3)
        H, S, V = hsv[:, 0].astype(np.int16), hsv[:, 1], hsv[:, 2]
        lo, hi = RED_HUE_RANGE
        hues.append(H[(S >= 110) & (V >= 90) & ((H >= lo) | (H <= hi))])
    h = np.concatenate(hues) if hues else np.zeros(0, np.int16)
    if len(h) < RED_MIN_PX:
        return RED_HUE_DEFAULT, len(h)
    peak = int(np.bincount(np.where(h < 128, h + 256, h)).argmax()) % 256
    d = abs(peak - RED_HUE_DEFAULT); d = min(d, 256 - d)
    return (RED_HUE_DEFAULT if d <= RED_HUE_TOLERANCE else peak), len(h)


def strong_mask(a):
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV_FULL)
    H, S, V = (hsv[..., i].astype(np.int16) for i in range(3))
    hd = _red_hue_dist(H)
    blue = (H >= 140) & (H <= 200) & (S >= 70) & (V <= 215)   # student blue ink, never red
    seed = (hd <= 22) & (S >= 90) & (V >= 90) & ~blue
    weak = (hd <= 30) & (S >= 28) & (V >= 55) & ~blue
    n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool); keep[np.unique(lab[seed])] = True; keep[0] = False
    m = keep[lab]
    ok = ((V >= 170) | ((hd <= 40) & (S >= 20))) & ~blue   # halo never onto dark/blue ink
    big = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return m | (big & ok)


def faint_region(a, seed=2.5, low=1.0, sigma=2.0, area=60):
    """faint pink tint (red marks soaked through from the reverse side)"""
    f = a.astype(np.float32)
    ds = cv2.GaussianBlur(f[..., 0] - np.maximum(f[..., 1], f[..., 2]), (0, 0), sigma)
    n, lab, st, _ = cv2.connectedComponentsWithStats((ds >= low).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool); keep[np.unique(lab[ds >= seed])] = True
    keep &= st[:, cv2.CC_STAT_AREA] >= area; keep[0] = False
    return keep[lab]


def paper_bg(a):
    s = cv2.resize(a.max(2), None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA)
    s = cv2.dilate(cv2.medianBlur(s, 21), np.ones((5, 5), np.uint8))
    return cv2.resize(s, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)


def red_halo(a, m, vdark=170, max_area=80, reach=3):
    """small dark specks touching a red stroke: the dark, colourless JPEG halo of darker
    (maroon) pens. Real pen strokes form large components and are left alone."""
    near = cv2.dilate(m.astype(np.uint8), np.ones((2 * reach + 1,) * 2, np.uint8)) > 0
    dark = ((a.max(2) < vdark) & ~m & ~_blue(a)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    small = st[:, cv2.CC_STAT_AREA] <= max_area; small[0] = False
    touch = np.zeros(n, bool); touch[np.unique(lab[near & (lab > 0)])] = True
    return (small & touch)[lab]


def remove_red(a):
    m = strong_mask(a)
    if m.any():
        m |= red_halo(a, m)
    out = cv2.inpaint(a, m.astype(np.uint8), 3, cv2.INPAINT_TELEA) if m.any() else a.copy()
    o = out.astype(np.int16)
    f = faint_region(a) & ~m & (o.max(2) >= 120) & (o[..., 2] - o[..., 0] < 15)
    out[f] = paper_bg(a)[f][:, None]
    return out, m


def residual_red(a):
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV_FULL)
    H, S, V = (hsv[..., i].astype(np.int16) for i in range(3))
    return ((_red_hue_dist(H) <= 22) & (S >= 90) & (V >= 90)).mean()


# =====================================================================  2. soaked-through ink
def _blue(a):
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV_FULL)
    H, S, V = (hsv[..., i].astype(np.int16) for i in range(3))
    return (H >= 140) & (H <= 200) & (S >= 70) & (V <= 215)


def own_ink(page, vcore=100, big=150):
    """a page's own writing (dark, blue or red strokes in large components), not its ghosts"""
    core = ((page.max(2) < vcore) | _blue(page) | strong_mask(page)).astype(np.uint8)
    _, lab, st, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    keep = st[:, cv2.CC_STAT_AREA] >= big; keep[0] = False
    return keep[lab]


def anymark(a, sc, vpaper=230):
    """1 wherever the paper is marked at all, resized to scale sc"""
    return cv2.resize((a.max(2) < vpaper).astype(np.float32), None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)


def coarse_init(front, back_m):
    """similarity transform (full res) mapping the mirrored back onto the front"""
    F = cv2.GaussianBlur(anymark(front, CS), (0, 0), 1.5)
    B = cv2.GaussianBlur(anymark(back_m, CS), (0, 0), 1.5)
    h, w = F.shape; win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    best = (-1.0, None)
    for ang in np.arange(-5, 5.01, 1.0):
        for s in np.arange(0.92, 1.081, 0.02):
            M = cv2.getRotationMatrix2D((B.shape[1] / 2, B.shape[0] / 2), ang, s)
            (dx, dy), resp = cv2.phaseCorrelate(cv2.warpAffine(B, M, (w, h)), F, win)
            if resp > best[0]:
                M = M.copy(); M[:, 2] += (dx, dy); best = (resp, M)
    return np.diag([1 / CS, 1 / CS, 1]) @ np.vstack([best[1], [0, 0, 1]]) @ np.diag([CS, CS, 1])


def register_back(front, back):
    """homography (full res) mirrored back -> front, and ECC score"""
    bm = np.ascontiguousarray(back[:, ::-1])
    H0 = coarse_init(front, bm)
    S = np.diag([SC, SC, 1])
    F = cv2.GaussianBlur(anymark(front, SC), (0, 0), 2)
    B = cv2.GaussianBlur(anymark(bm, SC), (0, 0), 2)
    W = (S @ np.linalg.inv(H0) @ np.linalg.inv(S)).astype(np.float32)   # ECC: F(x) ~ B(W x)
    best = -1.0
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    for motion in (cv2.MOTION_AFFINE, cv2.MOTION_HOMOGRAPHY):
        w0 = W.copy() if motion == cv2.MOTION_HOMOGRAPHY else W[:2].copy()
        try:
            cc, w = cv2.findTransformECC(F, B, w0, motion, crit, None, 5)
        except cv2.error:
            break
        W = w if motion == cv2.MOTION_HOMOGRAPHY else np.vstack([w, [0, 0, 1]]).astype(np.float32)
        best = cc
    if best < 0:
        return None, -1.0
    return np.linalg.inv(S) @ np.linalg.inv(W.astype(np.float64)) @ S, float(best)


def register_back_any(front, back):
    """register_back; if that direction fails, align the other way round and invert"""
    Hm, cc = register_back(front, back)
    if Hm is not None:
        return Hm, cc
    Hr, cc = register_back(back, front)
    if Hr is None:
        return None, -1.0
    def mirror(w): return np.array([[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]], np.float64)
    return mirror(front.shape[1]) @ np.linalg.inv(Hr) @ mirror(back.shape[1]), cc


def refine_flow(front, warped_back, sc=0.5):
    """dense residual shift (full res) so that front(x) ~ warped_back(x + flow)"""
    def ink(img, drop_dark):
        V = img.max(2).astype(np.float32); g = 255 - V
        if drop_dark: g[V < 90] = 0                 # the front's own strong ink is not bleed
        g = cv2.GaussianBlur(cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA), (0, 0), 1.5)
        return np.clip(g * (255.0 / max(np.percentile(g, 99.5), 1)), 0, 255).astype(np.uint8)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(ink(front, True), ink(warped_back, False), None)
    flow = cv2.GaussianBlur(np.clip(flow, -30 * sc, 30 * sc), (0, 0), 8)
    return cv2.resize(flow, (front.shape[1], front.shape[0]), interpolation=cv2.INTER_LINEAR) / sc


def lift(front, fp):
    """how much more grey (non-ink, non-paper) content the footprint holds than chance"""
    V = front.max(2); ghost = (V < 215) & (V >= 110)
    if fp.mean() == 0 or ghost.mean() == 0: return 0.0
    return float((ghost & fp).mean() / (fp.mean() * ghost.mean()))


def footprint(front, back, Hm, grow=4):
    """where the back's own ink lands on the front; flow refinement only if it fits better"""
    bm = np.ascontiguousarray(back[:, ::-1]); sz = (front.shape[1], front.shape[0])
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)
    own = cv2.warpPerspective(own_ink(bm).astype(np.uint8) * 255, Hm, sz)
    fp_plain = cv2.dilate((own > 127).astype(np.uint8), k) > 0
    wb = cv2.warpPerspective(bm, Hm, sz, borderValue=(255, 255, 255))
    fl = refine_flow(front, wb)
    gx, gy = np.meshgrid(np.arange(sz[0], dtype=np.float32), np.arange(sz[1], dtype=np.float32))
    fp_flow = cv2.dilate((cv2.remap(own, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR) > 127)
                         .astype(np.uint8), k) > 0
    l0, l1 = lift(front, fp_plain), lift(front, fp_flow)
    return (fp_flow, l1) if l1 > l0 else (fp_plain, l0)


def pair_pages(scores):
    """non-overlapping neighbour pairs (i, i+1) maximising total score, each >= LOW_ECC"""
    n = len(scores) + 1; best = [0.0] * (n + 1); take = [False] * (n + 1)
    for i in range(n - 2, -1, -1):
        skip = best[i + 1]
        use = scores[i] + best[i + 2] if scores[i] >= LOW_ECC else -1
        best[i], take[i] = (use, True) if use > skip else (skip, False)
    pairs, i = {}, 0
    while i < n - 1:
        if take[i]: pairs[i], pairs[i + 1] = i + 1, i; i += 2
        else: i += 1
    return pairs


def ruled_lines(a, vmax=215, length=250):
    dark = (a.max(2) < vmax).astype(np.uint8)
    h = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1)))
    v = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, length)))
    return cv2.dilate(h | v, np.ones((3, 3), np.uint8)) > 0


def remove_bleed(a, fp, inside=0.85, vnon=215, vlight=150, vdark=110, vcore=50, ncore=40, on_ink=0.95):
    """erase soaked-through marks inside footprint fp.
    A mark is erased whole if it lies inside fp, is not blue, is light overall, and has no
    real dark core (a student's letter can merge with a stain) -- unless nearly all of its
    dark pixels lie right on the back side's ink lines. Light fringes away from dark ink are
    cleared too. Ruled lines are never touched."""
    V = a.max(2); blue = _blue(a)
    n, lab, st, _ = cv2.connectedComponentsWithStats((V < vnon).astype(np.uint8), connectivity=8)
    area = st[:, cv2.CC_STAT_AREA]
    fin = np.bincount(lab.ravel(), fp.ravel().astype(np.float64), n) / np.maximum(area, 1)
    nblue = np.bincount(lab.ravel(), blue.ravel().astype(np.float64), n)
    order = np.argsort(lab.ravel(), kind="stable"); vv = V.ravel()[order]
    starts = np.concatenate([[0], np.cumsum(area)[:-1]])
    med = np.array([np.median(vv[s:s + k]) if k else 255 for s, k in zip(starts, area)])
    dk = np.array([np.percentile(vv[s:s + k], 3) if k else 255 for s, k in zip(starts, area)])
    D = V < 100
    tight = cv2.erode(fp.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0     # back ink lines
    ndark = np.bincount(lab.ravel(), D.ravel().astype(np.float64), n)
    ntight = np.bincount(lab.ravel(), (D & tight).ravel().astype(np.float64), n)
    stain_core = ntight >= on_ink * np.maximum(ndark, 1)
    is_bleed = (fin >= inside) & (nblue < 0.2 * area) & (med >= vdark) & \
               ((dk >= vcore) | (ndark < ncore) | stain_core); is_bleed[0] = False
    erase = is_bleed[lab]
    near_core = cv2.dilate(((V < 120) | blue).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    erase |= fp & (lab > 0) & ~is_bleed[lab] & (V >= vlight) & ~near_core
    erase |= fp & (V >= vnon)
    erase &= ~ruled_lines(a)
    out = a.copy(); out[erase] = paper_bg(a)[erase][:, None]
    return out, erase


# =====================================================================  PDF in / out
def _page_xref(page):
    imgs = [im for im in page.get_images(full=True) if max(im[2], im[3]) >= MIN_IMG_SIDE]
    return imgs[0][0] if len(imgs) == 1 else None


def _write_image(doc, page, xref, src_im, out):
    buf = io.BytesIO()
    if doc.xref_get_key(xref, "Filter")[1].strip("[ ]") == "/DCTDecode" and src_im.format == "JPEG" \
            and src_im.mode == "RGB":
        Image.fromarray(out).save(buf, "JPEG", qtables=src_im.quantization,
                                  subsampling=JpegImagePlugin.get_sampling(src_im))
        doc.update_stream(xref, buf.getvalue(), compress=False)
        doc.xref_set_key(xref, "Filter", "/DCTDecode")          # update_stream drops it
    else:
        Image.fromarray(out).save(buf, "JPEG", quality=JPEG_Q)
        page.replace_image(xref, stream=buf.getvalue())


def qa_sheet(pages, path):
    """before|after thumbnails, one row per page"""
    rows = []
    for a, b in pages:
        h = 700; w = int(a.shape[1] * h / a.shape[0])
        rows.append(np.hstack([cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA),
                               np.full((h, 10, 3), 255, np.uint8),
                               cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)]))
    W = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 14), (0, W - r.shape[1]), (0, 0)), constant_values=200) for r in rows]
    Image.fromarray(np.vstack(rows)).save(path, quality=80)


def clean_pdf(src, dst, report=None, qa_dir=None):
    """clean one PDF; report: optional csv.writer.
    returns (pages, stain-cleaned pages, list of warnings worth a manual look)"""
    global RED_HUE
    doc = pymupdf.open(src); name = os.path.splitext(os.path.basename(src))[0]
    xrefs = [_page_xref(p) for p in doc]
    srcs = [Image.open(io.BytesIO(doc.extract_image(x)["image"])) if x else None for x in xrefs]
    raw = [np.asarray(s.convert("RGB")) if s is not None else None for s in srcs]

    warnings = []
    RED_HUE, red_px = calibrate_red(raw)
    if red_px < RED_MIN_PX:
        warnings.append("no examiner red found -- pen may be another colour (check QA)")
    elif RED_HUE != RED_HUE_DEFAULT:
        warnings.append(f"red pen hue {RED_HUE} differs from usual {RED_HUE_DEFAULT} -- detection re-centred (check QA)")
    skipped = [i + 1 for i, a in enumerate(raw) if a is None]
    if skipped:
        warnings.append(f"pages {skipped} are not one scanned image -- left unchanged")

    # pair pages into sheets (front, back)
    scores, Hs = [], {}
    for i in range(len(raw) - 1):
        if raw[i] is None or raw[i + 1] is None: scores.append(-1.0); continue
        Hm, cc = register_back_any(raw[i], raw[i + 1]); scores.append(cc); Hs[(i, i + 1)] = Hm
    pairs = pair_pages(scores)

    fps = {}
    for i, j in pairs.items():
        Hm, cc = (Hs[(i, j)], scores[i]) if j > i else register_back_any(raw[i], raw[j])
        fps[i] = (footprint(raw[i], raw[j], Hm) if Hm is not None else (None, 0.0)) + (cc,)
    ok = set()
    for i, j in pairs.items():
        if i > j: continue
        li, lj = fps[i][1], fps[j][1]
        if scores[i] >= MIN_ECC or max(li, lj) >= STRONG_LIFT:
            ok |= {k for k, l in ((i, li), (j, lj)) if l >= MIN_LIFT}

    qa, cleaned, red_left = [], 0, []
    for i, page in enumerate(doc):
        if raw[i] is None:
            if report: report.writerow([name, i + 1, "", "", "", "", "", "", "not processed"])
            continue
        a = raw[i]
        out, m = remove_red(a)
        partner, cc, lf, er = "", "", "", 0.0
        if i in fps:
            fp, lf, cc = fps[i]
            if i in ok:
                out, e = remove_bleed(out, fp); er = e.mean(); partner = pairs[i] + 1; cleaned += 1
        _write_image(doc, page, xrefs[i], srcs[i], out)
        rr = residual_red(out)
        if rr > 0.0001: red_left.append(i + 1)                # > 0.01 % of the page still red
        if report:
            report.writerow([name, i + 1, partner, f"{cc:.3f}" if cc != "" else "",
                             f"{lf:.2f}" if lf != "" else "", f"{m.mean()*100:.3f}",
                             f"{er*100:.2f}", f"{rr*100:.4f}", "" if partner else "stains kept"])
        if qa_dir: qa.append((a, out))
    doc.save(dst, garbage=3)
    if qa_dir and qa: qa_sheet(qa, os.path.join(qa_dir, name + ".jpg"))

    done = len(doc) - len(skipped)
    if done >= 4 and cleaned < done / 2:
        warnings.append(f"only {cleaned}/{done} pages matched to their back side -- stains may remain")
    if red_left:
        warnings.append(f"red still visible on pages {red_left}")
    return len(doc), cleaned, warnings


def main():
    ap = argparse.ArgumentParser(description="Remove red marks and soaked-through ink from exam PDFs")
    ap.add_argument("input", help="PDF file or folder of PDFs")
    ap.add_argument("-o", "--out", help="output PDF (single input) or folder (folder input)")
    ap.add_argument("--report", help="write per-page CSV report here")
    ap.add_argument("--qa", help="folder for before|after JPEG sheets")
    args = ap.parse_args()

    src = args.input.rstrip("/\\")
    if os.path.isdir(src):
        files = sorted(glob.glob(os.path.join(src, "*.pdf")))
        out_dir = args.out or src + "_cleaned"
        os.makedirs(out_dir, exist_ok=True)
        jobs = [(f, os.path.join(out_dir, os.path.basename(f))) for f in files]
    else:
        dst = args.out or os.path.splitext(src)[0] + "_cleaned.pdf"
        if os.path.isdir(dst): dst = os.path.join(dst, os.path.basename(src))
        jobs = [(src, dst)]
    for s, d in jobs:
        if os.path.abspath(d) == os.path.abspath(s):
            raise SystemExit("output would overwrite the input; choose another --out")
    if args.qa: os.makedirs(args.qa, exist_ok=True)

    rep = w = None
    if args.report:
        rep = open(args.report, "w", newline=""); w = csv.writer(rep)
        w.writerow(["pdf", "page", "back_page", "align_score", "footprint_lift", "red_mask_%",
                    "bleed_erased_%", "red_after_%", "note"])
    flagged = []
    for k, (s, d) in enumerate(jobs, 1):
        n, cleaned, warns = clean_pdf(s, d, w, args.qa)
        print(f"[{k}/{len(jobs)}] {os.path.basename(s)}: {n} pages, {cleaned} stain-cleaned -> {d}", flush=True)
        for msg in warns:
            print(f"    ! {msg}", flush=True)
        if warns: flagged.append((os.path.basename(s), warns))
        if rep: rep.flush()
    if rep: rep.close()
    print(f"\ndone: {len(jobs)} PDF(s), {len(flagged)} flagged for a manual look")
    for f, warns in flagged:
        print(f"  {f}: " + "; ".join(warns))


if __name__ == "__main__":
    main()
