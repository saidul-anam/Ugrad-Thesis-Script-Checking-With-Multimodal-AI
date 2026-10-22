#!/usr/bin/env python3
"""
Reference transcriptions of scanned scripts with Claude Opus 5.5, through the Claude Code CLI on a Claude
subscription (`claude -p`, logged in with /login; no API key). One call per page, no tools.

The output has the same layout as data/ground_truth/transcripts/SE_10_Q1_extraction_opus_5.5/:

  <out-root>/<PAPER>_extraction_opus_5.5/<script_id>.txt            pages joined with "--- PAGE N ---"
  <out-root>/<PAPER>_extraction_opus_5.5/pages/<script_id>/page_N.jpg   the image sent (200 dpi)
                                                         page_N.txt   the page transcript
                                                         page_N.json  model notes, usage, timing

It is a silver reference, not ground truth: nobody has checked it by hand. The markup follows the hand-checked
GT pages ([struck: x], [illegible], [unclear: x]), so scripts/evaluate_transcription.py treats both alike.

Finished pages are skipped, so the script can be stopped and started again at any time. Usage limits:
every reply carries the plan's rate-limit state (five-hour and seven-day windows, with reset times). When a
limit is reached the script sleeps until the reset and carries on. It also pauses at --weekly-cap of the
seven-day window, so that the subscription stays usable for other work until the weekly reset.

Usage (run it in tmux, it takes many hours):
  python scripts/opus_transcribe.py                                   # classes 6 and 7, sample scripts first
  python scripts/opus_transcribe.py --papers SE_07_Q1 --scripts SE_07_Q1_0031
  python scripts/opus_transcribe.py --dry-run                          # list scripts and page counts only
  python scripts/opus_transcribe.py --weekly-cap 1.0                   # may use the whole weekly allowance
"""

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pymupdf
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
MODEL = "claude-opus-5-5"
CLAUDE = shutil.which("claude") or str(Path.home() / ".local/bin/claude")
DPI = 200            # same size as the hand-checked GT page images (1653 x 2339 for A4)
CALL_TIMEOUT = 900   # seconds for one page
# left out of every evaluation: SE_06_Q1_0004 is a 2nd Paper scan; SE_07_Q1_0000 has no teacher marks and no text
EXCLUDE = {"SE_06_Q1_0004", "SE_07_Q1_0000"}
# the seeded sample of the 2026-10 review, graded on both texts already: transcribed first
SAMPLE_DIR = REPO / "outputs/review_2026_10/scratchpad_backup/scans_sample"

SYSTEM_PROMPT = "You transcribe handwritten exam answer scripts exactly as written, for a research dataset."

PROMPT = """This is one page of a school student's handwritten English exam answer script (Bangladesh).
Transcribe the student's handwriting on this page exactly as written.

Rules:
- Verbatim: keep every spelling, grammar, capitalisation and punctuation mistake. Never correct or complete
  the student's words.
- Keep the line breaks of the page: one output line per handwritten line. Keep a blank line where the
  student left a gap between blocks.
- Crossed-out writing: [struck: the crossed-out text]. Keep it where it stands in the line.
- A word you cannot read at all: [illegible]. A word you can only partly read: [unclear: best reading].
- Question headers, numbers and labels exactly as written (for example "Ans to the Q No. 5", "5 Ans",
  "(a)", "b)"). Write circled letters or numbers as circled characters (ⓐ, ⓑ, ①, ⑥).
- A table or grid drawn by the student: a Markdown table with the student's rows and columns.
- Arrows: ->
- Ignore everything written by the teacher in red ink (ticks, crosses, marks, scores, comments,
  corrections), printed text of the answer sheet, and faint mirrored text showing through from the back
  of the sheet.
- If the page has no student writing, return an empty transcript.

Put the transcription, and nothing else, between <transcript> and </transcript>. After </transcript> you may
add short notes about anything uncertain."""

LIMIT_TEXT = re.compile(r"hit your .{0,20}limit|usage limit|session limit|weekly limit|rate.?limit|\b429\b",
                        re.IGNORECASE)
FATAL_TEXT = re.compile(r"not logged in|/login|invalid authentication|\b401\b|invalid model|model .* not found",
                        re.IGNORECASE)


def log(msg: str, fh=None):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def sleep_until(epoch: float, why: str, fh) -> None:
    epoch = max(epoch, time.time() + 60)
    log(f"waiting until {datetime.fromtimestamp(epoch):%a %H:%M} ({(epoch - time.time()) / 60:.0f} min): {why}", fh)
    while time.time() < epoch:
        time.sleep(min(300, max(1, epoch - time.time())))


def render_page(doc, index: int, path: Path) -> bytes:
    if not path.exists():
        pix = doc[index].get_pixmap(dpi=DPI)
        Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(path, quality=92)
    return path.read_bytes()


def call_opus(image: bytes, cwd: Path, model: str):
    """One page through `claude -p`. Returns (result message or None, last rate-limit info or None, stderr)."""
    line = json.dumps({"type": "user", "parent_tool_use_id": None, "message": {"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                     "data": base64.b64encode(image).decode()}},
        {"type": "text", "text": PROMPT}]}})
    env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE", "CLAUDE_CODE_CHILD_SESSION")}
    env.update(CLAUDE_CODE_DISABLE_CLAUDE_MDS="1", CLAUDE_CODE_DISABLE_AUTO_MEMORY="1")
    cmd = [CLAUDE, "-p", "--model", model, "--input-format", "stream-json", "--output-format", "stream-json",
           "--verbose", "--tools", "", "--strict-mcp-config", "--disable-slash-commands",
           "--no-session-persistence", "--system-prompt", SYSTEM_PROMPT]
    try:
        proc = subprocess.run(cmd, input=line + "\n", capture_output=True, text=True, cwd=cwd, env=env,
                              timeout=CALL_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, None, f"timed out after {CALL_TIMEOUT} s"
    result, limit = None, None
    for raw in proc.stdout.splitlines():
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if msg.get("type") == "rate_limit_event":
            limit = msg.get("rate_limit_info")
        elif msg.get("type") == "result":
            result = msg
    err = proc.stderr.strip()
    if result is None and not err:
        err = f"exit {proc.returncode}, no result message; stdout: {proc.stdout[-500:]}"
    return result, limit, err


def limit_reset(limit) -> float:
    """Reset time (epoch) of the window that is blocking, or 0 if unknown."""
    if not limit:
        return 0
    windows = limit.get("unifiedWindows") or {}
    kind = limit.get("rateLimitType")
    if kind in windows and windows[kind].get("resetsAt"):
        return float(windows[kind]["resetsAt"])
    return float(limit.get("resetsAt") or 0)


def split_reply(text: str):
    m = re.search(r"<transcript>(.*?)</transcript>", text, re.DOTALL)
    if not m:
        return None, text.strip()
    return m.group(1).strip("\n"), text[m.end():].strip()


def transcribe_page(doc, index: int, page_dir: Path, args, fh) -> bool:
    n = index + 1
    out_txt, out_json = page_dir / f"page_{n}.txt", page_dir / f"page_{n}.json"
    if out_txt.exists():
        return False
    image = render_page(doc, index, page_dir / f"page_{n}.jpg")
    failures = 0
    while True:
        t0 = time.time()
        result, limit, err = call_opus(image, page_dir, args.model)
        week = ((limit or {}).get("unifiedWindows") or {}).get("seven_day") or {}
        text = (result or {}).get("result") or ""
        ok = result is not None and not result.get("is_error") and result.get("subtype") == "success"
        transcript, notes = split_reply(text) if ok else (None, text)
        if ok and transcript is None and failures >= 2:
            transcript, notes = text.strip(), "(no <transcript> tags in the reply; whole reply kept)"
        if ok and transcript is not None:
            out_txt.write_text(transcript + "\n", encoding="utf-8")
            out_json.write_text(json.dumps({
                "model": args.model, "page": n, "dpi": DPI, "seconds": round(time.time() - t0, 1),
                "finished": datetime.now().isoformat(timespec="seconds"), "notes": notes,
                "usage": result.get("usage"), "cost_usd_list_price": result.get("total_cost_usd"),
                "rate_limit": limit}, indent=1, ensure_ascii=False), encoding="utf-8")
            if week.get("utilization") is not None and week["utilization"] >= args.weekly_cap:
                sleep_until(float(week.get("resetsAt") or 0),
                            f"seven-day usage at {week['utilization']:.0%} (cap {args.weekly_cap:.0%})", fh)
            return True
        failures += 1
        if ok:   # an answer without <transcript> tags: ask again (kept as it is after the third try)
            log(f"page {n}: no <transcript> tags in the reply ({failures}x); asking again", fh)
            continue
        why = (text or err or "")[:300].replace("\n", " ")
        if (limit and limit.get("status") == "rejected") or LIMIT_TEXT.search(why):
            reset = limit_reset(limit)
            sleep_until(reset + 60 if reset else time.time() + 900, f"usage limit ({why})", fh)
            failures = 0
            continue
        if FATAL_TEXT.search(why):
            log(f"FATAL on page {n}: {why}. Fix it (for example run `claude` and /login) and start the script again.", fh)
            sys.exit(2)
        wait = [30, 120, 300, 600, 1200][min(failures, 5) - 1]
        log(f"page {n} failed ({failures}x): {why or 'no output'}; retrying in {wait} s", fh)
        time.sleep(wait)


def assemble(sid: str, pages: int, script_dir: Path, out_dir: Path) -> None:
    blocks = []
    for n in range(1, pages + 1):
        text = (script_dir / f"page_{n}.txt").read_text(encoding="utf-8").rstrip("\n")
        blocks.append(f"--- PAGE {n} ---\n{text}\n")
    (out_dir / f"{sid}.txt").write_text("\n".join(blocks), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--papers", nargs="+", default=["SE_06_Q1", "SE_07_Q1"])
    ap.add_argument("--scripts", nargs="+", help="only these script ids")
    ap.add_argument("--pdf-root", default=str(REPO / "data/raw_pdfs/english"), help="holds <PAPER>/<script_id>.pdf")
    ap.add_argument("--out-root", default=str(REPO / "data/ground_truth/transcripts"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--weekly-cap", type=float, default=0.9,
                    help="pause when the seven-day window reaches this share (default 0.9; 1.0 = no pause)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = []   # (priority, paper, sid, pdf)
    for paper in args.papers:
        sample = {p.stem for p in (SAMPLE_DIR / paper.lower()).glob("*.pdf")}
        for pdf in sorted((Path(args.pdf_root) / paper).glob(f"{paper}_*.pdf")):
            sid = pdf.stem
            if sid in EXCLUDE or (args.scripts and sid not in args.scripts):
                continue
            jobs.append((0 if sid in sample else 1, paper, sid, pdf))
    jobs.sort(key=lambda j: (j[0], j[1], j[2]))
    total_pages = sum(pymupdf.open(j[3]).page_count for j in jobs)
    print(f"{len(jobs)} scripts, {total_pages} pages")
    if args.dry_run:
        for prio, paper, sid, pdf in jobs:
            print(f"  {'sample' if prio == 0 else 'rest  '} {sid}  {pymupdf.open(pdf).page_count} pages")
        return

    log_path = Path(args.out_root) / "opus_transcribe.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as fh:
        log(f"start: {len(jobs)} scripts, {total_pages} pages, model {args.model}, weekly cap {args.weekly_cap:.0%}", fh)
        done_pages, t_start, new_pages = 0, time.time(), 0
        for k, (prio, paper, sid, pdf) in enumerate(jobs, 1):
            out_dir = Path(args.out_root) / f"{paper}_extraction_opus_5.5"
            if (out_dir / f"{sid}.txt").exists():
                done_pages += pymupdf.open(pdf).page_count
                continue
            script_dir = out_dir / "pages" / sid
            script_dir.mkdir(parents=True, exist_ok=True)
            doc = pymupdf.open(pdf)
            for i in range(doc.page_count):
                if transcribe_page(doc, i, script_dir, args, fh):
                    new_pages += 1
                done_pages += 1
            assemble(sid, doc.page_count, script_dir, out_dir)
            rate = (time.time() - t_start) / max(new_pages, 1)
            log(f"[{k}/{len(jobs)}] {sid} done ({doc.page_count} pages); {done_pages}/{total_pages} pages in all; "
                f"{rate:.0f} s per new page including waits", fh)
        log("all scripts done", fh)


if __name__ == "__main__":
    main()
