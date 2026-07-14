"""End-to-end pipeline orchestration.

Stages per row:
  1. Erase red ink              (CV; always safe to run)
  2. Harvest red score          (LLM, original image)      -> ground-truth source
  3. Cross-check vs marks col    (deterministic)            -> GOLD / FLAG
  4. Resolve full marks          (rubric -> marks -> page)
  5. Route question text         (deterministic)
  6. Grade x2 conditions         (LLM, erased image)

Run stages 1 and (3-5) without an API key; stages 2 and 6 need one.
"""
from __future__ import annotations

import argparse
import csv

import pandas as pd

from . import config, redink, harvester, grader, ground_truth, fullmarks, normalize_bn, analyze


def load_rows(limit=None, subject=None):
    rows = list(csv.DictReader(open(config.CSV_PATH)))
    if subject:
        rows = [r for r in rows if r["subject"] == subject]
    if limit:
        rows = rows[:limit]
    return rows


def erase_all(rows):
    n = 0
    for r in rows:
        src = config.IMAGES_DIR / f"{r['id']}.jpg"
        dst = config.ERASED_DIR / f"{r['id']}.jpg"
        if src.exists() and not dst.exists():
            redink.erase_file(src, dst)
            n += 1
    return n


def _flat(v):
    """Collapse newlines/runs of whitespace so a value stays on one CSV line."""
    if isinstance(v, str):
        return " ".join(v.split())
    return v


def build_row_record(r, do_llm: bool) -> dict:
    q = normalize_bn.route_question(r)
    rec = {
        "id": r["id"], "subject": r["subject"], "marks_col": r["marks"],
        "question_source": q["source"], "question_text": q["text"],
    }

    red = None
    if do_llm:
        try:
            h = harvester.harvest(r["id"])
            red = harvester.red_score(h)
            rec["red_score"] = red
            rec["red_score_raw"] = h.get("red_score_raw", "")
            rec["full_marks_on_page"] = h.get("full_marks_on_page")
        except Exception as e:  # keep the batch alive; flag the row
            rec["error"] = f"harvest: {type(e).__name__}: {e}"

    cc = ground_truth.cross_check(red, r["marks"])
    rec.update(status=cc["status"], gold_score=cc["gold_score"],
               token_score=cc["token_score"], matched=cc["matched"])

    fm = fullmarks.resolve_full_marks(r, q["text"])
    full = fm["full_marks"]
    if full is None and do_llm:
        page = rec.get("full_marks_on_page")
        if isinstance(page, (int, float)) and page > 0:
            full, fm["source"] = float(page), "image"

    # Consistency guard: the awarded red score can never exceed the total. If it
    # does, the denominator is wrong (corrupt marks token / bad rubric parse), so
    # the row is unusable for normalized metrics and needs a human look.
    gold = rec.get("gold_score")
    reliable = not (full is not None and gold is not None and gold > full + 0.01)
    if not reliable:
        fm["source"] = "inconsistent(gold>full)"
    rec["full_marks"] = full
    rec["full_marks_source"] = fm["source"]
    rec["full_marks_reliable"] = reliable
    rec["gold_usable"] = (gold is not None) and reliable and (full is not None)

    if do_llm:
        for v in grader.VARIANTS:
            try:
                g = grader.grade(r, q["text"], full, v)
                rec[f"score_{v}"] = g.get("score")
                rec[f"transcription_{v}"] = g.get("transcription", "")
                rec[f"confidence_{v}"] = g.get("confidence")
                crit = g.get("criteria") or []
                rec[f"reasoning_{v}"] = " | ".join(
                    f"{c.get('criterion','')}: {c.get('awarded')}/{c.get('max')} {c.get('reason','')}"
                    for c in crit
                )
            except Exception as e:  # keep the batch alive; flag the row
                rec[f"score_{v}"] = None
                rec["error"] = f"{rec.get('error', '')} grade[{v}]: {type(e).__name__}: {e}".strip()
    return {k: _flat(v) for k, v in rec.items()}


def merge_grades(batch: pd.DataFrame) -> pd.DataFrame:
    """Union the current batch with the existing grades.csv, batch winning by id."""
    if not config.GRADES_CSV.exists():
        return batch
    old = pd.read_csv(config.GRADES_CSV, dtype={"id": str})
    keep = old[~old["id"].isin(batch["id"])]
    return pd.concat([keep, batch], ignore_index=True)


def run(limit=None, subject=None, do_llm=True):
    rows = load_rows(limit, subject)
    print(f"[1/6] erasing red ink on {len(rows)} scripts ...")
    print(f"      erased {erase_all(rows)} new images")

    records = []
    for i, r in enumerate(rows, 1):
        records.append(build_row_record(r, do_llm))
        if i % 25 == 0:
            print(f"      processed {i}/{len(rows)}")

    batch = pd.DataFrame(records)
    batch["id"] = batch["id"].astype(str)

    # Accumulate across runs: merge this batch into any existing grades.csv,
    # keyed by id (the current batch wins for rows it re-processed).
    df = merge_grades(batch)
    df.to_csv(config.GRADES_CSV, index=False)
    print(f"[done] processed {len(batch)} rows this run; "
          f"grades.csv now holds {len(df)} total rows → {config.GRADES_CSV}")

    if "error" in df:
        errs = df[df["error"].notna()]
        if len(errs):
            print(f"\n[warn] {len(errs)} row(s) failed and were skipped "
                  f"(re-run the same command to retry them from where they stopped):")
            for _, e in errs.iterrows():
                print(f"       {e['id']}: {e['error'][:120]}")

    # Per-subject grade files (derived from the full accumulated set).
    write_subject_files(df, do_llm)

    # Ground-truth / cross-check summary
    gerr = analyze.gold_label_error_rate(df)
    print("\n=== gold-label cross-check ===")
    print(gerr)

    if do_llm:
        usable = int(df["gold_usable"].sum()) if "gold_usable" in df else 0
        blank = int((df["status"] == "NO_REDMARK").sum())
        print(f"\ngradeable rows: {usable}/{len(df)} "
              f"({blank} have no red mark — blank/unmarked, excluded from metrics)")
        for v in grader.VARIANTS:
            print(f"\n=== metrics ({v}) ===")
            m = analyze.metrics_by_subject(df, v)
            if m.empty:
                print("  (no gradeable rows in this batch)")
            else:
                print(m.to_string(index=False))
            analyze.disagreement_table(df, v).to_csv(
                config.OUTPUTS_DIR / f"disagreements_{v}.csv", index=False)
        # combined metrics file
        allm = pd.concat([analyze.metrics_by_subject(df, v) for v in grader.VARIANTS])
        allm.to_csv(config.METRICS_CSV, index=False)
        print(f"\n[done] wrote {config.METRICS_CSV}, per-subject grades and "
              f"disagreement tables → {config.BY_SUBJECT_DIR}")

    return df


def _slug(subject: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(subject)).strip("_")


def write_subject_files(df: pd.DataFrame, do_llm: bool) -> None:
    """One grades CSV per subject (+ per-subject disagreement tables)."""
    for subject, sub in df.groupby("subject"):
        slug = _slug(subject)
        sub.to_csv(config.BY_SUBJECT_DIR / f"grades_{slug}.csv", index=False)
        if do_llm:
            for v in grader.VARIANTS:
                dt = analyze.disagreement_table(sub, v)
                dt.to_csv(config.BY_SUBJECT_DIR / f"disagreements_{slug}_{v}.csv",
                          index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--subject", default=None)
    ap.add_argument("--no-llm", action="store_true",
                    help="run CV + deterministic stages only (no API key needed)")
    args = ap.parse_args()
    run(limit=args.limit, subject=args.subject, do_llm=not args.no_llm)
