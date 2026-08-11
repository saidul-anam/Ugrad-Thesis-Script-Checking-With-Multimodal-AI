"""Stage 3: backend comparison report -> reports/backend_comparison.md

Read-only over the stored transcripts: nothing here modifies, corrects, or
normalises any transcript file. The spell-check is a count, not a fixer.

Usage:
    python compare.py
"""
from __future__ import annotations

import difflib
import re
import sys
from collections import defaultdict
from pathlib import Path

from spellchecker import SpellChecker

from common import (
    REPORTS_DIR,
    TRANSCRIPTS_DIR,
    read_run_log,
    utc_now,
)

BACKENDS = ["gemini", "mistral_ocr"]

# Pricing as of 2026-08-11 (sources in reports/backend_comparison.md).
# Per 1M tokens for Gemini (<=200k-token requests); per page for Mistral OCR.
GEMINI_PRICING = {  # model-version prefix -> ($/1M input, $/1M output incl. thinking)
    "gemini-3.6-flash": (1.50, 7.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
}
MISTRAL_OCR_PER_PAGE = 0.004  # $4 / 1000 pages (OCR 4, synchronous API)

FULL_RUN_SCRIPTS = 200

MARKERS = {
    "illegible": "[illegible",   # prefix: covers "[illegible]" and "[illegible: ~N words]"
    "struck": "[struck:",
    "cut": "[cut:",
}

# Red-ink leakage heuristics: examiner-ish content on short lines, mark
# fractions anywhere, or lone numbers. Deliberately over-flags; a human reads
# every flagged line against the scan.
LEAK_PATTERNS = [
    (re.compile(r"\b\d{1,2}\s*/\s*\d{1,2}\b"), "mark fraction (e.g. 6/10)"),
    (re.compile(r"^\s*\d{1,3}\s*$"), "lone number line"),
    (re.compile(r"[✓✔✗✘×☑]"), "tick/cross symbol"),
    (re.compile(r"(?i)^\W*(v\.?\s*)?(good|excellent|poor|weak|wrong|fine)\b\W*$"), "examiner comment word"),
    (re.compile(r"(?i)\b(marks?|total|spelling|grammar)\b"), "examiner vocabulary", 6),
    (re.compile(r"(?i)cam\s?scanner|^\s*CS\s*$"), "CamScanner watermark"),
]


def count_markers(text: str) -> dict:
    return {k: text.count(pat) for k, pat in MARKERS.items()}


def leak_lines(text: str) -> list[tuple[str, str]]:
    out = []
    for line in text.splitlines():
        for pat in LEAK_PATTERNS:
            rx, label, *maxwords = pat
            if maxwords and len(line.split()) > maxwords[0]:
                continue
            if rx.search(line):
                out.append((label, line.strip()))
                break
    return out


WORD_RE = re.compile(r"[A-Za-z']+")


def spell_unknown_count(text: str, checker: SpellChecker) -> tuple[int, int, list[str]]:
    """(non-dictionary word count, total word count, sample). Markers like
    [illegible] are excluded from the word list before counting."""
    cleaned = re.sub(r"\[(illegible|struck|cut)[^\]]*\]", " ", text)
    words = [w for w in WORD_RE.findall(cleaned) if len(w) > 1]
    unknown = [w for w in words if w.lower() in checker.unknown([w.lower()])]
    return len(unknown), len(words), unknown


def word_diff(a: str, b: str, context: int = 3) -> str:
    """Compact word-level diff of a (gemini) vs b (mistral_ocr)."""
    aw, bw = a.split(), b.split()
    sm = difflib.SequenceMatcher(a=aw, b=bw, autojunk=False)
    parts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            seg = aw[i1:i2]
            if len(seg) > 2 * context + 1:
                parts.append(" ".join(seg[:context]) + " … " + " ".join(seg[-context:]))
            else:
                parts.append(" ".join(seg))
        elif tag == "replace":
            parts.append(f"**[G: {' '.join(aw[i1:i2])} | M: {' '.join(bw[j1:j2])}]**")
        elif tag == "delete":
            parts.append(f"**[G only: {' '.join(aw[i1:i2])}]**")
        elif tag == "insert":
            parts.append(f"**[M only: {' '.join(bw[j1:j2])}]**")
    return " ".join(parts)


def diff_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=a.split(), b=b.split(), autojunk=False).ratio()


def cost_for_gemini(records: list[dict]) -> tuple[float, dict, list[str]]:
    total = 0.0
    tokens = defaultdict(int)
    notes = []
    for r in records:
        u = r.get("usage") or {}
        inp = u.get("prompt_tokens") or 0
        out = (u.get("output_tokens") or 0) + (u.get("thoughts_tokens") or 0)
        tokens["input"] += inp
        tokens["output_incl_thinking"] += out
        version = r.get("backend_version", "")
        price = next((p for k, p in GEMINI_PRICING.items() if version.startswith(k)), None)
        if price is None:
            notes.append(f"no pricing entry for model {version!r}; cost omitted for that call")
            continue
        total += inp / 1e6 * price[0] + out / 1e6 * price[1]
    return total, dict(tokens), sorted(set(notes))


def cost_for_mistral(records: list[dict]) -> tuple[float, dict, list[str]]:
    pages = sum((r.get("usage") or {}).get("pages_processed") or 0 for r in records)
    return pages * MISTRAL_OCR_PER_PAGE, {"pages_processed": pages}, []


def main() -> int:
    REPORTS_DIR.mkdir(exist_ok=True)
    checker = SpellChecker()

    units: dict[str, dict[str, str]] = defaultdict(dict)  # unit -> backend -> text
    for b in BACKENDS:
        d = TRANSCRIPTS_DIR / b
        if d.is_dir():
            for f in sorted(d.glob("*.txt")):
                units[f.stem][b] = f.read_text()

    run_logs = {b: read_run_log(b) for b in BACKENDS}
    ok_calls = {b: [r for r in run_logs[b] if not r.get("error")] for b in BACKENDS}

    lines: list[str] = []
    w = lines.append
    w(f"# Backend comparison report\n")
    w(f"Generated: {utc_now()}  ")
    w(f"Backends: {', '.join(BACKENDS)}\n")

    # ---- per-backend overview -------------------------------------------------
    w("## Overview\n")
    w("| metric | " + " | ".join(BACKENDS) + " |")
    w("|---|" + "---|" * len(BACKENDS))

    versions = {b: (ok_calls[b][0]["backend_version"] if ok_calls[b] else "—") for b in BACKENDS}
    w("| model version | " + " | ".join(versions[b] for b in BACKENDS) + " |")
    supports = {b: (str(ok_calls[b][0]["settings"].get("supports_prompt")) if ok_calls[b] else "—") for b in BACKENDS}
    w("| supports_prompt | " + " | ".join(supports[b] for b in BACKENDS) + " |")
    w("| pages transcribed | " + " | ".join(str(len(ok_calls[b])) for b in BACKENDS) + " |")
    failures = {b: len(run_logs[b]) - len(ok_calls[b]) for b in BACKENDS}
    w("| failed calls (incl. retries) | " + " | ".join(str(failures[b]) for b in BACKENDS) + " |")
    n_units = {b: sum(1 for u in units.values() if b in u) for b in BACKENDS}
    w("| task units segmented | " + " | ".join(str(n_units[b]) for b in BACKENDS) + " |")

    # wall-clock per script
    per_script: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for b in BACKENDS:
        for r in run_logs[b]:
            per_script[b][r["script_id"]] += r.get("duration_s") or 0
    for b in BACKENDS:
        if per_script[b]:
            detail = ", ".join(f"{s}: {t:.0f}s" for s, t in sorted(per_script[b].items()))
            w(f"| wall-clock ({b}) | " + f"{detail} |".ljust(1))
    w("")

    # ---- markers --------------------------------------------------------------
    w("## Fidelity markers per backend\n")
    w("A backend emitting zero markers on handwritten exam scripts is a signal,")
    w("not a success: it means unclear text was silently guessed or dropped.\n")
    w("| unit | " + " | ".join(f"{b} illegible/struck/cut" for b in BACKENDS) + " |")
    w("|---|" + "---|" * len(BACKENDS))
    marker_totals = {b: defaultdict(int) for b in BACKENDS}
    for u in sorted(units):
        cells = []
        for b in BACKENDS:
            if b in units[u]:
                c = count_markers(units[u][b])
                for k, v in c.items():
                    marker_totals[b][k] += v
                cells.append(f"{c['illegible']} / {c['struck']} / {c['cut']}")
            else:
                cells.append("—")
        w(f"| {u} | " + " | ".join(cells) + " |")
    w("| **total** | " + " | ".join(
        f"**{marker_totals[b]['illegible']} / {marker_totals[b]['struck']} / {marker_totals[b]['cut']}**"
        for b in BACKENDS) + " |")
    w("")

    # ---- spelling normalisation check ----------------------------------------
    w("## Spelling-normalisation check (read-only)\n")
    w("Count of non-dictionary words per transcript (pyspellchecker, en). These")
    w("students make frequent spelling errors, so the backend with MORE")
    w("non-dictionary words is likely the more faithful one. Nothing is corrected.\n")
    w("| unit | " + " | ".join(f"{b} non-dict/total" for b in BACKENDS) + " |")
    w("|---|" + "---|" * len(BACKENDS))
    spell_totals = {b: [0, 0] for b in BACKENDS}
    samples = {b: [] for b in BACKENDS}
    for u in sorted(units):
        cells = []
        for b in BACKENDS:
            if b in units[u]:
                n, tot, unk = spell_unknown_count(units[u][b], checker)
                spell_totals[b][0] += n
                spell_totals[b][1] += tot
                samples[b].extend(unk)
                cells.append(f"{n} / {tot}")
            else:
                cells.append("—")
        w(f"| {u} | " + " | ".join(cells) + " |")
    w("| **total** | " + " | ".join(f"**{spell_totals[b][0]} / {spell_totals[b][1]}**" for b in BACKENDS) + " |")
    w("")
    for b in BACKENDS:
        if samples[b]:
            uniq = sorted(set(w_.lower() for w_ in samples[b]))
            w(f"Non-dictionary words seen in **{b}**: {', '.join(uniq[:60])}"
              + (" …" if len(uniq) > 60 else ""))
            w("")

    # ---- red-ink leakage ------------------------------------------------------
    w("## Red-ink / annotation leakage check\n")
    w("Heuristically flagged lines that look like examiner annotation or scanner")
    w("watermark. Over-flagging is intentional — read each against the scan.\n")
    any_flag = False
    for u in sorted(units):
        for b in BACKENDS:
            if b not in units[u]:
                continue
            flags = leak_lines(units[u][b])
            if flags:
                any_flag = True
                w(f"**{u} — {b}:**")
                for label, line in flags:
                    w(f"- `{line}`  ← {label}")
                w("")
    if not any_flag:
        w("No lines flagged.\n")

    # ---- per-unit side-by-side + diff ----------------------------------------
    w("## Task units: side-by-side and word-level diff\n")
    for u in sorted(units):
        w(f"### {u}\n")
        have = [b for b in BACKENDS if b in units[u]]
        if len(have) == 1:
            w(f"*Only transcribed by {have[0]} — no comparison possible.*\n")
        else:
            w(f"Word-level similarity: {diff_ratio(units[u][BACKENDS[0]], units[u][BACKENDS[1]]):.3f}\n")
        for b in have:
            w(f"<details><summary><b>{b}</b> full text</summary>\n")
            w("```text")
            w(units[u][b].rstrip("\n"))
            w("```")
            w("</details>\n")
        if len(have) == 2:
            w("**Word diff (G = gemini, M = mistral_ocr):**\n")
            w(word_diff(units[u][BACKENDS[0]], units[u][BACKENDS[1]]))
            w("")

    # ---- cost -----------------------------------------------------------------
    w("## Cost, tokens, and projection\n")
    g_cost, g_tokens, g_notes = cost_for_gemini(ok_calls["gemini"])
    m_cost, m_tokens, _ = cost_for_mistral(ok_calls["mistral_ocr"])
    pilot_pages = {b: len(ok_calls[b]) for b in BACKENDS}
    pilot_scripts = {b: len(per_script[b]) for b in BACKENDS}
    w("| | gemini | mistral_ocr |")
    w("|---|---|---|")
    w(f"| pilot usage | {g_tokens or '—'} | {m_tokens or '—'} |")
    w(f"| pilot cost | ${g_cost:.4f} | ${m_cost:.4f} |")
    proj = {}
    for b, cost in (("gemini", g_cost), ("mistral_ocr", m_cost)):
        if pilot_scripts[b]:
            proj[b] = cost / pilot_scripts[b] * FULL_RUN_SCRIPTS
    w("| projected cost, "
      f"{FULL_RUN_SCRIPTS} scripts | "
      + " | ".join((f"${proj[b]:.2f}" if b in proj else "—") for b in BACKENDS) + " |")
    avg_cells = []
    for b in BACKENDS:
        if per_script[b]:
            avg_cells.append(f"{sum(per_script[b].values()) / len(per_script[b]):.0f}s")
        else:
            avg_cells.append("—")
    w("| avg wall-clock / script | " + " | ".join(avg_cells) + " |")
    w("")
    for n in g_notes:
        w(f"> NOTE: {n}")
    w("> Pricing used (2026-08-11): Gemini 3.6 Flash $1.50/M input, $7.50/M output,")
    w("> thinking tokens billed at the output rate (batch API is half);")
    w("> Mistral OCR 4 $4.00/1000 pages (synchronous; batch API is half).")
    w("")

    out = REPORTS_DIR / "backend_comparison.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out} ({len(lines)} lines, {len(units)} task units)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
