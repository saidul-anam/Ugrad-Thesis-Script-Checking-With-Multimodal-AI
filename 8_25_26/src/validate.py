"""Stage 4 — validation. Report, don't fix.

Reads every transcript in data/transcripts/ and flags scripts for human review.
Nothing here edits a transcript; the spell-check is measurement only, used for
the peer error-rate signal, and never writes back to the data.

Flags:
  low_confidence            any answer confidence < 0.7
  high_uncertainty          any answer illegible_count + unclear_count > 5
  missing_target_question   any of the target questions absent from the script
  ambiguous_authorship      any answer with non-empty ambiguous_authorship
  suspected_red_ink         isolated single-word segments (absorbed corrections),
                            or an implausibly low error rate vs peers
  wrap_as_paragraph         Q7 with >3 blank-line breaks or mean segment < 15 words
  duplicate_overlap         two questions with near-identical transcripts
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from src.config import Config

# --- thresholds (fixed once; documented) -----------------------------------
CONFIDENCE_MIN = 0.7
UNCERTAINTY_MAX = 5          # illegible + unclear
PARAGRAPH_BREAK_MAX = 3      # Q7 \n\n count
MEAN_SEGMENT_WORDS_MIN = 15  # Q7 mean words per \n\n segment
DUPLICATE_RATIO = 0.85       # difflib ratio between two transcripts
PARAGRAPH_QUESTIONS = {"7"}  # paragraph-type answers
LETTER_QUESTIONS = {"10"}    # letters legitimately have single-word lines
MAX_QUESTION = 11            # exam has questions 1..11; a numeric qno outside
                             # this range is a misread/split (e.g. '11' read '14')
# Absorbed-correction (isolated word) check runs only on continuous-prose target
# questions; short answers (Q1-6) and letters (Q10) legitimately have lone words.

# peer error-rate: flag a script whose misspelling rate is far below peers that
# themselves have plenty of errors (a sign examiner corrections were absorbed).
PEER_MIN_MEAN_RATE = 0.05    # peers must average >5% misspellings to compare
LOW_RATE_FRACTION = 0.4      # flag if script rate < 40% of the peer mean

# marker syntax stripped ONLY for measurement (word_count / spell-check).
_MARKER_RE = re.compile(
    r"\[illegible\]|\[unclear:\s*|\[cut:\s*|\{inserted:\s*|~~|\]|\}"
)


@dataclass
class ScriptFlags:
    script_id: str
    answers_found: int
    missing_targets: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)          # short flag names
    details: list[str] = field(default_factory=list)        # human-readable notes
    misspell_rate: float | None = None

    @property
    def needs_review(self) -> bool:
        return bool(self.flags)


def _strip_markers(text: str) -> str:
    """Remove editorial-marker syntax, keeping the words inside. Measurement only."""
    return _MARKER_RE.sub(" ", text)


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+", _strip_markers(text))


def _load_transcripts(cfg: Config) -> list[dict]:
    out = []
    for path in sorted(cfg.paths.transcripts.glob("*.json")):
        out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


def _misspell_rate(transcripts_text: str, checker) -> float | None:
    if checker is None:
        return None
    tokens = [t.lower() for t in _word_tokens(transcripts_text)]
    if len(tokens) < 20:  # too short to be meaningful
        return None
    unknown = checker.unknown(tokens)
    return len(unknown) / len(tokens)


def _qsort(q: str):
    """Sort key: numeric questions in numeric order, others after, alphabetically."""
    return (0, int(q)) if q.isdigit() else (1, q)


def normalize_qno(qno: str) -> str:
    """Question numbers are keys, not student text — normalize for matching.

    Zero-padded numerics collapse ('07' -> '7', '03' -> '3'); non-numeric forms
    ('1(A)', '14') are left as-is. This never touches transcript content.
    """
    q = str(qno).strip()
    return str(int(q)) if q.isdigit() else q


def _isolated_words(text: str) -> list[str]:
    """Single-word lines — student prose joins wraps with spaces, so a lone word
    is a likely absorbed examiner correction."""
    words = []
    for line in text.splitlines():
        seg = line.strip()
        if seg and re.fullmatch(r"[A-Za-z]+", seg):
            words.append(seg)
    return words


def evaluate_script(
    data: dict,
    target_questions: list[str],
    checker,
) -> ScriptFlags:
    answers = data.get("answers", []) or []
    script_id = data.get("_metadata", {}).get("script_id") or data.get("script_id", "?")
    sf = ScriptFlags(script_id=script_id, answers_found=len(answers))

    target_set = {normalize_qno(q) for q in target_questions}
    prose_targets = target_set - LETTER_QUESTIONS
    present = {normalize_qno(a.get("question_number")) for a in answers}

    # missing target questions
    missing = [q for q in sorted(target_set, key=_qsort) if q not in present]
    if missing:
        sf.missing_targets = missing
        sf.flags.append("missing_target_question")
        sf.details.append(f"missing targets: {', '.join(missing)}")

    for a in answers:
        qno = normalize_qno(a.get("question_number"))
        # numeric question number outside the exam's range -> misread / split
        if qno.isdigit() and not (1 <= int(qno) <= MAX_QUESTION):
            _add(sf, "unexpected_question_number", f"Q{qno} outside 1-{MAX_QUESTION}")
        conf = float(a.get("confidence", 0.0))
        if conf < CONFIDENCE_MIN:
            _add(sf, "low_confidence", f"Q{qno} confidence {conf:.2f}")
        unc = int(a.get("illegible_count", 0)) + int(a.get("unclear_count", 0))
        if unc > UNCERTAINTY_MAX:
            _add(sf, "high_uncertainty", f"Q{qno} illegible+unclear={unc}")
        if a.get("ambiguous_authorship"):
            _add(
                sf, "ambiguous_authorship",
                f"Q{qno}: {a['ambiguous_authorship']}",
            )

        transcript = a.get("transcript", "")
        # Absorbed-correction check: continuous-prose target questions only.
        if qno in prose_targets:
            iso = _isolated_words(transcript)
            if iso:
                _add(sf, "suspected_red_ink", f"Q{qno} isolated words: {iso}")

        # wrap-as-paragraph, paragraph-type questions only
        if qno in PARAGRAPH_QUESTIONS:
            breaks = transcript.count("\n\n")
            segments = [s for s in transcript.split("\n\n") if s.strip()]
            mean_words = (
                statistics.mean(len(_word_tokens(s)) for s in segments)
                if segments else 0.0
            )
            if breaks > PARAGRAPH_BREAK_MAX:
                _add(sf, "wrap_as_paragraph", f"Q{qno} has {breaks} blank-line breaks")
            if segments and mean_words < MEAN_SEGMENT_WORDS_MIN:
                _add(
                    sf, "wrap_as_paragraph",
                    f"Q{qno} mean segment {mean_words:.1f} words",
                )

    # duplicate / overlapping transcripts across questions
    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            ti = answers[i].get("transcript", "")
            tj = answers[j].get("transcript", "")
            if not ti or not tj:
                continue
            ratio = SequenceMatcher(None, ti, tj).ratio()
            if ratio >= DUPLICATE_RATIO:
                qi = answers[i].get("question_number")
                qj = answers[j].get("question_number")
                _add(
                    sf, "duplicate_overlap",
                    f"Q{qi} ~ Q{qj} similarity {ratio:.2f}",
                )

    # per-script misspelling rate (peer comparison added by caller)
    all_text = " ".join(a.get("transcript", "") for a in answers)
    sf.misspell_rate = _misspell_rate(all_text, checker)
    return sf


def _add(sf: ScriptFlags, flag: str, detail: str) -> None:
    if flag not in sf.flags:
        sf.flags.append(flag)
    sf.details.append(detail)


def _apply_peer_error_rate(results: list[ScriptFlags]) -> None:
    """Flag scripts whose error rate is implausibly low relative to peers."""
    rates = [r.misspell_rate for r in results if r.misspell_rate is not None]
    if len(rates) < 3:
        return  # not enough peers to compare
    peer_mean = statistics.mean(rates)
    if peer_mean < PEER_MIN_MEAN_RATE:
        return  # peers aren't error-prone; comparison is uninformative
    for r in results:
        if r.misspell_rate is None:
            continue
        if r.misspell_rate < LOW_RATE_FRACTION * peer_mean:
            _add(
                r, "suspected_red_ink",
                f"error rate {r.misspell_rate:.1%} << peer mean {peer_mean:.1%}",
            )


def build_report(results: list[ScriptFlags], target_questions: list[str]) -> str:
    lines: list[str] = []
    lines.append("# Validation Report\n")
    lines.append(f"Scripts evaluated: **{len(results)}**  ")
    review = [r for r in results if r.needs_review]
    lines.append(f"Needing review: **{len(review)}**\n")

    lines.append("| script_id | answers | missing targets | error rate | flags |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        rate = f"{r.misspell_rate:.1%}" if r.misspell_rate is not None else "n/a"
        flags = "; ".join(r.flags) if r.flags else "-"
        missing = ", ".join(r.missing_targets) if r.missing_targets else "-"
        lines.append(
            f"| {r.script_id} | {r.answers_found} | {missing} | {rate} | {flags} |"
        )

    # flag totals
    lines.append("\n## Flag totals\n")
    totals: dict[str, int] = {}
    for r in results:
        for f in r.flags:
            totals[f] = totals.get(f, 0) + 1
    if totals:
        lines.append("| flag | scripts |")
        lines.append("|---|---|")
        for f, n in sorted(totals.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {f} | {n} |")
    else:
        lines.append("_No flags raised._")

    # detail section
    lines.append("\n## Details\n")
    for r in results:
        if not r.details:
            continue
        lines.append(f"### {r.script_id}")
        for d in r.details:
            lines.append(f"- {d}")
        lines.append("")

    lines.append(f"\n_Target questions: {', '.join(target_questions)}_")
    return "\n".join(lines) + "\n"


def validate_all(cfg: Config) -> tuple[list[ScriptFlags], Path]:
    try:
        from spellchecker import SpellChecker

        checker = SpellChecker()
    except ImportError:
        checker = None

    targets = [str(q) for q in cfg.target_questions]
    transcripts = _load_transcripts(cfg)
    results = [evaluate_script(d, targets, checker) for d in transcripts]
    _apply_peer_error_rate(results)

    report = build_report(results, targets)
    report_path = cfg.paths.logs / "validation_report.md"
    report_path.write_text(report, encoding="utf-8")
    return results, report_path
