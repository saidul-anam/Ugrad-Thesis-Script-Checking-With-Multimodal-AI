import json
import re
import difflib
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Set
from PIL import Image
from src.engine.base_engine import BaseVLMEngine
from src.core.schemas import Stage2VerificationResult, AutocorrectionDiffItem, Stage2PatchItem
from src.prompts.stage2_verification import build_stage2_prompt, STAGE2_SYSTEM_PROMPT
from src.pipeline.allograph_calibrator import check_cursive_topology, CULTURAL_TERMS, FORBIDDEN_ALLOGRAPH_TARGETS
from src.pipeline.split_token_stitcher import stitch_pen_lift_splits
from src.utils.linguistic_sanitizer import get_english_lexicon


def sanitize_latex_json(text: str) -> str:
    """Ensure LaTeX commands like \\rightarrow don't get unescaped into control characters like \\r."""
    for cmd in ["rightarrow", "times", "frac", "bullet"]:
        text = re.sub(rf'(?<!\\)\\{cmd}', rf'\\\\{cmd}', text)
    return text


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from string even if surrounded by markdown code blocks."""
    text = sanitize_latex_json(text.strip())
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace + 1])
        except Exception:
            pass

    # Resilient fallback: attempt to repair truncated JSON by balancing braces
    if first_brace != -1:
        candidate = text[first_brace:].strip()
        candidate = re.sub(r',\s*"[^"]*"?\s*:\s*[^,}\]]*$', '', candidate)
        candidate = re.sub(r',\s*"[^"]*"?\s*$', '', candidate)
        candidate = re.sub(r',\s*$', '', candidate)
        open_curlies = candidate.count("{") - candidate.count("}")
        open_squares = candidate.count("[") - candidate.count("]")
        repaired = candidate + ("]" * max(0, open_squares)) + ("}" * max(0, open_curlies))
        try:
            return json.loads(repaired)
        except Exception:
            pass

    return None


# Protected closed-class English function words & high-frequency verbs
# Blocks VLM priming hallucinations mutating common function words into rare nouns or homoglyphs (e.g. 'have' -> 'hare')
# and prevents autoregressive transcript drift (e.g. 'see' -> 'the', 'the' -> 'the the')
PROTECTED_FUNCTION_WORDS: Set[str] = {
    # Auxiliaries & Modals
    "have", "has", "had", "having",
    "do", "does", "did", "doing", "done",
    "would", "could", "should", "will", "shall", "can", "may", "might", "must",
    "is", "are", "was", "were", "been", "being", "am", "be",
    # Common High-Frequency Content Verbs (Drift Shield)
    "see", "saw", "seen", "seeing",
    "go", "goes", "went", "gone", "going",
    "come", "comes", "came", "coming",
    "make", "makes", "made", "making",
    "take", "takes", "took", "taken", "taking",
    "give", "gives", "gave", "given", "giving",
    "get", "gets", "got", "getting",
    "know", "knows", "knew", "known",
    "feel", "feels", "felt", "feeling",
    "find", "finds", "found", "finding",
    # Prepositions & Particles
    "with", "from", "into", "onto", "about", "above", "after", "before",
    "under", "between", "through", "during", "without", "upon", "toward", "towards",
    "over", "against", "among", "across", "along", "behind", "beyond", "within",
    "to", "of", "in", "on", "at", "by", "for", "off", "out",
    # Pronouns & Determiners & Conjunctions
    "that", "this", "these", "those",
    "their", "there", "they", "them",
    "which", "where", "when", "what", "who", "whom", "whose", "why", "how",
    "some", "any", "every", "each", "both", "all", "such", "only", "other", "another",
    "then", "than", "and", "or", "but", "nor", "so", "yet",
    "the", "a", "an",
    "he", "she", "it", "we", "you", "i", "me", "him", "her", "us", "my", "his"
}
 
# Protected exam navigation headers, question numbers, and standard abbreviations (Problem 5)
PROTECTED_EXAM_TOKENS: Set[str] = {
    "ans", "ans:", "answer", "answer:", "q", "q.", "q:", "no", "no.", "no:",
    "question", "question:", "part", "part:", "sec", "sec:", "section", "section:"
}


def is_spurious_reversion(stage1_word: str, actual_word: str) -> bool:
    """
    Detect whether a proposed Stage 2 reversion is an invalid or spurious mutation.
    Blocks mutations of protected function words, common verbs, and exam navigation headers (Ans:, Q.),
    as well as cursive ligature hallucinations, minim doubling, and flourish additions.
    """
    s1 = stage1_word.strip().lower()
    act = actual_word.strip().lower()
    if not s1 or not act or s1 == act:
        return True

    # Tag additions like [struck: ...] or [unclear: ...] or [truncated] are valid auditing actions
    if act.startswith("[struck:") or act.startswith("[unclear:") or act.startswith("[truncated"):
        return False

    # 0. Function Word & Common Verb Shield:
    # High-frequency closed-class words and common verbs must NEVER mutate into different words (e.g. 'have' -> 'hare', 'see' -> 'the')
    s1_clean = re.sub(r'^[^\w]+|[^\w]+$', '', s1).strip().lower()
    if s1 in PROTECTED_FUNCTION_WORDS or s1_clean in PROTECTED_FUNCTION_WORDS:
        return True

    # Exam header / abbreviation protection (Problem 5):
    # Never mutate structural exam navigation labels into student words (e.g. 'Ans:' -> 'Ann:', 'Ans to the' -> 'Ann to the')
    if (
        s1 in PROTECTED_EXAM_TOKENS
        or s1_clean in PROTECTED_EXAM_TOKENS
        or s1.startswith("ans to")
        or s1.startswith("ans:")
        or s1.startswith("q.")
        or s1.startswith("q:")
    ):
        return True

    # 1. Multi-token phrase alignment (e.g. 'it remind' vs 'ut remind')
    s1_tokens = s1.split()
    act_tokens = act.split()
    if len(s1_tokens) > 1 and len(s1_tokens) == len(act_tokens):
        diff_tokens = [(w1, w2) for w1, w2 in zip(s1_tokens, act_tokens) if w1 != w2]
        if diff_tokens and all(is_spurious_reversion(w1, w2) for w1, w2 in diff_tokens):
            return True

    # 2. Generalized Vowel or Consonant duplication (e.g. 'work' -> 'woork', 'from' -> 'froom')
    for v in ["o", "e", "a", "u", "i", "l", "t"]:
        if v in s1 and s1.replace(v, v + v, 1) == act:
            return True

    # 3. Information-theoretic character similarity floor:
    # A genuine transcription verification edit cannot have arbitrary lexical divergence (< 50% similarity).
    if len(s1_tokens) == 1 and len(act_tokens) == 1 and not act.startswith("["):
        if difflib.SequenceMatcher(None, s1, act).ratio() < 0.50:
            return True

    return False


HEADER_PREFIX_RE = re.compile(
    r'^(?:Ans(?:\s*to|\s*\:)|\bAnswer\b|\bQuestion\b|\bQ[\s\.\:\-]|Part\b|\([a-z0-9]+\)|\d+[\.\)])',
    re.IGNORECASE
)


def is_header_strikethrough(stage1_target: str, replacement: str) -> bool:
    """Detect if a proposed patch attempts to wrap question headings/numbering in [struck: ...]."""
    if not replacement.startswith("[struck:"):
        return False
    inner = replacement[len("[struck:"): -1].strip()
    s1_clean = stage1_target.strip().lower()
    if HEADER_PREFIX_RE.search(s1_clean) or HEADER_PREFIX_RE.search(inner):
        return True
    if "question no" in s1_clean or "ans to" in s1_clean or s1_clean.startswith("ans"):
        return True
    return False


def is_student_error_autocorrection(
    stage1_target: str,
    replacement: str,
    combined_vocab: Set[str]
) -> bool:
    """
    Detect if a patch attempts to autocorrect an authentic student phonetic/lexical misspelling
    (e.g. desenibe -> describe, renny -> really, Pull-time -> Full-time, poverly -> poverty, parunts -> parents)
    into a standard dictionary word without physical cursive topology justification.
    """
    if replacement.startswith("[") or not replacement:
        return False

    s1_clean = re.sub(r'^[^\w]+|[^\w]+$', '', stage1_target).strip().lower()
    act_clean = re.sub(r'^[^\w]+|[^\w]+$', '', replacement).strip().lower()

    if not s1_clean or not act_clean or s1_clean == act_clean:
        return False

    # Only applies when Stage 1 had an out-of-vocabulary word and replacement is in vocabulary
    if s1_clean in combined_vocab or act_clean not in combined_vocab:
        return False

    # Allowed transformations:
    # 1. Physical cursive stroke topology match (e.g. curriter -> writer, remore -> remove)
    topo_res = check_cursive_topology(s1_clean, combined_vocab)
    if topo_res is not None and topo_res[1].lower() == act_clean:
        return False

    # 2. Pen-lift split stitching (e.g. 'Pen haps' -> 'Perhaps', 'resu lt' -> 'result')
    if " " in stage1_target and " " not in replacement and stage1_target.replace(" ", "").lower() == replacement.lower():
        return False

    # 3. Hyphenation/newline cleanup (e.g. 'oppor-\ntunities.' -> 'oppor-tunities.')
    if stage1_target.replace("\n", "").replace("-", "").strip().lower() == replacement.replace("-", "").strip().lower():
        return False

    # 4. Digit sequence / question item numbering (e.g. '(iii) blockade.' -> '(iv) blockade.')
    if re.search(r'\([ivx0-9]+\)', stage1_target):
        return False

    # Otherwise, this is a genuine student misspelling! Reject autocorrection.
    return True


def apply_anchored_diff(
    text: str,
    stage1_target: str,
    actual: str,
    context_snippet: str = ""
) -> Tuple[str, bool]:
    """
    Applies a verified diff (stage1_target -> actual) into text using context-anchored window matching.
    Resolves Problem 14: avoids naive re.sub(..., count=1) which corrupts the first occurrence on the page.
    Returns: (updated_text, success_boolean)
    """
    s1 = stage1_target.strip()
    act = actual.strip()
    if not s1 or not act or s1 == act or not text:
        return text, False

    ctx = context_snippet.strip()

    # Strategy 1: Exact context snippet match
    if ctx and ctx in text and s1 in ctx:
        updated_ctx = ctx.replace(s1, act, 1)
        return text.replace(ctx, updated_ctx, 1), True

    # Strategy 2: Whitespace-normalized regex context match
    if ctx and s1 in ctx:
        escaped_tokens = [re.escape(tok) for tok in ctx.split() if tok]
        if escaped_tokens:
            ctx_pattern = r'\s+'.join(escaped_tokens)
            m = re.search(ctx_pattern, text)
            if m:
                matched_span = m.group(0)
                if s1 in matched_span:
                    updated_span = matched_span.replace(s1, act, 1)
                    return text[:m.start()] + updated_span + text[m.end():], True

    # Find all occurrences of s1 in text
    if re.fullmatch(r'\w+', s1):
        raw_occurrences = [m.start() for m in re.finditer(rf'\b{re.escape(s1)}\b', text)]
    else:
        raw_occurrences = [m.start() for m in re.finditer(re.escape(s1), text)]

    if not raw_occurrences:
        return text, False

    # Prevent tag nesting: do not target an occurrence that is already inside an existing [struck: ...] tag
    occurrences = []
    for idx in raw_occurrences:
        prefix = text[:idx]
        last_open = prefix.rfind("[struck:")
        last_close = prefix.rfind("]")
        if last_open != -1 and (last_close == -1 or last_open > last_close):
            continue
        occurrences.append(idx)

    if not occurrences:
        if f"[struck: {s1}]" in text or f"[struck:{s1}]" in text:
            return text, True
        return text, False

    def _apply_at_idx(at_idx: int) -> str:
        prefix_text = text[:at_idx].rstrip()
        first_act_word = act.split()[0] if act.split() else ""
        if first_act_word and prefix_text.endswith(first_act_word) and not s1.startswith(first_act_word):
            # Strip duplicate leading word from replacement
            act_cleaned = act[len(first_act_word):].lstrip()
            return text[:at_idx] + act_cleaned + text[at_idx + len(s1):]
        return text[:at_idx] + act + text[at_idx + len(s1):]

    # Strategy 3: Single unambiguous occurrence on the page
    if len(occurrences) == 1:
        return _apply_at_idx(occurrences[0]), True

    # Strategy 4: Multiple occurrences with context snippet (sliding-window SequenceMatcher)
    if ctx:
        best_ratio = -1.0
        best_idx = -1
        ctx_len = len(ctx)
        for idx in occurrences:
            w_start = max(0, idx - ctx_len)
            w_end = min(len(text), idx + len(s1) + ctx_len)
            window = text[w_start:w_end]
            ratio = difflib.SequenceMatcher(None, window, ctx).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = idx

        if best_ratio >= 0.40 and best_idx != -1:
            return _apply_at_idx(best_idx), True

    # Ambiguous target with multiple occurrences and no resolving context snippet:
    # Abort to prevent corrupting an unrelated sentence earlier on the page.
    return text, False


def extract_ghost_corrections_from_notes(notes: str, current_text: str) -> List[AutocorrectionDiffItem]:
    """
    Mine corrections that the VLM explicitly identified in verification_notes
    but forgot to include in silent_corrections_fixed (Problem 17).
    """
    if not notes or not current_text:
        return []

    mined_diffs: List[AutocorrectionDiffItem] = []
    seen_targets = set()

    def _is_valid_ghost_target(tok: str) -> bool:
        t = tok.strip().lower()
        t_clean = re.sub(r'^[^\w]+|[^\w]+$', '', t)
        if not t or len(t) < 2:
            return False
        # Never target protected function words, common verbs, or exam headers
        if t in PROTECTED_FUNCTION_WORDS or t_clean in PROTECTED_FUNCTION_WORDS:
            return False
        if t in PROTECTED_EXAM_TOKENS or t_clean in PROTECTED_EXAM_TOKENS or t.startswith("ans"):
            return False
        # Disallow multi-word sentences (> 6 words)
        if len(t.split()) > 6:
            return False
        return True

    # Pattern A: "A strike-through was identified on the word 'TARGET' in the phrase 'CONTEXT'"
    for m in re.finditer(
        r"(?:strike-through|strikethrough|strike\s+through)\s+(?:was\s+)?identified\s+on\s+(?:the\s+word\s+)?['\"]([^'\"]+)['\"](?:\s+in\s+(?:the\s+phrase\s+)?['\"]([^'\"]+)['\"])?",
        notes,
        re.IGNORECASE
    ):
        target = m.group(1).strip()
        context = (m.group(2) or "").strip()
        if target and _is_valid_ghost_target(target) and target not in seen_targets and target in current_text and f"[struck: {target}]" not in current_text:
            mined_diffs.append(AutocorrectionDiffItem(
                stage1_output=target,
                actual_handwritten=f"[struck: {target}]",
                reason="Programmatic ghost-correction recovery from verification_notes",
                context_snippet=context
            ))
            seen_targets.add(target)

    # Pattern B: "applied [struck: ...] to 'TARGET1' ... and [the first instance of] 'TARGET2'"
    # or "Wrapped 'TARGET' in [struck: ...]"
    for m in re.finditer(
        r"(?:applied\s+\[struck:[^\]]*\]\s+to|wrapped)\s+['\"]([^'\"]+)['\"](?:\s*\([^\)]*\))?(?:\s+(?:in|as)\s+\[struck:[^\]]*\])?",
        notes,
        re.IGNORECASE
    ):
        target = m.group(1).strip()
        if target and _is_valid_ghost_target(target) and target not in seen_targets and target in current_text and f"[struck: {target}]" not in current_text:
            mined_diffs.append(AutocorrectionDiffItem(
                stage1_output=target,
                actual_handwritten=f"[struck: {target}]",
                reason="Programmatic ghost-correction recovery from verification_notes",
                context_snippet=""
            ))
            seen_targets.add(target)

    # Pattern B2: Second target in "to 'TARGET1' ... and (the first instance of )?'TARGET2'"
    for m in re.finditer(
        r"and\s+(?:the\s+(?:first|second)\s+instance\s+of\s+)?['\"]([^'\"]+)['\"](?:\s*\([^\)]*\))?(?:\s+as\s+per|\s+in\s+\[struck:|\s+as\s+struck)",
        notes,
        re.IGNORECASE
    ):
        target = m.group(1).strip()
        if target and _is_valid_ghost_target(target) and target not in seen_targets and target in current_text and f"[struck: {target}]" not in current_text:
            mined_diffs.append(AutocorrectionDiffItem(
                stage1_output=target,
                actual_handwritten=f"[struck: {target}]",
                reason="Programmatic ghost-correction recovery from verification_notes",
                context_snippet=""
            ))
            seen_targets.add(target)

    # Pattern C: General quoted target explicitly marked as struck
    for m in re.finditer(
        r"(?:marked\s+['\"]([^'\"]+)['\"]\s+as\s+struck|applied\s+strike-?through\s+to\s+['\"]([^'\"]+)['\"])",
        notes,
        re.IGNORECASE
    ):
        target = (m.group(1) or m.group(2) or "").strip()
        if target and _is_valid_ghost_target(target) and target not in seen_targets and target in current_text and f"[struck: {target}]" not in current_text:
            mined_diffs.append(AutocorrectionDiffItem(
                stage1_output=target,
                actual_handwritten=f"[struck: {target}]",
                reason="Programmatic ghost-correction recovery from verification_notes",
                context_snippet=""
            ))
            seen_targets.add(target)

    return mined_diffs


def extract_unlisted_strikethroughs(
    stage1_text: str,
    raw_verified_text: str,
    already_declared_targets: Optional[Set[str]] = None
) -> List[AutocorrectionDiffItem]:
    """
    Extract any genuine [struck: ...] tags that the model included in raw verified_transcript
    but forgot to declare in silent_corrections_fixed (Problem 17).
    """
    if not raw_verified_text:
        return []

    unlisted_diffs: List[AutocorrectionDiffItem] = []
    verified_tags = re.findall(r'\[struck:\s*([^\]]+)\]', raw_verified_text)
    stage1_tags = set(re.findall(r'\[struck:\s*([^\]]+)\]', stage1_text))
    declared = already_declared_targets or set()

    for inner in verified_tags:
        clean_inner = inner.strip()
        if not clean_inner or not re.search(r'[\w\u0980-\u09FF]', clean_inner):
            continue
        if clean_inner in declared:
            continue
        if clean_inner not in stage1_tags and clean_inner in stage1_text:
            unlisted_diffs.append(AutocorrectionDiffItem(
                stage1_output=clean_inner,
                actual_handwritten=f"[struck: {clean_inner}]",
                reason="Synthesized unlisted strikethrough from raw verified transcript",
                context_snippet=""
            ))
            stage1_tags.add(clean_inner)

    return unlisted_diffs


def purge_empty_struck_tags(text: str) -> str:
    """Purge empty, whitespace-only, punctuation-only, or nested strikethrough tags emitted by margin noise."""
    while re.search(r'\[struck:\s*\[struck:\s*([^\]]+)\]\s*\]', text):
        text = re.sub(r'\[struck:\s*\[struck:\s*([^\]]+)\]\s*\]', r'[struck: \1]', text)
    text = re.sub(r'\[struck:\s*\]', '', text)
    text = re.sub(r'\[struck:[^\w\u0980-\u09FF]*\]', '', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text


@dataclass
class Stage2PreAnalysisReport:
    """CPU-level pre-analysis findings to guide VLM verification and smart triggering."""
    anomalies: List[str]
    likely_glitches: List[Dict[str, str]]
    likely_student_errors: List[str]
    strike_gap_count: int
    split_candidates: List[Dict[str, Any]]
    formatted_prompt_block: str
    should_trigger_stage2: bool


def run_stage2_pre_analysis(
    stage1_transcript: str,
    strikethrough_regions: Optional[List[Any]] = None,
    strikethrough_blocks: Optional[List[Any]] = None,
    question_vocab: Optional[Set[str]] = None,
    lexicon: Optional[Set[str]] = None
) -> Stage2PreAnalysisReport:
    """
    Scan Stage 1 transcript programmatically on CPU before calling VLM:
    1. Lexicon scan + Allograph cursive topology checks (detect likely OCR glitches vs student misspellings)
    2. Strikethrough discrepancy check (optical strokes vs Stage 1 tags)
    3. Split token / pen-lift detection
    4. Ambiguity / illegible marker detection
    """
    if not stage1_transcript or not stage1_transcript.strip():
        return Stage2PreAnalysisReport(
            anomalies=[],
            likely_glitches=[],
            likely_student_errors=[],
            strike_gap_count=0,
            split_candidates=[],
            formatted_prompt_block="",
            should_trigger_stage2=False
        )

    lex = lexicon or get_english_lexicon()
    q_vocab = {w.lower() for w in (question_vocab or set())}
    combined_vocab = lex | q_vocab

    anomalies: List[str] = []
    likely_glitches: List[Dict[str, str]] = []
    likely_student_errors: List[str] = []

    # 1. Lexicon scan: Extract alphabetic words from transcript
    words_seen = set()
    oov_words = []
    for line in stage1_transcript.split("\n"):
        line_s = line.strip()
        if not line_s or line_s.startswith("|") or line_s.startswith("---") or re.match(r'^(?:ans|answer|dans|q(?:uestion)?|no)\b', line_s, re.IGNORECASE):
            continue
        for raw in line_s.split():
            clean = re.sub(r'^[^\w]+|[^\w]+$', '', raw)
            if clean and clean.isalpha() and len(clean) >= 3:
                low = clean.lower()
                if low not in words_seen:
                    words_seen.add(low)
                    if low not in combined_vocab and low not in CULTURAL_TERMS:
                        oov_words.append(clean)

    for w in oov_words:
        topo_match = check_cursive_topology(w, combined_vocab)
        if topo_match:
            rule, cand = topo_match
            likely_glitches.append({"word": w, "candidate": cand, "rule": rule})
            anomalies.append(
                f"- Word '{w}': Out-of-vocabulary non-word. Likely OCR ligature glitch via {rule} topology -> candidate: '{cand}'. VISUALLY VERIFY against handwriting strokes."
            )
        else:
            likely_student_errors.append(w)
            if len(likely_student_errors) <= 4:
                anomalies.append(
                    f"- Word '{w}': Out-of-vocabulary word. Plausible student misspelling. PRESERVE as written unless stroke evidence shows OCR misread."
                )

    # 2. Strikethrough Discrepancy Check
    stage1_strike_count = len(re.findall(r'\[struck:\s*[^\]]+\]', stage1_transcript))
    optical_strike_count = len(strikethrough_regions) if strikethrough_regions else 0
    # Suppress noisy background strikethrough priors (e.g. from lined notebook paper / checkmarks)
    if optical_strike_count > 6:
        strike_gap = 0
    else:
        strike_gap = max(0, optical_strike_count - stage1_strike_count)
    if strike_gap > 0:
        anomalies.append(
            f"- Strikethrough gap: Stage 0.5 detected {optical_strike_count} cross-out strokes, but Stage 1 only has {stage1_strike_count} [struck:] tags ({strike_gap} unverified). VISUALLY VERIFY candidate stroke locations."
        )

    # 2b. Multi-Line Block Discrepancy Check
    if strikethrough_blocks:
        for b in strikethrough_blocks:
            b_type = getattr(b, "stroke_type", "parallel_horizontal")
            line_cnt = getattr(b, "line_count", 2)
            anomalies.append(
                f"- Multi-line Strike Gap: Stage 0.5 detected a {b_type} cross-out block ({line_cnt} lines) at vertical ~{b.y_pct}%-{b.y2_pct}%. VISUALLY VERIFY if an entire draft paragraph was crossed out and wrap each line in [struck: ...]."
            )

    # 3. Pen-lift split token detection
    _, split_diffs = stitch_pen_lift_splits(stage1_transcript, lexicon=lex, question_vocab=question_vocab)
    split_candidates: List[Dict[str, Any]] = []
    for sd in split_diffs[:4]:
        split_candidates.append(sd)
        anomalies.append(
            f"- Split token candidate: '{sd['original']}' -> '{sd['stitched']}' ({sd['reason']}). VISUALLY VERIFY."
        )

    # 4. Ambiguity / illegible marker detection
    has_unclear = ("[unclear:" in stage1_transcript) or ("[illegible]" in stage1_transcript)
    if has_unclear:
        anomalies.append(
            "- Ambiguity markers: Stage 1 flagged [unclear: ...] or [illegible] tokens requiring visual cross-examination."
        )

    # Determine whether Stage 2 should be triggered
    has_strikes = bool(strikethrough_regions and len(strikethrough_regions) > 0)
    has_strike_tags = stage1_strike_count > 0
    has_strike_blocks = bool(strikethrough_blocks and len(strikethrough_blocks) > 0)
    should_trigger = (
        (strike_gap > 0)
        or has_strikes
        or has_strike_tags
        or has_strike_blocks
        or (len(likely_glitches) >= 1)
        or (len(split_candidates) >= 1)
        or has_unclear
        or (len(oov_words) >= 2)
    )

    formatted_block = "\n".join(anomalies) if anomalies else ""

    return Stage2PreAnalysisReport(
        anomalies=anomalies,
        likely_glitches=likely_glitches,
        likely_student_errors=likely_student_errors,
        strike_gap_count=strike_gap,
        split_candidates=split_candidates,
        formatted_prompt_block=formatted_block,
        should_trigger_stage2=should_trigger
    )


class Stage2Verifier:
    """Stage 2: Surgical Autocorrection Auditing (Image + Stage 1 Transcript -> Verified Transcript)."""

    def __init__(self, engine: BaseVLMEngine):
        self.engine = engine

    def run(
        self,
        image: Image.Image,
        stage1_transcript: str,
        pre_analysis: Optional[Stage2PreAnalysisReport] = None,
        question_syllabus: Optional[List[Dict[str, Any]]] = None,
        question_reference_vocab: Optional[List[str]] = None,
        question_reference_numerals: Optional[List[str]] = None,
        strikethrough_regions: Optional[List[Any]] = None,
        strikethrough_blocks: Optional[List[Any]] = None,
        teacher_mask: Optional[Any] = None,
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_new_tokens: int = 3072,
        thinking_mode: bool = False
    ) -> Stage2VerificationResult:
        # Phase 1: CPU Pre-Analysis
        if pre_analysis is None:
            pre_analysis = run_stage2_pre_analysis(
                stage1_transcript=stage1_transcript,
                strikethrough_regions=strikethrough_regions,
                strikethrough_blocks=strikethrough_blocks,
                question_vocab=set(question_reference_vocab) if question_reference_vocab else None
            )

        # Filter noisy strikethrough regions (ruled notebook lines or checkmark noise)
        filtered_strikes = strikethrough_regions if (strikethrough_regions and len(strikethrough_regions) <= 6) else None

        prompt = build_stage2_prompt(
            stage1_transcript=stage1_transcript,
            pre_analysis_report=pre_analysis.formatted_prompt_block,
            question_syllabus=question_syllabus,
            question_reference_vocab=question_reference_vocab,
            question_reference_numerals=question_reference_numerals,
            strikethrough_regions=filtered_strikes
        )

        try:
            response = self.engine.generate_multimodal(
                image=image,
                prompt=prompt,
                system_prompt=STAGE2_SYSTEM_PROMPT,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                thinking_mode=thinking_mode
            )
        except Exception as e:
            print(f"[Stage2Verifier] Warning: Autocorrection verification failed ({e}). Preserving Stage 1 transcript.")
            return Stage2VerificationResult(
                verified_transcript=stage1_transcript,
                silent_corrections_fixed=[],
                proposed_patches=[],
                total_corrections_count=0,
                verification_notes=f"Auto-verification preserved Stage 1 transcript (Error: {e})"
            )

        parsed_data = _extract_json_from_text(response)
        if parsed_data and ("proposed_patches" in parsed_data or "silent_corrections_fixed" in parsed_data or "verified_transcript" in parsed_data):
            raw_patches = parsed_data.get("proposed_patches") or parsed_data.get("silent_corrections_fixed") or []
            raw_verified = str(parsed_data.get("verified_transcript", "")).strip()
            notes = str(parsed_data.get("verification_notes", "")).strip()

            lex = get_english_lexicon()
            q_vocab = {w.lower() for w in (question_reference_vocab or [])}
            combined_vocab = lex | q_vocab

            applied_diffs: List[AutocorrectionDiffItem] = []
            applied_patches: List[Stage2PatchItem] = []
            seen_diff_keys = set()

            # Phase 3: CPU Post-Validation & Patch Application onto Immutable Stage 1 Base
            verified_text = stage1_transcript
            new_strikes_applied = 0

            # Cap at maximum 20 patches per page
            for patch in raw_patches[:20]:
                s1_out = str(patch.get("stage1_target") or patch.get("stage1_output") or patch.get("original") or "").strip()
                act_hw = str(patch.get("replacement") or patch.get("actual_handwritten") or patch.get("corrected") or "").strip()
                patch_type = str(patch.get("patch_type", "ligature_fix")).strip()
                reason = str(patch.get("reason", "")).strip()
                ctx = str(patch.get("context_anchor") or patch.get("context_snippet", "")).strip()
                conf = str(patch.get("confidence", "high")).strip()

                # Rule 1: Target Existence Check (eliminates hallucinated tokens like [struck: mpona])
                if not s1_out or s1_out not in verified_text:
                    continue

                # Rule 2: Idempotency Check
                if s1_out == act_hw or patch_type == "no_change":
                    continue

                # Rule 2b: Anti-Deletion Shield
                # Block silently dropping genuine student words/syllables (e.g. 'di desicions' -> 'desicions')
                # If a word is crossed out, it must be tagged as [struck: ...], never silently deleted!
                s1_tokens = [w for w in re.findall(r'[a-zA-Z\u0980-\u09FF]+', s1_out) if len(w) >= 2]
                act_tokens = [w for w in re.findall(r'[a-zA-Z\u0980-\u09FF]+', act_hw) if len(w) >= 2]
                if len(s1_tokens) > len(act_tokens) and not act_hw.startswith("[struck:"):
                    dropped = set(s1_tokens) - set(act_tokens)
                    if any(len(d) >= 2 for d in dropped):
                        continue

                # Disagreement / Ambiguity Synthesis
                if s1_out and act_hw:
                    if not (act_hw.startswith("[unclear:") or act_hw.startswith("[struck:") or act_hw.startswith("[illegible]")):
                        is_ambiguous_reason = bool(re.search(
                            r'\b(?:ambiguous|unclear|illegible|cannot\s+determine|unsure|either\b.*?\bor\b)\b',
                            reason,
                            re.IGNORECASE
                        ))
                        if is_ambiguous_reason:
                            act_hw = f"[unclear: {s1_out} | {act_hw}]"

                # Rule 3: Spurious Reversion & Function Word Shield
                if is_spurious_reversion(s1_out, act_hw):
                    continue

                # Rule 4: Non-Word Introduction Block
                # Never allow mutating a valid dictionary word into an out-of-lexicon non-word
                # UNLESS it is an explicit reversion of silent autocorrection (Direction A) on a non-function word
                if not (act_hw.startswith("[struck:") or act_hw.startswith("[unclear:") or act_hw.startswith("[truncated")):
                    is_reverting_autocorrection = (
                        (patch_type == "revert_autocorrection" or "revert" in reason.lower() or "autocorrect" in reason.lower())
                        and s1_out.lower() not in PROTECTED_FUNCTION_WORDS
                    )
                    if s1_out.lower() in combined_vocab and act_hw.lower() not in combined_vocab:
                        if not is_reverting_autocorrection:
                            continue

                # Block inserting fabricated words/verbs into student transcript (e.g. [unclear: do | make])
                if bool(re.search(r'\b(?:missing verb|omitted a verb|grammar completion|supply|insert missing)\b', reason, re.IGNORECASE)):
                    continue
                if "[unclear:" in act_hw and "|" in act_hw:
                    unclear_match = re.search(r'\[unclear:\s*(.*?)\s*\]', act_hw)
                    if unclear_match:
                        opts = [o.strip().lower() for o in unclear_match.group(1).split("|")]
                        if not any(difflib.SequenceMatcher(None, s1_out.lower(), o).ratio() >= 0.50 for o in opts):
                            continue

                # Rule 5: Question Header Strikethrough Shield
                if is_header_strikethrough(s1_out, act_hw):
                    continue

                # Rule 6: Strikethrough Quota Guard (Max 3 new strikethroughs per page)
                if act_hw.startswith("[struck:") and not s1_out.startswith("[struck:"):
                    if new_strikes_applied >= 3:
                        continue

                # Rule 6b: Teacher Underline / Mark Strikethrough Shield
                # Never allow teacher marks or underlines to be converted into student strikethroughs
                if act_hw.startswith("[struck:") and any(kw in reason.lower() for kw in ["teacher", "red ink", "grading", "tick", "red underline"]):
                    continue

                # Rule 7: Non-Cursive Student Error Shield (protects phonetic misspellings: desenibe, renny, Pull-time)
                if is_student_error_autocorrection(s1_out, act_hw, combined_vocab):
                    continue

                diff_key = (s1_out, act_hw, ctx)
                if diff_key in seen_diff_keys:
                    continue

                # Apply surgical anchored patch to immutable base
                updated_text, success = apply_anchored_diff(
                    text=verified_text,
                    stage1_target=s1_out,
                    actual=act_hw,
                    context_snippet=ctx
                )
                if success:
                    verified_text = updated_text
                    if act_hw.startswith("[struck:") and not s1_out.startswith("[struck:"):
                        new_strikes_applied += 1
                    applied_diffs.append(AutocorrectionDiffItem(
                        stage1_output=s1_out,
                        actual_handwritten=act_hw,
                        reason=reason,
                        context_snippet=ctx
                    ))
                    applied_patches.append(Stage2PatchItem(
                        patch_type=patch_type,
                        stage1_target=s1_out,
                        replacement=act_hw,
                        confidence=conf,
                        reason=reason,
                        context_anchor=ctx
                    ))
                    seen_diff_keys.add(diff_key)

            # Ghost correction recovery from verification_notes (safety net for unlisted notes)
            ghost_diffs = extract_ghost_corrections_from_notes(notes, verified_text)
            for gd in ghost_diffs:
                if (
                    gd.stage1_output in verified_text
                    and not is_spurious_reversion(gd.stage1_output, gd.actual_handwritten)
                    and not is_header_strikethrough(gd.stage1_output, gd.actual_handwritten)
                    and not is_student_error_autocorrection(gd.stage1_output, gd.actual_handwritten, combined_vocab)
                ):
                    if gd.actual_handwritten.startswith("[struck:") and not gd.stage1_output.startswith("[struck:"):
                        if new_strikes_applied >= 3:
                            continue
                    diff_key = (gd.stage1_output, gd.actual_handwritten, gd.context_snippet)
                    if diff_key not in seen_diff_keys:
                        updated_text, success = apply_anchored_diff(
                            text=verified_text,
                            stage1_target=gd.stage1_output,
                            actual=gd.actual_handwritten,
                            context_snippet=gd.context_snippet
                        )
                        if success:
                            verified_text = updated_text
                            if gd.actual_handwritten.startswith("[struck:") and not gd.stage1_output.startswith("[struck:"):
                                new_strikes_applied += 1
                            applied_diffs.append(gd)
                            applied_patches.append(Stage2PatchItem(
                                patch_type="strikethrough",
                                stage1_target=gd.stage1_output,
                                replacement=gd.actual_handwritten,
                                confidence="medium",
                                reason=gd.reason,
                                context_anchor=gd.context_snippet
                            ))
                            seen_diff_keys.add(diff_key)

            # Unlisted strikethrough recovery (for backward compatibility if raw_verified returned)
            if raw_verified:
                unlisted_diffs = extract_unlisted_strikethroughs(
                    stage1_text=verified_text,
                    raw_verified_text=raw_verified,
                    already_declared_targets={d.stage1_output for d in applied_diffs}
                )
                for ud in unlisted_diffs:
                    if (
                        ud.stage1_output in verified_text
                        and not is_spurious_reversion(ud.stage1_output, ud.actual_handwritten)
                        and not is_header_strikethrough(ud.stage1_output, ud.actual_handwritten)
                        and not is_student_error_autocorrection(ud.stage1_output, ud.actual_handwritten, combined_vocab)
                    ):
                        if ud.actual_handwritten.startswith("[struck:") and not ud.stage1_output.startswith("[struck:"):
                            if new_strikes_applied >= 3:
                                continue
                        diff_key = (ud.stage1_output, ud.actual_handwritten, ud.context_snippet)
                        if diff_key not in seen_diff_keys:
                            updated_text, success = apply_anchored_diff(
                                text=verified_text,
                                stage1_target=ud.stage1_output,
                                actual=ud.actual_handwritten,
                                context_snippet=ud.context_snippet
                            )
                            if success:
                                verified_text = updated_text
                                if ud.actual_handwritten.startswith("[struck:") and not ud.stage1_output.startswith("[struck:"):
                                    new_strikes_applied += 1
                                applied_diffs.append(ud)
                                applied_patches.append(Stage2PatchItem(
                                    patch_type="strikethrough",
                                    stage1_target=ud.stage1_output,
                                    replacement=ud.actual_handwritten,
                                    confidence="medium",
                                    reason=ud.reason,
                                    context_anchor=ud.context_snippet
                                ))
                                seen_diff_keys.add(diff_key)

            # Purge empty or punctuation-only struck tags caused by margin scribble noise
            verified_text = purge_empty_struck_tags(verified_text)

            # Purge VLM conversational meta-chatter and disclaimers about empty pages
            verified_text = re.sub(r'(?im)^The (?:provided )?image (?:is|contains|shows|depicts).*$', '', verified_text)
            verified_text = re.sub(r'(?im)^There is no (?:original |student |handwritten )?.*$', '', verified_text)
            verified_text = re.sub(r'(?im)^Note:\s*.*$', '', verified_text)
            verified_text = re.sub(r'\n{3,}', '\n\n', verified_text).strip()

            # Strict Tag & LaTeX Protection: Never drop genuine [struck: ...] tags or LaTeX arrows from Stage 1
            struck_tags = re.findall(r'\[struck:[^\]]+\]', stage1_transcript)
            for tag in struck_tags:
                inner = tag[len("[struck:"): -1].strip()
                if not re.search(r'[\w\u0980-\u09FF]', inner):
                    continue
                if tag not in verified_text:
                    if inner and inner in verified_text:
                        verified_text, _ = apply_anchored_diff(verified_text, inner, tag, "")

            # Preserve LaTeX arrow notations from Stage 1 if present
            if r"$\rightarrow$" in stage1_transcript and r"$\rightarrow$" not in verified_text:
                verified_text = re.sub(r'[\r\n\t\x0b\x0c]?ightarrow', r'\\rightarrow', verified_text)
                if r"\rightarrow" in verified_text and r"$\rightarrow$" not in verified_text:
                    verified_text = verified_text.replace(r"\rightarrow", r"$\rightarrow$")

            # Transcript Reconciliation Guard: Length Disparity Defense
            # If verification resulted in >15% length drift relative to Stage 1,
            # this indicates severe hallucinations or wholesale text truncation. Fallback to Stage 1 base!
            s1_len = len(stage1_transcript.strip())
            v_len = len(verified_text.strip())
            if s1_len > 60 and abs(v_len - s1_len) / float(s1_len) > 0.15:
                print(f"[Stage 2 Verifier] WARNING: Severe length disparity detected (Stage 1: {s1_len} chars vs Verified: {v_len} chars, diff={abs(v_len-s1_len)/s1_len:.2%}). Falling back to Stage 1 transcript to prevent hallucinated drift.")
                verified_text = stage1_transcript
                applied_diffs = []
                applied_patches = []

            return Stage2VerificationResult(
                verified_transcript=verified_text,
                silent_corrections_fixed=applied_diffs,
                proposed_patches=applied_patches,
                total_corrections_count=len(applied_diffs),
                verification_notes=notes
            )

        # Fallback if raw text returned without valid JSON structure
        return Stage2VerificationResult(
            verified_transcript=stage1_transcript,
            silent_corrections_fixed=[],
            proposed_patches=[],
            total_corrections_count=0,
            verification_notes=f"Auto-verification preserved Stage 1 transcript. (Raw response: {response[:150]}...)"
        )
