import json
import re
import difflib
from typing import Optional, List, Dict, Any
from PIL import Image
from src.engine.base_engine import BaseVLMEngine
from src.core.schemas import Stage2VerificationResult, AutocorrectionDiffItem
from src.prompts.stage2_verification import build_stage2_prompt, STAGE2_SYSTEM_PROMPT


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
        # Strip incomplete trailing key-value or key fragment (e.g. , "linguistic or "linguistic": )
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


def is_spurious_reversion(stage1_word: str, actual_word: str) -> bool:
    """
    Detect whether a proposed Stage 2 reversion is a hallucinated cursive ligature artifact.
    Rejects mutations of valid words into non-words caused by cursive pen loops or flourishes.
    """
    s1 = stage1_word.strip().lower()
    act = actual_word.strip().lower()
    if not s1 or not act or s1 == act:
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

    # 3. Leading or terminal cursive flourishes (e.g. 'wh' -> 'coh', 'that' -> 'thad')
    if s1.startswith("wh") and act.startswith("coh"):
        return True
    if s1.endswith("t") and act.endswith("d") and s1[:-1] == act[:-1]:
        return True
    if "az" in s1 and act == s1.replace("az", "raz", 1):
        return True

    # 4. Common word to non-word single-edit corruption (Lexicon Gate)
    from src.utils.linguistic_sanitizer import get_english_lexicon
    lex = get_english_lexicon()
    if s1 in lex and act not in lex:
        if difflib.SequenceMatcher(None, s1, act).ratio() >= 0.75:
            return True

    return False


class Stage2Verifier:
    """Stage 2: Autocorrection Verification (Image + Stage 1 Transcript -> Verified Transcript)."""

    def __init__(self, engine: BaseVLMEngine):
        self.engine = engine

    def run(
        self,
        image: Image.Image,
        stage1_transcript: str,
        question_syllabus: Optional[List[Dict[str, Any]]] = None,
        question_reference_vocab: Optional[List[str]] = None,
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_new_tokens: int = 3072,
        thinking_mode: bool = False
    ) -> Stage2VerificationResult:
        # Prompt with full transcript without artificial string clipping
        prompt = build_stage2_prompt(
            stage1_transcript=stage1_transcript,
            question_syllabus=question_syllabus,
            question_reference_vocab=question_reference_vocab
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
                total_corrections_count=0,
                verification_notes=f"Auto-verification preserved Stage 1 transcript (Error: {e})"
            )

        parsed_data = _extract_json_from_text(response)
        if parsed_data and ("verified_transcript" in parsed_data or "silent_corrections_fixed" in parsed_data):
            diffs = []
            declared_items = parsed_data.get("silent_corrections_fixed", [])

            for item in declared_items:
                s1_out = str(item.get("stage1_output") or item.get("original") or item.get("stage1") or "").strip()
                act_hw = str(item.get("actual_handwritten") or item.get("corrected") or item.get("actual") or "").strip()
                reason = str(item.get("reason", "")).strip()
                ctx = str(item.get("context_snippet", "")).strip()

                if is_spurious_reversion(s1_out, act_hw):
                    continue

                diffs.append(AutocorrectionDiffItem(
                    stage1_output=s1_out,
                    actual_handwritten=act_hw,
                    reason=reason,
                    context_snippet=ctx
                ))

            # Build verified transcript starting from Stage 1 (Declared-Only Modification Policy)
            verified_text = stage1_transcript
            for d in diffs:
                s1 = d.stage1_output
                act = d.actual_handwritten
                ctx = d.context_snippet
                if not s1 or not act or s1 == act:
                    continue
                if ctx and ctx in verified_text and s1 in ctx:
                    updated_ctx = ctx.replace(s1, act, 1)
                    verified_text = verified_text.replace(ctx, updated_ctx, 1)
                elif s1 in verified_text:
                    verified_text = re.sub(rf"\b{re.escape(s1)}\b", act, verified_text, count=1)

            # Strict Tag & LaTeX Protection: Never drop [struck: ...] tags or LaTeX arrows from Stage 1
            struck_tags = re.findall(r'\[struck:[^\]]+\]', stage1_transcript)
            for tag in struck_tags:
                if tag not in verified_text:
                    inner = tag[len("[struck:"): -1].strip()
                    if inner and inner in verified_text:
                        verified_text = re.sub(rf'\b{re.escape(inner)}\b', tag, verified_text, count=1)

            # Preserve LaTeX arrow notations from Stage 1 if present
            if r"$\rightarrow$" in stage1_transcript and r"$\rightarrow$" not in verified_text:
                verified_text = re.sub(r'[\r\n\t\x0b\x0c]?ightarrow', r'\\rightarrow', verified_text)
                if r"\rightarrow" in verified_text and r"$\rightarrow$" not in verified_text:
                    verified_text = verified_text.replace(r"\rightarrow", r"$\rightarrow$")

            notes = parsed_data.get("verification_notes", "")
            return Stage2VerificationResult(
                verified_transcript=verified_text,
                silent_corrections_fixed=diffs,
                total_corrections_count=len(diffs),
                verification_notes=notes
            )

        # Fallback if raw text returned without valid JSON structure
        return Stage2VerificationResult(
            verified_transcript=stage1_transcript,
            silent_corrections_fixed=[],
            total_corrections_count=0,
            verification_notes=f"Auto-verification preserved Stage 1 transcript. (Raw response: {response[:150]}...)"
        )
