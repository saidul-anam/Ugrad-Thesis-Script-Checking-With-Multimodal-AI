import json
import re
from typing import Optional, Dict, Any
from PIL import Image
from src.engine.base_engine import BaseVLMEngine
from src.core.schemas import Stage2VerificationResult, AutocorrectionDiffItem
from src.prompts.stage2_verification import build_stage2_prompt, STAGE2_SYSTEM_PROMPT


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from string even if surrounded by markdown code blocks."""
    text = text.strip()
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
    return None


class Stage2Verifier:
    """Stage 2: Autocorrection Verification (Image + Stage 1 Transcript -> Verified Transcript)."""

    def __init__(self, engine: BaseVLMEngine):
        self.engine = engine

    def run(
        self,
        image: Image.Image,
        stage1_transcript: str,
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_new_tokens: int = 4096,
        thinking_mode: bool = False
    ) -> Stage2VerificationResult:
        prompt = build_stage2_prompt(stage1_transcript=stage1_transcript)

        response = self.engine.generate_multimodal(
            image=image,
            prompt=prompt,
            system_prompt=STAGE2_SYSTEM_PROMPT,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            thinking_mode=thinking_mode
        )

        parsed_data = _extract_json_from_text(response)
        if parsed_data and "verified_transcript" in parsed_data:
            diffs = []
            for item in parsed_data.get("silent_corrections_fixed", []):
                diffs.append(AutocorrectionDiffItem(
                    stage1_output=item.get("stage1_output", ""),
                    actual_handwritten=item.get("actual_handwritten", ""),
                    reason=item.get("reason", ""),
                    context_snippet=item.get("context_snippet", "")
                ))
            
            return Stage2VerificationResult(
                verified_transcript=parsed_data["verified_transcript"],
                silent_corrections_fixed=diffs,
                total_corrections_count=len(diffs),
                verification_notes=parsed_data.get("verification_notes", "")
            )

        # Fallback if raw text returned without valid JSON structure
        return Stage2VerificationResult(
            verified_transcript=stage1_transcript,
            silent_corrections_fixed=[],
            total_corrections_count=0,
            verification_notes=f"Auto-verification preserved Stage 1 transcript. (Raw response: {response[:150]}...)"
        )
