import re
from typing import Optional, List, Dict
from PIL import Image
from src.engine.base_engine import BaseVLMEngine
from src.core.schemas import Stage1TranscriptionResult
from src.prompts.stage1_verbatim import build_stage1_prompt, STAGE1_SYSTEM_PROMPT


class Stage1Transcriber:
    """Stage 1: Verbatim Transcription (Image -> Text) preserving all handwritten errors."""

    def __init__(self, engine: BaseVLMEngine):
        self.engine = engine

    def run(
        self,
        image: Image.Image,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_new_tokens: int = 3072,
        thinking_mode: bool = False
    ) -> Stage1TranscriptionResult:
        prompt = build_stage1_prompt(few_shot_examples=few_shot_examples)
        
        raw_text = self.engine.generate_multimodal(
            image=image,
            prompt=prompt,
            system_prompt=STAGE1_SYSTEM_PROMPT,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            thinking_mode=thinking_mode
        )

        # Parse tags
        illegible_matches = re.findall(r"\[illegible\]", raw_text, re.IGNORECASE)
        unclear_matches = re.findall(r"\[unclear:[^\]]+\]", raw_text, re.IGNORECASE)

        # Detect script
        has_bangla = bool(re.search(r"[\u0980-\u09FF]", raw_text))
        has_english = bool(re.search(r"[a-zA-Z]", raw_text))
        if has_bangla and has_english:
            detected_script = "Mixed (Bangla + English)"
        elif has_bangla:
            detected_script = "Bangla"
        elif has_english:
            detected_script = "English"
        else:
            detected_script = "Unknown"

        words = raw_text.split()
        return Stage1TranscriptionResult(
            raw_transcript=raw_text,
            illegible_count=len(illegible_matches),
            unclear_count=len(unclear_matches),
            character_count=len(raw_text),
            word_count=len(words),
            detected_script=detected_script
        )
