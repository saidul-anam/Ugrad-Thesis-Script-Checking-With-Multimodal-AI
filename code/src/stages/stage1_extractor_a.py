"""
Stage 1: Extractor A (Cold OCR Read)
Performs first-pass, independent Multimodal OCR on raw student answer script images
without any external reference solution or rubric priming.
Extracts clean student text, raw text with strike-out markings, uncertainty areas, and word count.
"""

from typing import Any, Dict, List, Optional
import re
from PIL import Image

try:
    from src.schemas import Stage1Output, UncertaintyArea
    from src.model_client.base import ModelClient
    from src.stages.base_stage import extract_json_from_response
except ImportError:
    try:
        from schemas import Stage1Output, UncertaintyArea
        from model_client.base import ModelClient
        from stages.base_stage import extract_json_from_response
    except ImportError:
        from schemas import Stage1Output, UncertaintyArea
        from model_client.base import ModelClient
        from .base_stage import extract_json_from_response


STAGE1_SYSTEM_PROMPT = """You are Extractor A, an expert Multimodal OCR system specialized in transcribing handwritten exam answer scripts in English and Bangla (HSC Class 11 English 1st Paper).

YOUR TASK:
Read the provided handwritten student script image(s) and transcribe the text with 100% fidelity.

CRITICAL INSTRUCTIONS:
1. PURE EXTRACTION: Transcribe EXACTLY what the student wrote. Do NOT correct spelling errors, grammatical slips, capitalization, or punctuation mistakes.
2. STRIKE-OUTS / SCRATCH-OUTS: If a word, number, or phrase has been crossed out or struck through by the student, mark it as `[struck: <word>]` in `extracted_text_raw`. Omit struck words in the clean `STUDENT_ANSWER`.
3. ILLEGIBILITY / CUTS: If a word is unreadable due to scanning flaws or illegible handwriting, mark it as `[illegible]` or `[cut: <partial>]`.
4. PARAGRAPH STRUCTURE: Preserve paragraph breaks using double newlines (`\\n\\n`).
5. UNCERTAINTY AREAS: For any faint, messy, or ambiguous handwriting strokes, log them in `UNCERTAINTY_AREAS` with their estimated text, location, and reason.

OUTPUT FORMAT:
Respond ONLY with a valid JSON object matching this schema:
```json
{
  "QUESTION_TEXT": "<The question header or prompt number if written by student, e.g. Ans to Q. No. 8>",
  "STUDENT_ANSWER": "<Clean transcribed student answer text with original grammar/spelling preserved>",
  "extracted_text_raw": "<Raw transcribed text with [struck: ...] and [illegible] tags>",
  "UNCERTAINTY_AREAS": [
    {
      "location": "<line or paragraph description>",
      "text_guess": "<best guess>",
      "reason": "<e.g., faint stroke, overlapping ink, messy cursive>",
      "bbox": [ymin, xmin, ymax, xmax]
    }
  ],
  "struck_tokens": ["<struck token 1>", "<struck token 2>"],
  "word_count": 0
}
```
"""


def run_stage1_extractor_a(
    images: List[Image.Image],
    model_client: ModelClient,
    question_images: Optional[List[Image.Image]] = None,
    temperature: float = 0.0,
    **kwargs: Any
) -> Stage1Output:
    """
    Execute Stage 1: Extractor A (Cold OCR Read).
    """
    prompt = STAGE1_SYSTEM_PROMPT
    if question_images:
        prompt += f"\n\nNOTE: The first {len(question_images)} image(s) are the Question Paper / Stimulus. The subsequent {len(images)} image(s) are the Student's Handwritten Answer Script."

    all_images = (question_images or []) + images

    raw_response = model_client.generate(
        prompt=prompt,
        images=all_images,
        temperature=temperature,
        **kwargs
    )

    data = extract_json_from_response(raw_response)
    clean_ans = data.get("STUDENT_ANSWER", "")
    raw_ans = data.get("extracted_text_raw", clean_ans)
    
    # Calculate word count if not provided
    word_count = data.get("word_count", 0)
    if word_count == 0 and clean_ans:
        word_count = len(clean_ans.split())

    # Extract struck tokens if not provided
    struck_tokens = data.get("struck_tokens", [])
    if not struck_tokens and raw_ans:
        struck_tokens = re.findall(r"\[struck:\s*([^\]]+)\]", raw_ans)

    uncertainty_list = [
        UncertaintyArea.from_dict(u) if isinstance(u, dict) else u
        for u in data.get("UNCERTAINTY_AREAS", [])
    ]

    return Stage1Output(
        QUESTION_TEXT=data.get("QUESTION_TEXT", ""),
        STUDENT_ANSWER=clean_ans,
        UNCERTAINTY_AREAS=uncertainty_list,
        extracted_text_raw=raw_ans,
        struck_tokens=struck_tokens,
        word_count=word_count
    )
