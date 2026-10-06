import re
from typing import Optional, List, Dict, Any
from PIL import Image
from src.engine.base_engine import BaseVLMEngine
from src.core.schemas import Stage1TranscriptionResult
from src.prompts.stage1_verbatim import build_stage1_prompt, STAGE1_SYSTEM_PROMPT
from src.pipeline.band_tiler import BandTiler, stitch_band_transcripts
from src.utils.transcript_markup import normalize_markup


def sanitize_and_normalize_stage1_output(raw_text: str) -> str:
    """
    Standardize Stage 1 raw VLM transcript output:
    1. Strip presentation HTML tags (<u>, </u>, <b>, <i>).
    2. Normalize LaTeX arrow and math symbols ($\\rightarrow$ -> ->).
    3. Collapse consecutive [illegible] runs and cap per-page illegible markers.
    4. Squash autoregressive line repetitions and loop attractors on noisy/ruled pages.
    5. Clean empty [struck: ...] and VLM conversational disclaimers.
    """
    if not raw_text:
        return ""

    # 1. Strip presentation HTML tags (e.g. <u>, </u>, <b>, <i>, <span>) while preserving inner text
    text = re.sub(r'</?(?:u|b|i|strong|em|span|sub|sup)\b[^>]*>', '', raw_text, flags=re.IGNORECASE)

    # 2. Normalize LaTeX arrow/math notation to plain text symbols
    text = re.sub(r'\$(?:\\rightarrow|\\to|->)\$', '->', text)
    text = re.sub(r'\\rightarrow\b', '->', text)
    text = re.sub(r'\\to\b', '->', text)

    # 3. Purge empty or whitespace/punctuation-only struck tags
    text = re.sub(r'\[struck:\s*\]', '', text)
    text = re.sub(r'\[struck:[^\w\u0980-\u09FF]*\]', '', text)

    # 4. Purge VLM conversational meta-chatter, refusals, and overexposure disclaimers
    text = re.sub(r'(?im)^.*?(?:The image provided|The provided image|The image) is too (?:faint|overexposed|dark|blurry).*$', '', text)
    text = re.sub(r'(?im)^.*?(?:The image provided|The provided image) (?:is|contains|shows|depicts|appears).*$', '', text)
    text = re.sub(r'(?im)^.*?(?:No|There is no) (?:original |student |handwritten |legible )?handwriting (?:is )?(?:visible|present|readable|discernible|found).*$', '', text)
    text = re.sub(r'(?im)^.*?(?:Due to severe overexposure|Due to poor contrast|I cannot transcribe).*$', '', text)
    text = re.sub(r'(?im)^Note:\s*.*$', '', text)

    # 4b. Purge ruling line hallucinations (repeated underscores or hyphens with no text)
    text = re.sub(r'(?m)^[_\-\s]{3,}\s*$', '', text)


    # 5. Degeneracy suppression: collapse consecutive identical [illegible] tags
    text = re.sub(r'(?:\[illegible\][\s,;]*){2,}', '[illegible] ', text, flags=re.IGNORECASE)

    # Cap per-page [illegible] tokens (max 15 per page to prevent bleed-through pollution)
    max_page_illegible = 15
    illegible_spans = [m for m in re.finditer(r'\[illegible\]', text, re.IGNORECASE)]
    if len(illegible_spans) > max_page_illegible:
        pieces = []
        last_idx = 0
        for i, match in enumerate(illegible_spans):
            if i < max_page_illegible:
                pieces.append(text[last_idx:match.end()])
            else:
                pieces.append(text[last_idx:match.start()])
            last_idx = match.end()
        pieces.append(text[last_idx:])
        text = "".join(pieces)

    # 6. Degeneracy suppression: collapse consecutive identical or near-identical loop lines
    lines = text.splitlines()
    deduped_lines = []
    repeat_count = 0
    prev_norm = ""
    for line in lines:
        norm = re.sub(r'\s+', ' ', line).strip().lower()
        if norm and norm == prev_norm:
            repeat_count += 1
            if repeat_count < 2:  # allow at most 2 identical lines if student actually duplicated
                deduped_lines.append(line)
        else:
            repeat_count = 0
            prev_norm = norm
            deduped_lines.append(line)

    text = "\n".join(deduped_lines)

    # 6b. Multi-line cycle loop squashing (e.g. 2-line, 3-line, or 4-line repeating cycles)
    lines = text.splitlines()
    if len(lines) >= 6:
        for k in (2, 3, 4):
            new_lines = []
            i = 0
            while i < len(lines):
                if i + 2 * k <= len(lines):
                    block1 = [re.sub(r'\s+', ' ', l).strip().lower() for l in lines[i:i + k]]
                    block2 = [re.sub(r'\s+', ' ', l).strip().lower() for l in lines[i + k:i + 2 * k]]
                    if all(b1 and b1 == b2 for b1, b2 in zip(block1, block2)):
                        # Loop cycle detected: keep first block, skip consecutive repeating cycles
                        new_lines.extend(lines[i:i + k])
                        i += 2 * k
                        while i + k <= len(lines):
                            block_next = [re.sub(r'\s+', ' ', l).strip().lower() for l in lines[i:i + k]]
                            if all(b1 and b1 == bn for b1, bn in zip(block1, block_next)):
                                i += k
                            else:
                                break
                        continue
                new_lines.append(lines[i])
                i += 1
            lines = new_lines
        text = "\n".join(lines)

    # 7. Collapse inline phrase loop attractors (e.g. phrase repeated 3+ times consecutively within a paragraph)
    text = re.sub(r'(\b[\w\s\[\]:]{4,40}?\b)(?:\s+\1){2,}', r'\1', text)

    # 7b. Suppress runaway punctuation runs (e.g. ’s’’’’’’’’ -> ’s, ,,,,, -> ,)
    text = re.sub(r'([^\w\s])\1{3,}', r'\1', text)

    # 7c. Flatten nested struck tags: [struck: [struck: text]] -> [struck: text]
    while re.search(r'\[struck:\s*\[struck:', text, re.IGNORECASE):
        text = re.sub(r'\[struck:\s*\[struck:\s*([^\]]+)\]\s*\]', r'[struck: \1]', text, flags=re.IGNORECASE)

    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


class Stage1Transcriber:
    """Stage 1: Verbatim Transcription (Image -> Text) preserving all handwritten errors."""

    def __init__(self, engine: BaseVLMEngine, enable_banding: bool = True):
        self.engine = engine
        self.enable_banding = enable_banding
        self.tiler = BandTiler()

    def run(
        self,
        image: Image.Image,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        question_reference_vocab: Optional[List[str]] = None,
        question_reference_numerals: Optional[List[str]] = None,
        question_syllabus: Optional[List[Dict[str, Any]]] = None,
        strikethrough_detected: bool = False,
        strikethrough_region_count: int = 0,
        strikethrough_regions: Optional[List[Any]] = None,
        strikethrough_blocks: Optional[List[Any]] = None,
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_new_tokens: int = 3072,
        thinking_mode: bool = False
    ) -> Stage1TranscriptionResult:
        # Determine whether to use multi-band tiling for native DPI stroke preservation
        w, h = image.size
        use_bands = self.enable_banding and h > self.tiler.max_page_height_for_single_band

        if use_bands:
            band_images, band_metas = self.tiler.tile_page(image)
            band_transcripts: List[str] = []

            for b_img, b_meta in zip(band_images, band_metas):
                # Filter strikethrough regions belonging to this band
                band_strike_regions = []
                if strikethrough_regions:
                    for reg in strikethrough_regions:
                        ry = getattr(reg, "y_pct", 0.0)
                        if b_meta.y_start_pct <= ry <= b_meta.y_end_pct:
                            band_strike_regions.append(reg)

                band_strike_blocks = []
                if strikethrough_blocks:
                    for blk in strikethrough_blocks:
                        by = getattr(blk, "y_pct", 0.0)
                        if b_meta.y_start_pct <= by <= b_meta.y_end_pct:
                            band_strike_blocks.append(blk)

                b_prompt = build_stage1_prompt(
                    few_shot_examples=few_shot_examples,
                    question_reference_vocab=question_reference_vocab,
                    question_reference_numerals=question_reference_numerals,
                    question_syllabus=question_syllabus,
                    strikethrough_detected=len(band_strike_regions) > 0 or len(band_strike_blocks) > 0,
                    strikethrough_region_count=len(band_strike_regions),
                    strikethrough_regions=band_strike_regions,
                    strikethrough_blocks=band_strike_blocks,
                )

                b_text = self.engine.generate_multimodal(
                    image=b_img,
                    prompt=b_prompt,
                    system_prompt=STAGE1_SYSTEM_PROMPT,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max(1024, max_new_tokens // len(band_images)),
                    thinking_mode=thinking_mode
                )
                band_transcripts.append(b_text)

            raw_text = stitch_band_transcripts(band_transcripts)
        else:
            prompt = build_stage1_prompt(
                few_shot_examples=few_shot_examples,
                question_reference_vocab=question_reference_vocab,
                question_reference_numerals=question_reference_numerals,
                question_syllabus=question_syllabus,
                strikethrough_detected=strikethrough_detected or (strikethrough_blocks is not None and len(strikethrough_blocks) > 0),
                strikethrough_region_count=strikethrough_region_count,
                strikethrough_regions=strikethrough_regions,
                strikethrough_blocks=strikethrough_blocks,
            )

            raw_text = self.engine.generate_multimodal(
                image=image,
                prompt=prompt,
                system_prompt=STAGE1_SYSTEM_PROMPT,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                thinking_mode=thinking_mode
            )

        # Canonicalization and degeneracy suppression
        raw_text = normalize_markup(sanitize_and_normalize_stage1_output(raw_text))

        # Parse tags from canonicalized transcript
        illegible_matches = re.findall(r"\[illegible\]", raw_text, re.IGNORECASE)
        unclear_matches = re.findall(r"\[unclear:[^\]]+\]", raw_text, re.IGNORECASE)
        struck_matches = re.findall(r"\[struck:[^\]]+\]", raw_text, re.IGNORECASE)

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
            struck_count=len(struck_matches),
            character_count=len(raw_text),
            word_count=len(words),
            detected_script=detected_script
        )
