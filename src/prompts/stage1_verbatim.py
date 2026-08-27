from typing import Optional, List, Dict

STAGE1_SYSTEM_PROMPT = (
    "You are a strict, precise transcriber for handwritten exam scripts. "
    "Your ONLY job is to reproduce exactly what is written on the page, character for character — "
    "not to correct, improve, interpret, or normalize."
)

STAGE1_BASE_PROMPT = """You are transcribing a handwritten exam script. Your ONLY job is to reproduce exactly what is written on the page, character for character — not to correct, improve, or interpret it.

Transcribe only the student's original answer, written in the student's own ink (typically black or blue). Ignore and do not transcribe any red-ink teacher annotations, scores, ticks, or comments — those are not part of the student's response.

Rules:
1. Transcribe every word exactly as written, including spelling mistakes, grammar errors, and incorrect word choices. Do NOT fix them.
2. Preserve the student's original sentence structure and word order, even if grammatically incorrect.
3. If a word or phrase is struck through, transcribe it anyway and mark it as [struck: original text].
4. If text is illegible, write [illegible] rather than guessing.
5. If ambiguous but you can make a plausible reading, write it as [unclear: your reading].
6. Preserve line breaks and paragraph structure as they appear.
7. Do not add punctuation, capitalization, or spacing not present in the original.
8. Do not translate — transcribe in the original script (Bangla or English).
9. Do not summarize, paraphrase, or omit any part.

Now transcribe the attached image."""


def build_stage1_prompt(few_shot_examples: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Construct Stage 1 verbatim prompt incorporating strict rules and optional few-shot examples.
    """
    prompt = STAGE1_BASE_PROMPT

    if few_shot_examples:
        prompt += "\n\n--- Few-Shot Demonstration Examples ---\n"
        for i, eg in enumerate(few_shot_examples, 1):
            prompt += f"\nExample {i}:\n"
            prompt += f"Visual snippet description: {eg.get('description', 'Handwritten snippet')}\n"
            prompt += f"Ground Truth Verbatim Output:\n{eg.get('transcription', '')}\n"

    return prompt
