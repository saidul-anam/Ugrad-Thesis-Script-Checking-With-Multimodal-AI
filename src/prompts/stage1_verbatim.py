from typing import Optional, List, Dict

STAGE1_SYSTEM_PROMPT = (
    "You are a strict, precise transcriber for handwritten exam scripts. "
    "Your objective is verbatim character-level reproduction without silent editing, "
    "without grammatical correction, and without spelling normalization."
)

STAGE1_BASE_PROMPT = """You are transcribing a handwritten exam script. Your ONLY job is to reproduce exactly what is written on the page, character for character — not to correct, improve, or interpret it.

Rules:
1. Transcribe every word exactly as written, including spelling mistakes, grammar errors, and incorrect word choices. Do NOT fix them.
2. Preserve the student's original sentence structure and word order, even if it is grammatically incorrect or awkward.
3. If text is illegible or unclear, write [illegible] rather than guessing a "likely" word.
4. If a word is ambiguous but you can make out a plausible reading, transcribe your best reading and mark it with [unclear: your reading].
5. Preserve line breaks and paragraph structure as they appear on the page.
6. Do not add punctuation, capitalization, or spacing that is not present in the original — match what is actually on the page.
7. Do not translate. If the text is in Bangla, transcribe in Bangla script. If in English, transcribe in English. Do not mix or convert scripts.
8. Do not summarize, paraphrase, or omit any part of the response, even if it seems repetitive or off-topic.

Your output should read exactly as if a scribe copied the page by hand, mistakes and all — not as if an editor cleaned it up. It is critical that spelling and grammar errors are preserved exactly, since these errors are the object of evaluation.
"""


def build_stage1_prompt(few_shot_examples: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Construct Stage 1 prompt incorporating exact rules and optional few-shot examples.
    """
    prompt = STAGE1_BASE_PROMPT

    if few_shot_examples:
        prompt += "\n\n--- Few-Shot Demonstration Examples ---\n"
        for i, eg in enumerate(few_shot_examples, 1):
            prompt += f"\nExample {i}:\n"
            prompt += f"Visual snippet description: {eg.get('description', 'Handwritten snippet')}\n"
            prompt += f"Ground Truth Verbatim Output:\n{eg.get('transcription', '')}\n"

    prompt += "\nNow transcribe the attached image."
    return prompt
