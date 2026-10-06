"""
Stage 3b (EVIDENCE mode) prompts: line-crop re-reading, line localization, forced choice.

Design rules:
- No example words anywhere (label leakage).
- The line-crop re-read prompt carries no context and no candidate names, so the model reads the ink
  rather than the prior. Agreement across augmented re-reads is the confidence signal.
- Each prompt contains a unique phrase used by the mock engine to dispatch:
    line crop     -> "exactly one line of handwriting"
    bbox          -> "bounding box of the handwritten line"
    forced choice -> "either option a or option b"
"""

LINE_CROP_SYSTEM_PROMPT = (
    "You are a strict character-level transcriber of handwriting. You reproduce exactly the letters "
    "that are on the image, including misspellings and non-words. You never correct or complete words."
)

LINE_CROP_VERBATIM_PROMPT = (
    "The attached image contains exactly one line of handwriting from a student's exam script.\n"
    "Transcribe it character for character, exactly as the ink reads. Do not correct spelling, do not "
    "guess a dictionary word if the letters do not form one, do not add or remove words.\n"
    "Output only the transcribed line, nothing else."
)

LINE_BBOX_SYSTEM_PROMPT = (
    "You are a document layout analyser. You locate lines of handwriting on a page image and return "
    "their bounding boxes as JSON."
)


def build_line_bbox_prompt(context: str, target_token: str = "") -> str:
    """Ask for the bounding box of the handwritten line whose text best matches `context` and contains `target_token`."""
    ctx = " ".join(context.split())[:220]
    tok_clause = f" containing the target word \"{target_token}\"" if target_token else ""
    return (
        f"Find the bounding box of the single handwritten line on the attached page{tok_clause} whose text best matches:\n"
        f"\"{ctx}\"\n"
        "Return ONLY a JSON object of the form {\"bbox\": [y1, x1, y2, x2]} with integer coordinates "
        "on a 0-1000 scale relative to the full image (y1 top, x1 left, y2 bottom, x2 right). "
        "The box must tightly enclose that single line of writing. If the text spans multiple physical lines, "
        f"return the specific line that contains the target word{(' \"' + target_token + '\"') if target_token else ''}."
    )


FORCED_CHOICE_SYSTEM_PROMPT = (
    "You are a meticulous handwriting examiner. You decide which of two candidate readings the ink on a "
    "cropped line of handwriting actually shows, letter by letter. Output valid JSON only."
)


def build_forced_choice_prompt(option_a: str, option_b: str, context_hint: str) -> str:
    """
    Two-alternative forced choice on a line crop. Options are passed in a random order by the caller.
    `context_hint` is the surrounding words (without the token itself) to help the model find the word.
    """
    hint = " ".join(context_hint.split())[:160]
    return (
        "The attached image is one line of a student's handwriting. One word on this line is disputed. "
        f"Its neighbouring words are approximately: \"{hint}\".\n"
        "Look at the letters of the disputed word and decide whether the ink reads either option A or option B:\n"
        f"  A: \"{option_a}\"\n"
        f"  B: \"{option_b}\"\n"
        "Compare the letter shapes where the two options differ (ascenders, crossbars, loops, descenders, joins). "
        "Judge the ink only; do not prefer an option because it is a dictionary word.\n"
        "Note on cursive handwriting: Students writing quickly often produce cursive ligatures, such as blind loops on 'c' "
        "(resembling 'e'), double-peaked arches on 'r' (resembling 'rr'), or connected crossbars between 't' and 'e'. "
        "Do not interpret a single cursive letter's ligature loops as intentional extra letters unless there are distinct separate pen lifts. "
        "If the stroke represents an authentic cursive ligature of the intended word, or if the strokes are ambiguous, answer NEITHER.\n"
        "Return ONLY JSON: {\"choice\": \"A\" | \"B\" | \"NEITHER\", \"confidence\": 0.0-1.0, "
        "\"reason\": \"one sentence about the specific strokes\"}"
    )
