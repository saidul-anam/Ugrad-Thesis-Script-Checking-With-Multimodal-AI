"""Prompts for Stage 2b line reconciliation (independent line reads + order-debiased forced choice)."""

LINE_READ_SYSTEM_PROMPT = (
    "You are a strict character-level transcriber of handwriting. You reproduce exactly the letters "
    "that are on the image, including misspellings and non-words. You never correct or complete words."
)

LINE_READ_PROMPT = (
    "The image is one line of handwriting from a student's exam script (parts of the lines above or "
    "below may be visible at the edges; ignore them).\n"
    "Transcribe the line character for character, exactly as the ink reads. Do not correct spelling, "
    "do not substitute a dictionary word when the letters do not form one, do not add or remove words.\n"
    "If the student crossed out a letter, word or phrase with a pen stroke, write it as [struck: ...]. "
    "A crossbar that is part of a letter, and an underline beneath words, are not cross-outs.\n"
    "If the image contains no handwriting, output nothing. Output only the transcribed line."
)

FORCED_CHOICE_SYSTEM_PROMPT = "You verify transcriptions of handwriting against the image, letter by letter."

FORCED_CHOICE_PROMPT = (
    "The image shows one line of handwriting from a student's exam script. Two candidate transcriptions "
    "of this line are given; they differ only in a disputed part. Crossed-out text is marked as [struck: ...].\n\n"
    "{disputed_section}"
    "Candidate 1: {first}\n"
    "Candidate 2: {second}\n\n"
    "Look closely at the ink for the disputed text:\n"
    "- If a pen stroke (horizontal line, diagonal slash, or cross-out) clearly crosses through the letters, choose the candidate with [struck: ...].\n"
    "- If the letters are clean and untouched (or only underlined beneath), choose the candidate without [struck: ...].\n\n"
    "Format your response as:\n"
    "Observation: [1 brief sentence describing whether ink crosses through the letters of the disputed text]\n"
    "Choice: 1 or 2"
)
