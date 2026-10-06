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
    "of this line are given; they differ only in a small part. Crossed-out text is written as [struck: ...].\n"
    "Choose the candidate that matches the ink exactly: keep the student's own spelling even when it is "
    "wrong, and treat text as struck only if a pen stroke crosses it out.\n\n"
    "Candidate 1: {first}\nCandidate 2: {second}\n\nAnswer with 1 or 2 only."
)
