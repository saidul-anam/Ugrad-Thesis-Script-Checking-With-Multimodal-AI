# -*- coding: utf-8 -*-
"""One-off generator: data/Question/task_stems_SE_07_Q1.csv (Class VII, 13 questions).

The question text is transcribed from data/Question/SE_07_Q1.pdf (a scanned
paper — no text layer, so it was read off the page images).

`stimulus_data` carries the reading passage plus the ANSWER KEY that the grader
marks the item-scored questions against; without a key an LLM cannot mark an MCQ
or a rearrangement at all. `source_text` carries the passage that length-ratio
and verbatim-overlap metrics are computed against (Summary only).

Kept in the repo so the keys are reviewable and the CSV is regenerable; run it
from the project root: `python scripts/gen_task_stems_SE_07_Q1.py`.
"""

from __future__ import annotations

import csv
from pathlib import Path

PASSAGE_1 = """Mrs Nazma and Mr Joynul Ali live in a small village with their daughters, Mitu aged 7 and Nitu aged 5. Joynul is a carpenter. He is hired by the villagers to make chairs, tables, pira (low stool) and other furniture. He is also asked to do small repair work. But Joynul does not find work every day. He lives from hand to mouth.
Nazma does all the work at home from morning to night, rain or shine. Sometimes she sits with Mitu and Nitu, and teaches them Bangla and English alphabets and some numbers. They do not go to school. Nazma could study only up to class 5. Joynul could not study."""

PASSAGE_2 = """The show went very smoothly. The audience clapped and cheered, and all the performer did a great job. At last it was Pratap's turn. He walked onto the stage and everyone clapped. He was wearing a green robe and a tall pointed hat painted with stars and crescent moons. He looked like a real magician. But what was he going to do?
'Good evening, ladies and gentlemen', said Pratap. "Tonight I am going to perform a vanishing trick. I'm not going to make a coin disappear. I'm not going to make an egg disappear. No. I'm going to make my dog, Smokey, disappear!' Everyone cheered loudly."""

PASSAGE_3 = """On July 16, 1969, three American astronauts began their journey to the Moon. They were Neil Armstrong, Michael Collins, and Edwin Aldrin. Their spacecraft was called Apollo 11. It was launched from Florida, USA. People all over the world watched the news with great excitement.
After four days, on July 20, 1969, their spacecraft reached the Moon. Michael Collins stayed in the main ship, and Armstrong and Aldrin landed on the Moon. Neil Armstrong was the first person to walk there. He said. "That's one small step for man, one giant leap for mankind." They walked, took photos, and collected rocks.
On July 24, 1969, the astronauts came back safely to Earth. Their success made history. It was the first time humans had visited another world. The Apollo 11 mission showed the power of science, teamwork, and human courage."""

ITEM_NOTE = (
    "MARKING NOTE: this question is ITEM-SCORED. Mark each lettered item against the "
    "key and put the SUM of the item marks in context_content_data, leaving "
    "structure_format_brevity, language_mechanics and "
    "originality_comparisons_paraphrase at 0. Nothing is awarded or deducted for "
    "handwriting, style, or spelling beyond what the key requires. An item the "
    "student did not attempt scores 0 for that item only. No structural cap is ever "
    "applied to an item-scored question."
)

ROWS: list[dict] = [
    # ---- 1 -----------------------------------------------------------------
    dict(
        question_no=1,
        task_type="MCQ",
        marks=7,
        prompt_given=(
            "Read the passage. Then answer the question no. 1: Now choose the correct "
            "answer to each question from the given alternatives. (1 x 7 = 7)\n"
            "a. Joynul and Nazma live in — area. i. urban area  ii. municipal area  "
            "iii. rural area  iv. civic area\n"
            "b. The phrase “lives from hand to mouth” indicates— i. He is very rich  "
            "ii. He earns irregularly for daily needs  iii. He saves a lot of money  "
            "iv. He works in a factory\n"
            "c. In the sentence “Sometimes she sits with Mitu and Nitu,” the word "
            "‘sometimes’ is a — i. Adjective  ii. Preposition  iii. Pronoun  iv. Adverb\n"
            "d. Nazma’s working schedule is described as — i. From morning to night  "
            "ii. Limited to mornings only  iii. Only in the evening  iv. Occasionally\n"
            "e. In the sentence “Nazma does all the work at home,” the word ‘all’ is a — "
            "i. Pronoun  ii. Conjuction  iii. Adverb  iv. Adjective\n"
            "f. Which word best describes Joynul’s income condition? i. Stable  "
            "ii. Lofty  iii. Rough  iv. Fixed\n"
            "g. Nazma’s role in the family is mainly — i. Financial provider  "
            "ii. Household manager and caregiver  iii. Business owner  iv. Farmer"
        ),
        stimulus_data=(
            "READING PASSAGE (questions 1 and 2):\n" + PASSAGE_1 + "\n\n"
            "ANSWER KEY (1 mark per item, no part marks):\n"
            "a. iii. rural area\n"
            "b. ii. He earns irregularly for daily needs\n"
            "c. iv. Adverb\n"
            "d. i. From morning to night\n"
            "e. iv. Adjective\n"
            "f. iii. Rough\n"
            "g. ii. Household manager and caregiver\n\n"
            "Accept the option written as the roman numeral alone (“iii”), as the option "
            "text alone (“rural area”), or as both. Misspelling a correctly chosen "
            "option does NOT cost the mark.\n\n" + ITEM_NOTE
        ),
        source_text="",
    ),
    # ---- 2 -----------------------------------------------------------------
    dict(
        question_no=2,
        task_type="Short_Answer",
        marks=10,
        prompt_given=(
            "Answer the following questions from the passage above. (2 x 5 = 10)\n"
            "a. Where do Joynul and Nazma live?\n"
            "b. What types of work does Joynul do for the villagers?\n"
            "c. Why does Joynul Ali face financial difficulties?\n"
            "d. What does Nazma do throughout the day?\n"
            "e. How does Nazma contribute to her children’s early education?"
        ),
        stimulus_data=(
            "READING PASSAGE (questions 1 and 2):\n" + PASSAGE_1 + "\n\n"
            "ANSWER KEY — indicative content, 2 marks per item:\n"
            "a. They live in a small village (a rural area).\n"
            "b. He is a carpenter; he is hired to make chairs, tables, pira (low stools) "
            "and other furniture, and to do small repair work.\n"
            "c. He does not find work every day, so his income is irregular and he lives "
            "from hand to mouth.\n"
            "d. She does all the work at home from morning to night, rain or shine.\n"
            "e. She sits with Mitu and Nitu and teaches them the Bangla and English "
            "alphabets and some numbers, since they do not go to school.\n\n"
            "PART MARKS APPLY on this question: 2 = the required content is there in a "
            "sentence of the student’s own that a reader understands (a spelling slip or "
            "small grammatical error does not cost the second mark); 1 = partly correct, "
            "or the right content lifted word-for-word as a bare fragment, or a sentence "
            "so broken that the content survives only by inference; 0 = wrong or omitted. "
            "Wording need not match the key — any answer carrying the same content is "
            "correct. Put the sum of the five item marks (0–10) in context_content_data "
            "and leave the other three criteria at 0.\n\n" + ITEM_NOTE
        ),
        source_text="",
    ),
    # ---- 3 -----------------------------------------------------------------
    dict(
        question_no=3,
        task_type="Gap_Fill",
        marks=5,
        prompt_given=(
            "Read the passage. Then answer the question no. 3: Fill in each gap with a "
            "suitable word based on the information of the text above. (1 x 5 = 5)\n"
            "The audience watched the (a) — very carefully. The performers gave (b) — "
            "performances. Everyone clapped and (c) — loudly. The stage was decorated "
            "with lights. The show created great (d) — among the people. The show "
            "continued with loud (e) —."
        ),
        stimulus_data=(
            "READING PASSAGE (questions 3 and 4):\n" + PASSAGE_2 + "\n\n"
            "ANSWER KEY (1 mark per gap). No clue words are supplied, so ANY word that "
            "fits the sentence grammatically and agrees with the passage is correct:\n"
            "a. show / performance / magic show\n"
            "b. great / excellent / wonderful / fine\n"
            "c. cheered\n"
            "d. excitement / enthusiasm / interest\n"
            "e. cheers / applause / cheering / clapping\n\n"
            "A word that fits the sense but is misspelled still earns the mark. A word of "
            "the wrong part of speech, or one that contradicts the passage, does not.\n\n"
            + ITEM_NOTE
        ),
        source_text="",
    ),
    # ---- 4 -----------------------------------------------------------------
    dict(
        question_no=4,
        task_type="Vocabulary",
        marks=5,
        prompt_given=(
            "Read the above passage and write the antonyms or synonyms of the words as "
            "directed below. (1 x 5 = 5)\n"
            "a. Beautiful (Antonym), b. Brave (synonym), c. Clean (synonym), "
            "d. Old (synonym), e. Quiet (Antonym)"
        ),
        stimulus_data=(
            "READING PASSAGE (questions 3 and 4):\n" + PASSAGE_2 + "\n\n"
            "ANSWER KEY (1 mark per item). Any correct word of the required relation is "
            "acceptable; the words below are indicative, not exhaustive:\n"
            "a. Beautiful (Antonym): ugly / hideous / unattractive\n"
            "b. Brave (Synonym): courageous / bold / valiant / fearless\n"
            "c. Clean (Synonym): tidy / neat / spotless / pure\n"
            "d. Old (Synonym): aged / ancient / elderly / antique\n"
            "e. Quiet (Antonym): noisy / loud / boisterous\n\n"
            "Giving a synonym where an antonym was asked (or the reverse) scores 0 for "
            "that item. A correct word that is misspelled still earns the mark.\n\n"
            + ITEM_NOTE
        ),
        source_text="",
    ),
    # ---- 5 -----------------------------------------------------------------
    dict(
        question_no=5,
        task_type="Table_Completion",
        marks=5,
        prompt_given=(
            "Read the passage below and answer the question no. 5: Complete the following "
            "table with the information given in the passage. (1 x 5 = 5)\n"
            "| Who? | Event | When/Where |\n"
            "| Neil Armstrong, Michael Collins, and Edwin Aldrin | i. — | July 16, 1969 |\n"
            "| ii. — | Landed on the moon | July 20, 1969 |\n"
            "| Neil Armstrong | Became the first person to walk on the moon | iii. — |\n"
            "| The astronauts | iv. — | July 24, 1969 |\n"
            "| People | Watched the news | v. — |"
        ),
        stimulus_data=(
            "READING PASSAGE (questions 5, 6 and 7):\n" + PASSAGE_3 + "\n\n"
            "ANSWER KEY (1 mark per blank):\n"
            "i. Began their journey to the Moon (their spacecraft Apollo 11 was launched "
            "from Florida, USA)\n"
            "ii. Armstrong and Aldrin (Neil Armstrong and Edwin Aldrin)\n"
            "iii. July 20, 1969 (on the Moon)\n"
            "iv. Came back safely to Earth\n"
            "v. All over the world\n\n"
            "This question asks for information lifted from the passage, so copying the "
            "passage’s wording is CORRECT here and must not be penalised.\n\n" + ITEM_NOTE
        ),
        source_text="",
    ),
    # ---- 6 -----------------------------------------------------------------
    dict(
        question_no=6,
        task_type="True_False",
        marks=5,
        prompt_given=(
            "Read the following statements. Write ‘True’ in your answer script if the "
            "statement is true. Write ‘False’ if the statement is false. If false, give "
            "the correct answer. (1 x 5 = 5)\n"
            "a. Apollo 11 was a marine expedition to the Moon that began in 1969.\n"
            "b. Michael Collins landed on the Moon along with Neil Armstrong and Edwin "
            "Aldrin.\n"
            "c. Neil Armstrong was the final person to walk on the Moon’s surface.\n"
            "d. The spacecraft was propelled from Florida, USA, carrying three "
            "astronauts.\n"
            "e. The mission proved that humans had already achieved interplanetary travel "
            "to Mars before 1969."
        ),
        stimulus_data=(
            "READING PASSAGE (questions 5, 6 and 7):\n" + PASSAGE_3 + "\n\n"
            "ANSWER KEY (1 mark per item):\n"
            "a. False — Apollo 11 was a space expedition (a spaceflight to the Moon), not "
            "a marine expedition.\n"
            "b. False — Michael Collins stayed in the main ship; only Armstrong and Aldrin "
            "landed on the Moon.\n"
            "c. False — Neil Armstrong was the FIRST person to walk on the Moon’s "
            "surface.\n"
            "d. True\n"
            "e. False — the mission was the first time humans visited another world, the "
            "Moon; it had nothing to do with travel to Mars.\n\n"
            "A ‘False’ verdict with NO correction, or with a wrong correction, scores 0 "
            "for that item — the question requires both. The correction need not match "
            "the key’s wording; any correction carrying the right fact is accepted.\n\n"
            + ITEM_NOTE
        ),
        source_text="",
    ),
    # ---- 7 -----------------------------------------------------------------
    dict(
        question_no=7,
        task_type="Summary",
        marks=5,
        prompt_given="Summarize the text in your own words. (5)",
        stimulus_data="TEXT TO BE SUMMARIZED (questions 5, 6 and 7):\n" + PASSAGE_3,
        source_text=PASSAGE_3,
    ),
    # ---- 8 -----------------------------------------------------------------
    dict(
        question_no=8,
        task_type="Sentence_Transformation",
        marks=5,
        prompt_given=(
            "Rewrite the following sentences using “Do you know” at the beginning. "
            "(1 x 5 = 5)\n"
            "a. Where is the post office?\n"
            "b. What does the postmaster do?\n"
            "c. Why is the postmaster so happy?\n"
            "d. Where do his children go to school?\n"
            "e. When does the postmaster come to the post office?"
        ),
        stimulus_data=(
            "ANSWER KEY (1 mark per item). The transformation must turn the direct "
            "question into an embedded clause — subject before verb, no auxiliary "
            "do/does/did, question mark retained:\n"
            "a. Do you know where the post office is?\n"
            "b. Do you know what the postmaster does?\n"
            "c. Do you know why the postmaster is so happy?\n"
            "d. Do you know where his children go to school?\n"
            "e. Do you know when the postmaster comes to the post office?\n\n"
            "The mark is for the WORD ORDER of the embedded clause. An item that keeps "
            "the direct-question order (“Do you know where is the post office?”) or keeps "
            "the auxiliary (“…what does the postmaster do?”) is incorrect. A correct "
            "transformation with a spelling slip or a missing question mark still earns "
            "the mark.\n\n" + ITEM_NOTE
        ),
        source_text="",
    ),
    # ---- 9 -----------------------------------------------------------------
    dict(
        question_no=9,
        task_type="Rearrange",
        marks=8,
        prompt_given=(
            "Rearrange the following sentences in correct order and write them in a "
            "paragraph. (1 x 8 = 8)\n"
            "a. I waved my torch desperately, but the car did not stop.\n"
            "b. Feeling relieved and safe, I politely began to request the owner to drop "
            "me near Lalpur.\n"
            "c. Suddenly, I noticed two large yellow eyes approaching me.\n"
            "d. As the car passed by, I opened the door and jumped inside without "
            "hesitation.\n"
            "e. But soon, I realized the eyes were too large for that.\n"
            "f. At first, I was frightened and wondered if it was a tiger.\n"
            "g. As it came closer, I discovered it was a slow-moving Baby Austin car.\n"
            "h. Growing impatient, I made a quick decision."
        ),
        stimulus_data=(
            "ANSWER KEY — correct order (8 positions, 1 mark each):\n"
            "1. c  (Suddenly, I noticed two large yellow eyes approaching me.)\n"
            "2. f  (At first, I was frightened and wondered if it was a tiger.)\n"
            "3. e  (But soon, I realized the eyes were too large for that.)\n"
            "4. g  (As it came closer, I discovered it was a slow-moving Baby Austin "
            "car.)\n"
            "5. a  (I waved my torch desperately, but the car did not stop.)\n"
            "6. h  (Growing impatient, I made a quick decision.)\n"
            "7. d  (As the car passed by, I opened the door and jumped inside without "
            "hesitation.)\n"
            "8. b  (Feeling relieved and safe, I politely began to request the owner to "
            "drop me near Lalpur.)\n\n"
            "Award 1 mark for each sentence the student places in its correct POSITION in "
            "the sequence above — a position-by-position comparison: c in slot 1 earns a "
            "mark, c anywhere else does not. The student may write the letters, the full "
            "sentences, or both; mark the order either way, and do not penalise copying "
            "slips inside the sentences.\n\n" + ITEM_NOTE
        ),
        source_text="",
    ),
    # ---- 10 ----------------------------------------------------------------
    dict(
        question_no=10,
        task_type="Poem_Short_Answer",
        marks=5,
        prompt_given=(
            "Answer any 5 (five) of the following questions from poems. (1 x 5 = 5)\n"
            "a. What does the parent say about their responsibility in the poem “Whose "
            "Child is This”?\n"
            "b. What is the poet’s suggestion about nature in the poem “Leisure”?\n"
            "c. How does the poet feel about life without time for nature in the poem "
            "“Leisure”?\n"
            "d. How does the parent show care for the child in the poem “Whose Child is "
            "This”?\n"
            "e. Why is the childhood important according to the poem “Whose Child is "
            "This”?\n"
            "f. How does the teacher describe his role in the poem “Whose Child is "
            "This”?\n"
            "g. What role does nature play in the poem “Leisure”?\n"
            "h. What is the poet’s message in the poem “Leisure”?"
        ),
        stimulus_data=(
            "ANSWER KEY — indicative content, 1 mark per answer. The poems are "
            "W. H. Davies’s “Leisure” and “Whose Child is This?”, in which a parent, a "
            "teacher and the community each claim the child in turn.\n"
            "a. The parent claims the child as their own and accepts responsibility for "
            "loving, feeding, clothing and protecting the child.\n"
            "b. The poet suggests we should stop and enjoy the beauty of nature instead "
            "of hurrying past it.\n"
            "c. He calls such a life poor — a life full of care with no time to stand and "
            "stare is not worth living.\n"
            "d. By feeding, clothing, nursing, loving and guiding the child day after "
            "day.\n"
            "e. Because the care, love and teaching a child receives in childhood shape "
            "the person the child becomes.\n"
            "f. The teacher claims the child too — as a learner to be taught, guided and "
            "helped to grow in knowledge and character.\n"
            "g. Nature is the source of beauty and joy — boughs, sheep and cows, "
            "squirrels, streams full of stars, Beauty’s dance — that busy people never "
            "notice.\n"
            "h. Life is impoverished if we are too busy with worldly cares to pause and "
            "appreciate the beauty around us.\n\n"
            "ANY FIVE answers count, and only five are marked: if the student answers "
            "more than five, mark the FIRST five attempted and ignore the rest; if fewer "
            "than five, the unanswered ones score 0. Wording need not match the key — "
            "award the mark for any answer carrying the same idea. Award 0 for an answer "
            "about the wrong poem or one that merely repeats the question.\n\n" + ITEM_NOTE
        ),
        source_text="",
    ),
    # ---- 11 ----------------------------------------------------------------
    dict(
        question_no=11,
        task_type="Story",
        marks=10,
        prompt_given=(
            "Section-B : Writing. Read the opening of a story below and write at least "
            "ten new sentences to complete it. Give your story a suitable title. (10)\n"
            "Once upon a time two friends went on a journey. They had to go through a "
            "forest on the way. As they went the wood they saw a bear………"
        ),
        stimulus_data="",
        source_text="",
    ),
    # ---- 12 ----------------------------------------------------------------
    dict(
        question_no=12,
        task_type="Paragraph",
        marks=10,
        prompt_given=(
            "Write a paragraph on “Annual Prize-Giving Day of Your School” in about 150 "
            "words. (10)"
        ),
        stimulus_data="",
        source_text="",
    ),
    # ---- 13 ----------------------------------------------------------------
    dict(
        question_no=13,
        task_type="Dialogue",
        marks=10,
        prompt_given=(
            "Suppose, your friend Sheila has done very well in the last annual "
            "examination result. Your own performance is also very nice. Now write a "
            "dialogue between you two about your results. (10)"
        ),
        stimulus_data="",
        source_text="",
    ),
]

FIELDS = [
    "question_no", "task_type", "marks", "prompt_given", "stimulus_data",
    "source_text",
]


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "data" / "Question" / "task_stems_SE_07_Q1.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for row in ROWS:
            w.writerow(row)
    total = sum(int(r["marks"]) for r in ROWS)
    print(f"wrote {out}  ({len(ROWS)} rows, {total} marks + 10 Class Test = {total + 10})")


if __name__ == "__main__":
    main()
