# -*- coding: utf-8 -*-
"""One-off generator: data/Question/task_stems_SE_06_Q1.csv (Class VI, 13 questions).

Question text transcribed from data/Question/SE_06_Q1.pdf (a scanned paper — no
text layer, so it was read off the page images).

Same shape as the Class VII paper but NOT the same marks: here Q2 is 1x5=5 and
Q10 is 2x5=10 (Class VII has those the other way round), and Q8 asks the student
to compose sentences using given prepositions rather than to transform given
sentences. Marks: 7,5,5,5,5,5,5,5,8,10 (Part A = 60) + 10,10,10 (Part B = 30)
= 90, plus a 10-mark Class Test that is not written in the script.

See scripts/gen_task_stems_SE_07_Q1.py for the rationale on shipping answer keys
inside stimulus_data. Run from the project root.
"""

from __future__ import annotations

import csv
from pathlib import Path

PASSAGE_1 = """December 16 is a remarkable day for Bangladeshi people. On this day in 1971. Bangladesh got independence after a nine-month-long blood-soaked battle against the Pakistani army. It is a day of national pride as well as a day of commemorating the sacrifices of the millions of Bangladeshi people. Pakistani army killed three million people and assaulted thousands of women.
The Victory Day is celebrated with great enthusiasm across the country. The day begins with a 31-gun salute at dawn, followed by the hoisting of the national flag at government buildings and private institutions. Different political, cultural and educational programmes are organised to pay homage to the martyrs of the war. The entire nation celebrates the day with colourful flags, banners and parades.
The National Parade ground hosts a grand military parade where the president, the prime minister and dignitaries from government and non-government sectors join, The National Martyrs Memorial in Savar, our monument dedicated to the soldiers and civilians who lost their lives during the war, becomes a focal point for citizens to pay their respect. People from all walks of life gather to offer flowers, recite patriotic songs, and recall the sacrifices made by the nation's heroes."""

PASSAGE_2 = """That night Ravi dreamt about gold coins and jewels. The next morning they packed everything into the jeep and went back to Jaipur, Uncle Santosh went to the museum to show his friend the papers. When he came back, he told the boys, "Those papers were very valuable. The museum people bought them for ten thousand rupees." The boys were delighted. When Ravi met his father and mother once again, he told them about his adventures and gave them Rs 50,000. Rajendra and Ravi never forgot the kind owl who helped them to find the old box."""

PASSAGE_3 = """Raju is a student of class six. Last week, his school held class tests in several subjects. Raju worked very hard to do well. He got 8 in Bangla, 9 in English, and 10 in Mathematics out of 10. His teachers praised him for his good marks. However, he could not do well in ICT, Science, and Religion, getting only 6 marks in each.
Raju felt a little sad about his low marks but promised to study harder next time. His father told him not to lose hope. His mother advised him to read ICT and Science regularly. Raju made a plan to give more time to those subjects. He was happy that he did very well in Bangla, English, and Math. Raju believes that with more practice, he can improve all his subjects."""

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
            "Read the text and answer question 1: Choose the best answer from the "
            "alternatives. (1 x 7 = 7)\n"
            "a. On the 16th of December of the same year, the 26 March declaration of "
            "independence became a reality. Here the meaning of the word reality is — "
            "i. experience  ii. certainty  iii. virtuality  iv. actuality\n"
            "b. To every Bangladeshi the 16th December is a day of — i. self analysis  "
            "ii. sorrows and sufferings  iii. self questioning  iv. pride and joy\n"
            "c. We pay our deep respect to the martyrs who gave their lives for the "
            "independence of Bangladesh. Here respect means — i. neglect  ii. honour  "
            "iii. favour  iv. dignity\n"
            "d. We, the Bangalees celebrate our Victory Day by remembering the roles "
            "played by our fearless freedom fighters and the people at large. Which of "
            "the following words has a similar meaning of fearless? i. curious  "
            "ii. courageous  iii. fearful  iv. rebillion\n"
            "e. This is why December 16 is our Victory Day when the Pakistan army "
            "surrendered unconditionally. What does surrendered mean here? i. defended  "
            "ii. submitted  iii. captured  iv. with stood\n"
            "f. To build up our nation as a prosperous country we have to work — "
            "i. individually  ii. separately  iii. independently  iv. unitedly\n"
            "g. 1971 is the year of our — i. Liberation War  ii. Independence  "
            "iii. Victory day  iv. all of i, ii & iii"
        ),
        stimulus_data=(
            "READING PASSAGE (questions 1 and 2):\n" + PASSAGE_1 + "\n\n"
            "ANSWER KEY (1 mark per item, no part marks):\n"
            "a. iv. actuality\n"
            "b. iv. pride and joy\n"
            "c. ii. honour\n"
            "d. ii. courageous\n"
            "e. ii. submitted\n"
            "f. iv. unitedly\n"
            "g. iv. all of i, ii & iii\n\n"
            "Accept the option written as the roman numeral alone (“iv”), as the option "
            "text alone (“actuality”), or as both. Misspelling a correctly chosen option "
            "does NOT cost the mark.\n\n" + ITEM_NOTE
        ),
        source_text="",
    ),
    # ---- 2 -----------------------------------------------------------------
    dict(
        question_no=2,
        task_type="Short_Answer",
        marks=5,
        prompt_given=(
            "Write short answers to the following questions. (1 x 5 = 5)\n"
            "a. What happened during the war in 1971?\n"
            "b. How do people celebrate Victory Day?\n"
            "c. Why is the National Martyrs’ Memorial important?\n"
            "d. What do people sing at the National Martyrs’ Memorial?\n"
            "e. What does the 31-gun salute represent on Victory Day?"
        ),
        stimulus_data=(
            "READING PASSAGE (questions 1 and 2):\n" + PASSAGE_1 + "\n\n"
            "ANSWER KEY — indicative content, 1 mark per item (whole mark or nothing; "
            "there are no half marks on this question):\n"
            "a. Bangladesh fought a nine-month-long battle against the Pakistani army, "
            "who killed three million people and assaulted thousands of women; the "
            "country won independence on 16 December 1971.\n"
            "b. With a 31-gun salute at dawn, hoisting of the national flag, political, "
            "cultural and educational programmes, and parades with colourful flags and "
            "banners.\n"
            "c. It is the monument dedicated to the soldiers and civilians who lost their "
            "lives during the war, and it becomes the focal point where citizens pay "
            "their respect.\n"
            "d. Patriotic songs.\n"
            "e. It marks the start of Victory Day at dawn and honours the martyrs and the "
            "nation's victory.\n\n"
            "Wording need not match the key — award the mark for any answer carrying the "
            "same content, in the student's own words or lifted from the passage. Award 0 "
            "for a wrong, off-question or omitted answer.\n\n" + ITEM_NOTE
        ),
        source_text="",
    ),
    # ---- 3 -----------------------------------------------------------------
    dict(
        question_no=3,
        task_type="Gap_Fill",
        marks=5,
        prompt_given=(
            "Read the text and answer question 3: Complete the passage with suitable "
            "words. (1 x 5 = 5)\n"
            "After finding the box Rajendra and Ravi (a) — about valuable stones. Then "
            "they (b) — to Jaipur. After showing the papers to the museum people, Uncle "
            "Santosh (c) — to know that the papers were valuable. He (d) — the papers to "
            "the museum for ten thousand rupees. Later, (e) — the news, the boy were "
            "happy."
        ),
        stimulus_data=(
            "READING PASSAGE (questions 3 and 4):\n" + PASSAGE_2 + "\n\n"
            "ANSWER KEY (1 mark per gap). No clue words are supplied, so ANY word that "
            "fits the sentence grammatically and agrees with the passage is correct:\n"
            "a. dreamt / dreamed / talked / thought\n"
            "b. went back / returned / came back\n"
            "c. was surprised / was glad / was happy / came\n"
            "d. sold\n"
            "e. hearing / on hearing / after hearing / getting\n\n"
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
            "Read the passage carefully and replace the following words with their "
            "suitable synonyms or antonyms. (1 x 5 = 5)\n"
            "(a) Went back (antonym), (b) Museum (synonym), (c) Friend (synonym), "
            "(d) Show (antonym), (e) Bought (synonym)"
        ),
        stimulus_data=(
            "READING PASSAGE (questions 3 and 4):\n" + PASSAGE_2 + "\n\n"
            "ANSWER KEY (1 mark per item). Any correct word of the required relation is "
            "acceptable; the words below are indicative, not exhaustive:\n"
            "a. Went back (Antonym): came / went forward / advanced / set out / left\n"
            "b. Museum (Synonym): gallery / archive / repository / exhibition hall\n"
            "c. Friend (Synonym): companion / mate / comrade / ally / pal\n"
            "d. Show (Antonym): hide / conceal / cover\n"
            "e. Bought (Synonym): purchased\n\n"
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
            "Read the text and answer question 5: Complete the table with information "
            "from the passage. (1 x 5 = 5)\n"
            "| Topic | Information from the passage |\n"
            "| a. Class Raju reads in | (i) |\n"
            "| b. Subjects he did well in | (ii) |\n"
            "| c. Marks he got in those subjects | (iii) |\n"
            "| d. Subjects he did poorly in | (iv) |\n"
            "| e. Marks he got in weak subjects | (v) |"
        ),
        stimulus_data=(
            "READING PASSAGE (questions 5, 6 and 7):\n" + PASSAGE_3 + "\n\n"
            "ANSWER KEY (1 mark per blank):\n"
            "i. Class six\n"
            "ii. Bangla, English and Mathematics\n"
            "iii. 8 in Bangla, 9 in English and 10 in Mathematics (out of 10)\n"
            "iv. ICT, Science and Religion\n"
            "v. 6 marks in each\n\n"
            "This question asks for information lifted from the passage, so copying the "
            "passage's wording is CORRECT here and must not be penalised. A partial but "
            "unambiguous answer (e.g. naming all three subjects in ii) earns the mark; an "
            "answer that names only some of the required items does not.\n\n" + ITEM_NOTE
        ),
        source_text="",
    ),
    # ---- 6 -----------------------------------------------------------------
    dict(
        question_no=6,
        task_type="True_False",
        marks=5,
        prompt_given=(
            "Read the statements below. Write ‘True’ if the statement is correct and "
            "‘False’ if it is incorrect. If false, write the correct answer. "
            "(1 x 5 = 5)\n"
            "a. Raju thinks he cannot improve his weak subjects.\n"
            "b. He got 6 marks out of 10 in Mathematics.\n"
            "c. Raju got full marks in Mathematics.\n"
            "d. Raju plans to work lesser next time.\n"
            "e. His father scolded him for low marks."
        ),
        stimulus_data=(
            "READING PASSAGE (questions 5, 6 and 7):\n" + PASSAGE_3 + "\n\n"
            "ANSWER KEY (1 mark per item):\n"
            "a. False — Raju believes that with more practice he can improve all his "
            "subjects.\n"
            "b. False — he got 10 out of 10 in Mathematics; 6 was his mark in ICT, "
            "Science and Religion.\n"
            "c. True\n"
            "d. False — he promised to study harder and planned to give more time to his "
            "weak subjects.\n"
            "e. False — his father told him not to lose hope.\n\n"
            "A ‘False’ verdict with NO correction, or with a wrong correction, scores 0 "
            "for that item — the question requires both. The correction need not match "
            "the key's wording; any correction carrying the right fact is accepted.\n\n"
            + ITEM_NOTE
        ),
        source_text="",
    ),
    # ---- 7 -----------------------------------------------------------------
    dict(
        question_no=7,
        task_type="Summary",
        marks=5,
        prompt_given="Write a summary of the passage in your own words. (5)",
        stimulus_data="TEXT TO BE SUMMARIZED (questions 5, 6 and 7):\n" + PASSAGE_3,
        source_text=PASSAGE_3,
    ),
    # ---- 8 -----------------------------------------------------------------
    dict(
        question_no=8,
        task_type="Preposition_Use",
        marks=5,
        prompt_given=(
            "Use these prepositions in sentences of your own. (1 x 5 = 5)\n"
            "a. bottom of, b. top of, c. through, d. along, e. across"
        ),
        stimulus_data=(
            "ANSWER KEY (1 mark per item). There is no fixed answer: the student composes "
            "their own sentence for each preposition. Award the mark when the sentence "
            "(1) actually contains the given preposition, (2) uses it in a sense that is "
            "correct English, and (3) is a complete, intelligible sentence.\n\n"
            "Model sentences, for reference only — a student sentence need not resemble "
            "them:\n"
            "a. bottom of: The old box lay at the bottom of the river.\n"
            "b. top of: A small flag was fixed on the top of the hill.\n"
            "c. through: We walked through the forest before sunset.\n"
            "d. along: They strolled along the river bank in the evening.\n"
            "e. across: A wooden bridge runs across the canal.\n\n"
            "Award 0 for an item where the preposition is missing, is used in a way that "
            "is not English (e.g. “He is bottom of happy”), or where nothing resembling a "
            "sentence was written. A minor spelling or article slip does NOT cost the "
            "mark — the item tests the preposition, not general accuracy.\n\n" + ITEM_NOTE
        ),
        source_text="",
    ),
    # ---- 9 -----------------------------------------------------------------
    dict(
        question_no=9,
        task_type="Rearrange",
        marks=8,
        prompt_given=(
            "Put the following parts of the story in the correct order to make the whole "
            "story. Only the corresponding numbers of the sentences need to be written. "
            "(1 x 8 = 8)\n"
            "a. She asked Bayazid to give her a glass of water, but he could not find any "
            "water in the pitcher.\n"
            "b. Once Hazrat Bayazid Bustami came home to see his ailing mother.\n"
            "c. But she again fell asleep.\n"
            "d. So, he went to the well quite far from their house.\n"
            "e. She woke up some hours later.\n"
            "f. He filled the pitcher, came back and went to his mother with a glass of "
            "water.\n"
            "g. As he was still standing by her bed, his mother drank the glass of water "
            "and blessed him from the core of heart.\n"
            "h. He, instead of waking her up, stood by her bed with the glass of water in "
            "his hand."
        ),
        stimulus_data=(
            "ANSWER KEY — correct order (8 positions, 1 mark each):\n"
            "1. b  (Once Hazrat Bayazid Bustami came home to see his ailing mother.)\n"
            "2. a  (She asked Bayazid to give her a glass of water, but he could not find "
            "any water in the pitcher.)\n"
            "3. d  (So, he went to the well quite far from their house.)\n"
            "4. f  (He filled the pitcher, came back and went to his mother with a glass "
            "of water.)\n"
            "5. c  (But she again fell asleep.)\n"
            "6. h  (He, instead of waking her up, stood by her bed with the glass of "
            "water in his hand.)\n"
            "7. e  (She woke up some hours later.)\n"
            "8. g  (As he was still standing by her bed, his mother drank the glass of "
            "water and blessed him from the core of heart.)\n\n"
            "Award 1 mark for each sentence the student places in its correct POSITION in "
            "the sequence above — a position-by-position comparison: b in slot 1 earns a "
            "mark, b anywhere else does not. The student may give the letters, the full "
            "sentences, or both; mark the order either way, and do not penalise copying "
            "slips inside the sentences.\n\n" + ITEM_NOTE
        ),
        source_text="",
    ),
    # ---- 10 ----------------------------------------------------------------
    dict(
        question_no=10,
        task_type="Poem_Short_Answer",
        marks=10,
        prompt_given=(
            "Answer any five of the following questions from the poems in your text book. "
            "(2 x 5 = 10)\n"
            "a. What does Robin say about autumn?\n"
            "b. How is Robin a happy bird?\n"
            "c. How is Robin a hopeful bird?\n"
            "d. What reason does Robin give for singing during the winter?\n"
            "e. What does Robin teach us in the poem?\n"
            "f. Why does the playmate come only on a rainy day?\n"
            "g. Why does the boy think the playmate lives in another land?\n"
            "h. What does the Moon come there to do?"
        ),
        stimulus_data=(
            "ANSWER KEY — INDICATIVE ONLY, 2 marks per answer. The poem texts are not "
            "supplied with this paper, so the notes below describe the expected substance "
            "rather than fixed wording. Items a–e concern the poem about Robin, a bird "
            "who goes on singing as autumn turns to winter; items f–h concern the poem "
            "about a boy and an imagined playmate who visits on rainy days.\n"
            "a. That autumn is passing and the cold season is coming — leaves fall and the "
            "bright days are ending.\n"
            "b. He sings cheerfully whatever the season, finding joy in small things "
            "rather than complaining of the cold.\n"
            "c. He keeps singing through the bleak season because he trusts that spring "
            "and better days will return.\n"
            "d. He sings in winter to keep up his own and others' spirits, and because he "
            "believes the hard season will pass.\n"
            "e. To stay cheerful and hopeful in hard times instead of giving way to "
            "gloom.\n"
            "f. Because the rain keeps the boy indoors and alone, so that is when he longs "
            "for and imagines his playmate.\n"
            "g. Because the playmate is never seen among the people he knows and appears "
            "only in his imagination, so the boy supposes he must come from some distant, "
            "unknown land.\n"
            "h. It comes to look in on the child and keep him company — to shine on him "
            "and share the night with him.\n\n"
            "BECAUSE THE KEY IS INDICATIVE, mark generously on substance: award 2 for an "
            "answer that is clearly about the right poem and carries a sensible reading "
            "consistent with these notes, 1 for a partly relevant or very thin answer, and "
            "0 for an answer about the wrong poem, one that merely repeats the question, "
            "or one left blank.\n\n"
            "ANY FIVE answers count, and only five are marked: if the student answers more "
            "than five, mark the FIRST five attempted and ignore the rest; if fewer than "
            "five, the unanswered ones score 0. Put the sum of the five item marks (0–10) "
            "in context_content_data and leave the other three criteria at 0."
        ),
        source_text="",
    ),
    # ---- 11 ----------------------------------------------------------------
    dict(
        question_no=11,
        task_type="Story",
        marks=10,
        prompt_given=(
            "Part-B : Writing. Read the opening of a story below and complete it in at "
            "least ten new sentences. Give a suitable title to it. (10)\n"
            "Once upon a time two friends went on a journey. They had to go through a "
            "forest on the way. As they came through the wood, they saw a bear "
            "approaching…………"
        ),
        stimulus_data="",
        source_text="",
    ),
    # ---- 12 ----------------------------------------------------------------
    dict(
        question_no=12,
        task_type="Paragraph",
        marks=10,
        prompt_given='Write a paragraph in 120 words. "A Rickshaw-Puller" (10)',
        stimulus_data="",
        source_text="",
    ),
    # ---- 13 ----------------------------------------------------------------
    dict(
        question_no=13,
        task_type="Dialogue",
        marks=10,
        prompt_given=(
            "You are Shahid and you have been suffering from fever along with some "
            "Covid-19 symptoms for a week. Now, you are at the chamber of a doctor. Write "
            "a dialogue between you and the doctor. (10)"
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
    out = (
        Path(__file__).resolve().parent.parent
        / "data" / "Question" / "task_stems_SE_06_Q1.csv"
    )
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for row in ROWS:
            w.writerow(row)
    total = sum(int(r["marks"]) for r in ROWS)
    print(f"wrote {out}  ({len(ROWS)} rows, {total} marks + 10 Class Test = {total + 10})")


if __name__ == "__main__":
    main()
