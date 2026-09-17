# Exam sets — running more than one question paper

Until now the pipeline handled a single paper (Class XI, Section B writing only)
and the paper's shape was spread through the code: `target_questions` in the
config, question numbers hard-coded in the metric logic (`if qno == "3"`), one
extraction prompt, one rubric, one set of output paths. Adding the Class VII
paper — a different class, 13 questions, most of them objective — meant turning
all of that into data.

## The model

A script's **exam set** is its id minus the trailing serial:

```
SE_07_Q1_0003  ->  SE_07_Q1
SE_11_Q1_0017  ->  SE_11_Q1
```

`src.config.exam_set_of()` is the single implementation of that rule. Everything
that differs between papers hangs off the set, in `config.yaml`:

```yaml
exam_sets:
  SE_07_Q1:
    label: "Class VII English 1st Paper"
    task_stems: data/Question/task_stems_SE_07_Q1.csv
    prompt_file: prompts/extraction_prompt_c7_v1.md
    rubric_file: prompts/rubric_c7_v1.txt
    target_questions: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    max_question: 13
    extraction_csv: data/extraction_SE_07_Q1.csv
    extraction_runs_csv: data/extraction_runs_SE_07_Q1.csv
    eval_output_dir: output/SE_07_Q1
default_exam_set: SE_07_Q1
```

Adding a third paper is a config entry plus two prompt files, not a code change.

Every stage takes `--set`, defaulting to `default_exam_set`. `--set all` widens
the PDF-facing stages to every set at once; the CSV and grading stages are
always single-set, because their outputs are per-set files.

```
python scripts/01_convert.py        --set SE_07_Q1
python scripts/02_extract.py        --set SE_07_Q1
python scripts/03_validate.py       --set SE_07_Q1
python scripts/04_build_csv.py      --set SE_07_Q1
python scripts/eval/01_evaluate.py  --set SE_07_Q1 --k 3
python scripts/eval/02_emit_csv.py  --set SE_07_Q1
python scripts/eval/04_emit_avg.py  --set SE_07_Q1
python scripts/eval/03_validate.py  --set SE_07_Q1
```

or `scripts/run_set.sh SE_07_Q1` for all of it in order.

**The Class XI run is not disturbed.** `SE_11_Q1` keeps `data/extraction.csv` and
the flat `output/` directory where its completed 720-record grading run already
lives; only the new set writes anywhere new. Regenerating the Class XI
`extraction.csv` under the refactor reproduces the previous file field-for-field
(verified), with one added column, `exam_set`.

## Shared vs per-set storage

| Data | Location | Shared? |
|---|---|---|
| Source PDFs | `data/raw_pdfs/<set>/` | per-set directory (discovery is recursive) |
| Page images | `data/images/<script_id>/` | shared; script ids are already prefixed |
| Transcripts | `data/transcripts/<script_id>.json` | shared, same reason |
| LLM cache | `data/cache/` | shared; keyed by content hash |
| extraction.csv | per set (see config) | no |
| Grading records + CSVs | `<eval_output_dir>/` | no |
| Reports | `logs/*_<set>.md` | no |

## What became question-type-driven

Row metrics used to key off the question number. They now key off
`question_type`, which comes from the set's task stems:

| Metric | Before | Now |
|---|---|---|
| absorbed-correction (`suspected_red_ink`) check | `qno in {3,7,8,9,11}` | `question_type in PROSE_TYPES` |
| `wrap_as_paragraph` check | `qno == 7` | `question_type == "Paragraph"` |
| length ratio / verbatim overlap | `qno in {3, 11}` | any question whose stem supplies `source_text` |
| `exceeds_one_third` | `qno == 3` | `question_type == "Summary"` |
| letter-component count | `qno == 10` | `question_type == "Letter_Email"` |
| row-count assertion | `scripts * 6` | `scripts * len(target_questions)` |

On `SE_11_Q1` every one of these resolves to exactly the old question set, which
is why the regenerated CSV is identical.

## Sub-score ceilings are keyed by (type, max_mark)

`SUBSCORE_CEILINGS` was keyed by task type alone. That breaks the moment two
papers share a task name at different marks — a Class XI Summary is out of 10, a
Class VII Summary out of 5, and the splits are not proportional. The key is now
`(question_type, max_mark)`, and each row must sum to its `max_mark`.

Structural caps moved the same way. `CAP_VALUES` held fixed numbers (Summary →
5); it is now `CAP_FRACTIONS` plus `cap_value(reason, max_mark)`, so the summary
cap lands on 5 out of 10 and 2 out of 5. The fractions reproduce every previous
Class XI value exactly, including the deliberate 0.6 exception for
`Graph_External_Facts`.

## Class VII: two marking modes in one rubric

Ten of the thirteen Class VII questions are **item-scored** — MCQ, gap-fill,
synonyms, table completion, true/false, sentence transformation, sentence
rearrangement, poem questions, and the short comprehension answers. They have
right answers, so impression marking is meaningless for them.

Two consequences:

1. **The answer key ships with the question.** `stimulus_data` in
   `task_stems_SE_07_Q1.csv` carries the reading passage *and* the key, per item,
   with an explicit statement of what does and does not cost a mark (a misspelled
   but clearly intended correct synonym earns it; a "False" with no correction
   does not). Without a key an LLM cannot mark a rearrangement at all. The keys
   are generated by `scripts/gen_task_stems_SE_07_Q1.py` so they are reviewable
   and the CSV is regenerable.

2. **An item-scored question puts its whole mark in `context_content_data`** and
   fixes the other three criteria at 0, applying no structural cap. The
   evaluation validator asserts this. The four-criterion schema is unchanged, so
   every downstream CSV, aggregation and report works on both papers.

The remaining three — summary, story, paragraph, dialogue — are marked
holistically against the four criteria as before.

## Rate limiting

The AI Studio free tier caps **input tokens per minute per model** (16,000 for
`gemma-4-31b`). A grading call is ~6.3k input tokens because the whole rubric
rides in every one, so three workers firing together exhaust the budget
instantly. The 429s that follow carry a ~50s `retryDelay` that the old
exponential backoff exhausted long before it cleared — the first Class VII
grading attempt came back 12 of 13 FAILED.

`src.llm_client.TokenRateLimiter` paces calls client-side against a sliding
60-second window (`llm.tokens_per_minute` / `evaluation.tokens_per_minute`; 0
disables it on a paid tier), every attempt is charged rather than just the first,
and the backoff now never sleeps for less than the server asked for. Concurrency
was raised in step — both stages are latency-bound well before they are
token-bound, so too few workers leaves the budget idle.
