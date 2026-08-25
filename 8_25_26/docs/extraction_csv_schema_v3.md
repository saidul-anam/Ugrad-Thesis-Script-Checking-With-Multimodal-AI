# `extraction.csv` — Schema Specification v3

Merged from `extraction.csv` (draft) and the divergence-study requirements.
Grain: **one row per (script × question)**. In-scope questions only: 3, 7, 8, 9, 10, 11.
Expected row count at full scale: 22 scripts × 6 questions = **132 rows**.

Canonical storage is per-script JSON under `data/transcripts/`. This CSV is a
**derived, regenerable view** — never edit it by hand, never rebuild the pipeline from it.

---

## 1. IDENTITY

| Column | Type | Example | Notes |
|---|---|---|---|
| `task_id` | string | `CHART_001` | **Primary key.** `{TYPE}_{NNN}`, zero-padded. Prefixes: `SUMMARY`, `PARA`, `CHART`, `STORY`, `LETTER`, `THEME` |
| `script_id` | string | `SE_11_Q1_0001` | PDF filename stem |
| `question_no` | string | `8` | **String, not int** — `1A`/`1B` exist if scope widens |
| `question_type` | enum | `Graph_Chart` | `Summary`\|`Paragraph`\|`Graph_Chart`\|`Story`\|`Letter_Email`\|`Theme`. Must match `task_type` in rubric_v2 |
| `max_mark` | int | `10` | 3→10, 7→10, 8→10, 9→7, 10→5, 11→8. Carried so the grader never re-derives it |

## 2. GROUND TRUTH — reserved, populated later

Capturing the examiner's red-ink marks is deferred. These columns are **emitted on every row
but left empty**, so that adding marks later is a join rather than a schema migration.

| Column | Type | Notes |
|---|---|---|
| `teacher_mark` | int \| empty | Human examiner's awarded mark, from the red marginal number. Empty for now |
| `teacher_mark_confidence` | float \| empty | 0.0–1.0. Marginal digits are often ambiguous (3 vs 5 vs 8) |
| `teacher_mark_source` | enum | `manual` \| `model` \| `absent`. **`absent` until marks are captured** |

**Population hook:** if `data/reference/teacher_marks.csv` exists, left-join on
(`script_id`, `question_no`). If absent, proceed with empty columns — no error, no warning,
no placeholder file.

> When marks do arrive: a blank means *unknown*, an explicit `0` means the examiner awarded
> nothing. Never impute one as the other, and never fill blanks with 0 to make a join tidy.
> Until then, treat the study as a two-model agreement analysis; it becomes an accuracy
> analysis against a human reference only once this column is real.

## 3. QUESTION CONTEXT

| Column | Type | Notes |
|---|---|---|
| `question` | text | Full question text **plus** reference material inline — chart values for Q8, poem text for Q3 and Q11, story stem for Q9. Makes each row self-contained for the grading prompt |
| `source_text` | text \| empty | The passage/poem alone, without instructions. Used for overlap and length computation. Empty for Q7, Q9, Q10 |

## 4. TRANSCRIPT

| Column | Type | Notes |
|---|---|---|
| `extracted_text_raw` | text | Verbatim, retains `[struck: x]`, `{inserted: x}`, `[illegible]`, `[unclear: x]`, `[cut: x]` |
| `extracted_text` | text | Grading input. Editorial markers resolved: struck text removed, insertions applied. **Spelling and grammar errors preserved untouched** |
| `student_title` | string \| empty | Student's own heading, e.g. `Size doesn't Matter`. Q9 rubric requires a title |
| `page_range` | string | `3-4` or `3`. Hyphen range, not comma |

> `extracted_text` must never be spell-checked, normalised, or smoothed. The errors are the object of measurement.

## 5. ATTEMPT STATUS

| Column | Type | Notes |
|---|---|---|
| `extraction_status` | enum | `ok` \| `not_attempted` \| `error` |
| `error_message` | text \| empty | For `not_attempted`, the verification statement. For `error`, the exception |

> **Change from draft:** `not_attempted` was previously recorded as `failed`. A skipped question is not a pipeline failure — it is a legitimate Band 0 and **must reach the grader** and score 0. Only `error` rows are excluded, and they must be reconciled before analysis.

## 6. TRANSCRIPTION QUALITY

| Column | Type | Notes |
|---|---|---|
| `confidence` | float | 0.0–1.0, model's own legibility confidence. `< 0.7` → review |
| `illegible_count` | int | Count of `[illegible]` |
| `unclear_count` | int | Count of `[unclear:` |
| `cut_count` | int | Count of `[cut:` |
| `struck_count` | int | Student's own strikethroughs |
| `inserted_count` | int | Student's own caret insertions |
| `ambiguous_authorship_count` | int | Marks that could be student or examiner |
| `red_ink_suspected` | bool | Heuristic contamination flag |
| `needs_review` | bool | Any flag tripped |
| `review_flags` | string | Semicolon-joined: `low_confidence;suspected_red_ink;wrap_as_paragraph` |

> **Sanity check:** if `illegible_count + unclear_count + cut_count` is 0 across every row, the model is guessing rather than flagging uncertainty. Across ~130 handwritten answers, expect a non-trivial number. All-zero is a defect signal, not a quality signal.

## 7. COMPUTED METRICS — deterministic, Python-side

These exist so the two graders never disagree about **counting**. Each is passed to both
models as a stated fact, with the instruction not to recount.

| Column | Type | Applies to | Drives |
|---|---|---|---|
| `word_count` | int | all | length bands |
| `char_count` | int | all | — |
| `source_word_count` | int \| empty | Q3, Q11 | ratio denominator |
| `length_ratio` | float \| empty | Q3, Q11 | `word_count / source_word_count` |
| `exceeds_one_third` | bool \| empty | Q3 | Summary cap (5) |
| `verbatim_overlap_pct` | float \| empty | Q3, Q11 | Summary cap (5), Theme cap (4) |
| `paragraph_break_count` | int | Q7 | Paragraph cap (5) |
| `letter_components_found` | string | Q10 | semicolon-joined of: `address`, `date`, `salutation`, `body`, `close`, `signature` |
| `letter_components_missing_count` | int \| empty | Q10 | Letter cap (2) if ≥ 3 |

**Definitions — fix these once and never change them mid-study:**

- `word_count`: whitespace-split on `extracted_text`, editorial markers excluded
- `verbatim_overlap_pct`: proportion of student word-tokens inside a matched run of
  ≥ 7 consecutive tokens shared with `source_text`, case-folded, punctuation stripped
- `paragraph_break_count`: occurrences of `\n\n`. Requires the extraction prompt to have
  distinguished line-wrap from paragraph break, or this column is meaningless
- `letter_components_found`: regex/heuristic detection, spot-verified by hand on all Q10 rows
  (only 22 of them — cheap to check, and it drives a hard cap)

## 8. PROVENANCE

| Column | Type | Notes |
|---|---|---|
| `transcription_model_version` | string | e.g. `gemma-4-31b-it`. Pin exact, never an alias |
| `transcription_provider` | enum | `openrouter` \| `vertex` \| `local` |
| `transcription_prompt_version` | string | `extraction_prompt_v2` |
| `prompt_hash` | string | SHA256 of the prompt file. A version label is a claim; a hash is proof |
| `transcription_settings` | JSON string | e.g. `{"thinking_level": "high", "max_output_tokens": 32768}` |
| `extraction_timestamp` | ISO 8601 | With timezone |

> **Do not record `temperature`, `top_p`, or `top_k` for Gemini 3.x** — those parameters
> were removed from the API and recording them asserts a determinism you do not have.
> Record `thinking_level` instead.

## 9. DROPPED FROM DRAFT

- `notes` — empty on all 60 draft rows. Removed; `error_message` and `review_flags` cover it.

---

## FILE CONVENTIONS

- Encoding `utf-8-sig`, `QUOTE_ALL`, `\r\n` line terminator
- Read with `keep_default_na=False, dtype=str` so an empty transcript is `""` not `NaN`
- Transcript fields contain embedded newlines — **a real CSV parser is mandatory**,
  `split(',')` will silently corrupt rows
- **Never open in Excel and re-save.** It mangles embedded newlines and strips leading
  zeros from `task_id`. Open a copy if you need to look

## COMPANION FILE — `extraction_runs.csv`

One row per API call, not per answer. Keeps cost data out of the answer-grain file
where it would be double-counted on multi-question calls.

`run_id`, `script_id`, `model_id`, `provider`, `thinking_level`, `pages_sent`,
`input_tokens`, `output_tokens`, `cost_usd`, `latency_ms`, `attempt_number`,
`http_status`, `timestamp`

## VALIDATION ASSERTIONS

Run before any grading. Fail loudly.

1. `task_id` unique; row count == scripts × 6
2. Every (`script_id`, `question_no`) pair present exactly once
3. `question_type` ↔ `max_mark` mapping holds on every row
4. `extraction_status == 'error'` count is 0, or explicitly acknowledged
5. `teacher_mark` ≤ `max_mark` wherever present. While deferred, assert instead that the
   column is empty on every row and `teacher_mark_source` is `absent` — this catches a
   stray 0 being written in as a default
6. `extracted_text` non-empty for every `ok` row
7. `extracted_text` contains no `[struck:` or `{inserted` (those belong in `_raw` only)
8. Computed-metric columns are non-empty for the questions they apply to
9. `prompt_hash` identical across all rows — mixed hashes mean mixed prompt versions
10. `illegible_count + unclear_count + cut_count` summed over all rows is > 0
