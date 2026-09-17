# Evaluation Validation Report

Records: **455**  ·  Checks failed: **0**  ·  warnings: **0**

## Assertions

- ✅ every (task_id, model, run) present exactly once (455 records)
- ✅ row count == tasks(455) x models(1) x k(1) = 455; got 455
- ✅ raw_subscores sum == raw_total on every row
- ✅ every sub-score within its per-task ceiling
- ✅ total_score <= max_mark and <= applied cap
- ✅ performance_band contains total_score for its max_mark
- ✅ max_mark_applied == extraction.csv max_mark
- ✅ item-scored rows put the whole mark in context_content_data and apply no cap
- ✅ every (question_type, max_mark) has a ceiling row
- ✅ metadata max_mark matches extraction.csv per task
- ✅ all records come from one exam set (['SE_07_Q1'])
- ✅ rubric_hash identical across all rows (1 distinct)
- ✅ thinking_level identical across all rows (['high'])
- ✅ prompt_hash identical across models for same (task_id, run)
- ✅ synthesised rows: total 0, Band 0, no API cost (14 rows)

## Noise floor — within-model score range across runs

| model | mean range | max range | tasks |
|---|---|---|---|
| gemma | 0.0 | 0.0 | 455 |

_If within-model range routinely exceeds the cross-model gap below, raise k before making any divergence claim._

## Evidence-not-found rate (hallucination indicator)

| model | rate | graded rows |
|---|---|---|
| gemma | 74% (325/441) | 441 |

## Per-question mean score by model

| question | gemma |
|---|---|
| Q1 | 4.03 |
| Q2 | 8.80 |
| Q3 | 3.06 |
| Q4 | 3.63 |
| Q5 | 3.60 |
| Q6 | 2.49 |
| Q7 | 2.24 |
| Q8 | 2.23 |
| Q9 | 5.31 |
| Q10 | 2.97 |
| Q11 | 5.50 |
| Q12 | 6.53 |
| Q13 | 7.97 |

## Where the two graders disagree most (cross-model gap vs noise)

| task_id | cross-model gap | within-model noise | means |
|---|---|---|---|
