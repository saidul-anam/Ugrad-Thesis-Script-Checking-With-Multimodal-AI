# Evaluation Validation Report

Records: **286**  ·  Checks failed: **0**  ·  warnings: **0**

## Assertions

- ✅ every (task_id, model, run) present exactly once (286 records)
- ✅ row count == tasks(286) x models(1) x k(1) = 286; got 286
- ✅ raw_subscores sum == raw_total on every row
- ✅ every sub-score within its per-task ceiling
- ✅ total_score <= max_mark and <= applied cap
- ✅ performance_band contains total_score for its max_mark
- ✅ max_mark_applied == extraction.csv max_mark
- ✅ item-scored rows put the whole mark in context_content_data and apply no cap
- ✅ every (question_type, max_mark) has a ceiling row
- ✅ metadata max_mark matches extraction.csv per task
- ✅ all records come from one exam set (['SE_06_Q1'])
- ✅ rubric_hash identical across all rows (1 distinct)
- ✅ thinking_level identical across all rows (['high'])
- ✅ prompt_hash identical across models for same (task_id, run)
- ✅ synthesised rows: total 0, Band 0, no API cost (13 rows)

## Noise floor — within-model score range across runs

| model | mean range | max range | tasks |
|---|---|---|---|
| gemma | 0.0 | 0.0 | 286 |

_If within-model range routinely exceeds the cross-model gap below, raise k before making any divergence claim._

## Evidence-not-found rate (hallucination indicator)

| model | rate | graded rows |
|---|---|---|
| gemma | 71% (193/273) | 273 |

## Per-question mean score by model

| question | gemma |
|---|---|
| Q1 | 3.50 |
| Q2 | 2.91 |
| Q3 | 2.00 |
| Q4 | 1.68 |
| Q5 | 3.82 |
| Q6 | 4.14 |
| Q7 | 1.86 |
| Q8 | 2.55 |
| Q9 | 5.27 |
| Q10 | 5.68 |
| Q11 | 4.55 |
| Q12 | 5.20 |
| Q13 | 6.50 |

## Where the two graders disagree most (cross-model gap vs noise)

| task_id | cross-model gap | within-model noise | means |
|---|---|---|---|
