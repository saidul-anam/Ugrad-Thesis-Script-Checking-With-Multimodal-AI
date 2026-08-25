# Evaluation Validation Report

Records: **720**  ·  Checks failed: **1**  ·  warnings: **0**

## Assertions

- ✅ every (task_id, model, run) present exactly once (720 records)
- ✅ row count == tasks(120) x models(2) x k(3) = 720; got 720
- ❌ raw_subscores sum == raw_total on every row
- ✅ every sub-score within its per-task ceiling
- ✅ total_score <= max_mark and <= applied cap
- ✅ performance_band contains total_score for its max_mark
- ✅ max_mark_applied == extraction.csv max_mark
- ✅ metadata max_mark matches extraction.csv per task
- ✅ rubric_hash identical across all rows (1 distinct)
- ✅ thinking_level identical across all rows (['high'])
- ✅ prompt_hash identical across models for same (task_id, run)
- ✅ synthesised rows: total 0, Band 0, no API cost (42 rows)

## Noise floor — within-model score range across runs

| model | mean range | max range | tasks |
|---|---|---|---|
| gemini | 0.367 | 3.0 | 120 |
| gemma | 0.783 | 2.5 | 120 |

_If within-model range routinely exceeds the cross-model gap below, raise k before making any divergence claim._

## Evidence-not-found rate (hallucination indicator)

| model | rate | graded rows |
|---|---|---|
| gemini | 12% (40/339) | 339 |
| gemma | 71% (239/339) | 339 |

## Per-question mean score by model

| question | gemini | gemma |
|---|---|---|
| Q3 | 5.05 | 5.12 |
| Q7 | 4.78 | 4.82 |
| Q8 | 5.23 | 5.21 |
| Q9 | 3.43 | 3.58 |
| Q10 | 3.40 | 3.50 |
| Q11 | 4.02 | 4.28 |

## Where the two graders disagree most (cross-model gap vs noise)

| task_id | cross-model gap | within-model noise | means |
|---|---|---|---|
| CHART_017 | 1.67 | 1.00 | gemini=4.00, gemma=5.67 |
| SUMMARY_015 | 1.67 | 1.00 | gemini=6.67, gemma=8.33 |
| THEME_003 | 1.67 | 2.00 | gemini=3.33, gemma=5.00  ⚠️ noise≥gap |
| CHART_001 | 1.33 | 1.00 | gemini=6.00, gemma=4.67 |
| CHART_013 | 1.00 | 0.00 | gemini=5.00, gemma=4.00 |
| CHART_018 | 1.00 | 2.00 | gemini=5.33, gemma=4.33  ⚠️ noise≥gap |
| PARA_008 | 1.00 | 2.00 | gemini=6.00, gemma=7.00  ⚠️ noise≥gap |
| SUMMARY_001 | 1.00 | 2.00 | gemini=6.33, gemma=7.33  ⚠️ noise≥gap |
| SUMMARY_010 | 1.00 | 0.00 | gemini=4.00, gemma=3.00 |
| SUMMARY_020 | 1.00 | 0.00 | gemini=4.00, gemma=5.00 |
| THEME_009 | 1.00 | 1.50 | gemini=5.00, gemma=4.00  ⚠️ noise≥gap |
| CHART_002 | 0.67 | 1.00 | gemini=7.00, gemma=7.67  ⚠️ noise≥gap |
| CHART_004 | 0.67 | 2.00 | gemini=4.00, gemma=4.67  ⚠️ noise≥gap |
| CHART_006 | 0.67 | 1.00 | gemini=4.67, gemma=4.00  ⚠️ noise≥gap |
| CHART_007 | 0.67 | 1.00 | gemini=7.33, gemma=8.00  ⚠️ noise≥gap |
| CHART_009 | 0.67 | 2.00 | gemini=6.00, gemma=6.67  ⚠️ noise≥gap |
| CHART_016 | 0.67 | 2.00 | gemini=4.67, gemma=4.00  ⚠️ noise≥gap |
| CHART_020 | 0.67 | 2.00 | gemini=5.00, gemma=5.67  ⚠️ noise≥gap |
| PARA_015 | 0.67 | 1.00 | gemini=5.00, gemma=4.33  ⚠️ noise≥gap |
| PARA_017 | 0.67 | 2.00 | gemini=5.33, gemma=6.00  ⚠️ noise≥gap |
| STORY_010 | 0.67 | 1.00 | gemini=4.00, gemma=4.67  ⚠️ noise≥gap |
| STORY_014 | 0.67 | 1.00 | gemini=4.00, gemma=4.67  ⚠️ noise≥gap |
| SUMMARY_002 | 0.67 | 2.00 | gemini=7.33, gemma=8.00  ⚠️ noise≥gap |
| SUMMARY_012 | 0.67 | 1.00 | gemini=6.00, gemma=5.33  ⚠️ noise≥gap |
| THEME_012 | 0.67 | 1.00 | gemini=5.00, gemma=5.67  ⚠️ noise≥gap |
| THEME_017 | 0.67 | 1.00 | gemini=5.00, gemma=5.67  ⚠️ noise≥gap |
| THEME_019 | 0.67 | 1.00 | gemini=5.00, gemma=5.67  ⚠️ noise≥gap |
| STORY_003 | 0.67 | 1.00 | gemini=3.67, gemma=3.00  ⚠️ noise≥gap |
| STORY_013 | 0.67 | 1.00 | gemini=4.00, gemma=3.33  ⚠️ noise≥gap |
| STORY_015 | 0.67 | 1.00 | gemini=3.00, gemma=3.67  ⚠️ noise≥gap |
| SUMMARY_003 | 0.67 | 1.00 | gemini=4.00, gemma=3.33  ⚠️ noise≥gap |
| SUMMARY_004 | 0.67 | 2.00 | gemini=3.67, gemma=4.33  ⚠️ noise≥gap |
| SUMMARY_013 | 0.67 | 1.00 | gemini=4.00, gemma=3.33  ⚠️ noise≥gap |
| THEME_004 | 0.67 | 1.00 | gemini=3.00, gemma=3.67  ⚠️ noise≥gap |
| THEME_018 | 0.67 | 1.00 | gemini=4.67, gemma=5.33  ⚠️ noise≥gap |
| LETTER_010 | 0.50 | 1.00 | gemini=3.67, gemma=4.17  ⚠️ noise≥gap |
| CHART_003 | 0.50 | 1.50 | gemini=6.00, gemma=5.50  ⚠️ noise≥gap |
| CHART_015 | 0.50 | 2.00 | gemini=5.33, gemma=4.83  ⚠️ noise≥gap |
| PARA_004 | 0.50 | 1.00 | gemini=5.00, gemma=4.50  ⚠️ noise≥gap |
| STORY_007 | 0.50 | 1.00 | gemini=4.00, gemma=4.50  ⚠️ noise≥gap |
| STORY_019 | 0.50 | 1.00 | gemini=4.00, gemma=4.50  ⚠️ noise≥gap |
| THEME_005 | 0.50 | 2.50 | gemini=4.67, gemma=4.17  ⚠️ noise≥gap |
| CHART_012 | 0.33 | 1.00 | gemini=5.33, gemma=5.67  ⚠️ noise≥gap |
| CHART_014 | 0.33 | 2.00 | gemini=5.67, gemma=5.33  ⚠️ noise≥gap |
| SUMMARY_014 | 0.33 | 3.00 | gemini=5.67, gemma=5.33  ⚠️ noise≥gap |
| THEME_001 | 0.33 | 1.00 | gemini=4.33, gemma=4.67  ⚠️ noise≥gap |
| THEME_007 | 0.33 | 1.00 | gemini=5.33, gemma=5.67  ⚠️ noise≥gap |
| LETTER_003 | 0.33 | 1.00 | gemini=3.00, gemma=3.33  ⚠️ noise≥gap |
| LETTER_004 | 0.33 | 1.00 | gemini=3.00, gemma=3.33  ⚠️ noise≥gap |
| LETTER_014 | 0.33 | 1.00 | gemini=3.33, gemma=3.00  ⚠️ noise≥gap |
| LETTER_019 | 0.33 | 1.00 | gemini=3.00, gemma=3.33  ⚠️ noise≥gap |
| STORY_005 | 0.33 | 1.00 | gemini=3.67, gemma=4.00  ⚠️ noise≥gap |
| CHART_008 | 0.33 | 1.00 | gemini=6.00, gemma=6.33  ⚠️ noise≥gap |
| CHART_010 | 0.33 | 2.00 | gemini=5.00, gemma=4.67  ⚠️ noise≥gap |
| LETTER_012 | 0.33 | 1.00 | gemini=3.33, gemma=3.67  ⚠️ noise≥gap |
| PARA_001 | 0.33 | 1.00 | gemini=6.00, gemma=6.33  ⚠️ noise≥gap |
| PARA_002 | 0.33 | 1.00 | gemini=5.00, gemma=5.33  ⚠️ noise≥gap |
| PARA_003 | 0.33 | 2.00 | gemini=5.67, gemma=6.00  ⚠️ noise≥gap |
| PARA_013 | 0.33 | 1.00 | gemini=5.00, gemma=4.67  ⚠️ noise≥gap |
| PARA_019 | 0.33 | 1.00 | gemini=5.00, gemma=4.67  ⚠️ noise≥gap |
| STORY_011 | 0.33 | 0.50 | gemini=4.00, gemma=4.33  ⚠️ noise≥gap |
| STORY_016 | 0.33 | 0.50 | gemini=4.00, gemma=4.33  ⚠️ noise≥gap |
| SUMMARY_007 | 0.33 | 1.00 | gemini=6.00, gemma=5.67  ⚠️ noise≥gap |
| SUMMARY_009 | 0.33 | 1.00 | gemini=3.67, gemma=3.33  ⚠️ noise≥gap |
| SUMMARY_018 | 0.33 | 1.00 | gemini=5.67, gemma=6.00  ⚠️ noise≥gap |
| THEME_006 | 0.33 | 1.00 | gemini=2.33, gemma=2.67  ⚠️ noise≥gap |
| THEME_008 | 0.33 | 1.00 | gemini=4.33, gemma=4.00  ⚠️ noise≥gap |
| THEME_013 | 0.33 | 1.00 | gemini=4.00, gemma=4.33  ⚠️ noise≥gap |
| THEME_016 | 0.33 | 1.00 | gemini=5.00, gemma=5.33  ⚠️ noise≥gap |
| THEME_020 | 0.33 | 1.00 | gemini=5.00, gemma=5.33  ⚠️ noise≥gap |
| CHART_011 | 0.17 | 1.00 | gemini=4.33, gemma=4.50  ⚠️ noise≥gap |
| LETTER_005 | 0.17 | 0.50 | gemini=4.00, gemma=4.17  ⚠️ noise≥gap |
| LETTER_007 | 0.17 | 0.50 | gemini=4.00, gemma=4.17  ⚠️ noise≥gap |
| LETTER_016 | 0.17 | 0.50 | gemini=4.00, gemma=4.17  ⚠️ noise≥gap |
| PARA_006 | 0.17 | 1.00 | gemini=4.67, gemma=4.50  ⚠️ noise≥gap |
| STORY_008 | 0.17 | 0.50 | gemini=4.00, gemma=4.17  ⚠️ noise≥gap |
| STORY_009 | 0.17 | 0.50 | gemini=4.00, gemma=4.17  ⚠️ noise≥gap |
| SUMMARY_011 | 0.17 | 0.50 | gemini=4.00, gemma=4.17  ⚠️ noise≥gap |
| CHART_005 | 0.00 | 0.00 | gemini=4.00, gemma=4.00 |
| CHART_019 | 0.00 | 2.00 | gemini=4.00, gemma=4.00 |
| LETTER_001 | 0.00 | 0.00 | gemini=4.00, gemma=4.00 |
| LETTER_002 | 0.00 | 0.00 | gemini=4.00, gemma=4.00 |
| LETTER_006 | 0.00 | 0.00 | gemini=0.00, gemma=0.00 |
| LETTER_008 | 0.00 | 0.00 | gemini=4.00, gemma=4.00 |
| LETTER_009 | 0.00 | 0.00 | gemini=4.00, gemma=4.00 |
| LETTER_011 | 0.00 | 0.00 | gemini=4.00, gemma=4.00 |
| LETTER_013 | 0.00 | 0.00 | gemini=3.00, gemma=3.00 |
| LETTER_015 | 0.00 | 0.00 | gemini=3.00, gemma=3.00 |
| LETTER_017 | 0.00 | 0.00 | gemini=3.00, gemma=3.00 |
| LETTER_018 | 0.00 | 1.00 | gemini=3.67, gemma=3.67 |
| LETTER_020 | 0.00 | 0.00 | gemini=4.00, gemma=4.00 |
| PARA_005 | 0.00 | 0.00 | gemini=6.00, gemma=6.00 |
| PARA_007 | 0.00 | 0.00 | gemini=5.00, gemma=5.00 |
| PARA_009 | 0.00 | 0.00 | gemini=5.00, gemma=5.00 |
| PARA_010 | 0.00 | 0.00 | gemini=0.00, gemma=0.00 |
| PARA_011 | 0.00 | 0.00 | gemini=0.00, gemma=0.00 |
| PARA_012 | 0.00 | 0.00 | gemini=5.00, gemma=5.00 |
| PARA_014 | 0.00 | 0.00 | gemini=6.00, gemma=6.00 |
| PARA_016 | 0.00 | 0.00 | gemini=5.00, gemma=5.00 |
| PARA_018 | 0.00 | 2.00 | gemini=6.00, gemma=6.00 |
| PARA_020 | 0.00 | 0.00 | gemini=5.00, gemma=5.00 |
| STORY_001 | 0.00 | 0.00 | gemini=4.00, gemma=4.00 |
| STORY_002 | 0.00 | 1.00 | gemini=3.33, gemma=3.33 |
| STORY_004 | 0.00 | 0.00 | gemini=0.00, gemma=0.00 |
| STORY_006 | 0.00 | 0.00 | gemini=0.00, gemma=0.00 |
| STORY_012 | 0.00 | 1.00 | gemini=3.67, gemma=3.67 |
| STORY_017 | 0.00 | 1.00 | gemini=3.33, gemma=3.33 |
| STORY_018 | 0.00 | 0.00 | gemini=4.00, gemma=4.00 |
| STORY_020 | 0.00 | 0.00 | gemini=4.00, gemma=4.00 |
| SUMMARY_005 | 0.00 | 1.00 | gemini=5.67, gemma=5.67 |
| SUMMARY_006 | 0.00 | 1.00 | gemini=3.33, gemma=3.33 |
| SUMMARY_008 | 0.00 | 2.00 | gemini=6.00, gemma=6.00 |
| SUMMARY_016 | 0.00 | 2.00 | gemini=6.00, gemma=6.00 |
| SUMMARY_017 | 0.00 | 0.00 | gemini=3.00, gemma=3.00 |
| SUMMARY_019 | 0.00 | 0.00 | gemini=6.00, gemma=6.00 |
| THEME_002 | 0.00 | 1.00 | gemini=6.33, gemma=6.33 |
| THEME_010 | 0.00 | 0.00 | gemini=0.00, gemma=0.00 |
| THEME_011 | 0.00 | 0.00 | gemini=0.00, gemma=0.00 |
| THEME_014 | 0.00 | 0.00 | gemini=4.00, gemma=4.00 |
| THEME_015 | 0.00 | 2.00 | gemini=4.00, gemma=4.00 |
