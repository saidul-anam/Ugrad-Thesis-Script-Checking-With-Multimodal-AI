# Archive

Files the pipeline no longer uses, moved here on 2026-10-22 instead of being deleted. Nothing in `src/`, `scripts/`,
`configs/` or `tests/` reads from this folder.

| Folder | What it holds | In git? |
|---|---|---|
| `docs/` | Older plans and audit reports that later documents replaced | yes |
| `scripts/` | Scripts that no document or other script refers to | yes |
| `scratch/` | One-off check, debug and audit scripts from Sept–Oct 2026, with their logs | no (ignored, as before) |
| `logs/` | Logs of old runs (9–13 Oct 2026) that were in the repo root and `logs/`, and `nohup.out` | no (ignored, as before) |

## docs/

| File | What it was | See instead |
|---|---|---|
| `extraction_plan.md` | Plan for Stage 3 long-answer recall, handwriting calibration and a lighter arbitration step (19 Sept) | `docs/EXTRACTION_ACCURACY_GUIDE.md` |
| `IMPLEMENTATION_2026-09-12_extraction_first.md` | Notes on the extraction-first rebuild: CER harness and Stage 3b evidence gate (12 Sept) | `docs/EXTRACTION_ACCURACY_GUIDE.md` |
| `stage2_stage3_audit_plan.md`, `STAGE2_STAGE3_AUDIT_REPORT.md` | Stage 2 / Stage 3 audit plan and its report (4 Oct) | `docs/EXTRACTION_FIXES_2026-10-06.md` |
| `PIPELINE_PROBLEMS_AUDIT.md` | Problems audit P1–P10 (6 Oct) | `docs/EXTRACTION_FIXES_2026-10-06.md` (the response to it) |
| `TRIADIC_MULTI_AGENT_IMPLEMENTATION_PLAN.md` | Multi-agent design plan (6 Oct); no later document refers to it | — |
| `clean_pdf_integrate.md` | Plan for clean-PDF integration and standalone teacher marks (6 Oct) | `clean_pdf.py`, `scripts/extract_teacher_marks.py` |
| `clean_pdf_integrate.top_level_copy.md` | Identical copy of the above that was in the repo root | — |
| `current_changes.md` | Notes on the 9 Oct segmentation change (header rules, missing-question reconciliation) | `docs/PIPELINE_REVIEW_2026-10.md` |
| `ongoing_implementation.md` | Empty file | — |
| `pipeline_bloat_audit_plan.md` | Plan for the stage timing audit; already marked superseded on 17 Oct | `docs/PIPELINE_REVIEW_2026-10.md` |

## scripts/

| File | What it did |
|---|---|
| `create_sample_pdfs.py` | Made sample PDFs for mock runs during development (Aug) |
| `setup_env.py` | Printed an environment check: Python, PyTorch, CUDA and GPU memory (Aug) |
| `evaluate_benchmark.py` | Ran scripts with thinking mode on and off to compare silent corrections (Aug) |
| `evaluate_all_grades.sh` | Re-graded classes 6, 7, 8 and 10 in one go (16 Oct); the review runners in `outputs/review_2026_10/` do this now |
| `reformat_all_reports.py` | Re-rendered every `evaluation_report.md` under `outputs/evaluated/` from `complete_report.json` (16 Oct) |

These scripts find the repo root from their own location (`Path(__file__).parent.parent`), so to run one again,
move it back to `scripts/` first. The same goes for the files in `scratch/`.
