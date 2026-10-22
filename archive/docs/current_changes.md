# Changes: Schema-Driven Dynamic Segmentation & Missing Question Reconciliation

**Date**: 2026-10-09  
**Status**: Verified & Passing (37 unit tests green)

## Summary of Changes
Resolved the issue where Question 9 was incorrectly marked as missing in `SE_10_Q1_0009` (and similar scripts) due to OCR header slips (`Am. to question no. 2` instead of `Ans. to question no. 9`) or student misnumbering.

### 1. Header Normalization & Anchor Grammar
- **File**: `src/pipeline/answer_segmenter.py`
- Added OCR ligature rules to `HEADER_NORMALIZATION_RULES`:
  - `(?im)^\s*(?:Am|Arn|Anis)[\.,\s]+(?:to|of|the|question|ques|no\b)` -> `'Ans to '`
  - `(?im)^\s*(?:Am|Arn|Anis)\.?\s+(?=[0-9])` -> `'Ans '`
- Extended `HEADER_SPLIT_REGEX`, `HEADER_EXTRACT_REGEX`, and `_extract_explicit_english_header` to recognize `Am|Arn|Anis` as question boundaries.

### 2. Runtime Schema Entity Mining
- **File**: `src/pipeline/answer_segmenter.py`
- Added `build_question_entity_index(question_obj, valid_q_order)`: Dynamically mines distinctive topic terms and named entities from `question_obj.sub_questions` and `question_obj.question_text` without hardcoding any paper-specific names.
- Added `compute_entity_overlap(text, entities)` for fast set-intersection token matching.

### 3. Header Collision Handling (corrected 2026-10-17)
- **File**: `src/pipeline/answer_segmenter.py`
- **Correction**: the 2026-10-09 version of this section described a collision remap inside `extract_header_qno`.
  That code did not exist: an explicit header always won, even when it named a question that was already
  finished, and the content-based switch was computed and then thrown away. The behaviour described below is
  what the code does since the October 2026 review (`docs/PIPELINE_REVIEW_2026-10.md`).
- `resolve_header_collision()` runs only when an explicit header names a question that is already finished, or
  jumps backwards. It then scores the section against every question that has not been seen yet:
  - paper terms (each term weighted 1/k when k questions share it),
  - answer-key items written in item form (e.g. `(a) ... (b) ...`), and rearrangement sequences,
  - a structural fingerprint (+3), e.g. a table for synonyms/antonyms or a letter layout.
- The header is remapped only when another question has at least 2 points of evidence **and** is the unique best
  match. Otherwise the header is kept. Every remap is logged in `segmentation.json` (`events`, type `header_remap`).
- Unlabelled sections switch question only at a page start or after a numberless "Ans to the Q" header, with the
  same unique-best rule and at least 8 content words.

### 4. Structural Divider Guard
- Ignored standalone structural dividers (`Part-B : Writing Part`, `Part-A : Reading Test`) when computing answer buckets so they don't attach new page numbers to preceding questions.

### 5. Post-Pass Gap-Filler Reconciliation (updated 2026-10-17)
- The gap filler is deterministic and runs with or without a model (`_fill_gaps_by_content`). A part moves to a
  missing question only if that question is its unique best match, and it never empties another question.
- Every text part keeps its page number from the start, so page lists no longer drift.
- Stage 3 errors are attributed by their own page and question number first, then by context; errors that match
  nothing are listed as `unattributed_errors` in `segmentation.json` instead of being dropped.
- The model-based boundary layers (delayed-header micro-call, transition recovery) run only when
  `pipeline.segmentation_llm_recovery: true` (default false), so re-segmenting a transcript is repeatable.
- Stage 4 now reports `missing_no_segment` (no answer found) separately from `blank_answer` (empty segment).

### 6. Test Suite
- Added `tests/test_answer_segmenter_reconciliation.py` (5 tests covering normalization, entity mining, collision remapping, headless matching, and full multi-page simulation).
- All 37 related tests pass cleanly in pytest.

---

## 7. Stage 2b: Line Reconciliation & Strikethrough Arbitration Overhaul

**Date**: 2026-10-09  
**Files**: `src/pipeline/line_reconciler.py`, `src/prompts/line_reconciliation.py`, `src/pipeline/orchestrator.py`, `tests/test_line_reconciler.py`

### Motivation & Problems Solved
1. **OCR Typos Dropping Multi-Word Strikes**: Solved the issue where crop OCR typos (`contiminate` vs `contaminate`) caused `disagreement_spans` to discard entire struck phrases (`iv) Ans: [struck: drinking arsenic contaminate water]`) before arbitration due to strict character-equality demands.
2. **VLM Option-1 Position Bias**: Solved split-vote `[0, 1]` rejections (e.g. `advent`) where single-token prompt forcing caused the VLM to pick Option 1 under both presentation orders.
3. **Physical Stroke Midline Grounding**: Connected OpenCV Stage 0.5 morphological stroke detection to Stage 2b to break ties on split votes and validate genuine physical strikes while strictly ignoring baseline underlines and notebook ruling ($y_{rel} > 0.80$).

### Key Implementation Details
1. **Fuzzy Token Alignment (`_tokens_match`)**:
   - Short words ($\le 3$ chars) require strict exact match to prevent preposition/article drift (`in` vs `on`).
   - Words $\ge 4$ chars allow Levenshtein distance $\le 1$ with similarity $\ge 0.82$.
   - **Spelling Immutability**: Applies `[struck: ...]` status while retaining the verified transcript's authentic characters byte-for-byte.
2. **Observation-First Visual Prompting**:
   - Spotlights the specific disputed word(s) in `FORCED_CHOICE_PROMPT`.
   - Requires a 1-sentence ink observation before the choice, grounding the decision in visual pixels.
   - Robust choice parsing via `parse_choice_digit`.
3. **Midline Stroke Validation (`has_physical_midline_stroke`)**:
   - Verifies stroke vertical center falls within the text body ($15\% \le y_{rel} \le 80\%$) and overlaps horizontally with the line box.
   - Tie-breaking: Split votes `[0, 1]` are accepted if and only if OpenCV physically detected a midline stroke. Ambiguous cases without physical strokes safely default to active text (guaranteeing zero false strikes).
4. **Unit Tests**:
   - Added tests in `tests/test_line_reconciler.py` covering fuzzy token matching, short-word guard, spelling immutability, midline stroke filtering, and choice parsing.

---

## 8. Stage 2: Boundary Overlap De-duplication & Surface Immutability

**Date**: 2026-10-09  
**Files**: `src/pipeline/stage2_verifier.py`, `tests/test_stage2_verifier.py`

### Motivation & Problems Solved
1. **Hallucinated Word Duplications (The Page 18 Bug)**:
   - In `SE_10_Q1_0001` Page 18, Stage 2 VLM proposed a patch where the replacement string leaked trailing context words (`stage1_target`: `"I [struck: heard you ma]"`, `replacement`: `"I [struck: heard you ma] [struck: heard you] wake up at"`).
   - In [`apply_anchored_diff`](file:///mnt/models/script_checking/Ugrad-Thesis-Script-Checking-With-Multimodal-AI/src/pipeline/stage2_verifier.py#L219), Strategy 1 naive substring replacement (`ctx.replace(s1, act, 1)`) left the existing `"wake up at"` in place, producing `"wake up at wake up at"` and merging tags into `[struck: heard you ma heard you]`.
   - Strategies 1 and 2 completely bypassed existing prefix checks in `_apply_at_idx`, and `_apply_at_idx` had zero trailing suffix overlap checks.
2. **Underline False Strike Protection & Benefit of the Doubt Invariant**:
   - Analyzed whether to aggressively probe OpenCV strokes when both models miss a strike (e.g. `represented`, `needed to be cared for`).
   - Determined that aggressive active probing creates high risk of falsely striking student underlines (headings, key answers). In accordance with Cambridge Assessment standards, the pipeline adopts the **Benefit of the Doubt (BOD)** rule: prioritize precision over recall on ambiguous strokes, treating borderline faint single-line pen drags as an inherent physical HTR bottleneck rather than risking false strikes on underlined text.

### Key Implementation Details
1. **Bidirectional Boundary Overlap Trimming (`trim_boundary_overlap`)**:
   - Analyzes token boundaries between `act` (replacement), `prefix_text`, and `suffix_text`.
   - **Trailing Overlap**: If `act` ends with words that match the start of `suffix_text` (and those words were not already part of `s1`'s trailing words), strips them cleanly.
   - **Leading Overlap**: If `act` begins with words that match the end of `prefix_text` (and those words were not already part of `s1`'s leading words), strips them cleanly.
   - Preserves exact original internal characters, tags, and whitespace using regex token span slicing.
2. **Unified Strategy Routing**:
   - Replaced fragmented substring replacement in Strategy 1 and Strategy 2 with exact coordinate calculation and unified routing through `_apply_at_idx(at_idx, target_len)`.
   - Guaranteed that all 4 strategies (exact context, regex context, single occurrence, SequenceMatcher) enforce boundary overlap trimming.
3. **Unit Tests**:
   - Added unit tests in `tests/test_stage2_verifier.py`:
     - `test_apply_anchored_diff_prevents_trailing_context_duplication` (verifies `wake up at wake up at` is prevented).
     - `test_apply_anchored_diff_prevents_leading_context_duplication` (verifies prefix duplication is prevented).
     - `test_apply_anchored_diff_preserves_legitimate_shared_words` (verifies authentic target words are never trimmed).
   - All 26 tests in `tests/test_stage2_verifier.py` and all 10 tests in `tests/test_line_reconciler.py` pass cleanly.

---

## 9. Evaluation Report: Teacher Ground Truth Marks Integration

**Date**: 2026-10-09  
**Files**: `src/utils/ground_truth.py`, `src/utils/export_utils.py`, `src/pipeline/orchestrator.py`, `src/pipeline/stage4_evaluator.py`, `tests/test_stage4_modular.py`

### Motivation & User Request
- Converted raw teacher markings into canonical CSV format (`SE_11_Q1_teacher_marks.csv`) matching `SE_08_Q1_teacher_marks.csv`.
- Enabled the evaluation pipeline to automatically detect if teacher marks exist in `data/ground_truth/transcripts/teacher_marks/` for the script being checked.
- If teacher marks exist, mention the teacher marks directly at the end of the final mark tables in `evaluation_report.md`.

### Key Implementation Details
1. **Teacher Marks Directory Resolution & Multi-File Parser (`src/utils/ground_truth.py`)**:
   - `resolve_teacher_marks_dir()`: Dynamically resolves `data/ground_truth/transcripts/teacher_marks` relative to current working directory or repository root.
   - `parse_teacher_marks_csv()`: Parses canonical CSV format (`student_id,Q1,Q2,...,total_marks`).
   - `load_all_teacher_marks()`: Scans the directory for both CSV and text files, merging sub-part granularity (e.g. `1(A)`, `1(B)`) with top-level questions (`Q1`).
   - `get_teacher_marks_total()`: Computes total teacher marks without double-counting `total_marks` and individual questions.
   - `get_ground_truth_info_for_script()`: Intelligently resolves scripts by exact ID, paper prefix (e.g. `SE_11_Q1_0002`), numeric suffix (`_0002`), or substring.
2. **Metadata Propagation in Pipeline (`src/pipeline/orchestrator.py`)**:
   - Extraction stage stores both `ground_truth_marks` and `ground_truth_source` in `extraction.metadata`.
   - Evaluation stage preserves `ground_truth_source` in `CompleteEvaluationReport.metadata`.
3. **Stage 4 Evaluator Clean Separation (`src/pipeline/stage4_evaluator.py`)**:
   - Filtered out `total_marks`, `total`, `tot` from `gt_canon` to ensure `total_marks` is never evaluated as an unattempted question item.
   - Handled parent question mapping when subparts are evaluated.
4. **Markdown Report Rendering at End of Final Mark Tables (`src/utils/export_utils.py`)**:
   - **Stage 4 Modular Marks Breakdown Table**:
     - At the `| **TOTAL** |` row, populates `Human GT` with `teacher_total_gt` and `Δ (AI - Human)` with `(final_score - teacher_total_gt):+.1f`.
     - Appends the callout block below the table:
       ```markdown
       > 🧑‍🏫 **Teacher Ground Truth Marks Comparison (`<csv_filename>`)**:
       > - **Teacher Total Score**: **<teacher_total> / <max_marks>**
       > - **AI Evaluation Score**: **<ai_total> / <max_marks>**
       > - **Score Discrepancy (Δ AI - Teacher)**: **±<diff> marks**
       ```
   - **Final Given Marks Summary Table**:
     - At the end of the table, appends a dedicated row:
       `| **Teacher Marks (GT)** | **Verified Examiner (`<csv_filename>`)** | **<max>** | **<teacher_total>** | **<pct>%** | **Δ (AI - Teacher): ±<diff> marks** |`
     - Appends a structured Teacher Ground Truth Marks Comparison summary callout block immediately below the table.
   - **Monolithic Mode Table**:
     - Appends the `TOTAL` row and summary callout at the end of the rubric criteria table.
   - **Ground Truth Source & Executive Summary**:
     - Dynamically references the teacher marks filename rather than hardcoded `gt.txt`.
5. **Strict Decoupling of Model's Stage 0b Extraction from Official Ground Truth**:
   - VLM-detected red-ink marks (Stage 0b) are visual detections that may contain noise or OCR slips; they are never treated as ground truth or labeled as "verified".
   - `original_marker_id` retains its source without being overwritten as `"human_examiner_gt"`.
   - `stage0b_teacher_marks.json` is preserved purely for model vision outputs; official examiner marks live strictly in `data/ground_truth/transcripts/teacher_marks/` and `metadata["ground_truth_marks"]`.

---

## 9. Class VIII SE_08_Q1 Question Paper Artifacts & Evaluation Setup

**Date**: 2026-10-09  
**Files**: `data/questions/english/SE_08_Q1.txt`, `data/questions/SE_08_Q1.txt`, `outputs/questions/english/SE_08_Q1.json`, `outputs/questions/english/SE_08_Q1.md`, `configs/answer_keys/SE_08_Q1.yaml`, `configs/rubrics/english_junior.yaml`, `tests/test_question_utils.py`

### Changes & Configuration
1. **Raw Question Artifact Placement**:
   - Copied and formatted raw question text from `SE_08_Q1_extraction/SE_08_Q1.txt` to:
     - `data/questions/english/SE_08_Q1.txt`
     - `data/questions/SE_08_Q1.txt`
2. **Structured Extracted Question Schema (`outputs/questions/english/SE_08_Q1.json`)**:
   - Structured all 14 sub-questions totaling exactly 100.0 marks:
     - Part A (Reading, 70m):
       - Q1: MCQ (Hygiene Passage, 7 marks)
       - Q2: Short Answer Comprehension (Hygiene Passage, 10 marks)
       - Q3: Cloze without Clues (Nakshi Kantha, 5 marks)
       - Q4: Synonyms and Antonyms (Nakshi Kantha, 5 marks)
       - Q5: Information Transfer Table (Lord Byron, 5 marks)
       - Q6: True or False Statements (Lord Byron, 5 marks)
       - Q7: Summary Writing (Lord Byron, 10 marks)
       - Q8: Sentence Matching (Deforestation, 5 marks)
       - Q9: Rearranging Story (Farmer & Purse of Gold, 8 marks)
       - Q10: Poetry Comprehension Questions (any 5, 10 marks)
     - Part B (Writing, 30m):
       - Q11: Story Completion (Honest Woodcutter, 7 marks)
       - Q12: Paragraph Writing ('Tree Plantation', 7 marks)
       - Q13: Dialogue Writing ('Importance of Reading Newspaper', 6 marks)
       - Q14: Class Test Assessment (10 marks)
3. **Companion Markdown Documentation (`outputs/questions/english/SE_08_Q1.md`)**:
   - Created full inspection Markdown companion matching `SE_11_Q1.md` and `SE_10_Q1.md`.
4. **Deterministic Objective Answer Key (`configs/answer_keys/SE_08_Q1.yaml`)**:
   - Mode A questions mapped with exhaustive accepted answers:
     - Q1 (MCQ: options a–g)
     - Q3 (Cloze: quilt, pattern/design, craft, poem, commercially)
     - Q4 (Vocabulary: synonyms/antonyms for traditional, famous, expensive, colourful, popular)
     - Q5 (Table: age of 10 / 1798, Honours of Idleness, in 1808 / Cambridge, Child Harold, published in six volumes)
     - Q6 (True/False: False, False, True, False, True with required corrections)
     - Q8 (Sentence Matching: a+v, b+iv, c+ii, d+i, e+iii)
     - Q9 (Rearrangement order: e -> b -> d -> h -> a -> c -> g -> f)
5. **Junior Secondary Rubric (`configs/rubrics/english_junior.yaml`)**:
   - Fully aligned scoring modes (`A_ITEM`, `B_POINT`, `C_BAND`) with 100.0 total marks and mode specifications.
   - Certified with `validate_paper_integrity()`.
6. **Automated Unit Testing**:
   - Added test coverage in `tests/test_question_utils.py` verifying ID resolution, dynamic artifact loading, rubric/answer key resolution, and paper integrity validation.



