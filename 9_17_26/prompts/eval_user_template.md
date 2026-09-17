TASK_TYPE: %%QUESTION_TYPE%%
MAX_MARK: %%MAX_MARK%%

QUESTION AND REFERENCE MATERIAL:
%%QUESTION%%

COMPUTED METRICS — these are authoritative. Use them to decide whether any
structural cap applies. Do NOT recount, re-measure, or second-guess them.
  student_word_count: %%WORD_COUNT%%
  source_word_count: %%SOURCE_WORD_COUNT%%
  length_ratio: %%LENGTH_RATIO%%
  exceeds_one_third: %%EXCEEDS_ONE_THIRD%%
  verbatim_overlap_pct: %%VERBATIM_OVERLAP_PCT%%
  paragraph_break_count: %%PARAGRAPH_BREAK_COUNT%%
  letter_components_found: %%LETTER_COMPONENTS_FOUND%%
  letter_components_missing_count: %%LETTER_COMPONENTS_MISSING_COUNT%%%%CAP_NOTE%%

CANDIDATE RESPONSE:
%%EXTRACTED_TEXT%%

State your criterion reasoning BEFORE committing to any number. Reasoning written
after a score tends to rationalise it.

REQUIRED OUTPUT — extends the rubric's JSON schema with these fields:
- "raw_subscores": the four criteria scored strictly on merit BEFORE any cap,
  plus "raw_total" = their exact sum.
- "criterion_reasoning": for EACH of the four criteria, an object
  {"rationale": "<why this score>", "evidence": "<a span copied VERBATIM from the
  CANDIDATE RESPONSE that justifies it; use an empty string if none applies>"}.
- "score_breakdown": the four criteria plus "total_score" and "capped_from".
  Award the merit sub-scores; if a structural cap applies, reduce ONLY total_score
  to the cap and set "capped_from" to raw_total. If no cap applies, total_score
  equals raw_total and capped_from equals raw_total.
Return a single JSON object, no markdown fence.
