"""
Prompt templates and builders for the 4-stage Gemma multimodal grading pipeline.
"""
from src.prompts.stage1_verbatim import build_stage1_prompt, STAGE1_SYSTEM_PROMPT
from src.prompts.stage2_verification import build_stage2_prompt, STAGE2_SYSTEM_PROMPT
from src.prompts.stage3_errors import build_stage3_prompt, STAGE3_SYSTEM_PROMPT
from src.prompts.stage4_rubric import build_stage4_prompt, STAGE4_SYSTEM_PROMPT

__all__ = [
    "build_stage1_prompt", "STAGE1_SYSTEM_PROMPT",
    "build_stage2_prompt", "STAGE2_SYSTEM_PROMPT",
    "build_stage3_prompt", "STAGE3_SYSTEM_PROMPT",
    "build_stage4_prompt", "STAGE4_SYSTEM_PROMPT"
]
