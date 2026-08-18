"""
Deterministic Mock client for offline local testing, unit tests, and ablation validation.
Generates realistic, schema-compliant JSON responses matching rubric_v2.txt, extraction.csv, and evaluation.csv.
"""

from typing import Any, Dict, List, Optional
import json
from PIL import Image

try:
    from .base import ModelClient
    from schemas import ModelConfig
except ImportError:
    try:
        from model_client.base import ModelClient
        from schemas import ModelConfig
    except ImportError:
        from src.model_client.base import ModelClient
        from src.schemas import ModelConfig


class MockClient(ModelClient):
    """Deterministic Mock client returning valid JSON per stage aligned with rubric_v2.txt."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig(backend="mock")

    def generate(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        prompt_lower = prompt.lower()

        # Stage 1: Extractor A (Cold OCR Read)
        if "extractor a" in prompt_lower or "cold ocr" in prompt_lower:
            if "letter" in prompt_lower or "email" in prompt_lower:
                return json.dumps({
                    "QUESTION_TEXT": "Ans to Q. No. 10",
                    "STUDENT_ANSWER": "Jatrabari, Dhaka-1204\n27-06-2026\nDear Saiful,\nAt the beggining of the letter take my salam and the best wishes from the core of my heart. In your letter you wanted to know about my future plan after passing the Hsc Exam. Today I'm writing so. After my Hsc Exam I will get a lot of free time which I am thinking to use forc freelancing.\n\nNo more today. Hoping to hear from you soon.\n\nYours ever,\nKalam\n\nFrom: Kalam\nTo: Saiful\nStamp",
                    "extracted_text_raw": "Jatrabari, Dhaka-1204\n27-06-2026\nDear Saiful,\nAt the beggining of the letter take my salam [struck: and wishes] and the best wishes...",
                    "UNCERTAINTY_AREAS": [
                        {
                            "location": "line 12, word 4",
                            "text_guess": "freelancing",
                            "reason": "slight cursive trail"
                        }
                    ],
                    "struck_tokens": ["and wishes"],
                    "word_count": 155
                }, ensure_ascii=False)
            elif "paragraph" in prompt_lower or "artificial intelligence" in prompt_lower:
                return json.dumps({
                    "QUESTION_TEXT": "Ans to Q. No. 7",
                    "STUDENT_ANSWER": "Artificial Intelligence is the simulation of human intelligence in machines. It is one of the most prominent technological advancements of modern times.\n\nTeachers and students gain facilities from AI tools like ChatGPT and Gemini.",
                    "extracted_text_raw": "Artificial Intelligence is the simulation...",
                    "UNCERTAINTY_AREAS": [],
                    "struck_tokens": [],
                    "word_count": 140
                }, ensure_ascii=False)
            elif "story" in prompt_lower or "lion" in prompt_lower:
                return json.dumps({
                    "QUESTION_TEXT": "Ans to Q. No. 9",
                    "STUDENT_ANSWER": "Size doesn't Matter\n\nOnce a lion was sleeping in a forest. Suddenly a mouse came and disturbed him. The lion caught the mouse, but let him go after the mouse pleaded.\n\nLater, the lion was caught in a hunter's net. The mouse came and cut the ropes, freeing the lion.",
                    "extracted_text_raw": "Size doesn't Matter\n\nOnce a lion...",
                    "UNCERTAINTY_AREAS": [],
                    "struck_tokens": [],
                    "word_count": 110
                }, ensure_ascii=False)
            elif "summary" in prompt_lower or "feathers" in prompt_lower:
                return json.dumps({
                    "QUESTION_TEXT": "Ans to Q. No. 3",
                    "STUDENT_ANSWER": "Hope is an ever-present internal solace that sustains the human spirit through life's harshest adversities without demanding anything in return.",
                    "extracted_text_raw": "Hope is an ever-present...",
                    "UNCERTAINTY_AREAS": [],
                    "struck_tokens": [],
                    "word_count": 25
                }, ensure_ascii=False)
            elif "theme" in prompt_lower or "dream" in prompt_lower:
                return json.dumps({
                    "QUESTION_TEXT": "Ans to Q. No. 11",
                    "STUDENT_ANSWER": "The poem contrasts passive dreamers who achieve nothing with proactive visionaries who actively turn their daylight dreams into tangible reality.",
                    "extracted_text_raw": "The poem contrasts...",
                    "UNCERTAINTY_AREAS": [],
                    "struck_tokens": [],
                    "word_count": 28
                }, ensure_ascii=False)
            else:
                # Graph / Chart Default (Q8)
                return json.dumps({
                    "QUESTION_TEXT": "Ans to the Q. No. 8",
                    "STUDENT_ANSWER": "The sources of the USA electricity in 1980\n\nIn the given pie-chart we can see the sources of the USA electricity in 1980.\n\nIn 1980 The main source of generation electricity in USA was Coal which was the highest 46% of the total sources. It was the highest demand for generating electricity. After that Natural gas the second highest option for generating electricity which was 24%. Then it was Hydro-electric power which was 16% of total. 12% electricity was generated from oil. Now the least used one was Nuclear only 2%.\n\nSo we can make a conclusion that they used so much non-renewable energy which are limited in nature. If we want to save these natural resources we should increase the use of renewable energy.",
                    "extracted_text_raw": "The sources of the USA electricity in 1980\n\nIn the given pie-charet we can see that the sources of the USA electricity in 1980.\n\nIn 1980 The main source of generation electricity in USA was [struck: 24%] [struck: Natural] Coal [struck: gas] which was the highesst 46% of the total sources...\n\nSo we can make a [struck: col] conclusion that they [struck: sued] used so much non-renewal energy...",
                    "UNCERTAINTY_AREAS": [
                        {
                            "location": "line 3, word 5",
                            "text_guess": "pie-chart",
                            "reason": "slight spelling variation"
                        }
                    ],
                    "struck_tokens": ["24%", "Natural", "gas", "col", "sued"],
                    "word_count": 129
                }, ensure_ascii=False)

        # Stage 6: Compressor (Audit)
        elif "compressor" in prompt_lower or "auditor" in prompt_lower or "stage 6" in prompt_lower:
            return json.dumps({
                "task_type": "Graph_Chart",
                "max_mark_applied": 10.0,
                "final_marks": 5.0,
                "performance_band": "Band 2",
                "score_breakdown": {
                    "context_content_data": 2.0,
                    "structure_format_brevity": 1.0,
                    "language_mechanics": 1.0,
                    "originality_comparisons_paraphrase": 1.0,
                    "total_score": 5.0
                },
                "structural_audit": {
                    "cap_applied": True,
                    "applied_cap_reason": "Graph_External_Facts"
                },
                "attempt_status": {
                    "is_attempted": True,
                    "is_blank_or_unintelligible": False
                },
                "error_analysis": {
                    "frequent_errors": [
                        "Inclusion of external personal opinions and moral recommendations in graph analysis",
                        "Minor spelling errors ('charet', 'highesst', 'enery')",
                        "Incomplete sentence syntax in introductory statement"
                    ],
                    "positive_aspects": [
                        "Accurately extracted and reported all percentages from the given chart",
                        "Correctly arranged data in descending logical order from highest to lowest"
                    ]
                },
                "feedback_summary": "The student successfully reported all key data points from the chart in descending order. However, the response includes external moralizing opinions and policy recommendations, which violates the objective analytical structure required for chart descriptions and triggers a mandatory score cap of 6. Final score awarded is 5 out of 10 (Band 2).",
                "sum_check_passed": True,
                "band_check_passed": True,
                "error_detection": None
            }, ensure_ascii=False)

        # Stage 5: Examiner (Grading)
        elif "chief examiner" in prompt_lower or "examiner" in prompt_lower or "stage 5" in prompt_lower:
            if "task type: letter_email" in prompt_lower or "task_id: letter" in prompt_lower or "qno: 10" in prompt_lower:
                return json.dumps({
                    "task_type": "Letter_Email",
                    "max_mark_applied": 5.0,
                    "attempt_status": {
                        "is_attempted": True,
                        "is_blank_or_unintelligible": False
                    },
                    "structural_audit": {
                        "cap_applied": False,
                        "applied_cap_reason": "None"
                    },
                    "score_breakdown": {
                        "context_content_data": 1.0,
                        "structure_format_brevity": 2.0,
                        "language_mechanics": 0.5,
                        "originality_comparisons_paraphrase": 0.5,
                        "total_score": 4.0
                    },
                    "performance_band": "Band 4",
                    "error_analysis": {
                        "frequent_errors": ["Minor spelling slips ('beggining', 'forc')"],
                        "positive_aspects": ["Complete informal letter layout with envelope block", "Authentic modern future plans"]
                    },
                    "feedback_summary": "The letter strictly adheres to the standard structural format of an informal letter, including all layout elements and an envelope block. Final score awarded is 4 out of 5 (Band 4)."
                }, ensure_ascii=False)
            elif "task type: paragraph" in prompt_lower or "task_id: para" in prompt_lower or "qno: 7" in prompt_lower:
                return json.dumps({
                    "task_type": "Paragraph",
                    "max_mark_applied": 10.0,
                    "attempt_status": {
                        "is_attempted": True,
                        "is_blank_or_unintelligible": False
                    },
                    "structural_audit": {
                        "cap_applied": False,
                        "applied_cap_reason": "None"
                    },
                    "score_breakdown": {
                        "context_content_data": 2.0,
                        "structure_format_brevity": 2.0,
                        "language_mechanics": 1.0,
                        "originality_comparisons_paraphrase": 1.0,
                        "total_score": 6.0
                    },
                    "performance_band": "Band 3",
                    "error_analysis": {
                        "frequent_errors": ["Occasional spelling slips ('featers', 'presicely')"],
                        "positive_aspects": ["Single paragraph structure maintained", "Comprehensive coverage of guided prompt questions"]
                    },
                    "feedback_summary": "The candidate presents a well-focused single paragraph response covering AI features and applications. Final score awarded is 6 out of 10 (Band 3)."
                }, ensure_ascii=False)
            elif "task type: story" in prompt_lower or "task_id: story" in prompt_lower or "qno: 9" in prompt_lower:
                return json.dumps({
                    "task_type": "Story",
                    "max_mark_applied": 7.0,
                    "attempt_status": {
                        "is_attempted": True,
                        "is_blank_or_unintelligible": False
                    },
                    "structural_audit": {
                        "cap_applied": False,
                        "applied_cap_reason": "None"
                    },
                    "score_breakdown": {
                        "context_content_data": 2.0,
                        "structure_format_brevity": 1.5,
                        "language_mechanics": 0.5,
                        "originality_comparisons_paraphrase": 1.0,
                        "total_score": 5.0
                    },
                    "performance_band": "Band 3",
                    "error_analysis": {
                        "frequent_errors": ["Tense shifts between past and present"],
                        "positive_aspects": ["Logical narrative progression", "Apt title and clear moral resolution provided"]
                    },
                    "feedback_summary": "The story continues smoothly from the stem with a relevant title and constructive moral lesson. Score awarded is 5 out of 7 (Band 3)."
                }, ensure_ascii=False)
            elif "task type: summary" in prompt_lower or "task_id: summary" in prompt_lower or "qno: 3" in prompt_lower:
                return json.dumps({
                    "task_type": "Summary",
                    "max_mark_applied": 10.0,
                    "attempt_status": {
                        "is_attempted": True,
                        "is_blank_or_unintelligible": False
                    },
                    "structural_audit": {
                        "cap_applied": False,
                        "applied_cap_reason": "None"
                    },
                    "score_breakdown": {
                        "context_content_data": 2.0,
                        "structure_format_brevity": 2.0,
                        "language_mechanics": 1.0,
                        "originality_comparisons_paraphrase": 1.0,
                        "total_score": 6.0
                    },
                    "performance_band": "Band 3",
                    "error_analysis": {
                        "frequent_errors": ["Minor spelling slips ('alaways')"],
                        "positive_aspects": ["Zero verbatim line copying", "Kept within single paragraph length constraints"]
                    },
                    "feedback_summary": "The candidate captures the primary metaphor of hope in their own words. Final score awarded is 6 out of 10 (Band 3)."
                }, ensure_ascii=False)
            elif "task type: theme" in prompt_lower or "task_id: theme" in prompt_lower or "qno: 11" in prompt_lower:
                return json.dumps({
                    "task_type": "Theme",
                    "max_mark_applied": 8.0,
                    "attempt_status": {
                        "is_attempted": True,
                        "is_blank_or_unintelligible": False
                    },
                    "structural_audit": {
                        "cap_applied": False,
                        "applied_cap_reason": "None"
                    },
                    "score_breakdown": {
                        "context_content_data": 2.0,
                        "structure_format_brevity": 1.0,
                        "language_mechanics": 0.5,
                        "originality_comparisons_paraphrase": 0.5,
                        "total_score": 4.0
                    },
                    "performance_band": "Band 2",
                    "error_analysis": {
                        "frequent_errors": ["Partial restatement of poem lines"],
                        "positive_aspects": ["Identified core theme of dreams"]
                    },
                    "feedback_summary": "Identified the general theme of daylight dreamers versus night dreamers. Score awarded is 4 out of 8 (Band 2)."
                }, ensure_ascii=False)
            else:
                # Graph_Chart default
                return json.dumps({
                    "task_type": "Graph_Chart",
                    "max_mark_applied": 10.0,
                    "attempt_status": {
                        "is_attempted": True,
                        "is_blank_or_unintelligible": False
                    },
                    "structural_audit": {
                        "cap_applied": True,
                        "applied_cap_reason": "Graph_External_Facts"
                    },
                    "score_breakdown": {
                        "context_content_data": 2.0,
                        "structure_format_brevity": 1.0,
                        "language_mechanics": 1.0,
                        "originality_comparisons_paraphrase": 1.0,
                        "total_score": 5.0
                    },
                    "performance_band": "Band 2",
                    "error_analysis": {
                        "frequent_errors": [
                            "Inclusion of external personal opinions and moral recommendations in graph analysis",
                            "Minor spelling errors ('charet', 'highesst', 'enery')"
                        ],
                        "positive_aspects": [
                            "Accurately extracted and reported all percentages from the given chart",
                            "Correctly arranged data in descending logical order"
                        ]
                    },
                    "feedback_summary": "The student reported key data points from the chart. However, inclusion of external moralizing opinions triggers a score cap. Final score awarded is 5 out of 10 (Band 2)."
                }, ensure_ascii=False)

        # Stage 4: OCR Supervisor
        elif "ocr supervisor" in prompt_lower or "adjudication" in prompt_lower:
            return json.dumps({
                "QUESTION_TEXT": "Ans to the Q. No. 8",
                "STUDENT_ANSWER": "The sources of the USA electricity in 1980\n\nIn the given pie-chart we can see the sources of the USA electricity in 1980.\n\nIn 1980 The main source of generation electricity in USA was Coal which was the highest 46% of the total sources. Natural gas was 24%, Hydro-electric power was 16%, oil was 12%, and Nuclear was 2%.\n\nSo we can make a conclusion that they used so much non-renewable energy.",
                "adjudication_notes": "Verified against physical image strokes. Confirmed strike-out omissions and clean paragraph boundaries."
            }, ensure_ascii=False)

        # Stage 3: Extractor B (Reference-primed OCR)
        elif "extractor b" in prompt_lower or "resolved_via_reference" in prompt_lower:
            return json.dumps({
                "QUESTION_TEXT": "Ans to the Q. No. 8",
                "STUDENT_ANSWER": "The sources of the USA electricity in 1980\n\nIn the given pie-chart we can see the sources of the USA electricity in 1980.",
                "RESOLVED_VIA_REFERENCE": [
                    {
                        "location": "line 3, word 5",
                        "resolved_text": "pie-chart",
                        "reference_cue": "prompt refers to chart/pie-chart"
                    }
                ],
                "STILL_UNCERTAIN": []
            }, ensure_ascii=False)

        # Stage 2: RubricAligner
        elif "rubricaligner" in prompt_lower or "stage 2" in prompt_lower:
            return json.dumps({
                "decision": "KEEP",
                "operative_rubric": "Rubric v2: NCTB HSC English 1st Paper Chief Examiner Evaluation",
                "examiner_note": "Student followed standard prompt and task format. Apply official rubric directly.",
                "shadow_solution": None
            }, ensure_ascii=False)

        # Default fallback JSON
        return json.dumps({
            "status": "success",
            "message": "Mock completion generated."
        })
