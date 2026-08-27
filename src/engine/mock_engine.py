import json
from typing import Optional, Dict, Any
from PIL import Image
from src.engine.base_engine import BaseVLMEngine


class MockGemmaEngine(BaseVLMEngine):
    """
    Mock / Simulation Engine for Gemma 4 31B IT.
    Produces deterministic, structured responses for all 4 pipeline stages
    to enable fast local pipeline development and testing without GPU or model weights.
    """

    def __init__(self, model_id: str = "mock-google/gemma-4-31b-it"):
        self.model_id = model_id

    def generate_multimodal(
        self,
        image: Image.Image,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_new_tokens: int = 4096,
        thinking_mode: bool = False,
        **kwargs: Any
    ) -> str:
        prompt_lower = prompt.lower()
        
        # Stage 1: Verbatim Transcription Simulation
        if "transcribe every word exactly as written" in prompt_lower or "stage 1" in prompt_lower or "verbatim" in prompt_lower:
            return (
                "১. উদ্দীপকে বর্নিত বিষয়টি আমাদের সমাজের একটি গুরুত্বপূর্ন সমস্যাকে তুলে ধরে।\n"
                "লেখক এখানে সমাজ সংস্কারের প্রয়োজনীয়তার কথা উল্লেখ করেছেন।\n"
                "তিনি বলেন, কুসংস্কার দূর না হলে জাতি [unclear: উন্নতি] করতে পারবে না।\n"
                "কিন্তু আধুনিক যুগে মানুষ এখনো [illegible] রীতিনীতি মেনে চলছে।\n"
                "পরিশেষে বলা যায় যে, সমাজ পরিবর্তনের জন্য যুব সমাজের অবদান অপরিসিম।"
            )

        # Stage 2: Autocorrection Verification Simulation
        if "autocorrection" in prompt_lower or "cross-check" in prompt_lower or "stage 2" in prompt_lower:
            return json.dumps({
                "verified_transcript": (
                    "১. উদ্দীপকে বর্নিত বিষয়টি আমাদের সমাজের একটি গুরুত্বপূর্ন সমস্যাকে তুলে ধরে।\n"
                    "লেখক এখানে সমাজ সংস্কারের প্রয়োজনীয়তার কথা উল্লেখ করেছেন।\n"
                    "তিনি বলেন, কুসংস্কার দূর না হলে জাতি [unclear: উন্নতি] করতে পারবে না।\n"
                    "কিন্তু আধুনিক যুগে মানুষ এখনো [illegible] রীতিনীতি মেনে চলছে।\n"
                    "পরিশেষে বলা যায় যে, সমাজ পরিবর্তনের জন্য যুব সমাজের অবদান অপরিসিম।"
                ),
                "silent_corrections_fixed": [
                    {
                        "stage1_output": "বর্ণিত",
                        "actual_handwritten": "বর্নিত",
                        "reason": "Stage 1 silently corrected the student's spelling error বর্নিত to standard বর্ণিত.",
                        "context_snippet": "উদ্দীপকে বর্নিত বিষয়টি"
                    },
                    {
                        "stage1_output": "গুরুত্বপূর্ণ",
                        "actual_handwritten": "গুরুত্বপূর্ন",
                        "reason": "Stage 1 silently corrected র্ন to র্ণ.",
                        "context_snippet": "একটি গুরুত্বপূর্ন সমস্যাকে"
                    }
                ],
                "total_corrections_count": 2,
                "verification_notes": "Identified and reverted 2 silent spelling normalizations made during initial VLM transcription pass."
            }, ensure_ascii=False, indent=2)

        # Stage 0b: Teacher Mark Extraction Simulation
        if "red-ink numeric marks" in prompt_lower or "teacher mark" in prompt_lower or "stage 0b" in prompt_lower or "numeric marks written in red ink" in prompt_lower:
            return json.dumps([
                {
                    "question_no": "1",
                    "mark_value": "7/10",
                    "location": "margin next to answer 1"
                }
            ], ensure_ascii=False, indent=2)

        return "Mock multimodal response for Gemma 4 31B IT."

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_new_tokens: int = 4096,
        thinking_mode: bool = False,
        **kwargs: Any
    ) -> str:
        prompt_lower = prompt.lower()

        # Stage 3: Error Extraction Simulation
        if "error extraction" in prompt_lower or "linguistic errors" in prompt_lower or "stage 3" in prompt_lower:
            return json.dumps({
                "errors": [
                    {
                        "error_type": "spelling",
                        "erroneous_text": "বর্নিত",
                        "suggested_correction": "বর্ণিত",
                        "context_sentence": "উদ্দীপকে বর্নিত বিষয়টি আমাদের সমাজের একটি গুরুত্বপূর্ন সমস্যাকে তুলে ধরে।",
                        "explanation": "বাংলা বানানের নিয়ম অনুযায়ী রেফের পর ণ-ত্ব বিধান অনুসারে মূর্ধন্য-ণ (র্ণ) ব্যবহৃত হবে।"
                    },
                    {
                        "error_type": "spelling",
                        "erroneous_text": "গুরুত্বপূর্ন",
                        "suggested_correction": "গুরুত্বপূর্ণ",
                        "context_sentence": "আমাদের সমাজের একটি গুরুত্বপূর্ন সমস্যাকে তুলে ধরে।",
                        "explanation": "পূর্ণ বানানে দীর্ঘ-ঊ ও মূর্ধন্য-ণ (র্ণ) প্রযোজ্য।"
                    },
                    {
                        "error_type": "spelling",
                        "erroneous_text": "অপরিসিম",
                        "suggested_correction": "অপরিসীম",
                        "context_sentence": "যুব সমাজের অবদান অপরিসিম।",
                        "explanation": "সীম বানানে দীর্ঘ-ঈ কার (সী) ব্যবহৃত হয়।"
                    }
                ],
                "spelling_error_count": 3,
                "grammar_error_count": 0,
                "syntax_error_count": 0,
                "punctuation_error_count": 0,
                "total_error_count": 3,
                "linguistic_summary": "The student shows good structural sentence formation but struggles with Bangla Natwa-Bidhan and vowel length (ঈ-কার/ঊ-কার) spellings."
            }, ensure_ascii=False, indent=2)

        # Stage 4: Rubric Evaluation Simulation
        if "rubric" in prompt_lower or "grading" in prompt_lower or "stage 4" in prompt_lower:
            return json.dumps({
                "subject": "Bangla",
                "question_type": "Creative Question (সৃজনশীল প্রশ্ন - গ/ঘ)",
                "criteria_scores": [
                    {
                        "criterion_id": "knowledge",
                        "criterion_name": "জ্ঞানমূলক (Knowledge / Recall)",
                        "max_marks": 2.0,
                        "awarded_marks": 2.0,
                        "justification": "উদ্দীপক ও মূল পাঠের প্রেক্ষাপট স্পষ্টভাবে চিহ্নিত করেছে।",
                        "strengths": ["সরাসরি মূলভাব উল্লেখ"],
                        "weaknesses": []
                    },
                    {
                        "criterion_id": "comprehension",
                        "criterion_name": "অনুধাবনমূলক (Comprehension)",
                        "max_marks": 2.0,
                        "awarded_marks": 1.75,
                        "justification": "সমাজ সংস্কারের প্রয়োজনীয়তা যথাযথভাবে ব্যাখ্যা করেছে।",
                        "strengths": ["ব্যাখ্যার গভীরতা সন্তোষজনক"],
                        "weaknesses": ["একটি বাক্য অসম্পূর্ণ রয়ে গেছে"]
                    },
                    {
                        "criterion_id": "application",
                        "criterion_name": "প্রয়োগমূলক (Application)",
                        "max_marks": 3.0,
                        "awarded_marks": 2.5,
                        "justification": "উদ্দীপকের ভাবার্থের সাথে প্রাসঙ্গিক আলোচনা যুক্ত করা হয়েছে।",
                        "strengths": ["উত্তম উদাহরণ উপস্থাপন"],
                        "weaknesses": ["তুলনামূলক বিশ্লেষণ আরও স্পষ্ট হতে পারত"]
                    },
                    {
                        "criterion_id": "higher_ability",
                        "criterion_name": "উচ্চতর দক্ষতা (Higher Order Synthesis)",
                        "max_marks": 3.0,
                        "awarded_marks": 2.25,
                        "justification": "যুব সমাজের ভূমিকা নিয়ে যৌক্তিক সিদ্ধান্ত টেনেছে।",
                        "strengths": ["স্পষ্ট উপসংহার"],
                        "weaknesses": ["বিশ্লেষণাত্মক যুক্তির গভীরতা বৃদ্ধি করা প্রয়োজন"]
                    }
                ],
                "content_raw_score": 8.5,
                "linguistic_penalty": 0.75,
                "final_score": 7.75,
                "total_max_marks": 10.0,
                "percentage": 77.5,
                "overall_feedback": "উত্তরটি বিষয়বস্তুর দিক থেকে বেশ গোছানো ও যুক্তিগ্রাহ্য। তবে ণ-ত্ব বিধান ও বানানের নির্ভুলতার দিকে আরও মনোযোগী হতে হবে।",
                "actionable_recommendations": [
                    "রেফের পর মূর্ধন্য-ণ এর ব্যবহার (বর্ণিত, গুরুত্বপূর্ণ) নিয়মিত অনুশীলন করুন।",
                    "বাক্যের ধারাবাহিকতা বজায় রেখে বিশ্লেষণ আরও সমৃদ্ধ করুন।"
                ]
            }, ensure_ascii=False, indent=2)

        return "Mock text response for Gemma 4 31B IT."

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "engine_type": "MockGemmaEngine",
            "model_id": self.model_id,
            "quantization": "simulated-4bit",
            "torch_dtype": "bfloat16",
            "gpu_name": "Simulated NVIDIA RTX 5090 (32GB VRAM)",
            "vram_allocated_gb": 18.2,
            "vram_reserved_gb": 20.0,
        }
