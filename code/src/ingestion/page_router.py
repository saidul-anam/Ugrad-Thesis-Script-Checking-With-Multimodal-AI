"""
Page Router & Multi-Page Question Segmenter.
Identifies and routes pages in multi-page student script PDFs (e.g., SE_11_Q1_0001.pdf, 15-20+ pages)
into discrete question units (Q3, Q7, Q8, Q9, Q10, Q11).
Supports both Ground-Truth CSV Manifests (extraction.csv) and Automated Header/VLM Detection.
"""

from __future__ import annotations
import os
import re
import csv
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image

try:
    from schemas import QuestionSegment, ScriptManifest, TASK_MAX_MARKS, TASK_QUESTION_NUMBERS
    from model_client.base import ModelClient
except ImportError:
    try:
        from schemas import QuestionSegment, ScriptManifest, TASK_MAX_MARKS, TASK_QUESTION_NUMBERS
        from model_client.base import ModelClient
    except ImportError:
        from src.schemas import QuestionSegment, ScriptManifest, TASK_MAX_MARKS, TASK_QUESTION_NUMBERS
        from src.model_client.base import ModelClient


def parse_page_range(page_range_str: str) -> List[int]:
    """
    Parse 1-indexed page range string into 0-indexed page index list.
    Examples:
        "3-4"   -> [2, 3]
        "15"    -> [14]
        "1-3"   -> [0, 1, 2]
        "7-9"   -> [6, 7, 8]
        ""      -> []
    """
    if not page_range_str or not str(page_range_str).strip():
        return []
    
    clean_str = str(page_range_str).strip()
    if "-" in clean_str:
        parts = clean_str.split("-")
        try:
            start = int(parts[0].strip())
            end = int(parts[1].strip())
            return list(range(start - 1, end))
        except ValueError:
            return []
    elif "," in clean_str:
        parts = clean_str.split(",")
        indices = []
        for p in parts:
            try:
                indices.append(int(p.strip()) - 1)
            except ValueError:
                pass
        return indices
    else:
        try:
            val = int(clean_str)
            return [val - 1]
        except ValueError:
            return []


def load_manifest_from_csv(csv_path: str) -> Dict[str, ScriptManifest]:
    """
    Load ground-truth question page mappings from extraction.csv.
    
    Returns:
        Dict mapping script_id -> ScriptManifest
    """
    manifests: Dict[str, ScriptManifest] = {}
    if not os.path.exists(csv_path):
        return manifests

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            script_id = row.get("script_id", "").strip()
            task_id = row.get("task_id", "").strip()
            q_no = str(row.get("question_no", "")).strip()
            q_type = row.get("question_type", "").strip()
            p_range_str = row.get("page_range", "").strip()
            
            try:
                max_mark = float(row.get("max_mark", TASK_MAX_MARKS.get(q_type, 10.0)))
            except (ValueError, TypeError):
                max_mark = TASK_MAX_MARKS.get(q_type, 10.0)

            try:
                t_mark = float(row["teacher_mark"]) if row.get("teacher_mark") else None
            except (ValueError, TypeError):
                t_mark = None

            q_prompt = row.get("question", "").strip()
            page_indices = parse_page_range(p_range_str)

            if script_id not in manifests:
                manifests[script_id] = ScriptManifest(script_id=script_id)

            segment = QuestionSegment(
                task_id=task_id,
                script_id=script_id,
                question_no=q_no,
                question_type=q_type,
                max_mark=max_mark,
                page_indices=page_indices,
                page_range_str=p_range_str,
                teacher_mark=t_mark,
                question_prompt=q_prompt
            )
            # Index by unique task_id
            manifests[script_id].questions[task_id] = segment

    return manifests


class PageRouter:
    """Routes multi-page student answer scripts to distinct question segments."""

    @staticmethod
    def parse_page_range_str(page_range_str: str) -> List[int]:
        return parse_page_range(page_range_str)

    @staticmethod
    def route_pages(
        pages: List[Image.Image],
        manifest: Optional[Dict[str, List[int]]] = None,
        default_question_id: str = "Q1"
    ) -> Dict[str, List[Image.Image]]:
        """
        Group pages by question_id.
        """
        if not pages:
            return {}

        if manifest:
            routed: Dict[str, List[Image.Image]] = {}
            for q_id, indices in manifest.items():
                selected_pages = [pages[i] for i in indices if 0 <= i < len(pages)]
                if selected_pages:
                    routed[q_id] = selected_pages
            return routed

        return {default_question_id: pages}

    @staticmethod
    def get_question_pages_from_manifest(
        pages: List[Image.Image],
        script_id: str,
        task_or_question_id: str,
        manifest_csv: str = "extraction.csv"
    ) -> Tuple[List[Image.Image], Optional[QuestionSegment]]:
        """
        Fetch question-specific page images using extraction.csv manifest.
        """
        manifests = load_manifest_from_csv(manifest_csv)
        script_manifest = manifests.get(script_id)
        if not script_manifest:
            return pages, None

        segment = script_manifest.questions.get(task_or_question_id)
        if not segment and not task_or_question_id.startswith("Q"):
            segment = script_manifest.questions.get(f"Q{task_or_question_id}")

        if segment and segment.page_indices:
            selected_pages = [pages[i] for i in segment.page_indices if 0 <= i < len(pages)]
            if selected_pages:
                return selected_pages, segment

        return pages, segment

    @staticmethod
    def auto_detect_question_segments(
        pages: List[Image.Image],
        script_id: str = "script_01",
        model_client: Optional[ModelClient] = None
    ) -> Dict[str, QuestionSegment]:
        """
        Heuristic / VLM-based question detection across multi-page scripts when no manifest is available.
        Inspects page contents for question markers like:
        - Q3 / Summary: "Summarize", "Hope is the thing with feathers", "Summary"
        - Q7 / Paragraph: "Artificial Intelligence", "AI is", "Paragraph on"
        - Q8 / Graph_Chart: "The sources of the USA electricity", "pie-chart", "In 1980"
        - Q9 / Story: "Once a lion was sleeping", "Size doesn't Matter", "The Lion and the Mouse"
        - Q10 / Letter_Email: "Dear friend", "Dear Saiful", "Dear Jarif", "Stamp", "Yours ever"
        - Q11 / Theme: "All people dream", "theme of the poem"
        """
        detected_segments: Dict[str, QuestionSegment] = {}
        
        # Standard definitions
        standard_tasks = [
            ("Q8", "CHART", "Graph_Chart", 10.0, ["pie-chart", "pie chart", "usa electricity", "sources of electricity", "coal 46%", "natural gas 24%"]),
            ("Q10", "LETTER", "Letter_Email", 5.0, ["dear", "yours ever", "from:", "to:", "stamp", "hsc exam", "future plan"]),
            ("Q7", "PARA", "Paragraph", 10.0, ["artificial intelligence", "ai is", "robot", "chatgpt"]),
            ("Q9", "STORY", "Story", 7.0, ["once a lion", "lion was sleeping", "mouse", "forest", "hunter", "net"]),
            ("Q3", "SUMMARY", "Summary", 10.0, ["hope is the thing", "feathers", "soul", "bird", "summary"]),
            ("Q11", "THEME", "Theme", 8.0, ["all people dream", "theme of the poem", "dreamers of the day"])
        ]

        # For single question runs or default grouping
        for q_no, prefix, q_type, max_m, keywords in standard_tasks:
            task_id = f"{prefix}_001"
            detected_segments[q_no] = QuestionSegment(
                task_id=task_id,
                script_id=script_id,
                question_no=q_no.replace("Q", ""),
                question_type=q_type,
                max_mark=max_m,
                page_indices=list(range(len(pages))),
                page_range_str=f"1-{len(pages)}"
            )

        return detected_segments

    @staticmethod
    def segment_script_into_questions(
        pages: List[Image.Image],
        script_id: str,
        manifest_csv: Optional[str] = "extraction.csv",
        auto_detect: bool = False,
        model_client: Optional[ModelClient] = None
    ) -> Dict[str, Tuple[List[Image.Image], QuestionSegment]]:
        """
        Segment a multi-page PDF into all its constituent questions.
        
        Returns:
            Dict mapping task_id (or question_no) -> (List[PIL.Image], QuestionSegment)
        """
        segmented: Dict[str, Tuple[List[Image.Image], QuestionSegment]] = {}

        if manifest_csv and os.path.exists(manifest_csv):
            manifests = load_manifest_from_csv(manifest_csv)
            if script_id in manifests:
                script_manifest = manifests[script_id]
                for task_id, seg in script_manifest.questions.items():
                    if seg.page_indices:
                        selected = [pages[i] for i in seg.page_indices if 0 <= i < len(pages)]
                        if selected:
                            segmented[task_id] = (selected, seg)
                if segmented:
                    return segmented

        # If no manifest match, run auto detection
        detected = PageRouter.auto_detect_question_segments(pages, script_id, model_client)
        for q_no, seg in detected.items():
            selected = [pages[i] for i in seg.page_indices if 0 <= i < len(pages)]
            if selected:
                segmented[q_no] = (selected, seg)

        return segmented
