"""
Base utilities and JSON parsing helpers for pipeline stages.
"""

import json
import re
from typing import Any, Dict, Optional


def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """
    Extract JSON dict from LLM response text, handling markdown code fences,
    preceding commentary, or trailing text.
    """
    text = response_text.strip()

    # Case 1: Markdown code block ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        json_candidate = match.group(1).strip()
        try:
            parsed = json.loads(json_candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Case 2: Direct json parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Case 3: Search for outermost curly braces { ... }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_candidate = text[first_brace:last_brace + 1]
        try:
            parsed = json.loads(json_candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Fallback empty dict
    return {}
