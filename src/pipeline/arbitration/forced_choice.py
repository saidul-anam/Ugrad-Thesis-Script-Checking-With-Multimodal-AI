"""
Two-alternative forced choice on the line crop: does the ink read the candidate or the intended
token? Options are presented in a seeded random order to cancel position bias; NEITHER is allowed.
"""

import json
import random
import re
from typing import Dict, Any, Optional

from PIL import Image

from src.prompts.stage3b_arbitration import FORCED_CHOICE_SYSTEM_PROMPT, build_forced_choice_prompt


def _parse_choice(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    up = t.upper()
    if "NEITHER" in up:
        return {"choice": "NEITHER", "confidence": 0.5, "reason": t[:120]}
    m2 = re.search(r"\b([AB])\b", up)
    if m2:
        return {"choice": m2.group(1), "confidence": 0.6, "reason": t[:120]}
    return None


class ForcedChoiceJudge:
    def __init__(self, engine, seed: int = 0):
        self.engine = engine
        self.seed = seed
        self.model_calls = 0
        self.token_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _accumulate_usage(self) -> None:
        u = self.engine.get_last_usage() or {}
        for k in self.token_usage:
            self.token_usage[k] += int(u.get(k, 0) or 0)

    def judge(self, crop: Image.Image, candidate: str, intended: str, context: str, candidate_id: str = "") -> Dict[str, Any]:
        rng = random.Random(f"{self.seed}:{candidate_id}:{candidate}:{intended}")
        cand_first = rng.random() < 0.5
        option_a, option_b = (candidate, intended) if cand_first else (intended, candidate)
        order = "candidate_first" if cand_first else "intended_first"
        hint = re.sub(re.escape(candidate), "___", context, flags=re.IGNORECASE) if candidate else context

        try:
            raw = self.engine.generate_multimodal(
                image=crop,
                prompt=build_forced_choice_prompt(option_a, option_b, hint),
                system_prompt=FORCED_CHOICE_SYSTEM_PROMPT,
                temperature=0.0,
                top_p=0.1,
                max_new_tokens=160,
                thinking_mode=False,
            )
            self.model_calls += 1
            self._accumulate_usage()
        except Exception as ex:
            return {"choice": "invalid", "confidence": None, "reason": f"call failed: {ex}", "order": order, "signal": None}

        parsed = _parse_choice(raw)
        if not parsed:
            return {"choice": "invalid", "confidence": None, "reason": raw[:120], "order": order, "signal": None}

        letter = str(parsed.get("choice", "")).strip().upper()[:7]
        try:
            conf = float(parsed.get("confidence", 0.5))
        except Exception:
            conf = 0.5
        conf = min(1.0, max(0.0, conf))
        reason = str(parsed.get("reason", ""))[:200]

        if letter.startswith("NEITHER"):
            return {"choice": "neither", "confidence": conf, "reason": reason, "order": order, "signal": 0.5}
        if letter.startswith("A"):
            chosen = "candidate" if cand_first else "intended"
        elif letter.startswith("B"):
            chosen = "intended" if cand_first else "candidate"
        else:
            return {"choice": "invalid", "confidence": conf, "reason": reason, "order": order, "signal": None}

        # signal: probability that the ink shows the intended word (=> ambiguity / misread)
        signal = 0.5 + 0.5 * conf if chosen == "intended" else 0.5 - 0.5 * conf
        return {"choice": chosen, "confidence": conf, "reason": reason, "order": order, "signal": signal}
