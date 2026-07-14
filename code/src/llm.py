"""Thin wrapper around the Gemini client with on-disk response caching.

Cache key = (stage, id, variant). Every raw response is written to data/cache so
reruns and reprompts never re-spend (CLAUDE.md invariant).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

from . import config

_client = None


def client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(vertexai=True, api_key=config.get_api_key())
    return _client


def _cache_path(stage: str, id_: str, variant: str) -> Path:
    return config.CACHE_DIR / f"{stage}__{id_}__{variant}.json"


def generate_json(
    stage: str,
    id_: str,
    variant: str,
    parts: list[Any],
    response_schema: Optional[dict] = None,
    temperature: float = 0.0,
    force: bool = False,
) -> dict:
    """Call Gemini for a JSON response, caching the parsed result on disk.

    `parts` is the list of content parts (text strings and image Parts).
    Returns the parsed dict plus bookkeeping under `_meta`.
    """
    cache = _cache_path(stage, id_, variant)
    if cache.exists() and not force:
        return json.loads(cache.read_text())

    from google.genai import types, errors

    def _call(with_schema: bool):
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
        )
        if with_schema and response_schema is not None:
            cfg.response_schema = response_schema
        return client().models.generate_content(
            model=config.MODEL, contents=parts, config=cfg
        )

    def _code(e) -> Optional[int]:
        return getattr(e, "code", None) if isinstance(e, errors.APIError) else None

    resp = None
    use_schema = response_schema is not None
    for attempt in range(config.MAX_RETRIES):
        try:
            resp = _call(with_schema=use_schema)
            break
        except Exception as e:
            code = _code(e)
            # A rejected response_schema (400 INVALID_ARGUMENT) is permanent for
            # this schema — drop it and retry immediately in plain JSON mode.
            if code == 400 and use_schema:
                use_schema = False
                continue
            # Rate limit / server busy: wait and retry (self-heals a per-minute cap).
            if code in (429, 503) and attempt < config.MAX_RETRIES - 1:
                wait = min(config.BACKOFF_BASE * (2 ** attempt), config.BACKOFF_CAP)
                print(f"      [{stage} {id_}] {code}; backing off {wait:.0f}s "
                      f"(attempt {attempt + 1}/{config.MAX_RETRIES})")
                time.sleep(wait)
                continue
            raise
    raw = (resp.text if resp is not None else None) or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"_parse_error": True, "_raw": raw}

    data["_meta"] = {
        "stage": stage, "id": id_, "variant": variant,
        "model": config.MODEL,
        "prompt_sha": hashlib.sha1(
            "".join(p for p in parts if isinstance(p, str)).encode()
        ).hexdigest()[:12],
    }
    cache.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return data


def image_part(path: Path):
    from google.genai import types
    return types.Part.from_bytes(data=path.read_bytes(), mime_type="image/jpeg")
