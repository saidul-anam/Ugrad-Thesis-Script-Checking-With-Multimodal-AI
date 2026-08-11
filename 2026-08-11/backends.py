"""Pluggable transcription backends.

Everything downstream consumes PageResult and must not care which backend
produced it. No post-processing of backend output happens here or anywhere:
the text is stored exactly as returned.
"""
from __future__ import annotations

import base64
import os
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path

from common import PROMPT_FILE, PageResult


class TranscriptionBackend(ABC):
    name: str            # "gemini" | "mistral_ocr"
    version: str         # exact model/API version string
    supports_prompt: bool

    @abstractmethod
    def transcribe_page(self, image_path: Path) -> PageResult: ...

    @abstractmethod
    def settings(self) -> dict:
        """The settings actually used, for the run log and dataset."""


class GeminiBackend(TranscriptionBackend):
    supports_prompt = True

    def __init__(self, model: str = "gemini-3.6-flash"):
        from google import genai
        from google.genai import types  # noqa: F401 (validated at init)

        self.name = "gemini"
        self._genai_types = types
        # The project key is a Vertex AI express-mode key; it is not valid for
        # the Gemini Developer API endpoint.
        self.client = genai.Client(vertexai=True, api_key=os.environ["GOOGLE_API_KEY"])
        self.model = model
        self.version = model  # replaced by exact model_version from first response
        self.prompt = PROMPT_FILE.read_text()
        self.temperature = 0.0
        self.max_output_tokens = 32768
        self.top_p = None  # not set; API default
        self.max_429_retries = 5
        self.backoff_base_s = 2.0

    def settings(self) -> dict:
        return {
            "model": self.model,
            "model_version_reported": self.version,
            "api_mode": "vertex_ai_express",
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "top_p": self.top_p,
            "supports_prompt": True,
            "prompt_file": str(PROMPT_FILE.name),
            "backoff_429": {"max_retries": self.max_429_retries, "base_s": self.backoff_base_s},
        }

    @staticmethod
    def _is_rate_limit(e: Exception) -> bool:
        return getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e)

    def transcribe_page(self, image_path: Path) -> PageResult:
        t0 = time.time()
        types = self._genai_types
        try:
            image_part = types.Part.from_bytes(
                data=image_path.read_bytes(), mime_type="image/png"
            )
            retries_429 = 0
            while True:
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=[self.prompt, image_part],
                        config=types.GenerateContentConfig(
                            temperature=self.temperature,
                            max_output_tokens=self.max_output_tokens,
                        ),
                    )
                    break
                except Exception as e:
                    if not self._is_rate_limit(e) or retries_429 >= self.max_429_retries:
                        raise
                    delay = self.backoff_base_s * (2 ** retries_429) + random.uniform(0, 1)
                    retries_429 += 1
                    print(f"    429 rate limit; sleeping {delay:.1f}s "
                          f"(retry {retries_429}/{self.max_429_retries})")
                    time.sleep(delay)
            duration = time.time() - t0
            raw = response.model_dump(mode="json", exclude_none=True)
            if response.model_version:
                self.version = response.model_version
            um = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", None),
                "output_tokens": getattr(um, "candidates_token_count", None),
                "thoughts_tokens": getattr(um, "thoughts_token_count", None),
                "total_tokens": getattr(um, "total_token_count", None),
                "retries_429": retries_429,
            }
            text = response.text
            if not text or not text.strip():
                return PageResult(None, usage, raw, "empty response", duration)
            return PageResult(text, usage, raw, None, duration)
        except Exception as e:
            return PageResult(None, {}, None, repr(e), time.time() - t0)


class MistralOCRBackend(TranscriptionBackend):
    # Investigated 2026-08-11: the OCR endpoint accepts no instruction, system
    # prompt, or free-form guidance of any kind. The annotations feature only
    # extracts structured metadata after OCR and cannot change transcription
    # behaviour, so the prompt file is not used and no substitute is attempted.
    supports_prompt = False

    def __init__(self, model: str = "mistral-ocr-latest"):
        from mistralai.client import Mistral

        self.name = "mistral_ocr"
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        self.model = model
        self.version = model  # replaced by exact model string from first response

    def settings(self) -> dict:
        return {
            "model": self.model,
            "model_version_reported": self.version,
            "supports_prompt": False,
            "note": "OCR API accepts no transcription instructions; defaults used",
        }

    def transcribe_page(self, image_path: Path) -> PageResult:
        t0 = time.time()
        try:
            b64 = base64.b64encode(image_path.read_bytes()).decode()
            response = self.client.ocr.process(
                model=self.model,
                document={
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{b64}",
                },
            )
            duration = time.time() - t0
            raw = response.model_dump(mode="json")
            # Don't archive the echoed input image; it bloats the raw log.
            for page in raw.get("pages", []):
                page.pop("images", None)
            if getattr(response, "model", None):
                self.version = response.model
            usage = {
                "pages_processed": getattr(response.usage_info, "pages_processed", None),
                "doc_size_bytes": getattr(response.usage_info, "doc_size_bytes", None),
            }
            text = "\n\n".join(p.markdown for p in response.pages if p.markdown is not None)
            if not text.strip():
                return PageResult(None, usage, raw, "empty response", duration)
            return PageResult(text, usage, raw, None, duration)
        except Exception as e:
            return PageResult(None, {}, None, repr(e), time.time() - t0)


def get_backend(name: str, model: str | None = None) -> TranscriptionBackend:
    if name == "gemini":
        return GeminiBackend(**({"model": model} if model else {}))
    if name == "mistral_ocr":
        return MistralOCRBackend(**({"model": model} if model else {}))
    raise ValueError(f"unknown backend {name!r}")
