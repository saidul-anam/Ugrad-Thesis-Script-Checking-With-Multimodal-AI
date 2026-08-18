"""
Google Gemini Vision-Language Client using google-genai or google-generativeai SDK.
Enables real end-to-end multimodal OCR, rubric alignment, examiner grading, and auditing.
"""

from typing import Any, Dict, List, Optional
import os
import io
import json
from PIL import Image

try:
    from .base import ModelClient
    from ..schemas import ModelConfig
except (ImportError, ValueError):
    from model_client.base import ModelClient
    from schemas import ModelConfig


class GeminiClient(ModelClient):
    """Client implementing multimodal inference using Google Gemini API."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig(backend="gemini", checkpoint="gemini-2.5-flash")
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model_name = self.config.checkpoint or "gemini-2.5-flash"
        
        # Clean up model name if prefixed with google/
        if self.model_name.startswith("google/"):
            self.model_name = self.model_name.replace("google/", "")

        self._client = None
        self._init_client()

    def _init_client(self):
        if not self.api_key:
            return

        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            self._use_new_genai = True
        except ImportError:
            try:
                import google.generativeai as genai_old
                genai_old.configure(api_key=self.api_key)
                self._client = genai_old.GenerativeModel(self.model_name)
                self._use_new_genai = False
            except Exception:
                self._client = None

    def generate(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        if not self.api_key or not self._client:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is not set. "
                "Please set `set GEMINI_API_KEY=your_key` or export GEMINI_API_KEY=your_key."
            )

        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        try:
            if self._use_new_genai:
                from google.genai import types
                contents = []
                if images:
                    for img in images:
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=85)
                        image_bytes = buf.getvalue()
                        contents.append(
                            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                        )
                contents.append(prompt)

                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=temp,
                        max_output_tokens=max_tok
                    )
                )
                return response.text or ""
            else:
                # Legacy google.generativeai
                inputs = []
                if images:
                    inputs.extend(images)
                inputs.append(prompt)

                generation_config = {
                    "temperature": temp,
                    "max_output_tokens": max_tok
                }
                response = self._client.generate_content(
                    inputs,
                    generation_config=generation_config
                )
                return response.text or ""
        except Exception as e:
            raise RuntimeError(f"Gemini API generation failed: {e}") from e
