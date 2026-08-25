"""Stage 2 — provider-agnostic vision client.

One `VisionClient` protocol: `transcribe(images, prompt) -> RawResponse`.
Concrete `OpenRouterClient` and `VertexClient`, selected via config.yaml. The
grading stage will later reach Gemini through this same interface.

Retries with exponential backoff on 429/5xx (tenacity, max attempts from
config). Never swallows an exception — wraps and re-raises as `LLMError` with the
call label (script id) attached. Every call returns token counts, latency, cost,
and http status so the caller can persist an `extraction_runs.csv` row.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import httpx
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import Config

logger = logging.getLogger("extract.llm")


class LLMError(RuntimeError):
    """Fatal error from an LLM call. Carries the call label (usually script id)."""

    def __init__(self, label: str | None, message: str) -> None:
        self.label = label
        prefix = f"[{label}] " if label else ""
        super().__init__(f"{prefix}{message}")


class _RetryableError(Exception):
    """Internal: a 429/5xx or transport error worth retrying."""

    def __init__(self, status: int | None, message: str) -> None:
        self.status = status
        super().__init__(message)


@dataclass
class RawResponse:
    """Everything one vision call produced. `text` is the model's raw output.

    The transcript-quality/JSON parsing happens in Stage 3; this layer stays
    provider-agnostic and only reports transport + accounting facts.
    """

    text: str
    model_id: str
    provider: str
    thinking_level: str
    pages_sent: int
    input_tokens: int
    output_tokens: int
    latency_ms: int
    http_status: int
    attempt_number: int
    cost_usd: float
    settings: dict[str, object] = field(default_factory=dict)


class VisionClient(Protocol):
    """Provider-agnostic contract used by extraction (and later, grading)."""

    def transcribe(
        self,
        images: Sequence[bytes],
        prompt: str,
        *,
        label: str | None = None,
    ) -> RawResponse: ...


def _encode_data_uri(image: bytes, image_format: str = "png") -> str:
    b64 = base64.b64encode(image).decode("ascii")
    return f"data:image/{image_format};base64,{b64}"


def _run_with_retry(fn, max_attempts: int):
    """Run `fn` under tenacity, retrying only `_RetryableError`.

    Returns (result, attempt_number). `reraise=True` so the last real exception
    propagates unchanged rather than being wrapped in a RetryError.
    """
    retryer = Retrying(
        retry=retry_if_exception_type(_RetryableError),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    result = retryer(fn)
    attempt = retryer.statistics.get("attempt_number", 1)
    return result, attempt


# ---------------------------------------------------------------------------
# OpenRouter (OpenAI-compatible chat completions)
# ---------------------------------------------------------------------------
class OpenRouterClient:
    def __init__(self, cfg: Config, api_key: str) -> None:
        self.cfg = cfg
        self.provider_cfg = cfg.active_provider
        self.api_key = api_key
        self.model_id = self.provider_cfg.model_id
        self.image_format = cfg.pdf.image_format

    def transcribe(
        self,
        images: Sequence[bytes],
        prompt: str,
        *,
        label: str | None = None,
    ) -> RawResponse:
        content: list[dict[str, object]] = [
            {"type": "text", "text": "Pages of one exam script, in scan order:"}
        ]
        for img in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _encode_data_uri(img, self.image_format)},
                }
            )

        payload: dict[str, object] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ],
            "max_tokens": self.cfg.llm.max_output_tokens,
        }
        # thinking_level is passed through to providers that support it; harmless
        # (ignored) for those that don't.
        if self.cfg.thinking_level:
            payload["reasoning"] = {"effort": self.cfg.thinking_level}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.provider_cfg.api_base}/chat/completions"

        def _call() -> httpx.Response:
            try:
                resp = httpx.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.cfg.llm.request_timeout_s,
                )
            except httpx.HTTPError as exc:
                raise _RetryableError(None, f"transport error: {exc}") from exc
            if resp.status_code == 429 or resp.status_code >= 500:
                raise _RetryableError(
                    resp.status_code, f"HTTP {resp.status_code}: {resp.text[:500]}"
                )
            return resp

        start = time.monotonic()
        try:
            resp, attempt = _run_with_retry(_call, self.cfg.llm.max_retries)
        except _RetryableError as exc:
            raise LLMError(
                label, f"exhausted retries (last HTTP {exc.status}): {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMError(label, f"unexpected error: {exc}") from exc
        latency_ms = int((time.monotonic() - start) * 1000)

        if resp.status_code >= 400:
            raise LLMError(
                label, f"HTTP {resp.status_code} (non-retryable): {resp.text[:500]}"
            )

        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            input_tokens = int(usage.get("prompt_tokens", 0))
            output_tokens = int(usage.get("completion_tokens", 0))
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(label, f"malformed response body: {exc}") from exc

        if not text:
            raise LLMError(label, "empty completion content")

        cost = self.cfg.cost.cost_usd(input_tokens, output_tokens)
        settings = {
            "thinking_level": self.cfg.thinking_level,
            "max_output_tokens": self.cfg.llm.max_output_tokens,
        }
        logger.info(
            "call ok label=%s provider=%s model=%s pages=%d in=%d out=%d "
            "latency=%dms cost=$%.5f attempts=%d",
            label, "openrouter", self.model_id, len(images),
            input_tokens, output_tokens, latency_ms, cost, attempt,
        )
        return RawResponse(
            text=text,
            model_id=self.model_id,
            provider="openrouter",
            thinking_level=self.cfg.thinking_level,
            pages_sent=len(images),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            http_status=resp.status_code,
            attempt_number=attempt,
            cost_usd=cost,
            settings=settings,
        )


# ---------------------------------------------------------------------------
# google-genai client, covering two Google surfaces:
#   - "gemini": AI Studio / Gemini Developer API (api_key). Serves Gemma.
#   - "vertex": Vertex AI (express api_key, or ADC via project/location).
# The grading stage reuses this same client for Gemini judges.
# ---------------------------------------------------------------------------
class GoogleGenAIClient:
    def __init__(
        self,
        cfg: Config,
        *,
        provider_label: str,
        use_vertex: bool,
        api_key: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.provider_cfg = cfg.active_provider
        self.provider_label = provider_label
        self.model_id = self.provider_cfg.model_id
        self.image_format = cfg.pdf.image_format
        # Gemma on the Gemini API does NOT support a system role or thinking
        # config; the prompt must ride as the leading user text part instead.
        self.is_gemma = "gemma" in self.model_id.lower()

        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise LLMError(None, "google-genai not installed") from exc

        self._genai = genai
        if use_vertex and not api_key:
            # ADC path: project + location, no key.
            self._client = genai.Client(
                vertexai=True,
                project=self.provider_cfg.project,
                location=self.provider_cfg.location,
            )
        elif use_vertex:
            self._client = genai.Client(vertexai=True, api_key=api_key)
        else:
            # AI Studio / Gemini Developer API.
            self._client = genai.Client(api_key=api_key)

    def transcribe(
        self,
        images: Sequence[bytes],
        prompt: str,
        *,
        label: str | None = None,
    ) -> RawResponse:
        from google.genai import types

        lead = "Pages of one exam script, in scan order:"
        parts: list[object] = []
        config_kwargs: dict[str, object] = {
            "max_output_tokens": self.cfg.llm.max_output_tokens,
        }

        if self.is_gemma:
            # Prompt delivered verbatim as the first user text part.
            parts.append(types.Part.from_text(text=f"{prompt}\n\n{lead}"))
        else:
            config_kwargs["system_instruction"] = prompt
            parts.append(types.Part.from_text(text=lead))
            if self.cfg.thinking_level:
                try:
                    config_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_level=self.cfg.thinking_level
                    )
                except Exception:  # noqa: BLE001 - older SDKs lack thinking_level
                    pass

        for img in images:
            parts.append(
                types.Part.from_bytes(data=img, mime_type=f"image/{self.image_format}")
            )

        def _call():
            try:
                return self._client.models.generate_content(
                    model=self.model_id,
                    contents=parts,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
            except Exception as exc:  # noqa: BLE001
                status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                if status == 429 or (isinstance(status, int) and status >= 500):
                    raise _RetryableError(status, str(exc)) from exc
                raise

        start = time.monotonic()
        try:
            resp, attempt = _run_with_retry(_call, self.cfg.llm.max_retries)
        except _RetryableError as exc:
            raise LLMError(
                label, f"exhausted retries (last status {exc.status}): {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMError(label, f"{self.provider_label} error: {exc}") from exc
        latency_ms = int((time.monotonic() - start) * 1000)

        text = getattr(resp, "text", None)
        if not text:
            raise LLMError(label, "empty completion content")

        usage = getattr(resp, "usage_metadata", None)
        input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        cost = self.cfg.cost.cost_usd(input_tokens, output_tokens)
        settings = {
            "thinking_level": self.cfg.thinking_level,
            "max_output_tokens": self.cfg.llm.max_output_tokens,
        }
        logger.info(
            "call ok label=%s provider=%s model=%s pages=%d in=%d out=%d "
            "latency=%dms cost=$%.5f attempts=%d",
            label, self.provider_label, self.model_id, len(images),
            input_tokens, output_tokens, latency_ms, cost, attempt,
        )
        return RawResponse(
            text=text,
            model_id=self.model_id,
            provider=self.provider_label,
            thinking_level=self.cfg.thinking_level,
            pages_sent=len(images),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            http_status=200,
            attempt_number=attempt,
            cost_usd=cost,
            settings=settings,
        )


def _require_key(cfg: Config) -> str:
    env = cfg.active_provider.api_key_env
    api_key = os.environ.get(env) if env else None
    if not api_key:
        raise LLMError(None, f"env var {env} is not set")
    return api_key


def build_client(cfg: Config) -> VisionClient:
    """Construct the client for the active provider in config.yaml."""
    provider = cfg.provider
    if provider == "openrouter":
        return OpenRouterClient(cfg, _require_key(cfg))
    if provider == "gemini":
        # AI Studio / Gemini Developer API. Serves Gemma.
        return GoogleGenAIClient(
            cfg, provider_label="gemini", use_vertex=False, api_key=_require_key(cfg)
        )
    if provider == "vertex":
        env = cfg.active_provider.api_key_env
        api_key = os.environ.get(env) if env else None
        # api_key optional here: absent -> fall back to ADC (project/location).
        return GoogleGenAIClient(
            cfg, provider_label="vertex", use_vertex=True, api_key=api_key
        )
    raise LLMError(None, f"unknown provider '{provider}'")
