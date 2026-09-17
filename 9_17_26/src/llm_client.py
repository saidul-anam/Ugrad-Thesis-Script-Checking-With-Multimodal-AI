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
import re
import threading
import time
from collections import deque
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


# ---------------------------------------------------------------------------
# Client-side rate limiting
#
# The AI Studio free tier caps INPUT TOKENS PER MINUTE per model (16,000 for
# gemma-4-31b at the time of writing), and a grading prompt is ~6.3k tokens
# because the whole rubric rides in every call. Three workers firing together
# blow the budget instantly, and the resulting 429s carry a ~50s retryDelay that
# exponential backoff exhausts long before it clears — which is how a run ends
# up 12/13 FAILED rather than merely slow.
#
# So we pace ourselves instead of being paced by rejections: a shared sliding
# 60-second window over charged input tokens, blocking a worker until its call
# fits. Set `tokens_per_minute: 0` to disable (e.g. on a paid tier).
# ---------------------------------------------------------------------------
class Charge:
    """One entry in the limiter's window. Mutable so it can be refunded."""

    __slots__ = ("at", "tokens")

    def __init__(self, at: float, tokens: int) -> None:
        self.at = at
        self.tokens = tokens


class TokenRateLimiter:
    """Thread-safe sliding-window limiter over input tokens per minute."""

    WINDOW_S = 60.0

    def __init__(
        self,
        tokens_per_minute: int = 0,
        headroom: float = 0.9,
        requests_per_minute: int = 0,
    ) -> None:
        # Headroom absorbs the gap between our estimate and the server's count,
        # and the fact that its window and ours are not aligned.
        self.limit = int(tokens_per_minute * headroom) if tokens_per_minute else 0
        # Backstop on request COUNT. Because a 5xx is refunded, failures cost no
        # token budget — so against a fully broken endpoint the token window
        # alone would let retries spin without limit. This bounds the request
        # rate whatever the failure rate does.
        self.request_limit = (
            int(requests_per_minute * headroom) if requests_per_minute else 0
        )
        self._lock = threading.Lock()
        self._charges: deque[Charge] = deque()
        self._requests: deque[float] = deque()

    def _prune(self, now: float) -> None:
        while self._charges and now - self._charges[0].at >= self.WINDOW_S:
            self._charges.popleft()
        while self._requests and now - self._requests[0] >= self.WINDOW_S:
            self._requests.popleft()

    def _try_locked(self, tokens: int) -> tuple[Charge | None, float]:
        """Caller holds the lock. Returns (charge, seconds_until_room)."""
        now = time.monotonic()
        self._prune(now)
        waits = []
        if self.limit > 0 and sum(c.tokens for c in self._charges) + tokens > self.limit:
            waits.append(self.WINDOW_S - (now - self._charges[0].at))
        if self.request_limit > 0 and len(self._requests) + 1 > self.request_limit:
            waits.append(self.WINDOW_S - (now - self._requests[0]))
        if waits:
            return None, min(waits) + 0.05
        charge = Charge(now, tokens)
        self._charges.append(charge)
        self._requests.append(now)
        return charge, 0.0

    def _clamp(self, tokens: int) -> int:
        return max(1, min(int(tokens), self.limit)) if self.limit > 0 else 0

    def try_acquire(self, tokens: int) -> Charge | None:
        """Take budget only if it is free right now. Never blocks.

        The pool uses this to skip a saturated key instead of queueing behind
        it while another key sits idle.
        """
        if self.limit <= 0 and self.request_limit <= 0:
            return None
        with self._lock:
            return self._try_locked(self._clamp(tokens))[0]

    def acquire(self, tokens: int) -> Charge | None:
        """Block until `tokens` fit in the window. Returns the charge made."""
        if self.limit <= 0 and self.request_limit <= 0:
            return None
        tokens = self._clamp(tokens)
        while True:
            with self._lock:
                charge, wait = self._try_locked(tokens)
                if charge is not None:
                    return charge
            time.sleep(max(wait, 0.05))

    def refund(self, charge: Charge | None) -> None:
        """Give back budget for a request that did not consume the provider's.

        DELIBERATELY UNUSED against AI Studio — kept because the mechanism is
        sound and a different provider may well need it.

        The tempting theory is that a 5xx never consumed input budget, so its
        reservation should be returned. On the Class VII grading pass 51% of
        attempts came back 500 while the limiter sat at 88% "used", which made
        refunding look like a free doubling of throughput. It is not: measured
        2026-09-17, switching it on took 429s from 2-in-4-hours to 14-in-11-
        minutes and throughput DOWN from 1.0 to ~0.75 rows/min. Google counts
        the input tokens of a request that later fails server-side, so refunding
        simply overspends the quota and trades cheap 500s for 429s carrying a
        ~50s penalty. Treat every attempt as billed.
        """
        if self.limit <= 0 or charge is None:
            return
        with self._lock:
            charge.tokens = 0

    def reconcile(self, charge: Charge | None, actual: int) -> None:
        """Top up the window when the server counted more than we estimated."""
        if self.limit <= 0 or charge is None:
            return
        delta = int(actual) - int(charge.tokens)
        if delta <= 0:
            return
        with self._lock:
            self._charges.append(Charge(time.monotonic(), delta))


class KeySlot:
    """One API key: its client, its own quota window, and its env-var name."""

    __slots__ = ("env_name", "client", "limiter")

    def __init__(self, env_name: str, client: object, limiter: TokenRateLimiter) -> None:
        self.env_name = env_name
        self.client = client
        self.limiter = limiter


class KeyPool:
    """Several API keys worked in rotation, each metered independently.

    The per-minute token budget is per PROJECT, so keys issued from separate
    projects have separate budgets and multiply throughput; keys from the same
    project share one and buy nothing. Each slot therefore carries its own
    limiter — never a shared one, which would throttle the pool to a single
    key's quota.

    `acquire` first offers the request to each key that has room right now, in
    rotation, and only blocks if every key is saturated. Blocking on a busy key
    while another sits idle is exactly the waste the pool exists to avoid.
    """

    def __init__(self, slots: Sequence[KeySlot]) -> None:
        if not slots:
            raise LLMError(None, "key pool is empty")
        self._slots = list(slots)
        self._next = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._slots)

    @property
    def env_names(self) -> list[str]:
        return [s.env_name for s in self._slots]

    def _rotate(self) -> int:
        with self._lock:
            idx = self._next
            self._next = (self._next + 1) % len(self._slots)
            return idx

    def acquire(self, tokens: int) -> tuple[KeySlot, Charge | None]:
        """Reserve budget on whichever key can take the call soonest."""
        n = len(self._slots)
        start = self._rotate()
        for offset in range(n):
            slot = self._slots[(start + offset) % n]
            charge = slot.limiter.try_acquire(tokens)
            if charge is not None or slot.limiter.limit <= 0:
                return slot, charge
        # Every key is saturated; queue on one of them rather than spinning.
        slot = self._slots[start]
        return slot, slot.limiter.acquire(tokens)


def resolve_key_envs(primary: str | None, extras: Sequence[str] | None) -> list[str]:
    """The env-var names holding this provider's keys, in order, deduplicated."""
    names: list[str] = []
    for name in [primary, *(extras or [])]:
        if name and name not in names:
            names.append(name)
    return names


def load_keys(env_names: Sequence[str], *, required: bool = True) -> list[tuple[str, str]]:
    """[(env_name, key)] for every name that is actually set.

    A name listed in config but absent from the environment is skipped with a
    warning rather than failing the run — losing one key of four should slow the
    job down, not stop it. Only an empty result is fatal.
    """
    found: list[tuple[str, str]] = []
    for name in env_names:
        value = os.environ.get(name)
        if value:
            found.append((name, value))
        else:
            logger.warning("api key env var %s is not set - skipping that key", name)
    if not found and required:
        raise LLMError(None, f"none of these env vars are set: {list(env_names)}")
    return found


def estimate_tokens(text: str = "", images: Sequence[bytes] = (),
                    tokens_per_image: int = 560) -> int:
    """Rough input-token estimate used only to pace the limiter.

    ~4 characters per token for text; a flat per-image figure for page scans
    (measured at ~500 tokens for an A4 page at 300 DPI, rounded up). Accuracy
    only affects pacing — the true count is reconciled after every call.
    """
    return len(text) // 4 + len(images) * tokens_per_image


_RETRY_DELAY_RE = re.compile(r"retryDelay'?\s*:?\s*'?(\d+)s")


def _server_retry_delay(message: str) -> float | None:
    """The retryDelay the 429 body asked for, in seconds, if it carried one."""
    m = _RETRY_DELAY_RE.search(message)
    return float(m.group(1)) if m else None


def _encode_data_uri(image: bytes, image_format: str = "png") -> str:
    b64 = base64.b64encode(image).decode("ascii")
    return f"data:image/{image_format};base64,{b64}"


def _wait_honouring_server(retry_state) -> float:
    """Back off exponentially, but never for less than the server asked.

    A 429 from AI Studio names its own retryDelay (~50s when a per-minute token
    budget is exhausted). Sleeping the usual 2s and trying again just burns an
    attempt, so the server's figure wins whenever it is larger.
    """
    base = wait_exponential(multiplier=2, min=2, max=90)(retry_state)
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is not None:
        asked = _server_retry_delay(str(exc))
        if asked:
            return max(base, asked + 1.0)
    return base


def _run_with_retry(fn, max_attempts: int):
    """Run `fn` under tenacity, retrying only `_RetryableError`.

    Returns (result, attempt_number). `reraise=True` so the last real exception
    propagates unchanged rather than being wrapped in a RetryError.
    """
    retryer = Retrying(
        retry=retry_if_exception_type(_RetryableError),
        stop=stop_after_attempt(max_attempts),
        wait=_wait_honouring_server,
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
        keys: Sequence[tuple[str, str]] = (),
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

        def build(api_key: str | None):
            if use_vertex and not api_key:
                # ADC path: project + location, no key.
                return genai.Client(
                    vertexai=True,
                    project=self.provider_cfg.project,
                    location=self.provider_cfg.location,
                )
            if use_vertex:
                return genai.Client(vertexai=True, api_key=api_key)
            return genai.Client(api_key=api_key)  # AI Studio / Developer API

        def limiter() -> TokenRateLimiter:
            return TokenRateLimiter(
                cfg.llm.tokens_per_minute,
                requests_per_minute=cfg.llm.requests_per_minute,
            )

        # One slot per key, each metered on its own quota window. With no keys
        # at all (the Vertex ADC path) there is still exactly one slot.
        slots = [KeySlot(name, build(key), limiter()) for name, key in keys]
        if not slots:
            slots = [KeySlot("(adc)", build(None), limiter())]
        self._pool = KeyPool(slots)

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

        estimate = estimate_tokens(prompt + lead, images)
        charged: list[tuple[KeySlot, Charge | None]] = []

        def _call():
            # Pace every attempt, not just the first: a retry costs the provider
            # the same input tokens as the call that provoked it. The pool picks
            # whichever key has budget free.
            slot, charge = self._pool.acquire(estimate)
            charged.append((slot, charge))
            try:
                return slot.client.models.generate_content(
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
        used_slot = charged[-1][0] if charged else None
        if used_slot is not None:
            used_slot.limiter.reconcile(charged[-1][1], input_tokens)
        cost = self.cfg.cost.cost_usd(input_tokens, output_tokens)
        settings = {
            "thinking_level": self.cfg.thinking_level,
            "max_output_tokens": self.cfg.llm.max_output_tokens,
        }
        logger.info(
            "call ok label=%s provider=%s model=%s key=%s pages=%d in=%d out=%d "
            "latency=%dms cost=$%.5f attempts=%d",
            label, self.provider_label, self.model_id,
            used_slot.env_name if used_slot else "-", len(images),
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
    """Construct the client for the active provider in config.yaml.

    A provider may name several key env vars; each becomes a pool slot with its
    own quota window.
    """
    provider = cfg.provider
    pc = cfg.active_provider
    env_names = resolve_key_envs(pc.api_key_env, pc.api_key_envs)

    if provider == "openrouter":
        return OpenRouterClient(cfg, _require_key(cfg))
    if provider == "gemini":
        # AI Studio / Gemini Developer API. Serves Gemma.
        keys = load_keys(env_names)
        client = GoogleGenAIClient(
            cfg, provider_label="gemini", use_vertex=False, keys=keys
        )
    elif provider == "vertex":
        # Keys optional here: none -> fall back to ADC (project/location).
        keys = load_keys(env_names, required=False)
        client = GoogleGenAIClient(
            cfg, provider_label="vertex", use_vertex=True, keys=keys
        )
    else:
        raise LLMError(None, f"unknown provider '{provider}'")

    logger.info(
        "extraction: %d key(s) in pool (%s)",
        len(client._pool), ", ".join(client._pool.env_names),
    )
    return client
