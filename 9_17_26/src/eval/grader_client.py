"""Provider-agnostic text grader clients.

GraderClient protocol: grade(system_prompt, user_message) -> RawResponse.
  - VertexGeminiClient  (Grader A: gemini-3.7-flash via Vertex AI)
  - AIStudioGemmaClient (Grader B: gemma-4-31b-it via Google AI Studio)

Never sends temperature/top_p/top_k (removed from Gemini 3.x; kept off both to
keep the parameter surface identical). Never sees images. Reuses RawResponse,
retry, and error-wrapping behaviour from src.llm_client.

Prompt content is byte-identical across graders; only DELIVERY differs by model
constraint: Gemini takes the rubric as system_instruction, Gemma (no system role)
receives the same bytes (rubric + user) as a single user turn.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Protocol

from src.config import CostConfig, EvalConfig, GraderConfig
from src.llm_client import (
    KeyPool,
    KeySlot,
    LLMError,
    RawResponse,
    TokenRateLimiter,
    _RetryableError,
    _run_with_retry,
    estimate_tokens,
    load_keys,
    resolve_key_envs,
)

logger = logging.getLogger("eval.grader")


class GraderClient(Protocol):
    name: str

    def grade(
        self, user_content: str, *, label: str | None = None
    ) -> RawResponse: ...


def _usage_tokens(resp) -> tuple[int, int, int]:
    """(input, output_incl_thoughts, thoughts). Thinking tokens are billed."""
    u = getattr(resp, "usage_metadata", None)
    inp = int(getattr(u, "prompt_token_count", 0) or 0)
    cand = int(getattr(u, "candidates_token_count", 0) or 0)
    thoughts = int(getattr(u, "thoughts_token_count", 0) or 0)
    return inp, cand + thoughts, thoughts


class _GoogleGraderBase:
    """Shared grade() over google-genai; subclasses set client + delivery."""

    name: str
    provider: str
    is_gemma: bool

    def __init__(self, cfg: EvalConfig, grader_cfg: GraderConfig, cost: CostConfig,
                 keys: list[tuple[str, str]]) -> None:
        self.cfg = cfg
        self.model_id = grader_cfg.model_id
        self.cost = cost

        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise LLMError(None, "google-genai not installed") from exc
        self._genai = genai

        # One slot per key, each with its OWN quota window: the per-minute
        # budget is per project, so sharing a limiter across keys would throttle
        # the pool to a single key's allowance. Keys are never logged; only
        # their env-var names are.
        self._pool = KeyPool([
            KeySlot(
                env_name,
                self._make_client(key),
                TokenRateLimiter(
                    cfg.tokens_per_minute,
                    requests_per_minute=cfg.requests_per_minute,
                ),
            )
            for env_name, key in keys
        ])

    def _make_config(self):
        from google.genai import types

        # UNIFIED delivery: neither grader uses a system role; the rubric rides in
        # the single user turn. thinking_config is a model param, not delivery, so
        # Gemini still gets it; Gemma has no thinking config.
        kwargs: dict[str, object] = {"max_output_tokens": self.cfg.max_output_tokens}
        if not self.is_gemma:
            try:
                kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_level=self.cfg.thinking_level
                )
            except Exception:  # noqa: BLE001 - older SDKs
                pass
        return types.GenerateContentConfig(**kwargs)

    def grade(
        self, user_content: str, *, label: str | None = None
    ) -> RawResponse:
        config = self._make_config()
        contents = [user_content]

        estimate = estimate_tokens(user_content)
        charged: list[tuple[KeySlot, object]] = []

        def _call():
            # Pace every attempt: a retry costs the same input tokens. The pool
            # picks whichever key has budget, so a retry often lands on a
            # different key than the attempt that failed.
            slot, charge = self._pool.acquire(estimate)
            charged.append((slot, charge))
            try:
                return slot.client.models.generate_content(
                    model=self.model_id, contents=contents, config=config
                )
            except Exception as exc:  # noqa: BLE001
                status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                if status == 429 or (isinstance(status, int) and status >= 500):
                    raise _RetryableError(status, str(exc)) from exc
                raise

        start = time.monotonic()
        try:
            resp, attempt = _run_with_retry(_call, self.cfg.max_retries)
        except _RetryableError as exc:
            raise LLMError(label, f"exhausted retries (last status {exc.status}): {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMError(label, f"{self.provider} error: {exc}") from exc
        latency_ms = int((time.monotonic() - start) * 1000)

        text = getattr(resp, "text", None)
        if not text:
            raise LLMError(label, "empty completion content")

        inp, out, thoughts = _usage_tokens(resp)
        used_slot = charged[-1][0] if charged else None
        if used_slot is not None:
            used_slot.limiter.reconcile(charged[-1][1], inp)
        cost = self.cost.cost_usd(inp, out)
        logger.info(
            "grade ok label=%s grader=%s model=%s key=%s in=%d out=%d (think=%d) "
            "latency=%dms cost=$%.5f attempts=%d",
            label, self.name, self.model_id,
            used_slot.env_name if used_slot else "-",
            inp, out, thoughts, latency_ms, cost, attempt,
        )
        return RawResponse(
            text=text, model_id=self.model_id, provider=self.provider,
            thinking_level=self.cfg.thinking_level, pages_sent=0,
            input_tokens=inp, output_tokens=out, latency_ms=latency_ms,
            http_status=200, attempt_number=attempt, cost_usd=cost,
            settings={"thinking_level": self.cfg.thinking_level,
                      "thoughts_tokens": thoughts,
                      "max_output_tokens": self.cfg.max_output_tokens},
        )


class VertexGeminiClient(_GoogleGraderBase):
    provider = "vertex"
    is_gemma = False
    name = "gemini"

    def _make_client(self, api_key: str):
        return self._genai.Client(vertexai=True, api_key=api_key)


class AIStudioGemmaClient(_GoogleGraderBase):
    provider = "ai_studio"
    is_gemma = True
    name = "gemma"

    def _make_client(self, api_key: str):
        return self._genai.Client(api_key=api_key)


def build_grader(cfg: EvalConfig, name: str) -> GraderClient:
    """Construct the grader client for 'gemini' or 'gemma'.

    A grader may be given several keys (`api_key_envs`); each becomes a pool
    slot with its own quota window.
    """
    if name not in cfg.graders:
        raise LLMError(None, f"unknown grader '{name}' (have: {sorted(cfg.graders)})")
    gcfg = cfg.graders[name]
    env_names = resolve_key_envs(gcfg.api_key_env, gcfg.api_key_envs)
    keys = load_keys(env_names)
    cost = cfg.cost[name]
    cls = {"vertex": VertexGeminiClient, "ai_studio": AIStudioGemmaClient}.get(
        gcfg.provider
    )
    if cls is None:
        raise LLMError(None, f"unknown provider '{gcfg.provider}' for grader '{name}'")
    grader = cls(cfg, gcfg, cost, keys)
    logger.info(
        "grader %s: %d key(s) in pool (%s)",
        name, len(grader._pool), ", ".join(grader._pool.env_names),
    )
    return grader
