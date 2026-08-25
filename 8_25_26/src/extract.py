"""Stage 3 — extraction.

Sends all pages of one script in a single vision call so multi-page answers stay
coherent. The system prompt is the full contents of
prompts/extraction_prompt_v2.md, loaded verbatim from disk at runtime (never
inlined or paraphrased). Model output is fence-stripped, JSON-parsed (one retry
on invalid JSON with the parse error appended, then the script fails), and
validated against Pydantic models mirroring the prompt schema.

Nothing in this path corrects, normalises, or smooths the student text. The
`transcript` field is written byte-for-byte as the model returned it.

Caching: SHA256(all page bytes + prompt file contents + model id). Changing the
prompt changes the key and invalidates the cache. A completed transcript whose
stored cache_key still matches is skipped on re-run (resumable).

If a script exceeds `llm.max_pages_per_call` (when > 0), pages are chunked on
boundaries with `llm.chunk_overlap_pages` overlap and merged by question number
— logged loudly, since merged answers are a quality risk.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from pydantic import BaseModel

from src.config import Config
from src.llm_client import LLMError, RawResponse, VisionClient

logger = logging.getLogger("extract")

PROMPT_VERSION_FALLBACK = "extraction_prompt_v2"


# ---------------------------------------------------------------------------
# Pydantic models — mirror prompts/extraction_prompt_v2.md OUTPUT FORMAT exactly.
# No validators touch string content: transcripts must stay verbatim.
# ---------------------------------------------------------------------------
class Answer(BaseModel):
    question_number: str
    transcript: str
    pages: list[int] = []
    confidence: float
    illegible_count: int = 0
    unclear_count: int = 0
    ambiguous_authorship: list[str] = []
    student_title: str | None = None
    notes: str | None = None


class Transcript(BaseModel):
    script_id: str | None = None
    pages_processed: int = 0
    answers: list[Answer] = []
    questions_not_found: list[str] = []
    overall_confidence: float = 0.0
    extraction_warnings: list = []


class ExtractionError(RuntimeError):
    """A script failed to extract. Carries the script id (fail loudly)."""

    def __init__(self, script_id: str, message: str) -> None:
        self.script_id = script_id
        super().__init__(f"[{script_id}] {message}")


@dataclass
class CallRecord:
    """Per-API-call accounting; summed into transcript metadata."""

    chunk_index: int
    pages_sent: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    http_status: int
    attempt_number: int


@dataclass
class ExtractResult:
    script_id: str
    status: str  # "extracted" | "cached" | "skipped"
    answers_found: int = 0
    questions_not_found: list[str] = field(default_factory=list)
    total_cost_usd: float = 0.0
    wall_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_prompt(cfg: Config) -> tuple[str, str, str]:
    """Return (prompt_text, prompt_sha256, prompt_version). Loaded from disk."""
    path = cfg.paths.prompt_file
    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return text, digest, path.stem or PROMPT_VERSION_FALLBACK


def _page_files(cfg: Config, script_id: str) -> list[Path]:
    img_dir = cfg.paths.images / script_id
    pages = sorted(img_dir.glob(f"page_*.{cfg.pdf.image_format}"))
    if not pages:
        raise ExtractionError(
            script_id, f"no page images found in {img_dir} (run Stage 1 first)"
        )
    return pages


def cache_key(page_bytes: list[bytes], prompt_text: str, model_id: str) -> str:
    """SHA256 over page bytes (in order) + prompt file contents + model id."""
    h = hashlib.sha256()
    for b in page_bytes:
        h.update(b)
    h.update(prompt_text.encode("utf-8"))
    h.update(model_id.encode("utf-8"))
    return h.hexdigest()


def strip_code_fences(text: str) -> str:
    """Remove an outer ```json ... ``` wrapper if present.

    Operates only on the whole-response wrapper, never on parsed field content:
    the transcript strings inside the JSON are untouched by this and by
    json.loads, preserving verbatim fidelity.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    lines = lines[1:]  # drop opening ``` / ```json
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def _record(resp: RawResponse, chunk_index: int) -> CallRecord:
    return CallRecord(
        chunk_index=chunk_index,
        pages_sent=resp.pages_sent,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        cost_usd=resp.cost_usd,
        latency_ms=resp.latency_ms,
        http_status=resp.http_status,
        attempt_number=resp.attempt_number,
    )


def _call_and_parse(
    client: VisionClient,
    images: list[bytes],
    prompt_text: str,
    *,
    script_id: str,
    chunk_index: int,
    calls: list[CallRecord],
) -> dict:
    """One vision call with a single JSON-parse retry. Appends call records.

    On the retry, the parse error is appended to the prompt (a transient nudge,
    not a change to the on-disk prompt). Fails the script if both attempts yield
    invalid JSON.
    """
    resp = client.transcribe(images, prompt_text, label=script_id)
    calls.append(_record(resp, chunk_index))
    try:
        return json.loads(strip_code_fences(resp.text))
    except json.JSONDecodeError as first_err:
        logger.warning(
            "[%s] chunk %d invalid JSON, retrying once: %s",
            script_id, chunk_index, first_err,
        )
        retry_prompt = (
            f"{prompt_text}\n\n"
            f"Your previous response was not valid JSON ({first_err}). "
            f"Return ONLY the JSON object, no fence, no commentary."
        )
        resp2 = client.transcribe(images, retry_prompt, label=script_id)
        calls.append(_record(resp2, chunk_index))
        try:
            return json.loads(strip_code_fences(resp2.text))
        except json.JSONDecodeError as second_err:
            raise ExtractionError(
                script_id, f"invalid JSON after retry: {second_err}"
            ) from second_err


def _chunk_ranges(n_pages: int, max_per_call: int, overlap: int) -> list[range]:
    """Page-index ranges with overlap. Only used when max_per_call > 0."""
    ranges: list[range] = []
    start = 0
    while start < n_pages:
        end = min(start + max_per_call, n_pages)
        ranges.append(range(start, end))
        if end >= n_pages:
            break
        start = end - overlap
    return ranges


def _merge_chunks(script_id: str, parsed_chunks: list[dict]) -> dict:
    """Merge per-chunk parsed outputs by question number.

    Answers are joined in first-seen order; overlapping pages mean a question may
    appear in two chunks — the longer transcript wins (a truncated tail from a
    boundary chunk should not clobber the full answer). This is a quality risk,
    hence the loud warning at the call site.
    """
    merged_answers: dict[str, dict] = {}
    not_found: set[str] = set()
    warnings: list = []
    pages_processed = 0
    confidences: list[float] = []

    for chunk in parsed_chunks:
        pages_processed = max(pages_processed, int(chunk.get("pages_processed", 0)))
        warnings.extend(chunk.get("extraction_warnings", []) or [])
        if chunk.get("overall_confidence") is not None:
            confidences.append(float(chunk["overall_confidence"]))
        for ans in chunk.get("answers", []) or []:
            qno = str(ans.get("question_number"))
            existing = merged_answers.get(qno)
            if existing is None or len(ans.get("transcript", "")) > len(
                existing.get("transcript", "")
            ):
                merged_answers[qno] = ans
        for qno in chunk.get("questions_not_found", []) or []:
            not_found.add(str(qno))

    # A question found in any chunk is not "not found".
    not_found -= set(merged_answers)
    warnings.append(f"MERGED FROM {len(parsed_chunks)} CHUNKS — verify boundary answers")

    return {
        "script_id": script_id,
        "pages_processed": pages_processed,
        "answers": list(merged_answers.values()),
        "questions_not_found": sorted(not_found),
        "overall_confidence": min(confidences) if confidences else 0.0,
        "extraction_warnings": warnings,
    }


def _obtain_parsed(
    client: VisionClient,
    cfg: Config,
    script_id: str,
    page_bytes: list[bytes],
    prompt_text: str,
    calls: list[CallRecord],
) -> tuple[dict, bool]:
    """Single call, or chunk+merge when configured. Returns (parsed, chunked)."""
    max_per_call = cfg.llm.max_pages_per_call
    n = len(page_bytes)

    if max_per_call and n > max_per_call:
        ranges = _chunk_ranges(n, max_per_call, cfg.llm.chunk_overlap_pages)
        logger.warning(
            "[%s] %d pages exceeds max_pages_per_call=%d — CHUNKING into %d "
            "calls (overlap=%d). Merged answers are a QUALITY RISK.",
            script_id, n, max_per_call, len(ranges), cfg.llm.chunk_overlap_pages,
        )
        parsed_chunks = []
        for idx, rng in enumerate(ranges):
            chunk_imgs = [page_bytes[i] for i in rng]
            parsed_chunks.append(
                _call_and_parse(
                    client, chunk_imgs, prompt_text,
                    script_id=script_id, chunk_index=idx, calls=calls,
                )
            )
        return _merge_chunks(script_id, parsed_chunks), True

    parsed = _call_and_parse(
        client, page_bytes, prompt_text,
        script_id=script_id, chunk_index=0, calls=calls,
    )
    return parsed, False


def _transcript_path(cfg: Config, script_id: str) -> Path:
    return cfg.paths.transcripts / f"{script_id}.json"


def _existing_cache_key(path: Path) -> str | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("_metadata", {}).get("cache_key")
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def extract_script(
    script_id: str,
    cfg: Config,
    client: VisionClient,
    *,
    force: bool = False,
) -> ExtractResult:
    """Extract one script to data/transcripts/{script_id}.json. Fails loudly."""
    started = monotonic()
    prompt_text, prompt_hash, prompt_version = load_prompt(cfg)
    out_path = _transcript_path(cfg, script_id)

    page_files = _page_files(cfg, script_id)
    page_bytes = [p.read_bytes() for p in page_files]
    key = cache_key(page_bytes, prompt_text, cfg.model_id)

    # Resume: a completed transcript whose stored cache_key matches is up to date.
    if not force and out_path.exists() and _existing_cache_key(out_path) == key:
        data = json.loads(out_path.read_text(encoding="utf-8"))
        return ExtractResult(
            script_id=script_id,
            status="skipped",
            answers_found=len(data.get("answers", [])),
            questions_not_found=data.get("questions_not_found", []),
            total_cost_usd=data.get("_metadata", {}).get("total_cost_usd", 0.0),
            wall_time_s=0.0,
        )

    cache_file = cfg.paths.cache / f"{key}.json"
    calls: list[CallRecord] = []

    from_cache = False
    if not force and cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        parsed = cached["parsed"]
        chunked = cached.get("chunked", False)
        calls = [CallRecord(**c) for c in cached.get("calls", [])]
        from_cache = True
        logger.info("[%s] cache hit %s", script_id, key[:12])
    else:
        try:
            parsed, chunked = _obtain_parsed(
                client, cfg, script_id, page_bytes, prompt_text, calls
            )
        except LLMError as exc:
            raise ExtractionError(script_id, str(exc)) from exc
        cache_file.write_text(
            json.dumps(
                {
                    "parsed": parsed,
                    "chunked": chunked,
                    "calls": [c.__dict__ for c in calls],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # Validate against the prompt schema. Does not mutate transcript strings.
    try:
        transcript = Transcript.model_validate(parsed)
    except Exception as exc:  # noqa: BLE001 - pydantic ValidationError et al.
        raise ExtractionError(script_id, f"schema validation failed: {exc}") from exc

    total_in = sum(c.input_tokens for c in calls)
    total_out = sum(c.output_tokens for c in calls)
    total_cost = sum(c.cost_usd for c in calls)
    wall = monotonic() - started

    metadata = {
        "script_id": script_id,
        "provider": cfg.provider,
        "model_id": cfg.model_id,
        "thinking_level": cfg.thinking_level,
        "prompt_version": prompt_version,
        "prompt_file": str(cfg.paths.prompt_file.name),
        "prompt_hash": prompt_hash,
        "cache_key": key,
        "from_cache": from_cache,
        "chunked": chunked,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wall_time_s": round(wall, 3),
        "pages_available": len(page_files),
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_cost_usd": round(total_cost, 6),
        "settings": {
            "thinking_level": cfg.thinking_level,
            "max_output_tokens": cfg.llm.max_output_tokens,
        },
        "calls": [c.__dict__ for c in calls],
    }

    # model_dump preserves the verbatim strings exactly; ensure_ascii=False keeps
    # non-ASCII (e.g. Bangla) intact without escaping.
    output = transcript.model_dump()
    output["_metadata"] = metadata
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return ExtractResult(
        script_id=script_id,
        status="cached" if from_cache else "extracted",
        answers_found=len(transcript.answers),
        questions_not_found=transcript.questions_not_found,
        total_cost_usd=total_cost,
        wall_time_s=wall,
    )
