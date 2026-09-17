"""Configuration loading.

Model configuration is DATA, not code: everything the LLM was run with lives in
config.yaml and is stamped into every output record via `Config.stamp()`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel

# A script id is its exam set plus a four-digit serial: SE_07_Q1_0003 lives in
# exam set SE_07_Q1. One project holds several exam sets (different classes,
# different question papers), and almost everything downstream — which questions
# exist, which prompt transcribes them, which rubric grades them — is a property
# of the SET, not of the project.
_SERIAL_SUFFIX = re.compile(r"_\d+$")


def exam_set_of(script_id: str) -> str:
    """'SE_07_Q1_0003' -> 'SE_07_Q1'. The trailing serial is stripped."""
    return _SERIAL_SUFFIX.sub("", script_id)


def load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines into os.environ (no overwrite).

    Dependency-free. Ignores blanks and `#` comments, strips surrounding quotes.
    Existing environment values take precedence over the file.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


class ProviderConfig(BaseModel):
    model_id: str
    api_base: str | None = None
    api_key_env: str | None = None
    # Extra keys worked in rotation alongside api_key_env. Each gets its own
    # per-minute quota window, so keys from SEPARATE projects multiply
    # throughput; keys from one project share a budget and add nothing.
    api_key_envs: list[str] = []
    project: str | None = None
    location: str | None = None


class PdfConfig(BaseModel):
    dpi: int = 300
    image_format: str = "png"


class LlmConfig(BaseModel):
    max_retries: int = 5
    concurrency: int = 3
    request_timeout_s: int = 300
    max_output_tokens: int = 32768
    max_pages_per_call: int = 0
    chunk_overlap_pages: int = 2
    # Provider input-token budget per minute; 0 disables client-side pacing.
    tokens_per_minute: int = 0
    # Request-count backstop per minute; 0 disables it.
    requests_per_minute: int = 0


class CostConfig(BaseModel):
    input_per_million: float
    output_per_million: float

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        """Return USD cost for a call given token counts."""
        return (
            input_tokens / 1_000_000 * self.input_per_million
            + output_tokens / 1_000_000 * self.output_per_million
        )


class PathsConfig(BaseModel):
    raw_pdfs: Path
    images: Path
    transcripts: Path
    cache: Path
    logs: Path
    prompt_file: Path        # fallback only; a set's own prompt_file wins


_EXAM_SET_PATH_FIELDS = (
    "task_stems", "prompt_file", "rubric_file", "extraction_csv",
    "extraction_runs_csv", "eval_output_dir",
)


class ExamSetConfig(BaseModel):
    """One exam set: a class's question paper and everything derived from it.

    Every path here is resolved against the config file's directory, exactly as
    `paths:` is. `eval_output_dir` is the root of that set's grading output; the
    Class XI set points at the legacy flat `output/` so its completed run keeps
    working untouched.
    """

    label: str = ""
    task_stems: Path
    prompt_file: Path
    rubric_file: Path
    target_questions: list[int]
    max_question: int
    extraction_csv: Path
    extraction_runs_csv: Path
    eval_output_dir: Path

    def resolve(self, root: Path) -> None:
        for field in _EXAM_SET_PATH_FIELDS:
            value: Path = getattr(self, field)
            if not value.is_absolute():
                setattr(self, field, root / value)


class Config(BaseModel):
    provider: str
    thinking_level: str
    providers: dict[str, ProviderConfig]
    pdf: PdfConfig
    llm: LlmConfig
    cost: CostConfig
    exam_sets: dict[str, ExamSetConfig]
    default_exam_set: str
    paths: PathsConfig

    # ---- exam sets --------------------------------------------------------
    def exam_set(self, set_id: str) -> ExamSetConfig:
        try:
            return self.exam_sets[set_id]
        except KeyError as exc:
            raise KeyError(
                f"exam set '{set_id}' is not defined under config.exam_sets "
                f"(have: {sorted(self.exam_sets)})"
            ) from exc

    def exam_set_for_script(self, script_id: str) -> ExamSetConfig:
        """The set a script belongs to, derived from its id."""
        return self.exam_set(exam_set_of(script_id))

    def target_questions_for(self, set_id: str) -> list[int]:
        return self.exam_set(set_id).target_questions

    @property
    def active_provider(self) -> ProviderConfig:
        try:
            return self.providers[self.provider]
        except KeyError as exc:
            raise KeyError(
                f"provider '{self.provider}' is not defined under config.providers "
                f"(have: {sorted(self.providers)})"
            ) from exc

    @property
    def model_id(self) -> str:
        return self.active_provider.model_id

    def stamp(self) -> dict[str, str]:
        """The model-config fields stamped into every output record."""
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "thinking_level": self.thinking_level,
        }

    def ensure_dirs(self) -> None:
        """Create all output directories if they don't yet exist."""
        for p in (
            self.paths.images,
            self.paths.transcripts,
            self.paths.cache,
            self.paths.logs,
        ):
            p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Evaluation stage config (separate section of the same config.yaml).
# ---------------------------------------------------------------------------
class GraderConfig(BaseModel):
    model_id: str
    provider: str          # "vertex" | "ai_studio"
    api_key_env: str
    # Extra keys worked in rotation; see ProviderConfig.api_key_envs.
    api_key_envs: list[str] = []


class EvalPathsConfig(BaseModel):
    extraction_csv: Path
    rubric_file: Path
    user_template: Path
    evaluations: Path
    evaluation_csv: Path
    evaluation_runs_csv: Path
    logs: Path


class EvalConfig(BaseModel):
    thinking_level: str
    k: int
    concurrency: int = 3
    max_retries: int = 5
    request_timeout_s: int = 300
    max_output_tokens: int = 32768
    # Provider input-token budget per minute; 0 disables client-side pacing.
    tokens_per_minute: int = 0
    # Request-count backstop per minute; 0 disables it.
    requests_per_minute: int = 0
    summary_length_cap_exempt_below_words: int = 0
    graders: dict[str, GraderConfig]
    cost: dict[str, CostConfig]
    paths: EvalPathsConfig
    default_models: list[str] = ["gemini", "gemma"]
    # Set by load_eval_config once an exam set is chosen. The rubric, the
    # extraction CSV read, and every output path below are that set's.
    exam_set: str = ""
    exam_set_label: str = ""

    def stamp(self) -> dict[str, str]:
        """Config fields stamped into every evaluation record."""
        return {"thinking_level": self.thinking_level, "exam_set": self.exam_set}

    def ensure_dirs(self) -> None:
        for p in (self.paths.evaluations, self.paths.logs):
            p.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG_PATH = Path("config.yaml")


def _load_env(root: Path) -> None:
    """Load .env via python-dotenv without overriding existing environment."""
    from dotenv import load_dotenv as dotenv_load

    dotenv_load(dotenv_path=root / ".env", override=False)


def load_eval_config(
    path: Path | str = DEFAULT_CONFIG_PATH, exam_set: str | None = None
) -> EvalConfig:
    """Load and validate the `evaluation:` section. Resolves paths, loads .env.

    `exam_set` (defaulting to config.default_exam_set) selects which question
    paper is being graded. The set supplies the rubric, the extraction CSV to
    read, and the output directory, overriding the generic `evaluation.paths`
    defaults — so two classes never write over each other's grading records.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if "evaluation" not in raw:
        raise KeyError("config.yaml has no 'evaluation:' section")
    cfg = EvalConfig.model_validate(raw["evaluation"])

    root = path.resolve().parent
    _load_env(root)
    for field in cfg.paths.model_fields:
        value = getattr(cfg.paths, field)
        if not value.is_absolute():
            setattr(cfg.paths, field, root / value)

    sets = {k: ExamSetConfig.model_validate(v) for k, v in (raw.get("exam_sets") or {}).items()}
    set_id = exam_set or raw.get("default_exam_set")
    if not sets or not set_id:
        raise KeyError("config.yaml has no 'exam_sets:' / 'default_exam_set:'")
    if set_id not in sets:
        raise KeyError(
            f"exam set '{set_id}' is not defined under config.exam_sets "
            f"(have: {sorted(sets)})"
        )
    es = sets[set_id]
    es.resolve(root)

    cfg.exam_set = set_id
    cfg.exam_set_label = es.label
    cfg.paths.extraction_csv = es.extraction_csv
    cfg.paths.rubric_file = es.rubric_file
    cfg.paths.evaluations = es.eval_output_dir / "evaluations"
    cfg.paths.evaluation_csv = es.eval_output_dir / "evaluation.csv"
    cfg.paths.evaluation_runs_csv = es.eval_output_dir / "evaluation_runs.csv"
    return cfg


def require_grader_keys(cfg: EvalConfig, grader_names: list[str]) -> None:
    """Fail fast at startup if a grader has NO usable key at all.

    A grader may list several keys; some missing is survivable (the pool just
    runs narrower), none is not. Reports only variable NAMES, never values.
    """
    missing = []
    for name in grader_names:
        g = cfg.graders[name]
        candidates = [g.api_key_env, *g.api_key_envs]
        if not any(os.environ.get(c) for c in candidates if c):
            missing.append(f"{name} (none of: {', '.join(c for c in candidates if c)})")
    if missing:
        raise RuntimeError(
            "missing required API key env var(s): " + "; ".join(missing)
        )


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate config.yaml. Paths are resolved relative to the file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = Config.model_validate(raw)

    # Resolve relative paths against the config file's directory so the pipeline
    # runs the same regardless of the current working directory.
    root = path.resolve().parent

    # Load secrets from a project-root .env (e.g. VERTEX_API_KEY) if present.
    load_dotenv(root / ".env")
    for field in cfg.paths.model_fields:
        value = getattr(cfg.paths, field)
        if not value.is_absolute():
            setattr(cfg.paths, field, (root / value))
    for es in cfg.exam_sets.values():
        es.resolve(root)
    return cfg
