"""Configuration loading.

Model configuration is DATA, not code: everything the LLM was run with lives in
config.yaml and is stamped into every output record via `Config.stamp()`.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel


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
    prompt_file: Path


class Config(BaseModel):
    provider: str
    thinking_level: str
    providers: dict[str, ProviderConfig]
    pdf: PdfConfig
    llm: LlmConfig
    cost: CostConfig
    target_questions: list[int]
    paths: PathsConfig

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
    summary_length_cap_exempt_below_words: int = 0
    graders: dict[str, GraderConfig]
    cost: dict[str, CostConfig]
    paths: EvalPathsConfig

    def stamp(self) -> dict[str, str]:
        """Config fields stamped into every evaluation record."""
        return {"thinking_level": self.thinking_level}

    def ensure_dirs(self) -> None:
        for p in (self.paths.evaluations, self.paths.logs):
            p.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG_PATH = Path("config.yaml")


def _load_env(root: Path) -> None:
    """Load .env via python-dotenv without overriding existing environment."""
    from dotenv import load_dotenv as dotenv_load

    dotenv_load(dotenv_path=root / ".env", override=False)


def load_eval_config(path: Path | str = DEFAULT_CONFIG_PATH) -> EvalConfig:
    """Load and validate the `evaluation:` section. Resolves paths, loads .env."""
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
    return cfg


def require_grader_keys(cfg: EvalConfig, grader_names: list[str]) -> None:
    """Fail fast at startup if any needed grader's key env var is unset.

    Reports only the variable NAME, never the value.
    """
    missing = [
        f"{name} ({cfg.graders[name].api_key_env})"
        for name in grader_names
        if not os.environ.get(cfg.graders[name].api_key_env)
    ]
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
    return cfg
