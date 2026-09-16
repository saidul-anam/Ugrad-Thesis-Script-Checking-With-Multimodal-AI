import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

# Configure PyTorch CUDA Allocator early to prevent memory fragmentation
if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Auto-load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Minimal fallback parser if python-dotenv is not installed
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v


class ModelConfig(BaseModel):
    model_id: str = Field("google/gemma-4-31b-it", description="Hugging Face model ID or local directory")
    torch_dtype: str = Field("bfloat16", description="Weight precision (bfloat16 / float16 / float32)")
    quantization: str = Field("4bit", description="Quantization mode ('4bit', '8bit', or 'none')")
    engine_type: str = Field("cuda", description="Engine type: 'cuda', 'api', or 'mock'")
    api_url: Optional[str] = Field(None, description="Local or remote OpenAI-compatible API endpoint URL")
    device_map: str = Field("auto", description="Device placement strategy")
    trust_remote_code: bool = Field(True, description="Whether to allow remote code execution for custom models")
    use_flash_attention_2: bool = Field(False, description="Enable flash attention 2 if supported")
    attn_implementation: str = Field("sdpa", description="Attention implementation ('sdpa', 'flash_attention_2', 'eager')")



class DecodingConfig(BaseModel):
    temperature: float = Field(0.0, description="Sampling temperature (0.0 for greedy)")
    top_p: float = Field(0.1, description="Nucleus sampling cutoff")
    max_new_tokens: int = Field(3072, description="Maximum new generation tokens")
    do_sample: bool = Field(False, description="Whether sampling is active (False for greedy)")
    thinking_mode: bool = Field(False, description="Ablation flag for reasoning/thinking mode")


class RagConfig(BaseModel):
    enabled: bool = True
    thematic_context_dir: str = "configs/context/"


class PipelineStageConfig(BaseModel):
    stage0b_teacher_marks: bool = Field(True, description="Whether to execute Stage 0b red-ink teacher mark extraction")
    stage1_verbatim: bool = True
    stage2_verification: bool = True
    stage3_error_extraction: bool = True
    stage4_rubric_evaluation: bool = True
    stage4_max_new_tokens: int = Field(2048, description="Upper token budget for Stage 4 rubric generation")
    stage4_generation_timeout_sec: float = Field(180.0, description="Hard timeout for Stage 4 generation in seconds")
    rag: RagConfig = Field(default_factory=RagConfig)
    cache_intermediate_stages: bool = True
    output_dir: str = "outputs/extracted"
    parallel_workers: int = Field(1, description="Number of concurrent workers for page extraction (1=serial, >1=parallel)")


class ArbitrationWeights(BaseModel):
    """Logistic fusion weights. Provisional defaults; overwrite with values fitted by scripts/evaluate_arbitration.py --fit."""
    bias: float = Field(-0.3, description="Intercept (negative => default leans GENUINE when no evidence)")
    phonetic: float = Field(0.6, description="Weight of (1 - phonetic plausibility)")
    writer: float = Field(1.2, description="Weight of the per-writer learned confusion prior")
    consensus: float = Field(2.6, description="Weight of augmented re-read agreement signal")
    forced_choice: float = Field(2.0, description="Weight of the forced-choice crop verdict")


class ArbitrationConfig(BaseModel):
    """Stage 3b: evidence-fused handwriting ambiguity gate."""
    mode: str = Field("evidence", description="'evidence' (new gate), 'legacy' (single whole-page VLM call), or 'off'")
    max_token_edits: int = Field(2, description="Max character edits between read/intended token to be gated")
    use_bbox_localization: bool = Field(True, description="Ask the VLM for the line bounding box before projection fallback")
    localization_min_ratio: float = Field(0.6, description="Min fuzzy match between crop transcription and context to accept a crop")
    projection_search_window: int = Field(3, description="Lines searched either side of the expected line position")
    crop_pad_px: int = Field(14)
    crop_min_height_px: int = Field(96, description="Crops shorter than this are upscaled 2x")
    consensus_samples: int = Field(5, description="Number of augmented re-reads per crop")
    consensus_use_sampling_variant: bool = Field(True, description="Use one temperature-sampled variant among the re-reads")
    consensus_temperature: float = Field(0.7)
    consensus_parallel_workers: int = Field(1, description="Concurrency for consensus re-reads (1=serial, >1=parallel)")
    writer_profile_min_count: int = Field(2, description="Min observations of a confusion pair before it counts")
    threshold_ambiguity: float = Field(0.65, description="score >= => HANDWRITING_AMBIGUITY")
    threshold_genuine: float = Field(0.35, description="score <= => GENUINE_ERROR; between => UNCERTAIN")
    save_crops: bool = Field(True)
    lexicon_scan: bool = Field(True, description="Also gate non-dictionary tokens Stage 3 did not flag (dictionary lookup, no letter rules)")
    weights: ArbitrationWeights = Field(default_factory=ArbitrationWeights)


class PipelineConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    decoding: DecodingConfig = Field(default_factory=DecodingConfig)
    pipeline: PipelineStageConfig = Field(default_factory=PipelineStageConfig)
    arbitration: ArbitrationConfig = Field(default_factory=ArbitrationConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """Load configuration from specified path or default locations."""
    default_locations = [
        config_path,
        "configs/pipeline_config.yaml",
        "configs/default.yaml"
    ]
    for loc in default_locations:
        if loc and os.path.exists(loc):
            return PipelineConfig.from_yaml(loc)
    return PipelineConfig()
