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
    stage1_verbatim: bool = True
    stage2_verification: bool = True
    stage3_error_extraction: bool = True
    stage4_rubric_evaluation: bool = True
    stage4_max_new_tokens: int = Field(768, description="Upper token budget for Stage 4 rubric generation")
    stage4_generation_timeout_sec: float = Field(180.0, description="Hard timeout for Stage 4 generation in seconds")
    rag: RagConfig = Field(default_factory=RagConfig)
    cache_intermediate_stages: bool = True
    output_dir: str = "outputs/runs"


class PipelineConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    decoding: DecodingConfig = Field(default_factory=DecodingConfig)
    pipeline: PipelineStageConfig = Field(default_factory=PipelineStageConfig)

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
