import os
import gc
from typing import Optional, Dict, Any, List
from PIL import Image
from src.core.config import load_config
from src.engine.base_engine import BaseVLMEngine


class GemmaCudaEngine(BaseVLMEngine):
    """
    Local CUDA Inference Engine for Gemma 4 31B IT.
    Optimized for NVIDIA RTX 5090 (32GB VRAM) and 32GB RAM using
    bitsandbytes 4-bit NF4 / 8-bit quantization or bfloat16.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-4-31b-it",
        quantization: str = "4bit",
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        trust_remote_code: bool = True,
        use_flash_attention_2: bool = False
    ):
        self.model_id = model_id
        self.quantization = quantization
        self.torch_dtype_str = torch_dtype
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.use_flash_attention_2 = use_flash_attention_2
        
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._init_model()

    def _init_model(self):
        import torch
        from transformers import AutoProcessor, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device is not available. Ensure NVIDIA drivers and CUDA-enabled PyTorch are installed."
            )

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }
        resolved_dtype = dtype_map.get(self.torch_dtype_str.lower(), torch.bfloat16)

        # Configure Quantization
        quantization_config = None
        if self.quantization.lower() == "4bit":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=resolved_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif self.quantization.lower() == "8bit":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

        # Extract Hugging Face token from environment or .env file
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            from pathlib import Path
            candidate_env_paths = [
                ".env",
                "../.env",
                str(Path(__file__).resolve().parent.parent.parent / ".env")
            ]
            for env_path in candidate_env_paths:
                if os.path.exists(env_path):
                    with open(env_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("HF_TOKEN=") or line.startswith("HUGGING_FACE_HUB_TOKEN="):
                                val = line.split("=", 1)[1].strip().strip("'\"")
                                if val:
                                    hf_token = val
                                    os.environ["HF_TOKEN"] = val
                                    break
                if hf_token:
                    break
        hf_token = hf_token or None

        if hf_token:
            masked = hf_token[:6] + "..." + hf_token[-4:] if len(hf_token) > 10 else "***"
            print(f"[GemmaCudaEngine] Using Hugging Face token: {masked}")
        else:
            print("[GemmaCudaEngine] Warning: No HF_TOKEN detected. If downloading gated models, set HF_TOKEN in .env or run 'huggingface-cli login'.")

        print(f"[GemmaCudaEngine] Initializing Processor for '{self.model_id}'...")
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
                token=hf_token
            )
        except Exception as e:
            print(f"[GemmaCudaEngine] Fallback to AutoTokenizer: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
                token=hf_token
            )

        print(f"[GemmaCudaEngine] Loading '{self.model_id}' (quantization: {self.quantization}, dtype: {self.torch_dtype_str})...")
        
        model_kwargs: Dict[str, Any] = {
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
            "token": hf_token,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = resolved_dtype

        if self.use_flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Safe dynamic model loaders (prevents ImportError on missing classes in various transformers versions)
        loaders = []
        try:
            from transformers import AutoModelForImageTextToText
            loaders.append(("AutoModelForImageTextToText", AutoModelForImageTextToText))
        except ImportError:
            pass

        try:
            from transformers import AutoModelForVision2Seq
            loaders.append(("AutoModelForVision2Seq", AutoModelForVision2Seq))
        except ImportError:
            pass

        try:
            from transformers import AutoModelForCausalLM
            loaders.append(("AutoModelForCausalLM", AutoModelForCausalLM))
        except ImportError:
            pass

        try:
            from transformers import AutoModel
            loaders.append(("AutoModel", AutoModel))
        except ImportError:
            pass

        loaded = False
        last_error = None
        for loader_name, loader_cls in loaders:
            try:
                self.model = loader_cls.from_pretrained(self.model_id, **model_kwargs)
                print(f"[GemmaCudaEngine] Model successfully loaded via {loader_name}.")
                loaded = True
                break
            except Exception as e:
                last_error = e
                continue

        if not loaded:
            raise RuntimeError(
                f"Failed to load '{self.model_id}' across all candidate loaders ({[n for n, _ in loaders]}). "
                f"Error details: {last_error}"
            )

        self.model.eval()
        print(f"[GemmaCudaEngine] Model successfully initialized on CUDA.")

    def generate_multimodal(
        self,
        image: Image.Image,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_new_tokens: int = 4096,
        thinking_mode: bool = False,
        **kwargs: Any
    ) -> str:
        import torch

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Construct Chat Structure
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        user_content: List[Dict[str, Any]] = [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]
        
        if thinking_mode:
            user_content.append({"type": "text", "text": "\n(Include your analytical reasoning steps before concluding.)"})
        else:
            user_content.append({"type": "text", "text": "\n(Do NOT include any external commentary. Output only the requested result.)"})

        messages.append({"role": "user", "content": user_content})

        # Process inputs
        if self.processor and hasattr(self.processor, "apply_chat_template"):
            formatted_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=formatted_prompt,
                images=image,
                return_tensors="pt"
            )
        else:
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            )

        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            output_tokens = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        generated_tokens = output_tokens[0][input_len:]
        
        if self.processor and hasattr(self.processor, "decode"):
            response = self.processor.decode(generated_tokens, skip_special_tokens=True)
        elif self.tokenizer:
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            response = str(generated_tokens)

        return response.strip()

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_new_tokens: int = 4096,
        thinking_mode: bool = False,
        **kwargs: Any
    ) -> str:
        import torch

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_text = prompt
        if thinking_mode:
            user_text += "\n(Include your reasoning breakdown step by step.)"

        messages.append({"role": "user", "content": user_text})

        tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.tokenizer
        
        if tokenizer and hasattr(tokenizer, "apply_chat_template"):
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(formatted_prompt, return_tensors="pt")
        else:
            full_prompt = f"{system_prompt}\n\n{user_text}" if system_prompt else user_text
            inputs = tokenizer(full_prompt, return_tensors="pt")

        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            output_tokens = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        generated_tokens = output_tokens[0][input_len:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()

    def get_engine_info(self) -> Dict[str, Any]:
        import torch
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        vram_allocated_gb = torch.cuda.memory_allocated(0) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        vram_reserved_gb = torch.cuda.memory_reserved(0) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        
        return {
            "engine_type": "GemmaCudaEngine",
            "model_id": self.model_id,
            "quantization": self.quantization,
            "torch_dtype": self.torch_dtype_str,
            "gpu_name": gpu_name,
            "vram_allocated_gb": round(vram_allocated_gb, 2),
            "vram_reserved_gb": round(vram_reserved_gb, 2),
        }
