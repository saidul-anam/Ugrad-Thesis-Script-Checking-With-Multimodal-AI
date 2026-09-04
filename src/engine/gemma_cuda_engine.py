import os
import gc
from typing import Optional, Dict, Any, List
from PIL import Image
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
        self._configure_cuda_allocator()
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

    def _configure_cuda_allocator(self) -> None:
        """Reduce CUDA memory fragmentation for long-generation workloads."""
        if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    def clear_cuda_cache(self) -> None:
        """Release unreferenced CUDA memory back to the allocator."""
        try:
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            # Best-effort cleanup only.
            pass

    @staticmethod
    def _is_cuda_oom_error(exc: Exception) -> bool:
        return "cuda out of memory" in str(exc).lower()

    def _generate_with_oom_retry(
        self,
        inputs: Dict[str, Any],
        gen_kwargs: Dict[str, Any]
    ):
        import torch

        max_tokens = int(gen_kwargs.get("max_new_tokens", 512))
        attempts = [
            {"max_new_tokens": max_tokens, "use_cache": True},
            {"max_new_tokens": max(256, max_tokens // 2), "use_cache": True},
            {"max_new_tokens": max(128, max_tokens // 4), "use_cache": False},
        ]

        last_error: Optional[RuntimeError] = None
        for idx, candidate in enumerate(attempts, start=1):
            local_kwargs = dict(gen_kwargs)
            local_kwargs.update(candidate)
            try:
                with torch.inference_mode():
                    return self.model.generate(**inputs, **local_kwargs)
            except RuntimeError as exc:
                last_error = exc
                if not self._is_cuda_oom_error(exc):
                    raise

                is_last_attempt = idx >= len(attempts)
                if is_last_attempt:
                    raise

                next_candidate = attempts[idx]
                print(
                    "[GemmaCudaEngine] CUDA OOM during generation; retrying with "
                    f"max_new_tokens={next_candidate['max_new_tokens']} "
                    f"and use_cache={next_candidate['use_cache']}"
                )
                self.clear_cuda_cache()

        if last_error is not None:
            raise last_error
        raise RuntimeError("Generation failed before producing output.")

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
            "max_memory": {
                0: "30GiB",
                "cpu": "50GiB",
            },
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
                print(f"[GemmaCudaEngine] Trying loader: {loader_name}...")

                self.model = loader_cls.from_pretrained(
                    self.model_id,
                    **model_kwargs
                )

                print(
                    f"[GemmaCudaEngine] Model successfully loaded via "
                    f"{loader_name}."
                )

                if hasattr(self.model, "hf_device_map"):
                    print("[GemmaCudaEngine] Model device map:")
                    print(self.model.hf_device_map)

                loaded = True
                break

            except Exception as e:
                print(
                    f"[GemmaCudaEngine] {loader_name} failed: "
                    f"{type(e).__name__}: {e}"
                )
                last_error = e
                continue

        if not loaded:
            raise RuntimeError(
                f"Failed to load '{self.model_id}' across all candidate "
                f"loaders ({[n for n, _ in loaders]}). "
                f"Error details: {last_error}"
            )


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
        max_new_tokens: int = 3072,
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
            "use_cache": kwargs.get("use_cache", True),
        }
        max_time = kwargs.get("max_time")
        if max_time is not None:
            gen_kwargs["max_time"] = float(max_time)
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        output_tokens = None
        try:
            output_tokens = self._generate_with_oom_retry(inputs, gen_kwargs)

            input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
            generated_tokens = output_tokens[0][input_len:]

            if self.processor and hasattr(self.processor, "decode"):
                response = self.processor.decode(generated_tokens, skip_special_tokens=True)
            elif self.tokenizer:
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            else:
                response = str(generated_tokens)

            return response.strip()
        finally:
            if output_tokens is not None:
                del output_tokens
            del inputs
            self.clear_cuda_cache()

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 0.1,
        max_new_tokens: int = 3072,
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
            "use_cache": kwargs.get("use_cache", True),
        }
        max_time = kwargs.get("max_time")
        if max_time is not None:
            gen_kwargs["max_time"] = float(max_time)
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        output_tokens = None
        try:
            output_tokens = self._generate_with_oom_retry(inputs, gen_kwargs)

            input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
            generated_tokens = output_tokens[0][input_len:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return response.strip()
        finally:
            if output_tokens is not None:
                del output_tokens
            del inputs
            self.clear_cuda_cache()

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
