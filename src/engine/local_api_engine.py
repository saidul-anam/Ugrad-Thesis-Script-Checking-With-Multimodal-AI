import io
import base64
import requests
from typing import Optional, Dict, Any, List
from PIL import Image
from src.engine.base_engine import BaseVLMEngine


class LocalAPIEngine(BaseVLMEngine):
    """
    Inference Engine using a local OpenAI-compatible API endpoint (e.g. LM Studio, vLLM, Ollama).
    Enables GPU-accelerated inference without allocating additional VRAM when a model
    is already running locally on a shared machine.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:1234/v1",
        model_id: Optional[str] = None,
        timeout: float = 300.0,
    ):
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.context_window = 4096
        self.last_usage: Dict[str, Any] = {}
        self.model_id = self._resolve_model_id(model_id)
        print(f"[LocalAPIEngine] Connected to {self.api_url} (model: '{self.model_id}', ctx: {self.context_window})")

    def _resolve_model_id(self, requested: Optional[str]) -> str:
        """Query /v1/models and match the requested model ID or active server model."""
        try:
            resp = requests.get(f"{self.api_url}/models", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                available_ids = [m.get("id", "") for m in data if m.get("id")]
                if available_ids:
                    # 1. Exact match
                    if requested and requested in available_ids:
                        return requested
                    # 2. Fuzzy match (e.g. 'google/gemma-4-31b-it' -> 'google/gemma-4-31b')
                    if requested:
                        clean_req = requested.lower().replace("-it", "").replace("_it", "").split("/")[-1]
                        for aid in available_ids:
                            clean_aid = aid.lower().split("/")[-1]
                            if clean_req in clean_aid or clean_aid in clean_req:
                                print(f"[LocalAPIEngine] Mapping model '{requested}' -> '{aid}' on API server.")
                                return aid
                    # 3. Prefer gemma if present
                    for aid in available_ids:
                        if "gemma" in aid.lower():
                            return aid
                    return available_ids[0]
        except Exception as e:
            print(f"[LocalAPIEngine] Note: Could not query {self.api_url}/models ({e}). Using requested ID.")
        return requested or "google/gemma-4-31b"

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
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_text = prompt
        if thinking_mode:
            user_text += "\n(Include your reasoning breakdown step by step.)"

        messages.append({"role": "user", "content": user_text})

        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
        }
        if not thinking_mode:
            payload["reasoning_effort"] = "none"

        timeout_val = kwargs.get("max_time") or self.timeout
        try:
            resp = requests.post(
                f"{self.api_url}/chat/completions",
                json=payload,
                timeout=timeout_val
            )
            if not resp.ok:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
            data = resp.json()
            usage = data.get("usage", {})
            self.last_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "context_window": self.context_window
            }
            choice = data["choices"][0]["message"]
            content = choice.get("content") or ""
            # If content is empty but reasoning is present, use reasoning as fallback
            if not content.strip() and "reasoning_content" in choice:
                content = choice.get("reasoning_content") or ""
            return content.strip()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to local API at {self.api_url}. "
                "Ensure LM Studio or your local API server is active and the model is loaded."
            )
        except Exception as e:
            raise RuntimeError(f"[LocalAPIEngine] API request failed: {e}")

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
        img_b64 = getattr(image, "_cached_b64", None)
        if not img_b64:
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Constrain image resolution to keep vision tokens within context budget
            max_dim = 1280
            if max(image.width, image.height) > max_dim:
                scale = max_dim / max(image.width, image.height)
                new_size = (int(image.width * scale), int(image.height * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            try:
                setattr(image, "_cached_b64", img_b64)
            except Exception:
                pass

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content: List[Dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": prompt}
        ]
        if thinking_mode:
            user_content.append({"type": "text", "text": "\n(Include your analytical reasoning steps before concluding.)"})

        messages.append({"role": "user", "content": user_content})

        # Cap max_tokens to preserve context headroom for image tokens
        safe_tokens = min(max_new_tokens, 1536)

        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": safe_tokens,
        }

        if not thinking_mode:
            payload["reasoning_effort"] = "none"

        timeout_val = kwargs.get("max_time") or self.timeout
        try:
            resp = requests.post(
                f"{self.api_url}/chat/completions",
                json=payload,
                timeout=timeout_val
            )
            if not resp.ok:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
            data = resp.json()
            usage = data.get("usage", {})
            self.last_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "context_window": self.context_window
            }
            choice = data["choices"][0]["message"]
            content = choice.get("content") or ""
            if not content.strip() and "reasoning_content" in choice:
                content = choice.get("reasoning_content") or ""
            return content.strip()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to local API at {self.api_url}. "
                "Ensure LM Studio or your local API server is active and the model is loaded."
            )
        except Exception as e:
            raise RuntimeError(f"[LocalAPIEngine] Multimodal API request failed: {e}")

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "engine": "LocalAPIEngine",
            "api_url": self.api_url,
            "model_id": self.model_id,
            "backend": "OpenAI-compatible HTTP API (LM Studio / vLLM / Ollama)"
        }
