"""Run an open-weight VLM locally with PyTorch + Hugging Face Transformers.

`TransformersVlm` loads a model once and answers (image, prompt) pairs. torch and
transformers are imported only when it is created, so the rest of the project (and the
unit tests) still work without them. Install them from requirements-vlm.txt.

Default model: Qwen3-VL-4B-Instruct in 4-bit, which fits an 8 GB GPU (bf16 would need ~9 GB).
Other image-text-to-text models on the Hugging Face Hub (e.g. Qwen3-VL-2B/8B, Qwen3.5) can be
passed by name; the chat format comes from each model's own processor.
"""

import time
from dataclasses import dataclass

import numpy as np

DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
QUANTIZE_OPTIONS = ("4bit", "8bit", "none")


@dataclass
class VlmReply:
    """One model answer.

    text: the generated text only (the prompt is cut off).
    seconds: wall-clock time for preprocessing + generation (model loading not included).
    prompt_tokens: tokens fed to the model, image tokens included.
    new_tokens: tokens the model generated.
    """

    text: str
    seconds: float
    prompt_tokens: int
    new_tokens: int


class TransformersVlm:
    """A vision-language model on the local GPU (or CPU, very slowly, if there is no GPU)."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        quantize: str = "4bit",
        max_new_tokens: int = 512,
        quantize_vision: bool = False,
    ):
        """quantize_vision: also quantize the vision encoder (saves ~1 GB for 8B models, may cost accuracy)."""
        if quantize not in QUANTIZE_OPTIONS:
            raise ValueError(f"quantize must be one of {QUANTIZE_OPTIONS}, got {quantize!r}")
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.model_id = model_id
        self.quantize = quantize
        self.max_new_tokens = max_new_tokens
        self.on_gpu = torch.cuda.is_available()

        kwargs = {"dtype": torch.bfloat16, "device_map": "cuda" if self.on_gpu else "cpu"}
        if quantize != "none":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quantize == "4bit",
                load_in_8bit=quantize == "8bit",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                # By default keep the vision encoder in bf16: it is small, and seeing the image well matters most here.
                llm_int8_skip_modules=["lm_head"] if quantize_vision else ["visual", "lm_head"],
            )

        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self.load_seconds = time.perf_counter() - started

    def generate(self, image_rgb: np.ndarray, prompt: str, max_new_tokens: int | None = None) -> VlmReply:
        """Answer `prompt` about one (H, W, 3) uint8 RGB image. Greedy decoding, so runs are repeatable."""
        from PIL import Image

        torch = self.torch
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": Image.fromarray(image_rgb)}, {"type": "text", "text": prompt}],
        }]
        self._sync()
        started = time.perf_counter()
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                temperature=None, top_p=None, top_k=None,  # silence "sampling settings ignored" warnings
            )
        self._sync()
        seconds = time.perf_counter() - started

        prompt_tokens = inputs["input_ids"].shape[1]
        new_ids = output[:, prompt_tokens:]
        text = self.processor.batch_decode(new_ids, skip_special_tokens=True)[0]
        return VlmReply(text=text, seconds=seconds, prompt_tokens=int(prompt_tokens), new_tokens=int(new_ids.shape[1]))

    def peak_memory_gb(self) -> float:
        """Most GPU memory used so far by PyTorch, GB (0 on CPU)."""
        return self.torch.cuda.max_memory_allocated() / 2**30 if self.on_gpu else 0.0

    def _sync(self):
        # GPU work runs in the background; wait for it so the timing is honest.
        if self.on_gpu:
            self.torch.cuda.synchronize()
