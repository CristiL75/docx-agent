import json
import os
import random
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFModel:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device_map: str = "auto",
        seed: int = 42,
        hf_token: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.seed = seed
        self.device_map = device_map
        self._set_seed(seed)
        token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device_map,
            token=token,
        )

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _extract_json(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return "{}"
        candidate = text[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            return "{}"

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        messages = [
            {
                "role": "system",
                "content": "Return ONLY strict JSON. No prose, no code fences.",
            },
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
        return self._extract_json(decoded)

    def available(self) -> bool:
        return self.model is not None and self.tokenizer is not None
