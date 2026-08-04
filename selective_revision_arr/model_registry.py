from __future__ import annotations

from typing import Any, Dict

import torch

MODEL_DTYPE = torch.bfloat16
MODEL_DTYPE_NAME = "bfloat16"

DEFAULT_MODELS = [
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    "google/medgemma-27b-text-it",
    "EPFLiGHT/EuroLLM-22B-MeditronFO",
    "Qwen/Qwen3.6-35B-A3B",
    "EPFLiGHT/Gemma-3-27B-MeditronFO",
    "EPFLiGHT/OLMo-2-32B-MeditronFO",
    "google/gemma-4-31B-it",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "sarvamai/sarvam-30b",
    "unsloth/Qwen2.5-72B-bnb-4bit",
]

PREQUANTIZED_4BIT_MODELS = {
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
    "unsloth/Qwen2.5-72B-bnb-4bit",
}

MULTIMODAL_MODELS = {
    "zai-org/GLM-4.7-Flash",
    "google/medgemma-1.5-4b-it",
    "Qwen/Qwen3.6-35B-A3B",
    "google/gemma-4-31B-it",
    "EPFLiGHT/Gemma-3-27B-MeditronFO",
}
MULTIMODAL_GENERATE_ONLY_MODELS = {"google/medgemma-1.5-4b-it"}

TRUST_REMOTE_CODE_MODELS = {
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "sarvamai/sarvam-30b",
}

CHAT_TEMPLATE_KWARGS_BY_MODEL: Dict[str, Dict[str, Any]] = {
    "Qwen/Qwen3.6-35B-A3B": {"enable_thinking": False},
    "google/gemma-4-31B-it": {"enable_thinking": False},
    "zai-org/GLM-4.7-Flash": {"enable_thinking": False},
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": {"enable_thinking": False},
    "sarvamai/sarvam-30b": {"enable_thinking": False},
}

MODEL_NOTES = {
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit": "Pre-quantized 4-bit instruction model.",
    "google/medgemma-27b-text-it": "Text-only medical instruction model; access may require authentication.",
    "EPFLiGHT/EuroLLM-22B-MeditronFO": "Clinical instruction model loaded in BF16.",
    "Qwen/Qwen3.6-35B-A3B": "Text-only evaluation path with thinking disabled.",
    "EPFLiGHT/Gemma-3-27B-MeditronFO": "Text-only evaluation path through the multimodal processor.",
    "EPFLiGHT/OLMo-2-32B-MeditronFO": "Medical instruction model loaded in BF16.",
    "google/gemma-4-31B-it": "Text-only evaluation path with thinking disabled.",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "Remote model code required; thinking disabled.",
    "sarvamai/sarvam-30b": "Remote model code required; loaded in BF16.",
    "unsloth/Qwen2.5-72B-bnb-4bit": "Pre-quantized 4-bit model with BF16 compute.",
    "zai-org/GLM-4.7-Flash": "Remote chat template supports non-thinking evaluation.",
    "google/medgemma-1.5-4b-it": "Generation-only multimodal baseline for text prompts.",
}
