"""
config.py
=========
Single source of truth for paths, hyperparameters, and constants used
across the whole Ecommerce-LLM-Finetuning project (notebooks, src/, app/).

Import pattern:
    from src.config import Config
    cfg = Config()
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Set all relevant random seeds for reproducibility.

    Args:
        seed: The seed value to apply to python's random, numpy, and torch
            (if installed) RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # torch may not be installed yet when config is first imported
        pass


# ---------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Central configuration for the Ecommerce-LLM-Finetuning project."""

    # ---- Project root / paths -------------------------------------------------
    project_root: Path = field(
        default_factory=lambda: Path(
            os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1])
        )
    )

    # ---- Random seed ------------------------------------------------------
    seed: int = 42

    # ---- Dataset ------------------------------------------------------
    # Primary HF dataset. Swappable; notebook 01 also has a synthetic fallback
    # so the pipeline never hard-fails if the HF Hub is unreachable.
    hf_dataset_name: str = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    hf_dataset_config: Optional[str] = None
    train_split_ratio: float = 0.8
    val_split_ratio: float = 0.1
    test_split_ratio: float = 0.1

    # ---- Model ------------------------------------------------------
    # Small, Colab-T4-friendly instruct model. Swap for Qwen2.5-3B-Instruct,
    # gemma-2-2b-it, or TinyLlama if you want to experiment.
    base_model_name: str = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    # ---- LoRA / QLoRA ------------------------------------------------------
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0  # 0 is optimized in Unsloth
    lora_target_modules: tuple = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    use_gradient_checkpointing: str = "unsloth"  # "unsloth" | True | False

    # ---- Training ------------------------------------------------------
    output_dir: str = "outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 2
    bf16: bool = True   # auto-fallback to fp16 handled in trainer.py
    fp16: bool = False
    optim: str = "adamw_8bit"
    report_to: str = "none"

    # ---- Prompt template ------------------------------------------------------
    system_prompt: str = (
        "You are a helpful, polite Ecommerce Customer Support Assistant. "
        "You answer questions about orders, shipping, refunds, returns, "
        "payments, coupons, delivery, and account management. "
        "If a question is unrelated to ecommerce customer support, politely "
        "say that you can only help with ecommerce support questions."
    )
    instruction_header: str = "### Instruction:"
    response_header: str = "### Response:"

    # ---- Evaluation ------------------------------------------------------
    eval_num_samples: int = 50
    eval_max_new_tokens: int = 128

    # ---- Inference / app ------------------------------------------------------
    default_temperature: float = 0.7
    default_max_new_tokens: int = 256
    default_top_p: float = 0.9

    def __post_init__(self) -> None:
        set_seed(self.seed)

    # ---- Derived paths ------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def sample_data_dir(self) -> Path:
        return self.data_dir / "sample"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def base_model_dir(self) -> Path:
        return self.models_dir / "base_model"

    @property
    def lora_adapter_dir(self) -> Path:
        return self.models_dir / "lora_adapter"

    @property
    def merged_model_dir(self) -> Path:
        return self.models_dir / "merged_model"

    def ensure_dirs(self) -> None:
        """Create every directory this config points to, if missing."""
        for p in (
            self.raw_data_dir,
            self.processed_data_dir,
            self.sample_data_dir,
            self.base_model_dir,
            self.lora_adapter_dir,
            self.merged_model_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    cfg = Config()
    cfg.ensure_dirs()
    print(cfg)
