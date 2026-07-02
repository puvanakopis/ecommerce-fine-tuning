"""
trainer.py
==========
Wraps Unsloth model loading + PEFT/LoRA + TRL's SFTTrainer into a single
reusable `EcommerceFineTuner` class used by notebook 05.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.config import Config
from src.utils import get_logger, supports_bf16, detect_device

logger = get_logger(__name__)


class EcommerceFineTuner:
    """End-to-end QLoRA fine-tuning wrapper around Unsloth + TRL."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.model = None
        self.tokenizer = None
        self.trainer = None

    # ------------------------------------------------------------------
    def load_base_model(self):
        """Load the 4-bit base model + tokenizer via Unsloth's `FastLanguageModel`.

        Returns:
            (model, tokenizer) tuple.
        """
        from unsloth import FastLanguageModel

        cfg = self.config
        device = detect_device()
        if device != "cuda":
            logger.warning(
                "No CUDA GPU detected. 4-bit QLoRA training requires a GPU "
                "(Colab: Runtime > Change runtime type > T4 GPU)."
            )

        logger.info(f"Loading base model: {cfg.base_model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.base_model_name,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit,
            dtype=None,  # let Unsloth auto-select bf16/fp16
        )
        self.model, self.tokenizer = model, tokenizer
        return model, tokenizer

    # ------------------------------------------------------------------
    def apply_lora(self):
        """Attach a LoRA adapter to the loaded base model via Unsloth.

        Returns:
            The PEFT-wrapped model.
        """
        from unsloth import FastLanguageModel

        if self.model is None:
            raise RuntimeError("Call load_base_model() before apply_lora().")

        cfg = self.config
        logger.info(
            f"Applying LoRA: r={cfg.lora_r}, alpha={cfg.lora_alpha}, "
            f"targets={cfg.lora_target_modules}"
        )
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=cfg.lora_r,
            target_modules=list(cfg.lora_target_modules),
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            use_gradient_checkpointing=cfg.use_gradient_checkpointing,
            random_state=cfg.seed,
        )
        return self.model

    # ------------------------------------------------------------------
    def build_trainer(self, train_dataset, eval_dataset=None):
        """Configure a TRL `SFTTrainer` with the project's training args.

        Args:
            train_dataset: HF `Dataset` with a `text` column.
            eval_dataset: Optional HF `Dataset` with a `text` column.

        Returns:
            A configured `SFTTrainer` instance (not yet trained).
        """
        from trl import SFTTrainer, SFTConfig

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call load_base_model() and apply_lora() first.")

        cfg = self.config
        use_bf16 = cfg.bf16 and supports_bf16()
        use_fp16 = not use_bf16

        sft_config = SFTConfig(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.num_train_epochs,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.per_device_eval_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_scheduler_type,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            logging_steps=cfg.logging_steps,
            eval_strategy=cfg.eval_strategy if eval_dataset is not None else "no",
            eval_steps=cfg.eval_steps if eval_dataset is not None else None,
            save_strategy=cfg.save_strategy,
            save_steps=cfg.save_steps,
            save_total_limit=cfg.save_total_limit,
            bf16=use_bf16,
            fp16=use_fp16,
            optim=cfg.optim,
            report_to=cfg.report_to,
            seed=cfg.seed,
            max_seq_length=cfg.max_seq_length,
            dataset_text_field="text",
            packing=False,
        )

        logger.info(
            f"Building SFTTrainer (bf16={use_bf16}, fp16={use_fp16}, "
            f"epochs={cfg.num_train_epochs}, effective_batch="
            f"{cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps})"
        )

        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_config,
        )
        return self.trainer

    # ------------------------------------------------------------------
    def train(self):
        """Run training. Returns the HF `TrainOutput` object with metrics/logs."""
        if self.trainer is None:
            raise RuntimeError("Call build_trainer() before train().")
        logger.info("Starting training...")
        result = self.trainer.train()
        logger.info(f"Training finished. Final metrics: {result.metrics}")
        return result

    # ------------------------------------------------------------------
    def save_adapter(self, save_dir: Optional[str] = None) -> Path:
        """Save the LoRA adapter + tokenizer to disk (small, portable)."""
        save_dir = Path(save_dir or self.config.lora_adapter_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))
        logger.info(f"Saved LoRA adapter + tokenizer -> {save_dir}")
        return save_dir

    # ------------------------------------------------------------------
    def get_loss_history(self) -> "pd.DataFrame":
        """Extract per-step training/eval loss history from the trainer's
        log history, for plotting in notebook 05.
        """
        import pandas as pd

        if self.trainer is None:
            raise RuntimeError("No trainer available yet.")
        return pd.DataFrame(self.trainer.state.log_history)
