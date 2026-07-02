"""
inference.py
=============
Loads a fine-tuned ecommerce support model (base + LoRA adapter, or a
merged model) and exposes a simple `generate()` interface used by
notebook 07 and the Streamlit app.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from src.config import Config
from src.prompt_template import PromptFormatter
from src.utils import get_logger, detect_device

logger = get_logger(__name__)


class EcommerceSupportBot:
    """Wraps a fine-tuned LLM for ecommerce customer-support inference."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.formatter = PromptFormatter(self.config)
        self.model = None
        self.tokenizer = None
        self._loaded_from: Optional[str] = None

    # ------------------------------------------------------------------
    def load_merged_model(self, model_dir: Optional[Union[str, Path]] = None):
        """Load a fully merged (base + LoRA folded in) model for standalone
        inference — no PEFT/Unsloth dependency required at serve time.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        model_dir = str(model_dir or self.config.merged_model_dir)
        logger.info(f"Loading merged model from {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16 if detect_device() == "cuda" else torch.float32,
        )
        self._loaded_from = "merged"
        return self.model, self.tokenizer

    # ------------------------------------------------------------------
    def load_adapter_model(
        self,
        base_model_name: Optional[str] = None,
        adapter_dir: Optional[Union[str, Path]] = None,
    ):
        """Load the base model + LoRA adapter directly with Unsloth
        (fast path, recommended on Colab GPU for immediate testing right
        after training, without a separate merge step).
        """
        from unsloth import FastLanguageModel

        cfg = self.config
        base_model_name = base_model_name or cfg.base_model_name
        adapter_dir = str(adapter_dir or cfg.lora_adapter_dir)

        logger.info(f"Loading base model '{base_model_name}' + adapter '{adapter_dir}'")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit,
        )
        model.load_adapter(adapter_dir)
        FastLanguageModel.for_inference(model)  # enable Unsloth's fast generation path

        self.model, self.tokenizer = model, tokenizer
        self._loaded_from = "adapter"
        return model, tokenizer

    # ------------------------------------------------------------------
    def merge_and_save(self, save_dir: Optional[Union[str, Path]] = None) -> Path:
        """Merge the currently loaded LoRA adapter into the base weights
        and save a standalone model (for deployment without PEFT/Unsloth).
        """
        if self.model is None:
            raise RuntimeError("Load a base+adapter model first via load_adapter_model().")
        save_dir = Path(save_dir or self.config.merged_model_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Merging LoRA weights and saving to {save_dir} ...")
        merged = self.model.merge_and_unload() if hasattr(self.model, "merge_and_unload") else self.model
        merged.save_pretrained(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))
        logger.info("Merge complete.")
        return save_dir

    # ------------------------------------------------------------------
    def generate(
        self,
        instruction: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        use_chat_template: bool = True,
    ) -> str:
        """Generate a support response for a single customer message.

        Args:
            instruction: The customer's question/message.
            max_new_tokens: Generation length cap (defaults from config).
            temperature: Sampling temperature (defaults from config).
            top_p: Nucleus sampling parameter (defaults from config).
            use_chat_template: If the tokenizer has a chat template
                (Llama-3/Qwen2.5 instruct models), use it; otherwise fall
                back to the raw ### Instruction/### Response format.

        Returns:
            The generated response text (assistant turn only).
        """
        import torch

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model loaded. Call load_merged_model() or load_adapter_model().")

        cfg = self.config
        max_new_tokens = max_new_tokens or cfg.default_max_new_tokens
        temperature = temperature if temperature is not None else cfg.default_temperature
        top_p = top_p if top_p is not None else cfg.default_top_p

        if use_chat_template and getattr(self.tokenizer, "chat_template", None):
            messages = self.formatter.format_chat_messages(instruction)
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)
            attention_mask = torch.ones_like(input_ids)
        else:
            prompt = self.formatter.format_prompt_only(instruction)
            enc = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 1e-3),
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[-1] :]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()

    # ------------------------------------------------------------------
    def chat(self, history: list[tuple[str, str]], new_message: str, **gen_kwargs) -> str:
        """Simple multi-turn helper: ignores prior turns for the *model*
        input (single-turn instruct model) but keeps `history` for UI
        display purposes in the Streamlit app.
        """
        return self.generate(new_message, **gen_kwargs)
