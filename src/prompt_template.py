"""
prompt_template.py
====================
Converts clean instruction/response pairs into the training prompt format,
and provides the matching inference-time prompt builder so training and
inference never drift apart.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.config import Config


class PromptFormatter:
    """Builds instruction-formatted prompts for training and inference."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    # ------------------------------------------------------------------
    def format_training_example(self, instruction: str, response: str) -> str:
        """Build a single full training string (instruction + response +
        EOS-ready) in the project's instruction/response format.

        Example:
            ### Instruction:
            Customer:
            Where is my order?

            ### Response:
            Your order has been shipped ...
        """
        cfg = self.config
        return (
            f"{cfg.instruction_header}\n"
            f"Customer:\n{instruction.strip()}\n\n"
            f"{cfg.response_header}\n"
            f"{response.strip()}"
        )

    # ------------------------------------------------------------------
    def format_prompt_only(self, instruction: str) -> str:
        """Build the prompt half only (no response) — used at inference
        time; the model generates everything after the response header.
        """
        cfg = self.config
        return f"{cfg.instruction_header}\n" f"Customer:\n{instruction.strip()}\n\n" f"{cfg.response_header}\n"

    # ------------------------------------------------------------------
    def format_chat_messages(self, instruction: str, response: Optional[str] = None) -> list[dict]:
        """Build a chat-style messages list, for use with
        `tokenizer.apply_chat_template()` on Llama-3.2 / Qwen2.5-style models.

        Args:
            instruction: The customer's message.
            response: If provided, appended as the assistant turn (training).
                If None, the messages list ends after the user turn
                (inference — the model will generate the assistant turn).
        """
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": instruction.strip()},
        ]
        if response is not None:
            messages.append({"role": "assistant", "content": response.strip()})
        return messages

    # ------------------------------------------------------------------
    def add_text_column(self, df: pd.DataFrame, style: str = "instruction") -> pd.DataFrame:
        """Add a `text` column with the fully formatted training string
        for every row, ready to be wrapped in a HF `Dataset` for TRL's
        `SFTTrainer`.

        Args:
            df: DataFrame with `instruction` and `response` columns.
            style: "instruction" for the raw ### Instruction/### Response
                format, or "chat" to store chat-message lists (useful if
                you'll call `apply_chat_template` inside the tokenizing step).

        Returns:
            DataFrame with an added `text` (or `messages`) column.
        """
        df = df.copy()
        if style == "instruction":
            df["text"] = df.apply(
                lambda r: self.format_training_example(r["instruction"], r["response"]), axis=1
            )
        elif style == "chat":
            df["messages"] = df.apply(
                lambda r: self.format_chat_messages(r["instruction"], r["response"]), axis=1
            )
        else:
            raise ValueError(f"Unknown style: {style!r}. Use 'instruction' or 'chat'.")
        return df

    # ------------------------------------------------------------------
    def tokenize_lengths(self, df: pd.DataFrame, tokenizer, text_col: str = "text") -> pd.DataFrame:
        """Add a `token_length` column using the given tokenizer — useful
        for picking `max_seq_length` and for the token-distribution plot
        in notebook 02/04.
        """
        df = df.copy()
        df["token_length"] = df[text_col].apply(lambda t: len(tokenizer(t)["input_ids"]))
        return df
