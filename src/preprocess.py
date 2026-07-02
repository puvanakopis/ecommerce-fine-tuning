"""
preprocess.py
==============
Cleaning and preprocessing utilities that turn raw ecommerce support data
into clean instruction/response pairs ready for prompt formatting.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from src.utils import clean_text, get_logger, word_count

logger = get_logger(__name__)

# Common column-name aliases seen across public support datasets.
INSTRUCTION_ALIASES = ["instruction", "question", "query", "customer", "input", "text"]
RESPONSE_ALIASES = ["response", "answer", "response_text", "output", "reply"]


class Preprocessor:
    """Cleans and normalizes raw ecommerce support conversation data."""

    def __init__(self, min_words: int = 2, max_words: int = 512):
        self.min_words = min_words
        self.max_words = max_words

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
        """Find the first matching column name from a list of aliases."""
        for alias in aliases:
            if alias in df.columns:
                return alias
        return None

    # ------------------------------------------------------------------
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename whichever instruction/response-like columns exist to
        canonical `instruction` / `response` names.

        Raises:
            ValueError: if neither an instruction-like nor response-like
                column can be found.
        """
        instr_col = self._resolve_column(df, INSTRUCTION_ALIASES)
        resp_col = self._resolve_column(df, RESPONSE_ALIASES)
        if instr_col is None or resp_col is None:
            raise ValueError(
                f"Could not resolve instruction/response columns from {list(df.columns)}"
            )
        out = df.rename(columns={instr_col: "instruction", resp_col: "response"})
        keep_cols = [c for c in ["instruction", "response", "category"] if c in out.columns]
        return out[keep_cols].copy()

    # ------------------------------------------------------------------
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop exact and near-duplicate rows based on instruction+response."""
        before = len(df)
        df = df.drop_duplicates(subset=["instruction", "response"]).reset_index(drop=True)
        logger.info(f"Removed {before - len(df)} duplicate rows ({before} -> {len(df)}).")
        return df

    # ------------------------------------------------------------------
    def clean_columns(self, df: pd.DataFrame, lowercase: bool = False) -> pd.DataFrame:
        """Apply full text cleaning (HTML/URL/unicode/noise/whitespace) to
        the instruction and response columns.
        """
        df = df.copy()
        df["instruction"] = df["instruction"].apply(lambda t: clean_text(t, lowercase=lowercase))
        df["response"] = df["response"].apply(lambda t: clean_text(t, lowercase=lowercase))
        return df

    # ------------------------------------------------------------------
    def remove_empty_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows where instruction or response is empty after cleaning."""
        before = len(df)
        df = df[
            (df["instruction"].str.len() > 0) & (df["response"].str.len() > 0)
        ].reset_index(drop=True)
        logger.info(f"Removed {before - len(df)} empty rows ({before} -> {len(df)}).")
        return df

    # ------------------------------------------------------------------
    def filter_by_length(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows whose instruction or response is too short/too long."""
        before = len(df)

        def ok(row):
            iw, rw = word_count(row["instruction"]), word_count(row["response"])
            return self.min_words <= iw <= self.max_words and self.min_words <= rw <= self.max_words

        df = df[df.apply(ok, axis=1)].reset_index(drop=True)
        logger.info(f"Removed {before - len(df)} rows outside length bounds ({before} -> {len(df)}).")
        return df

    # ------------------------------------------------------------------
    def run(self, df: pd.DataFrame, lowercase: bool = False) -> pd.DataFrame:
        """Run the full preprocessing pipeline end-to-end.

        Steps: standardize columns -> dedup -> clean text -> drop empties
        -> filter by length -> dedup again (post-clean duplicates can
        appear once noise/casing is normalized).

        Args:
            df: Raw DataFrame.
            lowercase: Whether to lowercase cleaned text.

        Returns:
            Fully cleaned instruction/response DataFrame.
        """
        logger.info(f"Starting preprocessing on {len(df)} raw rows.")
        df = self.standardize_columns(df)
        df = self.remove_duplicates(df)
        df = self.clean_columns(df, lowercase=lowercase)
        df = self.remove_empty_rows(df)
        df = self.filter_by_length(df)
        df = self.remove_duplicates(df)
        logger.info(f"Preprocessing complete: {len(df)} clean rows remain.")
        return df

    # ------------------------------------------------------------------
    def build_instruction_response_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the DataFrame is in a clean `instruction` / `response`
        pair format ready for prompt_template.py. This is effectively an
        alias for `run()` kept for notebook readability.
        """
        return self.run(df)
