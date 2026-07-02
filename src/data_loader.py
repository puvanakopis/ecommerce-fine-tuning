"""
data_loader.py
===============
Handles downloading (or loading from disk) the ecommerce customer support
dataset, plus train/val/test splitting utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.config import Config
from src.utils import get_logger, save_jsonl, load_jsonl

logger = get_logger(__name__)


class EcommerceDataLoader:
    """Loads, splits, and persists the ecommerce customer-support dataset."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.config.ensure_dirs()

    # ------------------------------------------------------------------
    def download_from_hub(self, dataset_name: Optional[str] = None) -> pd.DataFrame:
        """Download the dataset from Hugging Face Hub as a pandas DataFrame.

        Falls back to raising a clear error if the `datasets` library or
        network access is unavailable — callers should catch this and use
        `build_synthetic_fallback()` instead (see notebook 01).

        Args:
            dataset_name: Optional override of the HF dataset repo id.

        Returns:
            A pandas DataFrame with at least `instruction`/`question` and
            `response`/`answer` columns (raw, un-normalized).
        """
        from datasets import load_dataset

        name = dataset_name or self.config.hf_dataset_name
        logger.info(f"Downloading dataset '{name}' from Hugging Face Hub...")
        ds = load_dataset(name)
        split = "train" if "train" in ds else list(ds.keys())[0]
        df = ds[split].to_pandas()
        logger.info(f"Downloaded {len(df):,} rows, columns={list(df.columns)}")
        return df

    # ------------------------------------------------------------------
    def build_synthetic_fallback(self, n: int = 500) -> pd.DataFrame:
        """Build a small, realistic synthetic ecommerce-support dataset.

        Used only if the Hugging Face download fails (e.g. no internet in
        a restricted Colab session), so the pipeline never hard-blocks.

        Args:
            n: Number of synthetic examples to generate.

        Returns:
            DataFrame with `instruction` and `response` columns.
        """
        import random

        random.seed(self.config.seed)

        topics = {
            "order tracking": [
                ("Where is my order #{oid}?",
                 "Your order #{oid} has shipped and is expected to arrive within 3-5 business days. You can track it using the link in your confirmation email."),
                ("How can I track my package?",
                 "You can track your package by logging into your account and visiting the 'My Orders' section, or using the tracking number sent to your email."),
            ],
            "refunds": [
                ("I want a refund for my last purchase.",
                 "I'm sorry to hear that. Refunds are processed within 5-7 business days once we receive the returned item. Would you like me to start the return process?"),
                ("When will I get my refund?",
                 "Refunds typically appear in your original payment method within 5-7 business days after we process your return."),
            ],
            "returns": [
                ("How do I return an item?",
                 "You can return an item within 30 days of delivery. Go to 'My Orders', select the item, and click 'Return Item' to print a prepaid label."),
                ("Can I exchange a product instead of returning it?",
                 "Yes, exchanges are available for items in original condition within 30 days. Select 'Exchange' instead of 'Return' in your order history."),
            ],
            "shipping": [
                ("How long does shipping take?",
                 "Standard shipping takes 3-5 business days. Expedited shipping (1-2 days) is available at checkout for an additional fee."),
                ("Do you ship internationally?",
                 "Yes, we ship to over 50 countries. International orders typically take 7-14 business days depending on the destination."),
            ],
            "payment issues": [
                ("My payment was declined, what should I do?",
                 "Please double check your card details and billing address. If the issue persists, try an alternate payment method or contact your bank."),
                ("Can I pay with PayPal?",
                 "Yes, we accept PayPal, major credit/debit cards, and select digital wallets at checkout."),
            ],
            "coupons": [
                ("My coupon code isn't working.",
                 "Coupon codes are case-sensitive and may have minimum purchase requirements or expiration dates. Could you share the code so I can check it for you?"),
                ("Do you have any active discounts?",
                 "You can find current promotions on our homepage banner or by subscribing to our newsletter for exclusive codes."),
            ],
            "account management": [
                ("How do I reset my password?",
                 "Click 'Forgot Password' on the login page and follow the link sent to your registered email to reset it."),
                ("How can I update my shipping address?",
                 "Go to 'Account Settings' > 'Addresses' to add, edit, or remove a shipping address at any time."),
            ],
            "delivery": [
                ("My package says delivered but I never received it.",
                 "I'm sorry for the trouble. Please check with neighbors or your building's front desk first. If it's still missing after 24 hours, contact us and we'll open an investigation."),
            ],
        }

        rows = []
        keys = list(topics.keys())
        for i in range(n):
            topic = keys[i % len(keys)]
            q_template, a_template = topics[topic][i % len(topics[topic])]
            oid = f"{100000 + i}"
            rows.append(
                {
                    "instruction": q_template.format(oid=oid),
                    "response": a_template.format(oid=oid),
                    "category": topic,
                }
            )
        random.shuffle(rows)
        df = pd.DataFrame(rows)
        logger.info(f"Built synthetic fallback dataset with {len(df)} rows.")
        return df

    # ------------------------------------------------------------------
    def get_dataset(self) -> pd.DataFrame:
        """Try Hugging Face first, fall back to synthetic data on failure."""
        try:
            return self.download_from_hub()
        except Exception as e:  # noqa: BLE001 - deliberate broad catch w/ fallback
            logger.warning(f"HF download failed ({e}); using synthetic fallback dataset.")
            return self.build_synthetic_fallback()

    # ------------------------------------------------------------------
    @staticmethod
    def split_dataset(
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Shuffle and split a DataFrame into train/val/test sets.

        Args:
            df: Input DataFrame.
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation (test = remainder).
            seed: Random seed for the shuffle.

        Returns:
            (train_df, val_df, test_df) tuple, each with a reset index.
        """
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_df = df.iloc[:n_train].reset_index(drop=True)
        val_df = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
        test_df = df.iloc[n_train + n_val :].reset_index(drop=True)
        logger.info(
            f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
        )
        return train_df, val_df, test_df

    # ------------------------------------------------------------------
    def save_splits(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        """Persist train/val/test splits as JSONL under data/processed/."""
        out_dir = self.config.processed_data_dir
        save_jsonl(train_df.to_dict(orient="records"), out_dir / "train.jsonl")
        save_jsonl(val_df.to_dict(orient="records"), out_dir / "val.jsonl")
        save_jsonl(test_df.to_dict(orient="records"), out_dir / "test.jsonl")

    def load_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load previously saved train/val/test JSONL splits."""
        out_dir = self.config.processed_data_dir
        train_df = pd.DataFrame(load_jsonl(out_dir / "train.jsonl"))
        val_df = pd.DataFrame(load_jsonl(out_dir / "val.jsonl"))
        test_df = pd.DataFrame(load_jsonl(out_dir / "test.jsonl"))
        return train_df, val_df, test_df
