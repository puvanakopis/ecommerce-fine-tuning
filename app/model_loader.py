"""
model_loader.py
================
Cached model-loading helpers for the Streamlit app. Kept separate from
`chatbot.py` so `st.cache_resource` has a stable function signature.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make the project root importable when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.inference import EcommerceSupportBot


@st.cache_resource(show_spinner="Loading Ecommerce Support model... (first load can take a minute)")
def load_bot(prefer_merged: bool = True) -> EcommerceSupportBot:
    """Load and cache the `EcommerceSupportBot`, preferring a merged model
    if one exists on disk, falling back to base+adapter, and finally to a
    warning stub if no fine-tuned weights are present yet (so the UI still
    boots for local development).

    Args:
        prefer_merged: Try the merged model directory first.

    Returns:
        A ready-to-use `EcommerceSupportBot`.
    """
    cfg = Config()
    bot = EcommerceSupportBot(cfg)

    merged_has_weights = any(cfg.merged_model_dir.glob("*.safetensors")) or any(
        cfg.merged_model_dir.glob("*.bin")
    )
    adapter_has_weights = any(cfg.lora_adapter_dir.glob("adapter_model.*"))

    try:
        if prefer_merged and merged_has_weights:
            bot.load_merged_model()
        elif adapter_has_weights:
            bot.load_adapter_model()
        elif merged_has_weights:
            bot.load_merged_model()
        else:
            st.warning(
                "No fine-tuned weights found in models/. Run notebooks 05 & 07 "
                "first, or point Config at a Hugging Face Hub repo. "
                "Falling back to the base instruct model (untuned)."
            )
            bot.load_adapter_model()  # will still load the base model even w/o an adapter dir with weights
        return bot
    except Exception as e:  # noqa: BLE001
        st.error(f"Failed to load model: {e}")
        raise
