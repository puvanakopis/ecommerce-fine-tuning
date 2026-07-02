"""
chatbot.py
==========
Chatbot orchestration logic used by streamlit_app.py: turns a user
message + generation settings into a bot reply, with a lightweight
"is this an ecommerce question?" guardrail and optional FAQ-PDF context.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from src.inference import EcommerceSupportBot

# Keywords that signal an on-topic ecommerce support question. This is a
# cheap, transparent guardrail layered on top of the model's own system
# prompt (which already instructs it to decline unrelated topics) — belt
# and suspenders for a portfolio-quality demo.
ECOMMERCE_KEYWORDS = [
    "order", "ship", "deliver", "refund", "return", "exchange", "payment",
    "pay", "coupon", "discount", "promo", "account", "password", "cart",
    "checkout", "tracking", "track", "invoice", "receipt", "cancel",
    "subscription", "warranty", "product", "item", "package", "address",
    "size", "stock", "availability", "price", "review", "wishlist",
]

OFF_TOPIC_REPLY = (
    "I'm your Ecommerce Customer Support Assistant, so I'm only able to help "
    "with questions about orders, shipping, refunds, returns, payments, "
    "coupons, delivery, or your account. Could you rephrase your question "
    "around one of those topics?"
)


def looks_ecommerce_related(message: str) -> bool:
    """Cheap keyword-based topic check (defense-in-depth alongside the
    model's own system-prompt instructions).

    Args:
        message: The user's raw message.

    Returns:
        True if the message contains at least one ecommerce-support signal
        word, or is short enough that it's likely a greeting/follow-up.
    """
    text = message.lower()
    if len(text.split()) <= 3:
        return True  # greetings / short follow-ups, let the model handle it
    return any(re.search(rf"\b{kw}\b", text) for kw in ECOMMERCE_KEYWORDS)


def extract_pdf_context(pdf_bytes: bytes, max_chars: int = 3000) -> str:
    """Extract text from an uploaded FAQ PDF to optionally prepend as
    extra context for the model.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF.
        max_chars: Truncate extracted text to this many characters.

    Returns:
        Extracted plain text (possibly truncated).
    """
    from pypdf import PdfReader
    import io

    reader = PdfReader(io.BytesIO(pdf_bytes))
    text_parts = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(text_parts).strip()
    return full_text[:max_chars]


def build_message_with_context(message: str, faq_context: Optional[str]) -> str:
    """Optionally prepend uploaded FAQ context to the user's message so
    the model can ground its answer in store-specific policy text.
    """
    if not faq_context:
        return message
    return (
        f"Store FAQ reference (use only if relevant, otherwise ignore):\n"
        f"{faq_context}\n\n"
        f"Customer question:\n{message}"
    )


def get_bot_reply(
    bot: EcommerceSupportBot,
    message: str,
    temperature: float,
    max_new_tokens: int,
    faq_context: Optional[str] = None,
) -> str:
    """Produce a single bot reply for the chat UI.

    Args:
        bot: A loaded `EcommerceSupportBot`.
        message: The user's message.
        temperature: Sampling temperature from the UI slider.
        max_new_tokens: Max generation length from the UI slider.
        faq_context: Optional extracted FAQ PDF text for grounding.

    Returns:
        The assistant's reply text.
    """
    if not looks_ecommerce_related(message):
        return OFF_TOPIC_REPLY

    full_message = build_message_with_context(message, faq_context)
    return bot.generate(
        full_message,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )


def format_history_for_display(history: List[Tuple[str, str]]) -> List[dict]:
    """Convert (role, content) tuples into Streamlit chat-message dicts."""
    return [{"role": role, "content": content} for role, content in history]
