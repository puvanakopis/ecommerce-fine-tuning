"""
streamlit_app.py
=================
Professional Streamlit chatbot UI for the fine-tuned Ecommerce Customer
Support Assistant.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chatbot import get_bot_reply, extract_pdf_context
from app.model_loader import load_bot

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Ecommerce Support Assistant",
    page_icon="🛍️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main { background-color: #fafafa; }
    .stChatMessage { border-radius: 12px; }
    .assistant-badge {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        background: #eef2ff; color: #4338ca; font-size: 0.75rem; font-weight: 600;
        margin-bottom: 0.5rem;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛍️ Support Assistant")
    st.caption("Fine-tuned open-source LLM · QLoRA · Ecommerce support")

    st.markdown("### Generation settings")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05, help="Higher = more creative/varied replies.")
    max_new_tokens = st.slider("Max new tokens", 32, 512, 256, 16, help="Upper bound on reply length.")

    st.markdown("### Store FAQ (optional)")
    uploaded_pdf = st.file_uploader("Upload a FAQ PDF for grounded answers", type=["pdf"])
    faq_context = None
    if uploaded_pdf is not None:
        with st.spinner("Reading FAQ PDF..."):
            faq_context = extract_pdf_context(uploaded_pdf.read())
        st.success(f"Loaded {len(faq_context)} chars of FAQ context.")

    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption(
        "This assistant answers questions about orders, shipping, refunds, "
        "returns, payments, coupons, delivery, and account management."
    )

# ---------------------------------------------------------------------------
# Load model (cached)
# ---------------------------------------------------------------------------
bot = load_bot()

# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! 👋 I'm your Ecommerce Support Assistant. Ask me about "
            "orders, shipping, refunds, returns, payments, coupons, or your account.",
        }
    ]

st.title("Ecommerce Customer Support Assistant")
st.markdown('<span class="assistant-badge">🤖 Fine-tuned LLM · QLoRA</span>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Render history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
user_input = st.chat_input("Type your question, e.g. 'Where is my order #12345?'")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Thinking..."):
            reply = get_bot_reply(
                bot,
                user_input,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                faq_context=faq_context,
            )

        # Simple typing animation
        displayed = ""
        for word in reply.split(" "):
            displayed += word + " "
            placeholder.markdown(displayed + "▌")
            time.sleep(0.02)
        placeholder.markdown(displayed.strip())

    st.session_state.messages.append({"role": "assistant", "content": reply})
