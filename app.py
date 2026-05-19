"""
app.py — SmartDesk AI Streamlit chat interface.

Run with:
    streamlit run app.py
"""

import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Load .env for local development (no-op when running on Streamlit Cloud)
load_dotenv()

# ─────────────────────────────────────────────
# Secrets: Streamlit Cloud → st.secrets
#          Local dev       → .env / os.environ
# This block runs before any other st.* calls
# because set_page_config must be the first one.
# We write secrets into os.environ so all
# downstream code (LangChain, OpenAI SDK, etc.)
# picks them up transparently.
# ─────────────────────────────────────────────
_SECRET_KEYS = [
    "OPENAI_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_TRACING_V2",
    "SIMILARITY_THRESHOLD",
    "TICKETS_FILE",
    "FAISS_INDEX_PATH",
]

for _key in _SECRET_KEYS:
    if _key in st.secrets:
        os.environ[_key] = st.secrets[_key]

# ─────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SmartDesk AI",
    page_icon="🖥️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Guard: index must exist
# ─────────────────────────────────────────────
INDEX_FILE = Path(os.getenv("FAISS_INDEX_PATH", "index/faiss_index")).with_suffix(".faiss")

if not INDEX_FILE.exists():
    st.error(
        "Knowledge base index not found. "
        "Please run `python build_index.py` before starting the app.",
        icon="⚠️",
    )
    st.stop()

# ─────────────────────────────────────────────
# Lazy import agent (after index check)
# ─────────────────────────────────────────────
from agent.graph import build_graph, new_session

# ─────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────
if "agent_state" not in st.session_state:
    st.session_state.agent_state = new_session()

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/headset.png",
        width=64,
    )
    st.title("SmartDesk AI")
    st.caption("NovaTech Solutions IT & HR Help Desk")
    st.divider()

    st.markdown("**What I can help with:**")
    st.markdown(
        "- 🔐 Password resets, VPN, MFA\n"
        "- 💻 Software, hardware, email\n"
        "- 🏖️ Leave policies and HR queries\n"
        "- 📋 Raise and track support tickets\n"
        "- 💰 Payroll, tax, and benefits"
    )
    st.divider()

    email = st.session_state.agent_state.get("user_email")
    if email:
        st.success(f"Session email: {email}", icon="✅")

    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.agent_state = new_session()
        st.session_state.display_messages = []
        st.rerun()

    st.divider()
    st.caption("Powered by GPT-4o · LangGraph · FAISS")

# ─────────────────────────────────────────────
# Main header
# ─────────────────────────────────────────────
st.title("🖥️ SmartDesk AI")
st.caption("Your intelligent IT & HR help desk — powered by NovaTech AI")
st.divider()

# ─────────────────────────────────────────────
# Welcome message (first load)
# ─────────────────────────────────────────────
if not st.session_state.display_messages:
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(
            "Hello! I'm **SmartDesk AI**, your NovaTech IT and HR assistant. "
            "I can answer questions about IT support, HR policies, leave, payroll, "
            "and more.\n\n"
            "I can also **raise a support ticket** if I can't find the answer, "
            "and **check the status** of tickets you've already raised.\n\n"
            "How can I help you today?"
        )

# ─────────────────────────────────────────────
# Render conversation history
# ─────────────────────────────────────────────
for msg in st.session_state.display_messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ─────────────────────────────────────────────
# Chat input
# ─────────────────────────────────────────────
if prompt := st.chat_input("Type your question here…"):
    # Show user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.display_messages.append({"role": "user", "content": prompt})

    # Add user message to agent state
    state = st.session_state.agent_state
    state["messages"].append(HumanMessage(content=prompt))

    # Run the graph
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            try:
                result = st.session_state.graph.invoke(state)

                # Update session state with new state
                st.session_state.agent_state = result

                # Extract the last AI message (skip tool messages)
                ai_response = ""
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                        ai_response = msg.content
                        break

                if not ai_response:
                    ai_response = "I'm sorry, I encountered an issue processing your request. Please try again."

                st.markdown(ai_response)
                st.session_state.display_messages.append(
                    {"role": "assistant", "content": ai_response}
                )

                # Update sidebar email display
                if result.get("user_email"):
                    st.session_state.agent_state["user_email"] = result["user_email"]

            except Exception as exc:
                err_msg = (
                    f"I'm sorry, something went wrong: `{exc}`. "
                    "Please try again or contact IT at ext. 2020."
                )
                st.error(err_msg)
                st.session_state.display_messages.append(
                    {"role": "assistant", "content": err_msg}
                )

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🔐 IT Help Desk: ext. 2020")
with col2:
    st.caption("👥 HR Support: ext. 3000")
with col3:
    st.caption("🚨 Emergency: ext. 2021")
