"""
Streamlit Frontend — Healthcare KPI RAG Chatbot
Doctor-specific analytics assistant with multi-turn conversation support.
"""

import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare KPI Assistant",
    page_icon="🏥",
    layout="centered",
)

st.title("🏥 Healthcare KPI Assistant")
st.caption("Ask me about doctor performance, patient metrics, revenue trends, and more.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Documents to retrieve", min_value=1, max_value=5, value=3)
    show_sources = st.toggle("Show sources", value=True)
    show_latency = st.toggle("Show latency", value=True)

    st.markdown("---")
    st.markdown("**Example questions:**")
    examples = [
        "What is Dr. Smith's patient wait time this month?",
        "Which doctor has the highest satisfaction score?",
        "What is the clinic's total revenue for May 2026?",
        "How many exams did Dr. Johnson complete?",
        "What's the appointment utilization rate?",
        "Which departments improved their no-show rate?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["prefill"] = ex

    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄 Rebuild Index", use_container_width=True):
        with st.spinner("Rebuilding vector index..."):
            resp = requests.post(f"{API_URL}/rebuild-index")
            if resp.status_code == 200:
                st.success(f"Index rebuilt: {resp.json()['documents_indexed']} docs")
            else:
                st.error("Failed to rebuild index.")

# ── Chat State ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Chat History Display ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources") and show_sources:
            with st.expander("📎 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")
        if msg.get("latency_ms") and show_latency:
            st.caption(f"⚡ {msg['latency_ms']} ms | 🔤 {msg.get('tokens_used', '?')} tokens")

# ── Input ─────────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", "")
user_input = st.chat_input("Ask about any doctor or clinic KPI...") or prefill

if user_input:
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build history payload (last 6 messages)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-7:-1]
    ]

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Retrieving KPI data..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "question": user_input,
                        "top_k": top_k,
                        "chat_history": history,
                    },
                    timeout=30,
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    latency = data["latency_ms"]
                    tokens = data["tokens_used"]

                    st.markdown(answer)
                    if show_sources and sources:
                        with st.expander("📎 Sources"):
                            for src in sources:
                                st.markdown(f"- {src}")
                    if show_latency:
                        st.caption(f"⚡ {latency} ms | 🔤 {tokens} tokens")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "latency_ms": latency,
                        "tokens_used": tokens,
                    })
                else:
                    err = f"API error {response.status_code}: {response.text}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})

            except requests.exceptions.ConnectionError:
                msg = "⚠️ Cannot connect to API. Make sure the FastAPI server is running (`uvicorn api.main:app --reload`)."
                st.error(msg)
