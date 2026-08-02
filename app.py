import os
import sys
from pathlib import Path

import streamlit as st

# allow imports from src/
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from src.filter import load_vectorstore
from src.chain import run_chatbot

st.set_page_config(page_title="SEU Chatbot", page_icon="🎓")
st.title("🎓 Southeast University Chatbot")
st.caption("Ask me anything about Southeast University, Bangladesh")


@st.cache_resource
def get_vectorstore():
    return load_vectorstore()


try:
    vectorstore = get_vectorstore()
except Exception as exc:
    st.error(f"The knowledge base could not be loaded: {exc}")
    st.stop()

# Chat history stored as list of (user_msg, bot_msg) tuples
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Render past messages
for user_msg, bot_msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(user_msg)
    with st.chat_message("assistant"):
        st.write(bot_msg)

# Chat input
user_question = st.chat_input("Ask a question about SEU...")

if user_question:
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = run_chatbot(
                user_question,
                vectorstore,
                st.session_state.chat_history,
            )
        st.write(answer)

    st.session_state.chat_history.append((user_question, answer))