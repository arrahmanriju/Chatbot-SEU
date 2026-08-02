import os
import sys

import streamlit as st

# allow imports from src/
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from src.filter import load_vectorstore
from src.chain import run_chatbot

st.set_page_config(page_title="SEU Chatbot", page_icon="🎓")
st.title("🎓 Southeast University Chatbot")
st.caption("Ask me anything about Southeast University, Bangladesh")

# Load vectorstore once and cache it across reruns
@st.cache_resource
def get_vectorstore():
    return load_vectorstore()

vectorstore = get_vectorstore()

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