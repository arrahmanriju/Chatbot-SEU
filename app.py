"""Streamlit frontend for the Chatbot-SEU API."""
import os

import httpx
import streamlit as st

from src.config import load_environment

load_environment()
API = (os.getenv("SEU_API_URL") or "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="Southeast University Assistant", page_icon="🎓")
st.title("Southeast University Assistant")

language = st.sidebar.selectbox("Language / ভাষা", ["English", "বাংলা"])
bn = language == "বাংলা"
st.caption("সাউথইস্ট ইউনিভার্সিটি, বাংলাদেশ সম্পর্কে প্রশ্ন করুন।" if bn else
           "Ask about Southeast University, Bangladesh.")
if "turns" not in st.session_state:
    st.session_state.turns = []
if st.sidebar.button("নতুন কথোপকথন" if bn else "New conversation"):
    st.session_state.turns = []
    st.rerun()


def send(path, payload):
    response = httpx.post(f"{API}/{path}", json=payload, timeout=90)
    response.raise_for_status()
    return response.json()


def render_answer(reply):
    st.write(reply["answer"])
    if reply["citations"]:
        with st.expander("সূত্র" if bn else "Sources"):
            for citation in reply["citations"]:
                title = citation["title"] or ("সূত্র" if bn else "Source")
                if citation.get("page"):
                    page_label = "পৃষ্ঠা" if bn else "page"
                    title = f"{title} ({page_label} {citation['page']})"
                if citation["url"]:
                    st.link_button(title, citation["url"])
                else:
                    st.write(title)
                if citation.get("potentially_outdated"):
                    st.caption("সূত্রটির তথ্য পুরোনো হতে পারে।" if bn else "This source may be outdated.")
    request_id = reply["request_id"]
    if request_id not in st.session_state:
        rating = st.feedback("thumbs", key=f"rating-{request_id}")
        if rating is not None and st.button("জমা দিন" if bn else "Submit feedback", key=f"send-{request_id}"):
            try:
                send("feedback", {"request_id": request_id, "helpful": bool(rating)})
                st.session_state[request_id] = True
                st.rerun()
            except (httpx.HTTPError, ValueError):
                st.error("মতামত জমা হয়নি।" if bn else "Feedback could not be saved.")
    else:
        st.caption("মতামত জমা হয়েছে।" if bn else "Feedback saved.")


for turn in st.session_state.turns:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        render_answer(turn["reply"])

question = st.chat_input("বিশ্ববিদ্যালয় সম্পর্কে প্রশ্ন করুন" if bn else
                         "Ask a question about SEU", max_chars=2000)
if question and question.strip():
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("উত্তর খোঁজা হচ্ছে…" if bn else "Finding an answer…"):
                history = [{"question": turn["question"], "answer": turn["reply"]["answer"][:8000]}
                           for turn in st.session_state.turns[-5:]]
                reply = send("chat", {"question": question, "history": history,
                                      "language": "bn" if bn else "en"})
            st.session_state.turns.append({"question": question, "reply": reply})
            render_answer(reply)
        except (httpx.HTTPError, ValueError):
            st.error("সহকারী এখন উত্তর দিতে পারছে না। আবার চেষ্টা করুন।" if bn else
                     "The assistant is temporarily unavailable. Please try again.")
