"""Run Chatbot-SEU directly in Streamlit or against an optional HTTP backend."""
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import httpx
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from src.config import load_environment
from src.runtime import create_retriever, create_service, load_or_build_index, save_feedback
from src.store import INDEX_FILE, index_directory
from src.student import CATEGORIES, DATA, private_request, safe_question

load_environment()
API = (os.getenv("SEU_API_URL") or "").strip().rstrip("/")
logger = logging.getLogger(__name__)
st.set_page_config(page_title="Southeast University Assistant", page_icon="🎓")
st.title("Southeast University Assistant")

language = st.sidebar.selectbox("Language / ভাষা", ["Auto", "English", "বাংলা"])
bn = language == "বাংলা"
st.caption("সাউথইস্ট ইউনিভার্সিটি, বাংলাদেশ সম্পর্কে প্রশ্ন করুন।" if bn else
           "Get help with registration, fees, campus services, and academic questions.")
st.sidebar.caption("পাসওয়ার্ড বা যাচাই কোড শেয়ার করবেন না।" if bn else "Keep passwords and verification codes private.")
category = st.sidebar.selectbox("প্রশ্নের বিষয়" if bn else "Question category", list(CATEGORIES))
example = None
with (DATA / "student_questions.jsonl").open(encoding="utf-8") as questions:
    options = [row["question"] for line in questions if line.strip()
               for row in [json.loads(line)]
               if row["category"] == category and row["language"] == ("bn" if bn else "en")
               and not row["requires_personal_student_data"]][:2]
for option in options:
    if st.sidebar.button(option):
        example = option

if "turns" not in st.session_state:
    st.session_state.turns = []
if st.sidebar.button("নতুন কথোপকথন" if bn else "New conversation"):
    st.session_state.turns = []
    st.rerun()


@st.cache_resource(show_spinner=False, max_entries=1)
def standalone_service(directory: Path):
    return create_service(create_retriever(load_or_build_index(directory)))


service = None
if not API:
    if not os.getenv("GROQ_API_KEY", "").strip():
        try:
            key = st.secrets.get("GROQ_API_KEY")
        except StreamlitSecretNotFoundError:
            key = None
        if isinstance(key, str) and key.strip():
            os.environ["GROQ_API_KEY"] = key.strip()
    if not os.getenv("GROQ_API_KEY", "").strip():
        st.warning("Set GROQ_API_KEY in Streamlit secrets or the environment to enable generated answers.")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    directory = index_directory()
    building = not (directory / INDEX_FILE).is_file()
    try:
        with st.spinner("Preparing the university knowledge base?" if building else "Loading the knowledge base?"):
            service = standalone_service(directory)
    except Exception as exc:
        logger.error("Knowledge-base startup failed (%s)", type(exc).__name__)
        st.error("The university knowledge base could not be built or loaded. Check the deployment logs, "
                 "available memory, model download access, and the bundled JSONL files.")
        st.stop()


def send(path, payload):
    if API:
        response = httpx.post(f"{API}/{path}", json=payload, timeout=90)
        response.raise_for_status()
        return response.json()

    if path == "feedback":
        save_feedback(payload["request_id"], payload["helpful"])
        return {"saved": True}

    history = [(turn["question"], turn["answer"]) for turn in payload["history"]]
    answer = service.chat(payload["question"], history, payload["language"])
    return {"request_id": str(uuid4()), **asdict(answer)}


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
            except Exception as exc:
                logger.error("Feedback request failed (%s)", type(exc).__name__)
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
question = question or example
if question and question.strip():
    display_question = safe_question(question)
    history_question = "[private student record request]" if private_request(question) else display_question
    with st.chat_message("user"):
        st.write(display_question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("উত্তর খোঁজা হচ্ছে…" if bn else "Finding an answer…"):
                history = [{"question": turn["question"], "answer": turn["reply"]["answer"][:8000]}
                           for turn in st.session_state.turns[-5:]]
                reply = send("chat", {"question": question, "history": history,
                                      "language": {"Auto": "auto", "English": "en", "বাংলা": "bn"}[language]})
            st.session_state.turns.append({"question": history_question, "reply": reply})
            render_answer(reply)
        except Exception as exc:
            logger.error("Chat request failed (%s)", type(exc).__name__)
            st.error("সহকারী এখন উত্তর দিতে পারছে না। আবার চেষ্টা করুন।" if bn else
                     "The assistant is temporarily unavailable. Please try again.")
