import logging

from src.filter import classify_message
from src.llm import ask_llm
from src.prompt_builder import build_prompt
from src.retriever import retrieve_top_chunks

logger = logging.getLogger(__name__)

GREETING_REPLY = "Hi! I'm the SEU assistant. Ask me about Southeast University."
META_REPLY = "I can answer questions about Southeast University in Bangladesh."
OFF_TOPIC_REPLY = "I can help with questions about Southeast University."
UNVERIFIED_REPLY = "I could not verify the answer from the available university information."
ERROR_REPLY = "I couldn’t generate a response right now. Please try again."


def run_chatbot(question: str, vectorstore, chat_history: list | None = None) -> str:
    question = question.strip()
    if not question:
        return "Please enter a question about Southeast University."
    category = classify_message(question)
    if category == "greeting":
        return GREETING_REPLY
    if category == "meta":
        return META_REPLY

    try:
        chunks = retrieve_top_chunks(question, vectorstore)
        if not chunks:
            return UNVERIFIED_REPLY
        if chunks[0][1] < 0.1:
            return OFF_TOPIC_REPLY
        return ask_llm(build_prompt(question, chunks, chat_history))
    except Exception as exc:
        logger.error("Chat request failed (%s)", type(exc).__name__)
        return ERROR_REPLY
