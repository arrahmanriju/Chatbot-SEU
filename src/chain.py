try:
    from src.filter import load_vectorstore, classify_message, is_university_question
    from src.retriever import retrieve_top_chunks
    from src.prompt_builder import build_prompt
    from src.llm import ask_llm
except ModuleNotFoundError:
    from filter import load_vectorstore, classify_message, is_university_question
    from retriever import retrieve_top_chunks
    from prompt_builder import build_prompt
    from llm import ask_llm

GREETING_REPLY = "Hi! I'm the SEU assistant. Ask me anything about Southeast University — programs, admissions, faculty, and more."
META_REPLY = "I can answer questions about Southeast University (SEU) in Bangladesh — things like academic programs, departments, admissions, vision/mission, and faculty. What would you like to know?"
OFF_TOPIC_REPLY = "Sorry, only Southeast University related questions are supported."


def run_chatbot(question: str, vectorstore, chat_history: list = None):
    """
    Full query-time pipeline:
    classify -> (greeting/meta short-circuit) OR (filter -> retrieve -> prompt -> LLM)
    """
    category = classify_message(question)

    if category == "greeting":
        return GREETING_REPLY

    if category == "meta":
        return META_REPLY

    if not is_university_question(question, vectorstore):
        return OFF_TOPIC_REPLY

    chunks = retrieve_top_chunks(question, vectorstore, k=5)
    prompt = build_prompt(question, chunks, chat_history)

    try:
        reply = ask_llm(prompt)
    except Exception as exc:
        return f"Sorry, I couldn’t generate a response right now. {exc}"

    return reply


if __name__ == "__main__":
    vectorstore = load_vectorstore()
    chat_history = []

    test_questions = [
        "hi",
        "What programs does SEU offer?",
        "What information do you have?",
        "What is the capital of France?",
        "Tell me about SEU's vision.",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        answer = run_chatbot(q, vectorstore, chat_history)
        print(f"A: {answer}")
        chat_history.append((q, answer))