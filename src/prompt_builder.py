def build_prompt(question: str, retrieved_chunks, chat_history: list = None) -> str:
    """
    Combines retrieved context chunks + chat history + the user's question
    into a single prompt string for the LLM.

    retrieved_chunks: list of (Document, score) tuples from retriever.py
    chat_history: list of (user_msg, bot_msg) tuples, most recent last
    """
    context_text = "\n\n".join(
        f"[Source: {doc.metadata.get('title', 'Unknown')}]\n{doc.page_content}"
        for doc, score in retrieved_chunks
    )

    history_text = ""
    if chat_history:
        history_lines = []
        for user_msg, bot_msg in chat_history[-5:]:  # last 5 exchanges max
            history_lines.append(f"User: {user_msg}")
            history_lines.append(f"Assistant: {bot_msg}")
        history_text = "\n".join(history_lines)

    prompt = f"""You are a helpful assistant for Southeast University (SEU) in Bangladesh.
Answer the user's question using ONLY the context provided below. If the answer
isn't in the context, say you don't have that information rather than guessing.

Context:
{context_text}

Previous conversation:
{history_text if history_text else "(none)"}

Current question: {question}

Answer:"""

    return prompt


if __name__ == "__main__":
    from filter import load_vectorstore
    from retriever import retrieve_top_chunks

    vectorstore = load_vectorstore()
    question = "What programs does SEU offer?"
    chunks = retrieve_top_chunks(question, vectorstore, k=3)

    prompt = build_prompt(question, chunks, chat_history=[])
    print(prompt)