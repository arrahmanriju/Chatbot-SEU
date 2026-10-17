# Chatbot-SEU 🎓

A RAG-based chatbot that answers questions about Southeast University (SEU), Bangladesh, using the university's website content, FAISS vector search, and Llama 3.3 on Groq.

🔗 **Live demo:** `<add your live link, or remove this line>`

---

## Summary and Problem

Information about a university, such as its programs, departments, admissions, vision, and faculty, is spread across many website pages. Students and applicants have to search through them to find a simple answer.

Chatbot-SEU lets them ask in plain language instead. It searches a knowledge base built from the university's website, finds the most relevant passages, and has a language model answer using only those passages. If the answer is not in the data, the bot says so instead of guessing. Questions that are not about SEU get a polite refusal.

## Features

- **University Q&A:** answers questions about SEU programs, departments, admissions, vision/mission, and faculty
- **Retrieval-Augmented Generation (RAG):** answers are grounded in retrieved website content, not the model's memory
- **Off-topic filtering:** unrelated questions (for example "What is the capital of France?") are rejected using a similarity check
- **Greeting and "about the bot" handling:** greetings and questions like "who are you?" get instant replies without calling the LLM
- **Conversation memory:** the last 5 exchanges are included in the prompt, so follow-up questions work
- **Chat interface:** a Streamlit chat UI
- **Docker support:** a Dockerfile is included for deployment

## Architecture

```
                        ── Ingestion (run once) ──
 data/old_seu_website_cleaned.jsonl
        │
        ▼
 loader.py ──► splitter.py ──► embedder.py ──► store.py
 (JSON Lines)  (500-char       (MiniLM        (FAISS index saved
               chunks)         embeddings)     to vectorstore/)

                        ── Query time ──
 User question (Streamlit chat)
        │
        ▼
 classify_message ──► greeting / meta ──► instant reply
        │
        ▼
 is_university_question (similarity check) ──► off-topic ──► refusal
        │
        ▼
 retriever.py (top 5 chunks from FAISS)
        │
        ▼
 prompt_builder.py (context + last 5 exchanges + question)
        │
        ▼
 llm.py (Llama 3.3 70B on Groq) ──► answer
```

- **Ingestion** turns the website data into a searchable FAISS index. It runs once, or again when the data changes.
- **Query time** is handled by `run_chatbot` in `chain.py`, which connects all the steps above.

## Database (ERD)

This project has no relational database, so there is no ERD. It uses two data stores:

**1. Knowledge base: `data/old_seu_website_cleaned.jsonl`** (one JSON record per line, 333 records)

| Field | Description |
|---|---|
| id | Record number |
| source_url | Page the text came from |
| title | Page or section title |
| clean_text | Cleaned page text used for retrieval |

**2. Vector store: FAISS index** (`vectorstore/faiss_index`)

| Stored item | Description |
|---|---|
| Chunk text | A 500-character piece of a record, with the title prepended |
| Embedding | 384-dimension vector from `all-MiniLM-L6-v2` |
| Metadata | `id`, `source_url`, `title` |

The index is generated, not committed. Build it with the steps in **Run and Tests**.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| UI | Streamlit |
| RAG framework | LangChain |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face) |
| Vector store | FAISS (cosine distance) |
| LLM | Llama 3.3 70B Versatile via Groq API |
| Config | python-dotenv |
| Containerization | Docker |

## Project Structure

```
Chatbot-SEU/
├── app.py                 # Streamlit chat interface
├── src/
│   ├── loader.py          # Loads the JSONL knowledge base
│   ├── splitter.py        # Splits text into overlapping chunks
│   ├── embedder.py        # Loads the embedding model
│   ├── store.py           # Builds and saves the FAISS index
│   ├── filter.py          # Loads the index, classifies messages, off-topic check
│   ├── retriever.py       # Retrieves the top matching chunks
│   ├── prompt_builder.py  # Builds the LLM prompt
│   ├── llm.py             # Calls the Groq API
│   └── chain.py           # Full question-to-answer pipeline
├── data/
│   └── old_seu_website_cleaned.jsonl
├── .streamlit/config.toml
├── Dockerfile
└── requirements.txt
```

## Demo Credentials

There is no login. The only credential you need is your own Groq API key:

| Item | Value |
|---|---|
| `GROQ_API_KEY` | Get a free key from [console.groq.com](https://console.groq.com) |

## Run and Tests

**Prerequisites:** Python 3.11, pip, Git, and a Groq API key.

**1. Clone the repository**
```bash
git clone https://github.com/arrahmanriju/Chatbot-SEU.git
cd Chatbot-SEU
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set the environment variable**

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key
```

**5. Build the vector index (first time only)**
```bash
python -m src.store
```
This reads the JSONL data, splits it into chunks, and saves the FAISS index to `vectorstore/faiss_index`.

**6. Run the app**
```bash
streamlit run app.py
```
Open `http://localhost:8501`.

**Run with Docker**
```bash
docker build -t chatbot-seu .
docker run -p 8501:8501 -e GROQ_API_KEY=your_groq_api_key chatbot-seu
```
Build the index first (step 5) so the `vectorstore/` folder exists before building the image.

**Tests:** there is no automated test suite yet. Each module in `src/` can be run on its own as a quick check, and the full pipeline can be tried with sample questions:
```bash
python -m src.chain
```

## API Overview

This project has no REST API. The pipeline is a set of Python functions:

| Function | File | Purpose |
|---|---|---|
| `build_vectorstore()` | `store.py` | Runs load → split → embed → save |
| `load_vectorstore()` | `filter.py` | Loads the saved FAISS index |
| `classify_message(question)` | `filter.py` | Detects greetings and "about the bot" questions |
| `is_university_question(question, vectorstore)` | `filter.py` | Similarity check to reject off-topic questions |
| `retrieve_top_chunks(question, vectorstore, k=5)` | `retriever.py` | Returns the top-k chunks with scores |
| `build_prompt(question, chunks, chat_history)` | `prompt_builder.py` | Builds the final prompt |
| `ask_llm(prompt)` | `llm.py` | Sends the prompt to Groq and returns the reply |
| `run_chatbot(question, vectorstore, chat_history)` | `chain.py` | Full pipeline |

## Deployment

- The app is containerized with the included `Dockerfile` (Python 3.11, Streamlit on port 8501).
- It can be deployed to any Docker host. The code is set up for Hugging Face Spaces, where `GROQ_API_KEY` is added as a Space secret.
- Set `GROQ_API_KEY` as an environment variable or secret on the host. Never commit it.
- The `vectorstore/` folder is not in Git, so it must be built and included in the deployment image, or built at startup.

## Key Decisions

- **RAG instead of a fine-tuned model:** the university data can be updated by editing the JSONL file and rebuilding the index, with no retraining.
- **FAISS for the vector store:** it runs locally with no external database, which keeps setup and deployment simple.
- **`all-MiniLM-L6-v2` embeddings:** small and fast, and good enough for a few hundred records on a free CPU host.
- **Groq with Llama 3.3 70B:** fast hosted inference with a strong model, and no GPU needed on the server.
- **500-character chunks with 100 overlap:** keeps each chunk focused while not cutting facts in half. Chunks under 40 characters (navigation fragments) are dropped.
- **Title added to each chunk:** helps queries like "vision" match the record titled "Vision".
- **Context-only prompt:** the model is told to answer only from the retrieved text and to admit when it doesn't know, which reduces made-up answers.
- **Greeting and meta shortcuts:** simple messages skip retrieval and the LLM, saving time and API calls.
- **Similarity check for off-topic questions:** the top match score decides whether a question is about SEU, with no extra classifier model.
- **Low temperature (0.3):** keeps answers factual and consistent.

## Known Limitations

- The knowledge base comes from the old SEU website, so some information may be outdated
- The embedding model is English-focused, so Bangla questions may retrieve poorly
- Off-topic detection uses a low similarity threshold, so some unrelated questions may still pass
- Greeting and "about the bot" detection is keyword-based and can misfire (for example, any message containing "help")
- Answers do not show their sources
- Chat history lasts only for the current browser session
- No automated tests
- The FAISS index must be built manually before the first run

## Next Improvements

- Update the knowledge base from the current SEU website
- Show source links with each answer
- Use a multilingual embedding model for better Bangla support
- Tune the off-topic threshold or replace it with a proper intent classifier
- Add a feedback button to measure answer quality
- Build the index automatically on startup
- Add unit tests and a retrieval evaluation set
- Deploy a public live demo

## Contributing

Contributions, issues, and feature requests are welcome. Check the [issues page](https://github.com/arrahmanriju/Chatbot-SEU/issues).

## License

`<add your license, for example MIT, and a LICENSE file>`
