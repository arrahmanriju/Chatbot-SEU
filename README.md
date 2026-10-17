# 🤖 Chatbot-SEU — Southeast University AI Assistant

An AI-powered chatbot designed to help students quickly find information about **Southeast University (SEU)**. The chatbot uses a university-specific knowledge base to answer student questions and make university information easier to access.

## ✨ Features

- 🎓 **University-Focused Q&A** — Answers questions related to Southeast University.
- 🌐 **English & Bangla Knowledge** — Supports university information in both languages, depending on the query and available data.
- 📚 **Knowledge-Based Responses** — Uses university information stored in a structured JSONL dataset.
- 🔎 **Relevant Information Retrieval** — Retrieves relevant knowledge passages to help generate answers.
- 💬 **Conversational Interface** — Provides a simple way for students to ask questions in natural language.
- 🛡️ **Domain-Focused Assistant** — Designed to prioritize university-related questions.

## 🛠️ Technology Stack

- **Language:** Python
- **Interface:** Streamlit
- **AI/LLM:** Large Language Model (LLM)
- **Retrieval:** Semantic search / Retrieval-Augmented Generation (RAG)
- **Vector Database:** ChromaDB
- **Knowledge Base:** JSONL

## 🏗️ How It Works

1. **User Query:** A student asks a question about Southeast University.
2. **Information Retrieval:** The system searches the university knowledge base for relevant information.
3. **Context Preparation:** The most relevant retrieved passages are selected as context.
4. **Answer Generation:** The language model uses the retrieved information to generate a response.
5. **Response:** The chatbot presents the answer to the student.

## 📂 Project Structure

The exact file structure may vary depending on the current version of the project.

```text
Chatbot-SEU/
├── app.py                  # Streamlit application entry point
├── data/                   # University knowledge data
├── requirements.txt        # Python dependencies
├── .env.example            # Example environment configuration
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or a compatible version
- pip
- Git
- An API key for the LLM provider, if required by your implementation

### 1. Clone the Repository

```bash
git clone https://github.com/arrahmanriju/Chatbot-SEU.git
cd Chatbot-SEU
```

### 2. Create a Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

If the project requires an API key, create a `.env` file in the project root and add the variable names required by your implementation.

Example:

```env
LLM_API_KEY=your_api_key_here
```

Replace the example variable with the actual variable name expected by your application. Never commit API keys or other secrets to GitHub.

### 5. Run the Application

If `app.py` is the application entry point, run:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to interact with the chatbot.

## 💡 Example Questions

You can ask questions such as:

- What is Southeast University?
- What programs does SEU offer?
- Where is Southeast University located?
- How can students find university-related information?
- Can you explain university information in Bangla?

*The answers depend on the information available in the knowledge base.*

## 🎯 Project Goals

- Make university information easier for students to access.
- Explore practical applications of Large Language Models.
- Learn how retrieval-based AI systems connect external knowledge with language models.
- Build experience in Python, NLP, and AI application development.

## 🔮 Future Improvements

- Expand and regularly update the university knowledge base.
- Improve retrieval accuracy and answer relevance.
- Add source references to chatbot responses.
- Improve Bangla language understanding.
- Add a feedback system to evaluate answer quality.
- Deploy the chatbot for easier access.

## 👨‍💻 Author

**Azizur Rahman**

- GitHub: [@arrahmanriju](https://github.com/arrahmanriju)
- Project Repository: [Chatbot-SEU](https://github.com/arrahmanriju/Chatbot-SEU)

---

*Built as an AI application project focused on university information retrieval and conversational assistance.*
