```
---
title: Diabetes RAG Chatbot
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---
```
# 🔬 Research Paper RAG Chatbot

A conversational AI chatbot that lets you **chat with any research PDF** 
using Retrieval Augmented Generation (RAG).

Built as a portfolio project to demonstrate end-to-end GenAI engineering — 
from document ingestion to deployed web application.

🚀 **[Live Demo →](YOUR_HUGGINGFACE_URL_HERE)**

---

## What it does

Upload any academic PDF and ask questions about it in natural language. 
The app retrieves the most relevant sections and generates grounded, 
cited answers — it won't hallucinate content that isn't in your document.

Tested with a diabetes classification research paper (Masters Project, 
UMass Dartmouth 2025).

---

## Architecture
```
PDF Upload
    │
    ▼
PyPDFLoader → RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
    │
    ▼
HuggingFace Embeddings (all-MiniLM-L6-v2)  ← runs locally, no API cost
    │
    ▼
ChromaDB Vector Store  ← in-memory per session
    │
    ▼
Similarity Search (top-k=4 chunks retrieved per query)
    │
    ▼
Custom PromptTemplate + ConversationBufferMemory
    │
    ▼
Groq LLM (Llama 3 8B)  ← free, fast inference API
    │
    ▼
Streamlit UI  ← streaming output + source citations
```

---

## Tech stack

| Layer | Technology |
|---|---|
| LLM | Groq API — Llama 3 8B |
| Orchestration | LangChain — ConversationalRetrievalChain |
| Vector DB | ChromaDB (in-memory) |
| Embeddings | HuggingFace — all-MiniLM-L6-v2 |
| Document loading | LangChain PyPDFLoader |
| Frontend | Streamlit |
| Deployment | Hugging Face Spaces |

---

## Run locally
```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/diabetes-rag-chatbot.git
cd diabetes-rag-chatbot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# 5. Run
streamlit run app.py
```

---

## Key concepts demonstrated

- **RAG pipeline** — retrieval grounded generation prevents hallucination
- **Vector embeddings** — semantic search over document chunks
- **Prompt engineering** — system prompt constrains LLM to document context
- **Conversational memory** — multi-turn context with ConversationBufferMemory
- **Streamlit session state** — persistent state across UI reruns
- **Production patterns** — temp file handling, cached resources, .env secrets

---

