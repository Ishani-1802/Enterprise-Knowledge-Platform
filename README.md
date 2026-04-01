# 🧠 Enterprise Knowledge Assistant

A **production-grade, fully local Retrieval-Augmented Generation (RAG)** system that lets users upload internal PDF documents and interact with them through a conversational AI interface — with **zero API costs** and **full offline capability**.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Tech Stack & Modules](#-tech-stack--modules)
- [How It Works — Step by Step](#-how-it-works--step-by-step)
- [Key Design Decisions](#-key-design-decisions)
- [Setup & Installation](#-setup--installation)
- [Running the Application](#-running-the-application)

---

## 🎯 Project Overview

The **Enterprise Knowledge Assistant** eliminates the need for manual document search inside organizations. Instead of reading through PDFs manually, users can simply upload a document and ask natural language questions — the system retrieves the most relevant sections and generates a grounded, cited answer using a local LLM.

**Core Principles:**
- 🔒 **Privacy-first** — all processing happens locally, nothing leaves your machine
- 💸 **Zero cost** — no OpenAI, no paid APIs, fully open-source stack
- ⚡ **Production-quality** — semantic chunking, lazy DB initialization, conversation memory, proper error handling

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   INGESTION PIPELINE                │
│                                                     │
│  PDF Upload → pdfplumber → RecursiveTextSplitter   │
│       → MiniLM Embeddings → ChromaDB (local)       │
└─────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────┐
│                    QUERY PIPELINE                   │
│                                                     │
│  User Question → MiniLM Embed → ChromaDB TopK      │
│    → build_prompt (context + history) → Mistral    │
│              → Answer in Streamlit UI               │
└─────────────────────────────────────────────────────┘
```

**Data Flow:**
1. User uploads PDF via Streamlit
2. `pdfplumber` extracts clean text preserving layout
3. `RecursiveCharacterTextSplitter` creates semantically coherent chunks
4. `all-MiniLM-L6-v2` converts chunks to dense vectors
5. Vectors stored in **ChromaDB** (persistent local disk)
6. User asks a question → query embedded → top-5 similar chunks retrieved
7. Retrieved chunks + conversation history injected into prompt
8. **Mistral** (via Ollama HTTP API) generates a grounded answer
9. Answer displayed in a chat-style Streamlit UI

---

## 🧩 Tech Stack & Modules

### Core Libraries

| Module | Version | Purpose |
|--------|---------|---------|
| `streamlit` | latest | Web UI — chat interface, file uploader, session state |
| `pdfplumber` | latest | PDF text extraction with layout preservation |
| `langchain-text-splitters` | latest | `RecursiveCharacterTextSplitter` for semantic chunking |
| `sentence-transformers` | latest | Local embedding model (`all-MiniLM-L6-v2`) |
| `chromadb` | latest | Local vector database for storing and querying embeddings |
| `requests` | latest | HTTP calls to Ollama's local API |
| `hashlib` | stdlib | MD5-based chunk ID generation (deduplication) |
| `pathlib` | stdlib | Cross-platform file path handling |
| `ollama` | (service) | Hosts Mistral LLM locally, exposes REST API on port 11434 |

### Why These Choices?

**`pdfplumber` over `pypdf`**  
`pypdf` concatenates all text without respecting paragraph structure. `pdfplumber` preserves whitespace, newlines, and multi-column layouts — feeding cleaner text into the chunker.

**`RecursiveCharacterTextSplitter` over naïve character slicing**  
Plain character slicing cuts mid-sentence or mid-word. The recursive splitter tries boundaries in priority order: `\n\n` (paragraphs) → `\n` (lines) → `. ` (sentences) → ` ` (words) → `""` (characters). Chunks stay semantically complete.

**`all-MiniLM-L6-v2` for embeddings**  
A 22M parameter model that runs on CPU in milliseconds. It produces 384-dimensional vectors with strong semantic accuracy for English text — the best free, local option for this use case.

**ChromaDB for vector storage**  
Fully embedded in Python (no separate server needed), persists to disk, and natively supports batch upserts and cosine similarity search. Ideal for single-machine deployment.

**Ollama HTTP API over `subprocess`**  
`subprocess.run(["ollama", "run", ...])` cold-starts the model on every query (5–15 seconds overhead). Calling `http://localhost:11434/api/generate` hits the already-loaded model, reducing response time to pure inference latency.

---

## 🔍 How It Works — Step by Step

### Step 1 — PDF Upload (`ui.py`)
The user uploads a PDF via Streamlit's sidebar file uploader. Streamlit's session state (`st.session_state.doc_processed`) ensures the document is only processed once per upload, not on every UI rerender.

### Step 2 — Text Extraction (`loaders.py`)
`pdfplumber` opens the PDF page by page. Each page's text is extracted and separated by double newlines (`\n\n`), which act as natural paragraph boundary signals for the next step.

### Step 3 — Semantic Chunking (`loaders.py`)
`RecursiveCharacterTextSplitter` splits the full document text into chunks of ~500 characters with 100-character overlap. It respects natural language boundaries — paragraphs first, then sentences, then words — so no chunk ever cuts a thought in half.

### Step 4 — Embedding Generation (`embeddings.py`)
The `LocalEmbeddingFunction` class wraps `SentenceTransformer("all-MiniLM-L6-v2")`. Each chunk is passed through the model and converted to a 384-dimensional floating point vector. This vector numerically captures the semantic meaning of the text.

### Step 5 — Vector Storage (`embeddings.py`)
All chunk vectors are stored in ChromaDB under a persistent local collection (`enterprise_docs`). Chunks are stored in batches of 500. Each chunk's ID is generated as an MD5 hash of its content — this means identical chunks are never stored twice (deduplication), and re-runs won't overwrite data with wrong IDs.

### Step 6 — Query Embedding + Retrieval (`retriever.py`)
When the user types a question, the same embedding model converts it into a query vector. ChromaDB performs cosine similarity search and returns the 5 most relevant chunks. The `get_collection()` function uses lazy initialization — the DB connection is only opened when the first query arrives, preventing crashes if the DB doesn't exist yet.

### Step 7 — Prompt Construction (`chat.py`)
`build_prompt()` assembles three components into a single structured prompt:
- **System rules** — instructs the model to answer only from context, not hallucinate
- **Retrieved context** — the 5 relevant document chunks
- **Conversation history** — the last 6 messages (3 turns) so the model can handle follow-up questions

### Step 8 — LLM Generation (`chat.py`)
`ask_llm()` sends the assembled prompt to Ollama's HTTP API (`POST /api/generate`) with `stream: false`. Ollama keeps Mistral loaded in memory, so responses are returned within seconds rather than cold-starting the model each time.

### Step 9 — Display (`ui.py`)
The answer is rendered in Streamlit's native `st.chat_message` component. Both the question and the answer are appended to `st.session_state.chat_history`, which powers the full conversation display and feeds the next turn's history.

---

## 🔑 Key Design Decisions

### Lazy Database Initialization
`retriever.py` uses a module-level `_collection = None` with a `get_collection()` function. The ChromaDB connection is only created the first time a query runs — not at import time. This prevents the application from crashing in a fresh environment where no documents have been ingested yet.

### MD5-Based Chunk IDs
Instead of sequential IDs (`id_0`, `id_1`...) that would silently overwrite existing data on re-ingestion, each chunk is hashed with MD5. The same content always produces the same ID, enabling safe re-runs and natural deduplication.

### Conversation Memory in Prompt
The last 6 messages of conversation history are injected directly into the LLM prompt. This allows the model to handle follow-up questions ("can you elaborate on point 2?") without losing context between turns.

### Single Shared `LocalEmbeddingFunction`
The embedding class is defined once in `embeddings.py` and imported wherever needed. Previously it was duplicated across `embeddings.py` and `retriever.py` — a maintenance hazard if the model or interface ever changes.

### Session State–Gated Document Processing
`st.session_state.doc_processed` prevents `create_vector_store()` from running on every Streamlit rerender (which happens on every user interaction). Document processing runs exactly once per upload.

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Mistral model pulled: `ollama pull mistral`

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/enterprise-knowledge-assistant.git
cd enterprise-knowledge-assistant
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env if needed — defaults work out of the box
```

### 5. Start Ollama

Make sure Ollama is running in a separate terminal:

```bash
ollama serve
```

Verify Mistral is available:

```bash
ollama list
```

---

## 🚀 Running the Application

```bash
streamlit run app/ui.py
```

Open your browser at `http://localhost:8501`.

1. Upload a PDF using the left sidebar
2. Wait for the "Document Indexed!" confirmation
3. Type your question in the chat box
4. Ask follow-up questions — the system remembers conversation context

---

*Built as a learning + portfolio project demonstrating production RAG system design using fully open-source, local-first tooling.*