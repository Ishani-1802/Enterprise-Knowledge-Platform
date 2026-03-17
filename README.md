# Enterprise-Knowledge-Platform using RAG

The Enterprise Knowledge Assistant is an AI-powered chatbot built using Retrieval-Augmented Generation (RAG) to provide accurate, document-grounded answers from enterprise data such as PDFs, policies, manuals, and academic syllabi.
The system is fully containerized using Docker and integrated with CI/CD pipelines via GitHub Actions, enabling automated build, test, and deployment on every code update.
This project demonstrates the practical integration of AI + DevOps, closely resembling real-world enterprise knowledge systems.

## Problem Statement

Organizations store critical information across multiple documents (policies, guidelines, notes, manuals). Searching and retrieving accurate answers manually is time-consuming and inefficient.
This project solves the problem by:

Converting documents into searchable embeddings
Retrieving only the most relevant content
Generating reliable answers using an LLM
Automating deployment using DevOps best practices

## Architecture Overview

1. Document ingestion and preprocessing
2. Text chunking and embedding generation
3. Vector storage using FAISS/Chroma
4. Query-based retrieval of relevant chunks
5. LLM-based response generation grounded in retrieved context
6. Automated deployment using Docker and GitHub Actions


## Tech Stack

- Language: Python
- Backend: FastAPI
- LLM: OpenAI / Gemini / Claude
- Embeddings: OpenAI Embeddings
- Vector Store: FAISS / Chroma
- Containerization: Docker
- CI/CD: GitHub Actions
- Deployment: Cloud VM / Platform-as-a-Service


## Project Structure

enterprise-knowledge-rag/
├── app/
├── data/
├── tests/
├── .github/workflows
├── Dockerfile
├── requirements.txt
└── README.md

Final System Architecture (Full Vision)
User Question
      │
      ▼
Embed Question
      │
      ▼
Vector Search (ChromaDB)
      │
      ▼
Top-K Chunks Retrieved
      │
      ▼
Prompt Builder
      │
      ▼
Local LLM
      │
      ▼
Answer Returned to User
