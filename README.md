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

Core Technologies
•Python 3.11+ – Core programming language for backend and pipeline development
•Sentence Transformers – For generating semantic embeddings using transformer models
•ChromaDB – Vector database for storing and retrieving document embeddings
•PyPDF (pypdf) – PDF parsing and text extraction

Machine Learning / NLP
•Transformer-based Embeddings (all-MiniLM-L6-v2) – For semantic similarity search
•Text Chunking Strategy – Custom sliding window approach for context preservation

Retrieval System
•Semantic Search – Context-aware retrieval using vector similarity
•Retrieval-Augmented Generation (RAG) – Architecture for combining retrieval with LLMs (LLM integration in progress)

Data Processing
•Document Ingestion Pipeline – Automated loading and processing of PDF documents
•Text Preprocessing & Chunking – Splitting large text into overlapping chunks

Dev & Tooling
•Git & GitHub – Version control and collaboration
•Virtual Environment (venv) – Dependency isolation
•VS Code – Development environment

Deployment (Planned / Optional)
•Docker – Containerization for reproducible deployment (planned)
•CI/CD (GitHub Actions) – Automated workflows (planned)

LLM Integration (In Progress)
•Ollama (Local LLM Runtime) – Running open-source models locally
•Mistral Model – Lightweight LLM for answer generation


## Project Structure

enterprise-knowledge-rag/
├── app/
├── data/
├── tests/
├── .github/workflows
├── Dockerfile
├── requirements.txt
└── README.md
