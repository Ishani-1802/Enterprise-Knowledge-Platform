import streamlit as st
from loaders import chunk_text, load_pdfs_from_directory
from embeddings import create_vector_store, LocalEmbeddingFunction
from app.retriever import retrieve
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess

def ask_llm(question, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an enterprise knowledge assistant.

Answer the question ONLY using the context below.
If the answer is not found, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode(),
        stdout=subprocess.PIPE
    )

    return result.stdout.decode()

# --- Sidebar ---
st.sidebar.title("📂 Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

st.sidebar.markdown("---")
reset = st.sidebar.button("🧹 Reset Chat")

# --- Main ---
st.title("🧠 Enterprise Knowledge Assistant")

# Session state for chat memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if reset:
    st.session_state.chat_history = []
    st.experimental_rerun()

# Process uploaded PDF
if uploaded_file:
    st.sidebar.info("Processing document...")

    # Save uploaded file temporarily
    temp_path = os.path.join("data/raw_docs", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and chunk text
    text = load_pdfs_from_directory("data/raw_docs")
    chunks = chunk_text(text)
    st.sidebar.success(f"Document loaded! {len(chunks)} chunks created.")

    # Create embeddings
    create_vector_store()
    st.sidebar.success("Embeddings created!")

    # Input for user queries
    query = st.text_input("Ask a question about the document:")

    if query:
        chunks = retrieve(query)
        answer = ask_llm(query, chunks)
        
        st.session_state.chat_history.append(("User", query))
        st.session_state.chat_history.append(("Bot", answer))

# Display chat history
for sender, msg in st.session_state.chat_history:
    if sender == "User":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")
