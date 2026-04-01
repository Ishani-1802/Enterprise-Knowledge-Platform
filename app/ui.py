import streamlit as st
import os
from chat import ask_llm, build_prompt
from retriever import retrieve
from embeddings import create_vector_store

# --- Page Config ---
st.set_page_config(page_title="Enterprise RAG Assistant", page_icon="🧠")

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False

# --- Sidebar ---
st.sidebar.title("📂 Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if st.sidebar.button("🧹 Reset Chat"):
    st.session_state.chat_history = []
    # FIX 3: Use st.rerun() instead of experimental_rerun()
    st.rerun()

# --- Document Processing Logic ---
# FIX 2: Only run this if it hasn't been processed yet for this file
if uploaded_file and not st.session_state.doc_processed:
    with st.status("Processing document...", expanded=True) as status:
        st.write("Saving file...")
        os.makedirs("data/raw_docs", exist_ok=True)
        temp_path = os.path.join("data/raw_docs", uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.write("Creating embeddings (this may take a moment)...")
        # This calls your updated embeddings.py which uses semantic chunking
        create_vector_store()
        
        st.session_state.doc_processed = True
        status.update(label="Document Indexed!", state="complete", expanded=False)

# --- Main Chat Interface ---
st.title("🧠 Enterprise Knowledge Assistant")
st.info("Ask questions based on your uploaded documents.")

# Display chat history using Streamlit's native chat UI
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# FIX 4: Move query input outside the upload block so it's always accessible
if query := st.chat_input("Ask a question about the document:"):
    
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # FIX 5: Use st.spinner for loading state
    with st.spinner("Searching knowledge base and generating answer..."):
        # 1. Retrieve chunks
        chunks = retrieve(query)
        
        # 2. Build prompt (FIX 6: Now uses the stronger prompt + history from chat.py)
        prompt = build_prompt(query, chunks, st.session_state.chat_history)
        
        # 3. Get LLM response (FIX 1: Uses the shared function from chat.py)
        answer = ask_llm(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})