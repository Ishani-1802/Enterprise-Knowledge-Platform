import requests
from retriever import retrieve

def ask_llm(prompt: str) -> str:
    """
    Calls Ollama via HTTP API instead of subprocess.
    This keeps the model loaded in memory for much faster responses.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False  # Set to True if you want to implement streaming later
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status() # Check for HTTP errors
        return response.json().get("response", "Error: No response text found.")
    except requests.exceptions.RequestException as e:
        return f"Connection Error: Ensure Ollama is running. ({e})"

def build_prompt(question, context_docs, history=None):
    """
    Builds the prompt with context and conversation history.
    """
    context = "\n\n".join(context_docs)
    
    # Format the last 6 messages (3 turns) of history
    history_text = ""
    if history:
        for msg in history[-6:]:
            role = "User" if msg['role'] == 'user' else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    return f"""You are an enterprise knowledge assistant.
Answer the question ONLY using the context and conversation history below.

Rules:
- Answer ONLY from the context provided.
- If the answer is not in the context, say: "I don't have enough information in the knowledge base."
- Do not add outside knowledge.
- Be concise and professional.

Context:
{context}

Conversation History:
{history_text}

Question:
{question}

Answer:"""

if __name__ == "__main__":
    # Internal list to track history during CLI testing
    chat_history = []
    
    while True:
        question = input("\nAsk a question (or type exit): ")

        if question.lower() == "exit":
            break

        docs = retrieve(question)
        print("\nGenerating answer...\n")

        # Pass history to the prompt builder
        prompt = build_prompt(question, docs, chat_history)
        answer = ask_llm(prompt)

        # Update history for the next turn
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})

        print(f"Answer:\n{answer}")