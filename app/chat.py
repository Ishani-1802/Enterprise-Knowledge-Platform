import subprocess
from app.retriever import retrieve


def ask_llm(prompt: str):
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode(),
        stdout=subprocess.PIPE
    )
    return result.stdout.decode()


def build_prompt(question, context_docs):
    context = "\n\n".join(context_docs)

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
    return prompt


if _name_ == "_main_":
    while True:
        question = input("\nAsk a question (or type exit): ")

        if question.lower() == "exit":
            break

        docs = retrieve(question)

        print("\nGenerating answer...\n")

        prompt = build_prompt(question, docs)
        answer = ask_llm(prompt)

        print("Answer:\n")
        print(answer)