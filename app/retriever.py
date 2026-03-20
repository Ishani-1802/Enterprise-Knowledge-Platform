import chromadb
from sentence_transformers import SentenceTransformer

# Load same model
model = SentenceTransformer("all-MiniLM-L6-v2")


class LocalEmbeddingFunction:
    def __call__(self, input):
        return model.encode(input).tolist()

    def embed_documents(self, texts):
        return model.encode(texts).tolist()


    def embed_query(self, input):  # Added this method for query embedding
        return model.encode(input).tolist()

    def name(self):
        return "local-sentence-transformer"


# Connect to DB
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
    name="enterprise_docs",
    embedding_function=LocalEmbeddingFunction()
)


def retrieve(query: str, top_k: int = 5):
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return results["documents"][0]


if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or type exit): ")

        if query.lower() == "exit":
            break

        docs = retrieve(query)

        print("\nTop Retrieved Chunks:\n")
        for i, doc in enumerate(docs):
            print(f"--- Result {i+1} ---")
            print(doc[:300])
            print()
