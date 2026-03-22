from sentence_transformers import SentenceTransformer
import chromadb
from app.loaders import load_pdfs_from_directory, chunk_text

# Load local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


class LocalEmbeddingFunction:
    def __call__(self, input):
        return model.encode(input).tolist()

    def embed_documents(self, texts):
        return model.encode(texts).tolist()

    def embed_query(self, input):  
        return model.encode(input).tolist()

    def name(self):
        return "local-sentence-transformer"


def create_vector_store():
    print("Loading documents...")
    documents = load_pdfs_from_directory("data/raw_docs")
    print(f"Loaded text length: {len(documents)}")
    chunks = chunk_text(documents)

    print(f"Total chunks to embed: {len(chunks)}")

    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(
        name="enterprise_docs",
        embedding_function=LocalEmbeddingFunction()
    )

    print("Creating embeddings and storing in vector DB...")

    batch_size = 5000
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_ids = [f"id_{j}" for j in range(i, i + len(batch_chunks))]

        collection.add(
            documents=batch_chunks,
            ids=batch_ids
        )

    print("Vector store created successfully!")


if __name__ == "__main__":
    create_vector_store()
