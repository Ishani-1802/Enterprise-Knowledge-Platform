import hashlib
from sentence_transformers import SentenceTransformer
import chromadb
from loaders import load_pdfs_from_directory, chunk_text # Use your new chunking function

# Load local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# In embeddings.py

# In embeddings.py

class LocalEmbeddingFunction:
    def __call__(self, input):
        """
        Used for embedding document chunks. 
        input is usually a list of strings.
        """
        return model.encode(input).tolist()

    def embed_documents(self, texts):
        return model.encode(texts).tolist()

    def embed_query(self, input):  
        """
        Used for embedding user queries.
        ChromaDB expects a list of embeddings back.
        """
        # If the input is just one string, wrap it in a list.
        # This ensures model.encode returns a 2D list [[...]] instead of 1D [...].
        if isinstance(input, str):
            input = [input]
            
        return model.encode(input).tolist()

    def name(self):
        return "local-sentence-transformer"


def create_vector_store():
    print("Loading documents...")
    # It's better to use your new loader that handles metadata/better chunking
    # For now, we'll focus on the ID fix:
    documents = load_pdfs_from_directory("data/raw_docs")
    from loaders import chunk_text # Assuming your original function name
    chunks = chunk_text(documents)

    print(f"Total chunks to embed: {len(chunks)}")

    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(
        name="enterprise_docs",
        embedding_function=LocalEmbeddingFunction()
    )

    print("Creating embeddings and storing in vector DB...")

    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        
        # CHANGE 2: Use MD5 Hashing for IDs
        # WHY: 'id_0', 'id_1' will overwrite previous docs if you run this twice.
        # Hashing the text makes the ID unique to the content.
        batch_ids = [hashlib.md5(chunk.encode()).hexdigest() for chunk in batch_chunks]

        collection.add(
            documents=batch_chunks,
            ids=batch_ids
        )

    print("Vector store created successfully!")

if __name__ == "__main__":
    create_vector_store()