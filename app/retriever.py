import chromadb
# CHANGE 1: Import the shared class instead of redefining it
from embeddings import LocalEmbeddingFunction 

# CHANGE 2: Use Lazy Initialization to prevent crashes at import time
_collection = None

def get_collection():
    """
    Ensures the database connection is only made when actually needed.
    """
    global _collection
    if _collection is None:
        try:
            client = chromadb.PersistentClient(path="chroma_db")
            _collection = client.get_collection(
                name="enterprise_docs",
                # Use the class we imported from embeddings.py
                embedding_function=LocalEmbeddingFunction()
            )
        except Exception as e:
            print(f"Error connecting to ChromaDB: {e}")
            return None
    return _collection

def retrieve(query: str, top_k: int = 5):
    """
    Retrieves the top_k relevant document chunks.
    """
    collection = get_collection()
    if collection is None:
        return ["Database not initialized."]
        
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    # Check if we actually got results
    if results and results["documents"]:
        return results["documents"][0]
    return []

if __name__ == "__main__":
    # Test the retriever
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