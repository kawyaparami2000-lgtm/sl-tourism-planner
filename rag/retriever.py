# RAG Retriever Function Module for Sri Lanka Tourism Planner

from typing import List, Dict, Any
from rag.embed_store import get_vector_store

def retrieve(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Embeds the user/agent query using SentenceTransformers and retrieves top-k matching chunks
    from the persistent Chroma vector store along with text content and metadata.
    """
    print(f"[Retriever] Executing vector search query (k={k}): '{query}'")
    vector_store = get_vector_store()
    
    # Perform similarity search with metadata
    docs = vector_store.similarity_search(query, k=k)
    
    results = []
    for doc in docs:
        results.append({
            "text": doc.page_content,
            "category": doc.metadata.get("category", "general"),
            "source_file": doc.metadata.get("source_file", "unknown")
        })
    
    return results

if __name__ == "__main__":
    test_query = "best time of year to visit Nuwara Eliya"
    retrieved = retrieve(test_query, k=2)
    for i, res in enumerate(retrieved, 1):
        print(f"\nResult #{i} [{res['category']} / {res['source_file']}]:\n{res['text']}")
