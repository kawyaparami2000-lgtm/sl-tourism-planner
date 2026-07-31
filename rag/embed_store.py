# Embedding and Persistent Chroma Vector Store Module for Sri Lanka Tourism RAG Pipeline

import os
from rag.ingest import load_and_chunk_documents

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_PERSIST_DIR = os.path.abspath("./chroma_db")

def get_embedding_model():
    """
    Initializes and returns the local free SentenceTransformers embedding model (all-MiniLM-L6-v2).
    """
    print(f"[EmbedStore] Initializing local embedding model: {EMBEDDING_MODEL_NAME}")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def get_chroma_class():
    """
    Returns the Chroma vector store class.
    """
    return Chroma

def build_vector_store(persist_directory: str = DEFAULT_PERSIST_DIR):
    """
    Ingests corpus text files, chunks documents, embeds chunks via sentence-transformers,
    and builds/persists the Chroma vector store collection.
    """
    print(f"[EmbedStore] Building persistent Chroma vector store at: {persist_directory}")
    chunks = load_and_chunk_documents()
    embeddings = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="sri_lanka_tourism"
    )
    print(f"[EmbedStore] Successfully embedded {len(chunks)} chunks into ChromaDB.")
    return vector_store

def get_vector_store(persist_directory: str = DEFAULT_PERSIST_DIR):
    """
    Loads and returns the persisted Chroma collection.
    Self-healing: If the chroma_db folder is missing or empty, automatically builds it from data/.
    """
    is_missing_or_empty = not os.path.exists(persist_directory) or len(os.listdir(persist_directory)) == 0

    if is_missing_or_empty:
        print(f"[EmbedStore] First-time run detected: Vector store directory '{persist_directory}' is missing or empty.")
        print("[EmbedStore] Automatically executing build_vector_store() from data/ corpus for self-healing deployment...")
        return build_vector_store(persist_directory)

    embeddings = get_embedding_model()
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="sri_lanka_tourism"
    )
    print(f"[EmbedStore] Successfully loaded persisted ChromaDB store from {persist_directory}")
    return vector_store

if __name__ == "__main__":
    build_vector_store()
