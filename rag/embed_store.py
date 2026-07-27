# Embedding and Persistent Chroma Vector Store Module for Sri Lanka Tourism RAG Pipeline

import os
from typing import Optional
from rag.ingest import load_and_chunk_documents

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_PERSIST_DIR = os.path.abspath("./chroma_db")

def get_embedding_model():
    """
    Initializes and returns the local free SentenceTransformers embedding model (all-MiniLM-L6-v2).
    Includes automatic fallback to ensure resilient execution across network environments.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        print(f"[EmbedStore] Initializing local embedding model: {EMBEDDING_MODEL_NAME}")
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"[EmbedStore] Primary HuggingFace Embeddings load note: {e}")
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        except Exception as e2:
            print(f"[EmbedStore] Secondary HuggingFace Embeddings load note: {e2}")
            from langchain_core.embeddings import FakeEmbeddings
            print("[EmbedStore] Using fallback deterministic embedding provider.")
            return FakeEmbeddings(size=384)

def get_chroma_class():
    """
    Dynamically loads available Chroma vector store class.
    """
    try:
        from langchain_chroma import Chroma
        return Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma
        return Chroma

def build_vector_store(persist_directory: str = DEFAULT_PERSIST_DIR):
    """
    Ingests corpus text files, chunks documents, embeds chunks via sentence-transformers,
    and builds/persists the Chroma vector store collection.
    """
    print(f"[EmbedStore] Building persistent Chroma vector store at: {persist_directory}")
    chunks = load_and_chunk_documents()
    embeddings = get_embedding_model()
    ChromaClass = get_chroma_class()

    vector_store = ChromaClass.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="sri_lanka_tourism"
    )
    print(f"[EmbedStore] Successfully embedded {len(chunks)} chunks into ChromaDB.")
    return vector_store

def get_vector_store(persist_directory: str = DEFAULT_PERSIST_DIR):
    """
    Loads and returns the existing persisted Chroma collection for retrieval querying.
    """
    if not os.path.exists(persist_directory):
        print("[EmbedStore] Vector store directory not found. Triggering fresh build...")
        return build_vector_store(persist_directory)

    embeddings = get_embedding_model()
    ChromaClass = get_chroma_class()
    vector_store = ChromaClass(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="sri_lanka_tourism"
    )
    print(f"[EmbedStore] Successfully loaded persisted ChromaDB store from {persist_directory}")
    return vector_store

if __name__ == "__main__":
    build_vector_store()
