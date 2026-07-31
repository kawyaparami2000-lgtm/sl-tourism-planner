# Document Ingestion and Chunking Module for Sri Lanka Tourism RAG Pipeline

import os
from typing import List
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_documents(data_dir: str = None) -> List[Document]:
    """
    Recursively scans the data/ directory for text files, loads their contents,
    splits them into chunks, and attaches metadata (category and source_file).
    """
    documents: List[Document] = []
    if data_dir is None:
        # Resolve path relative to this file's location, not the current working directory
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(this_file_dir, "..", "data")
    
    # Text splitter configured for ~300-500 characters with ~50 character overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    base_path = os.path.abspath(data_dir)
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Data directory not found at: {base_path}")

    print(f"[Ingest] Scanning directory for domain corpus: {base_path}")

    for root, _, files in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)
            category = os.path.basename(root)  # Subfolder name (e.g. destinations, transport)
            content = ""

            if file.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
            elif file.endswith(".pdf"):
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(file_path)
                    pages_text = [page.extract_text() or "" for page in reader.pages]
                    content = "\n".join(pages_text).strip()
                except Exception as e:
                    print(f"  - Warning: Failed to parse PDF '{file}': {e}")
                    content = ""

            if content:
                # Create parent document with metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        "category": category,
                        "source_file": file
                    }
                )
                # Split into chunks preserving metadata
                chunks = text_splitter.split_documents([doc])
                documents.extend(chunks)
                print(f"  - Loaded '{file}' [{category}]: Split into {len(chunks)} chunks.")

    print(f"[Ingest] Total document chunks generated across all categories: {len(documents)}")
    return documents

if __name__ == "__main__":
    chunks = load_and_chunk_documents()
    if chunks:
        print(f"\nSample chunk:\n{chunks[0]}")
