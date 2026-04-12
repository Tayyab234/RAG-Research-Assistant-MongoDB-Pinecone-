import os
import tiktoken
import re
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredFileLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -------------------------------
# 1. Select Loader
# -------------------------------
def get_loader(file_path, filename):
    filename = filename.lower()

    if filename.endswith(".pdf"):
        return PyPDFLoader(file_path)

    elif filename.endswith(".txt"):
        return TextLoader(
            file_path,
            encoding="utf-8",
            autodetect_encoding=True
        )

    elif filename.endswith(".docx"):
        return Docx2txtLoader(file_path)

    elif filename.endswith(".doc"):
        # fallback for old Word files
        return UnstructuredFileLoader(file_path)

    else:
        return UnstructuredFileLoader(file_path)


# -------------------------------
# 2. Load Documents (Lazy or Full)
# -------------------------------
def load_documents(loader, file_path, lazy_threshold_mb=5):
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    documents = []

    if file_size_mb > lazy_threshold_mb:
        for doc in loader.lazy_load():
            documents.append(doc)
    else:
        documents = loader.load()

    return documents


# -------------------------------
# 3. Token Estimation
# -------------------------------
def estimate_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# -------------------------------
# 4. Validate Token Limit
# -------------------------------
def validate_tokens(documents, max_tokens=2_000_000, warn_threshold=0.8):
    total_tokens = 0

    for i, doc in enumerate(documents):
        tokens = estimate_tokens(doc.page_content)
        total_tokens += tokens

        # 🔴 Hard limit
        if total_tokens > max_tokens:
            raise ValueError(
                f"Document exceeds token limit: {total_tokens} > {max_tokens} at document index {i}"
            )

    # 🟡 Warning if nearing limit
    if total_tokens > max_tokens * warn_threshold:
        print(
            f"⚠️ Warning: Document is large ({total_tokens} tokens, "
            f"{(total_tokens / max_tokens) * 100:.2f}% of limit)"
        )

    return total_tokens


# -------------------------------
# 5. Clean Documents (Optional but Recommended)
# -------------------------------
import re

def clean_documents(documents):
    cleaned_docs = []

    for doc in documents:
        text = doc.page_content

        # Basic cleanup
        text = text.strip()
        text = " ".join(text.split())

        # Fix hyphenated words
        text = re.sub(r'-\s+', '', text)

        # Remove excessive symbols
        text = re.sub(r'[-=_]{3,}', ' ', text)

        # Normalize unicode spaces
        text = text.replace("\u00a0", " ")

        doc.page_content = text
        cleaned_docs.append(doc)

    return cleaned_docs

# -------------------------------
# 6. Split into Chunks
# -------------------------------
def split_documents(documents, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return splitter.split_documents(documents)


# -------------------------------
# 🔥 MAIN FUNCTION
# -------------------------------
def process_document(file_path, filename):
    # Step 1: Loader
    loader = get_loader(file_path, filename)

    # Step 2: Load
    documents = load_documents(loader, file_path)

    if not documents:
        raise ValueError("No content extracted from document")

    # Step 3: Validate tokens
    total_tokens = validate_tokens(documents)

    # Step 4: Clean
    documents = clean_documents(documents)

    # Step 5: Chunking
    chunks = split_documents(documents)

    return {
        "chunks": chunks,
        "num_chunks": len(chunks),
        "total_tokens": total_tokens
    }