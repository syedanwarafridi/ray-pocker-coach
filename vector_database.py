import os
import docx
import tiktoken
from chromadb import Client
from chromadb.config import Settings
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import uuid
import numpy as np

def read_docx(file_path: str) -> str:
    """Read text from a .docx file."""
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

def get_chunks_with_overlap(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into chunks with overlap based on token count."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(encoding.decode(chunk_tokens))
        start += chunk_size - overlap
    return chunks

def get_embeddings(texts: List[str]) -> List[np.ndarray]:
    """Generate embeddings for a list of texts using sentence-transformers."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(texts, show_progress_bar=True)

def create_vector_db(directory: str, collection_name: str = "poker_knowledgebase"):
    """Create a Chroma DB vector database from .docx files in the directory."""
    # client = Client(Settings(persist_directory="./rag_db"))
    client = PersistentClient(
    path="rag_db",
    settings=Settings()
)
    collection = client.get_or_create_collection(name=collection_name)
    
    # Collect all .docx files
    docx_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".docx"):
                docx_files.append(os.path.join(root, file))
    
    # Process each file
    for file_path in docx_files:
        text = read_docx(file_path)
        if not text:
            continue
        chunks = get_chunks_with_overlap(text)
        embeddings = get_embeddings(chunks)
        
        # Add to Chroma DB
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            collection.add(
                documents=[chunk],
                embeddings=[embedding.tolist()],
                ids=[f"{os.path.basename(file_path)}_{i}_{uuid.uuid4()}"],
                metadatas=[{"file": file_path, "chunk_index": i}]
            )
    
    print(f"Vector database created with {collection.count()} documents.")

if __name__ == "__main__":
    directory = "E:\\LinkedIn\\texaxSolver-chatbot\\Ray Knowledgebase"
    create_vector_db(directory)