from chromadb import PersistentClient
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

class Retriever:
    def __init__(self, collection_name: str = "poker_knowledgebase", persist_directory: str = "./rag_db"):
        self.client = PersistentClient(path=persist_directory, settings=Settings())
        self.collection = self.client.get_collection(name=collection_name)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def query(self, query_text: str, top_k: int = 3) -> list:
        """Retrieve top_k relevant documents for the query."""
        query_embedding = self.model.encode([query_text])[0].tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        for doc, dist, meta in zip(results['documents'][0], results['distances'][0], results['metadatas'][0]):
            formatted_results.append({
                'text': doc,
                'distance': dist,
                'file': meta['file'],
                'chunk_index': meta['chunk_index']
            })
        return formatted_results

    def list_all_documents(self, limit: int = 10) -> list:
        """List all documents in the collection (up to a limit for display)."""
        results = self.collection.get(limit=limit)
        formatted_results = []
        for doc, meta, doc_id in zip(results['documents'], results['metadatas'], results['ids']):
            formatted_results.append({
                'id': doc_id,
                'text': doc[:200] + "..." if len(doc) > 200 else doc,
                'file': meta['file'],
                'chunk_index': meta['chunk_index']
            })
        return formatted_results

# # 🔍 Function to use the retriever
# def top_docs_retriever(query: str) -> str:
#     """Retrieve top 3 similar document chunks and return them as a single combined string."""
#     r = Retriever()
#     similar_docs = r.query(query, top_k=1)
    
#     combined_text = ""
#     for res in similar_docs:
#         combined_text += res["text"].strip() + "\n\n"
    
#     return combined_text.strip()


# # Example usage
# if __name__ == "__main__":
#     query_text = "How to bluff effectively in poker?"
#     top_docs = top_docs_retriever(query_text)
#     print(top_docs)
