import sys
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import json

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import CHROMA_PATH, COLLECTION_NAME, TOP_K, PROCESSED_DIR, RERANKER_MODEL
from src.utilities.model_loader import get_huggingface_embedding_model

# Load Vector Store
embedding_model = get_huggingface_embedding_model()
vector_store = Chroma(
    persist_directory=str(CHROMA_PATH),
    embedding_function=embedding_model,
    collection_name=COLLECTION_NAME,
)

# Global instances cached for retrieval
_bm25_retriever = None
_reranker_instance = None

def get_vector_retriever(k=TOP_K * 2):
    """Returns dense ChromaDB similarity retriever."""
    return vector_store.as_retriever(search_kwargs={"k": k})

def get_bm25_retriever(k=TOP_K * 2):
    """Initializes and returns lexical BM25 retriever using chunked_docs.json."""
    global _bm25_retriever
    if _bm25_retriever is not None:
        return _bm25_retriever

    chunk_file = PROCESSED_DIR / "chunked_docs.json"
    if not chunk_file.exists():
        print(f"Warning: chunked_docs.json not found at {chunk_file}. Lexical fallback disabled.")
        return None

    try:
        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        documents = []
        for c in chunks:
            documents.append(Document(page_content=c["text"], metadata=c["metadata"]))
        
        _bm25_retriever = BM25Retriever.from_documents(documents)
        _bm25_retriever.k = k
        return _bm25_retriever
    except Exception as e:
        print(f"Error loading BM25 index: {str(e)}")
        return None

def get_reranker():
    """Initializes and returns CrossEncoder reranker model."""
    global _reranker_instance
    if _reranker_instance is not None:
        return _reranker_instance
    
    print(f"Loading reranker model {RERANKER_MODEL} on CPU...")
    _reranker_instance = CrossEncoder(RERANKER_MODEL, device="cpu")
    return _reranker_instance

class HybridRetriever:
    def __init__(self, top_k=TOP_K, rerank=True):
        self.top_k = top_k
        self.rerank = rerank
        
    def retrieve(self, query: str) -> list:
        # Retrieve candidates from Chroma vector search (dense)
        vector_ret = get_vector_retriever(k=self.top_k * 2)
        dense_docs = vector_ret.invoke(query)
        
        # Retrieve candidates from BM25 (lexical)
        lexical_docs = []
        bm25_ret = get_bm25_retriever(k=self.top_k * 2)
        if bm25_ret:
            try:
                lexical_docs = bm25_ret.invoke(query)
            except Exception as e:
                print(f"BM25 retrieval failed: {str(e)}")

        # Deduplicate combined results
        seen_texts = set()
        combined_docs = []
        for doc in dense_docs + lexical_docs:
            if doc.page_content not in seen_texts:
                seen_texts.add(doc.page_content)
                combined_docs.append(doc)
                
        # Rerank if enabled
        if self.rerank and combined_docs:
            try:
                reranker = get_reranker()
                pairs = [[query, doc.page_content] for doc in combined_docs]
                scores = reranker.predict(pairs)
                
                # Zip scores and documents, sort by score descending
                ranked_docs = sorted(zip(scores, combined_docs), key=lambda x: x[0], reverse=True)
                
                # Annotate document metadata with the relevance score
                final_docs = []
                for score, doc in ranked_docs[:self.top_k]:
                    doc.metadata["relevance_score"] = float(score)
                    final_docs.append(doc)
                return final_docs
            except Exception as e:
                print(f"Reranking failed: {str(e)}. Returning raw candidates.")
                return combined_docs[:self.top_k]
        
        return combined_docs[:self.top_k]

def get_retriever(top_k=TOP_K, rerank=False):
    """Returns a simple wrapper interface that mimics standard retriever behavior."""
    hybrid = HybridRetriever(top_k=top_k, rerank=rerank)
    
    class Wrapper:
        def invoke(self, query: str):
            return hybrid.retrieve(query)
    
    return Wrapper()
