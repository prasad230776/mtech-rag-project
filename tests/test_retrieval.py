import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utilities.retriever import get_retriever, vector_store

def test_retrieval():
    query = "Where is SEAT located?"
    print(f"Querying vector store collection size...")
    try:
        count = vector_store._collection.count()
        print(f"Collection document count: {count}")
    except Exception as e:
        print(f"Error checking count: {str(e)}")
        
    print(f"\n--- Testing Simple Vector Search directly ---")
    try:
        results = vector_store.similarity_search(query, k=3)
        print(f"Direct Chroma search returned {len(results)} docs:")
        for idx, doc in enumerate(results):
            print(f"Doc {idx+1}: Content: {doc.page_content[:150]}... (Source: {doc.metadata.get('source')})")
    except Exception as e:
        print(f"Direct Chroma search failed: {str(e)}")

    print(f"\n--- Testing Hybrid Retriever wrapper ---")
    try:
        retriever = get_retriever(rerank=False)
        docs = retriever.invoke(query)
        print(f"Retriever wrapper returned {len(docs)} docs:")
        for idx, doc in enumerate(docs):
            print(f"Doc {idx+1}: Content: {doc.page_content[:150]}... (Source: {doc.metadata.get('source')})")
    except Exception as e:
        print(f"Retriever wrapper failed: {str(e)}")

if __name__ == "__main__":
    test_retrieval()
