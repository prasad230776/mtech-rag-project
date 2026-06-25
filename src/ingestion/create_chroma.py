from pathlib import Path
import json
import sys
from langchain_core.documents import Document

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import CHROMA_PATH, COLLECTION_NAME, PROCESSED_DIR
from src.utilities.model_loader import get_huggingface_embedding_model
from langchain_chroma import Chroma

CHUNK_FILE = PROCESSED_DIR / "chunked_docs.json"

def create_chroma_db():
    if not CHUNK_FILE.exists():
        print(f"Error: Chunk file does not exist at {CHUNK_FILE}. Please run prepare_data.py first.")
        return

    print("Loading embedding model BAAI/bge-base-en-v1.5...")
    embedding_model = get_huggingface_embedding_model()

    print(f"Loading chunks from {CHUNK_FILE}...")
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        raw_docs = json.load(f)
    print(f"Loaded {len(raw_docs)} chunks")

    documents = []
    for item in raw_docs:
        document = Document(page_content=item["text"], metadata=item["metadata"])
        documents.append(document)

    print(f"Initializing Chroma DB collection '{COLLECTION_NAME}' at {CHROMA_PATH}...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=str(CHROMA_PATH),
        collection_name=COLLECTION_NAME,
    )
    print("\nChromaDB created and persisted successfully")

if __name__ == "__main__":
    create_chroma_db()
