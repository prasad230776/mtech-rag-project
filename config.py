from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "mtech_rag"
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Models
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
CHAT_MODEL = "llama-3.3-70b-versatile"
REASONING_MODEL = "llama-3.1-8b-instant"

# Parameters
TOP_K = 5
TEMPERATURE = 0
ESCALATION_THRESHOLD = 0.70
