from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add project root to sys.path to allow config import
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import CHAT_MODEL, EMBEDDING_MODEL, REASONING_MODEL

load_dotenv()

def get_groq_chat_model():
    """Returns the chat generation model from Groq (Llama 3.3)."""
    return init_chat_model(model=CHAT_MODEL, model_provider="groq", temperature=0)

def get_huggingface_embedding_model():
    """Returns the Hugging Face BGE embedding model running locally on CPU."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def get_qwen_reasoning_model():
    """Returns the Qwen reasoning model from Groq (Qwen QwQ 32B)."""
    return init_chat_model(model=REASONING_MODEL, model_provider="groq", temperature=0)
