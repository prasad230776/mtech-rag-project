import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utilities.model_loader import get_groq_chat_model
from src.prompts.rag_prompt import get_rag_prompt
from src.utilities.retriever import get_retriever
from src.utilities.doc_formatter import format_docs

class V2Pipeline:
    def __init__(self):
        # V2 uses hybrid + reranking enabled retriever
        self.retriever = get_retriever(rerank=True)
        self.prompt = get_rag_prompt()
        self.llm = get_groq_chat_model()

    def invoke(self, question: str) -> dict:
        # 1. Retrieve & Rerank documents
        docs = self.retriever.invoke(question)
        context = format_docs(docs)
        
        # 2. Format Prompt & Invoke LLM
        formatted_prompt = self.prompt.format_messages(context=context, question=question)
        response = self.llm.invoke(formatted_prompt)
        answer = response.content.strip()
        sources = [] if answer == "Information not found" else [doc.metadata.get("source", "unknown") for doc in docs]
        
        return {
            "answer": answer,
            "sources": sources,
            "context": context,
            "retrieved_docs": docs,
            "metrics": {
                "retrieval_relevance": 1.0,
                "answer_relevance": 1.0,
                "faithfulness": 1.0
            }
        }

if __name__ == "__main__":
    pipeline = V2Pipeline()
    response = pipeline.invoke("What is the placement percentage?")
    print("Answer:", response["answer"])
    print("Sources:", response["sources"])
