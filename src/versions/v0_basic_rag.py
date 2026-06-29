import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utilities.model_loader import get_groq_chat_model
from src.prompts.rag_prompt import get_rag_prompt
from src.utilities.retriever import get_retriever
from src.utilities.doc_formatter import format_docs

class V0Pipeline:
    def __init__(self):
        # V0 uses simple dense retriever (no reranking)
        self.retriever = get_retriever(rerank=False)
        self.prompt = get_rag_prompt()
        self.llm = get_groq_chat_model()

    def invoke(self, question: str) -> dict:
        # 1. Retrieve documents
        docs = self.retriever.invoke(question)
        context = format_docs(docs)
        
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
                "retrieval_relevance": 1.0,  # Placeholder for evaluators
                "answer_relevance": 1.0,
                "faithfulness": 1.0
            }
        }

if __name__ == "__main__":
    pipeline = V0Pipeline()
    response = pipeline.invoke("What is the placement rate?")
    print("Answer:", response["answer"])
    print("Sources:", response["sources"])
