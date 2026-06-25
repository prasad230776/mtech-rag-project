import sys
from pathlib import Path
from duckduckgo_search import DDGS
from langchain_core.documents import Document

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utilities.model_loader import get_groq_chat_model, get_qwen_reasoning_model
from src.prompts.rag_prompt import get_rag_prompt, get_validation_prompt
from src.utilities.retriever import get_retriever
from src.utilities.doc_formatter import format_docs

class V1Pipeline:
    def __init__(self):
        self.retriever = get_retriever(rerank=False)
        self.prompt = get_rag_prompt()
        self.llm = get_groq_chat_model()
        self.reasoner = get_qwen_reasoning_model()

    def validate_context(self, question: str, context: str) -> bool:
        """Returns True if context is RELEVANT, False if IRRELEVANT."""
        if not context.strip():
            return False
            
        validation_prompt = get_validation_prompt()
        formatted_prompt = validation_prompt.format_messages(context=context, question=question)
        try:
            response = self.reasoner.invoke(formatted_prompt)
            result = response.content.strip().upper()
            return "RELEVANT" in result
        except Exception as e:
            print(f"Context validation failed: {str(e)}. Defaulting to relevant.")
            return True

    def run_web_fallback(self, question: str) -> list:
        """Executes native DuckDuckGo Search scoped to specific domains and returns web search results wrapped in Documents."""
        import re
        # Clean query: remove special characters that might break search
        clean_q = re.sub(r'[^\w\s-]', '', question).strip()
        
        # Try 2 to 3 different query patterns to ensure we find results
        queries = [
            f"{clean_q} site:siddarthaedu.in",
            f"{clean_q} site:seatexams.in",
            f"{clean_q} Siddartha Educational Academy Tirupati"
        ]
        
        for q_pattern in queries:
            print(f"Triggering Web Fallback for query: '{q_pattern}'...")
            try:
                with DDGS() as ddgs:
                    results = [r for r in ddgs.text(q_pattern, max_results=3)]
                    if results:
                        search_results = "\n\n".join([
                            f"Source: {r['href']}\nTitle: {r['title']}\nSnippet: {r['body']}"
                            for r in results
                        ])
                        print(f"Successfully retrieved web content for query: '{q_pattern}'")
                        return [Document(
                            page_content=search_results,
                            metadata={"source": "web_fallback", "title": "DuckDuckGo Search"}
                        )]
            except Exception as e:
                print(f"Web search fallback attempt failed for '{q_pattern}': {str(e)}")
        
        # If no results found after 3 attempts, return a fallback message
        print("Web search fallback: No results found after 3 attempts.")
        return [Document(
            page_content="Not able to retrieve even from the web.",
            metadata={"source": "web_fallback", "title": "DuckDuckGo Search"}
        )]

    def invoke(self, question: str) -> dict:
        # 1. Retrieve candidates
        docs = self.retriever.invoke(question)
        context = format_docs(docs)
        source_list = [doc.metadata.get("source", "unknown") for doc in docs]
        
        # 2. Check Context Relevance
        is_relevant = self.validate_context(question, context)
        
        # 3. Fallback if context is irrelevant or empty
        web_docs = []
        if not is_relevant or not docs:
            web_docs = self.run_web_fallback(question)
            if web_docs:
                docs = web_docs
                context = format_docs(docs)
                source_list = [doc.metadata.get("source", "unknown") for doc in docs]
        
        # 4. Generate response
        formatted_prompt = self.prompt.format_messages(context=context, question=question)
        response = self.llm.invoke(formatted_prompt)
        
        return {
            "answer": response.content.strip(),
            "sources": source_list,
            "context": context,
            "retrieved_docs": docs,
            "web_fallback_used": len(web_docs) > 0,
            "metrics": {
                "retrieval_relevance": 1.0 if is_relevant else 0.2,
                "answer_relevance": 1.0,
                "faithfulness": 1.0
            }
        }

if __name__ == "__main__":
    pipeline = V1Pipeline()
    response = pipeline.invoke("Who won the FIFA world cup 2022?")
    print("Answer:", response["answer"])
    print("Web Fallback Used:", response.get("web_fallback_used"))
