import sys
from pathlib import Path
import re

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utilities.model_loader import get_groq_chat_model, get_qwen_reasoning_model
from src.prompts.rag_prompt import get_rag_prompt, get_claim_extraction_prompt, get_claim_verification_prompt
from src.utilities.retriever import get_retriever
from src.utilities.doc_formatter import format_docs

class V3Pipeline:
    def __init__(self):
        self.retriever = get_retriever(rerank=True)
        self.prompt = get_rag_prompt()
        self.llm = get_groq_chat_model()
        self.reasoner = get_qwen_reasoning_model()

    def extract_claims(self, response_text: str) -> list:
        """Decomposes response_text into a list of atomic assertions using Qwen."""
        if not response_text.strip() or response_text.strip() == "Information not found":
            return []
            
        extraction_prompt = get_claim_extraction_prompt()
        formatted_prompt = extraction_prompt.format_messages(response=response_text)
        try:
            response = self.reasoner.invoke(formatted_prompt)
            content = response.content.strip()
            
            # Strip reasoning/thought block if model uses chain-of-thought
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            
            lines = content.split("\n")
            claims = []
            for line in lines:
                cleaned_line = re.sub(r'^[-\*\s\d\.\)]+', '', line).strip()
                if cleaned_line:
                    claims.append(cleaned_line)
            return claims
        except Exception as e:
            print(f"Claim extraction failed: {str(e)}")
            # Fallback: split by sentences
            sentences = re.split(r'(?<=[.!?]) +', response_text)
            return [s.strip() for s in sentences if s.strip()]

    def verify_claim(self, context: str, claim: str) -> str:
        """Verifies a single claim against the context. Returns SUPPORTED, CONTRADICTED, or UNSUPPORTED."""
        verification_prompt = get_claim_verification_prompt()
        formatted_prompt = verification_prompt.format_messages(context=context, claim=claim)
        try:
            response = self.reasoner.invoke(formatted_prompt)
            result = response.content.strip().upper()
            if "SUPPORTED" in result:
                return "SUPPORTED"
            elif "CONTRADICTED" in result:
                return "CONTRADICTED"
            else:
                return "UNSUPPORTED"
        except Exception as e:
            print(f"Claim verification failed for '{claim}': {str(e)}")
            return "UNSUPPORTED"

    def invoke(self, question: str) -> dict:
        # 1. Retrieve & Rerank documents
        docs = self.retriever.invoke(question)
        context = format_docs(docs)
        
        # 2. Format Prompt & Generate Raw Answer
        formatted_prompt = self.prompt.format_messages(context=context, question=question)
        raw_response = self.llm.invoke(formatted_prompt)
        raw_answer = raw_response.content.strip()
        
        # If model outputs "Information not found", skip verification
        if raw_answer == "Information not found":
            return {
                "answer": raw_answer,
                "raw_answer": raw_answer,
                "sources": [],
                "context": context,
                "retrieved_docs": docs,
                "claims": [],
                "verified_claims": [],
                "metrics": {
                    "retrieval_relevance": 1.0,
                    "answer_relevance": 1.0,
                    "faithfulness": 1.0
                }
            }

        # 3. Extract Claims
        claims = self.extract_claims(raw_answer)
        print(f"Extracted {len(claims)} claims: {claims}")
        
        # 4. Verify Chunks
        supported_claims = []
        verified_claims_info = []
        
        for claim in claims:
            verdict = self.verify_claim(context, claim)
            verified_claims_info.append({"claim": claim, "verdict": verdict})
            if verdict == "SUPPORTED":
                supported_claims.append(claim)
                
        # 5. Compile Filtered Response
        if supported_claims:
            filtered_answer = " ".join(supported_claims)
        else:
            filtered_answer = "Information not found"
            
        # Calculate Faithfulness: Supported Claims / Total Claims
        faithfulness = len(supported_claims) / len(claims) if claims else 1.0
        
        return {
            "answer": filtered_answer,
            "raw_answer": raw_answer,
            "sources": [doc.metadata.get("source", "unknown") for doc in docs],
            "context": context,
            "retrieved_docs": docs,
            "claims": claims,
            "verified_claims": verified_claims_info,
            "metrics": {
                "retrieval_relevance": 1.0,
                "answer_relevance": 1.0,
                "faithfulness": faithfulness
            }
        }

if __name__ == "__main__":
    pipeline = V3Pipeline()
    response = pipeline.invoke("What is the hostel fee and placement percentage?")
    print("Filtered Answer:", response["answer"])
    print("Faithfulness:", response["metrics"]["faithfulness"])
