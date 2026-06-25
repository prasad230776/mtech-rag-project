import sys
from pathlib import Path
import json
import re

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.versions.v4_escalated_rag import V4Pipeline
from src.utilities.model_loader import get_qwen_reasoning_model, get_groq_chat_model
from src.prompts.rag_prompt import get_query_intelligence_prompt
from config import ESCALATION_THRESHOLD

class V5Pipeline:
    def __init__(self):
        self.v4_pipeline = V4Pipeline()
        self.reasoner = get_qwen_reasoning_model()
        self.llm = get_groq_chat_model()

    def analyze_query(self, question: str) -> dict:
        """Classifies and decomposes the question using Qwen reasoning model."""
        prompt_tmpl = get_query_intelligence_prompt()
        formatted_prompt = prompt_tmpl.format_messages(question=question)
        try:
            response = self.reasoner.invoke(formatted_prompt)
            content = response.content.strip()
            
            # Strip reasoning/thought block
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            
            # Clean markdown code block decorators
            cleaned = content
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()
            
            # Find the outermost curly braces
            start_idx = cleaned.find('{')
            end_idx = cleaned.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = cleaned[start_idx:end_idx+1]
                return json.loads(json_str)
                
            return json.loads(cleaned)
        except Exception as e:
            print(f"Query analysis failed: {str(e)}. Defaulting to SINGLE.")
            return {"intent_type": "SINGLE", "sub_queries": []}

    def invoke(self, question: str, threshold: float = ESCALATION_THRESHOLD) -> dict:
        # 1. Query Intelligence classification
        analysis = self.analyze_query(question)
        intent_type = analysis.get("intent_type", "SINGLE")
        sub_queries = analysis.get("sub_queries", [])
        
        print(f"Query Analysis - Intent: {intent_type}, Sub-queries: {sub_queries}")
        
        if intent_type == "MULTI" and sub_queries:
            sub_results = []
            for sq in sub_queries:
                print(f"Processing sub-query: '{sq}'...")
                sub_res = self.v4_pipeline.invoke(sq, threshold=threshold)
                sub_results.append((sq, sub_res))
                
            # Aggregate answers
            answers_text = []
            sources_set = set()
            all_docs = []
            escalated_count = 0
            cs_sum = 0.0
            
            for sq, res in sub_results:
                answers_text.append(f"For '{sq}': {res['answer']}")
                sources_set.update(res.get("sources", []))
                all_docs.extend(res.get("retrieved_docs", []))
                cs_sum += res.get("confidence_score", 1.0)
                if res.get("decision") == "ESCALATE":
                    escalated_count += 1
            
            aggregated_answer = "\n".join(answers_text)
            avg_cs = cs_sum / len(sub_results)
            
            # If any sub-query was escalated, flag the overall response
            decision = "ESCALATE" if escalated_count > 0 else "ACCEPT"
            message = f"Merged response from {len(sub_queries)} queries. ({escalated_count} escalated)" if escalated_count > 0 else "Merged reliable response."
            
            return {
                "answer": aggregated_answer,
                "confidence_score": avg_cs,
                "decision": decision,
                "message": message,
                "sources": list(sources_set),
                "retrieved_docs": all_docs,
                "sub_queries_processed": sub_queries,
                "intent_type": "MULTI",
                "metrics": {
                    "retrieval_relevance": avg_cs,  # Average confidence proxy
                    "answer_relevance": 1.0,
                    "faithfulness": 1.0 - (escalated_count / len(sub_queries))
                }
            }
        else:
            # Process as SINGLE
            res = self.v4_pipeline.invoke(question, threshold=threshold)
            res["intent_type"] = "SINGLE"
            return res

if __name__ == "__main__":
    pipeline = V5Pipeline()
    response = pipeline.invoke("What is the hostel fee and placement percentage?")
    print("Answer:", response["answer"])
    print("Decision:", response["decision"])
