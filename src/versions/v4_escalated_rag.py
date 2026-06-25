import sys
from pathlib import Path
import math

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.versions.v3_verified_rag import V3Pipeline
from config import ESCALATION_THRESHOLD

def sigmoid(x):
    """Applies sigmoid function to map raw model outputs to [0,1]."""
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

class V4Pipeline:
    def __init__(self):
        self.v3_pipeline = V3Pipeline()

    def calculate_confidence(self, retrieved_docs, faithfulness) -> tuple:
        """
        Calculates confidence score: CS = (0.60 * CR) + (0.40 * RQ)
        CR = Average relevance score of retrieved documents after sigmoid scaling.
        RQ = Faithfulness score (ratio of supported claims to total claims).
        """
        # CR calculation
        if not retrieved_docs:
            cr = 0.0
        else:
            scores = []
            for doc in retrieved_docs:
                raw_score = doc.metadata.get("relevance_score", 0.0)
                # Map raw reranker logit to [0, 1] using sigmoid
                scores.append(sigmoid(raw_score))
            cr = sum(scores) / len(scores)

        # RQ calculation
        rq = faithfulness

        # CS calculation
        cs = (0.60 * cr) + (0.40 * rq)
        return cs, cr, rq

    def invoke(self, question: str, threshold: float = ESCALATION_THRESHOLD) -> dict:
        # Run V3 pipeline to get retrieved documents, claims, and verified answers
        v3_result = self.v3_pipeline.invoke(question)
        
        # Calculate confidence score
        faithfulness = v3_result["metrics"]["faithfulness"]
        cs, cr, rq = self.calculate_confidence(v3_result["retrieved_docs"], faithfulness)
        
        # Determine escalation
        decision = "ACCEPT"
        final_answer = v3_result["answer"]
        message = "Reliable answer generated."
        
        # Escalate if confidence is low OR if no claims were verified and the output was not "Information not found"
        if cs < threshold:
            decision = "ESCALATE"
            final_answer = "The response could not be confidently verified. Please consult the institution office."
            message = f"Escalated due to low confidence score ({cs:.2f} < {threshold:.2f})."
        elif final_answer == "Information not found":
            decision = "ACCEPT"
            message = "No context was found to verify this answer."
            
        result = v3_result.copy()
        result.update({
            "answer": final_answer,
            "confidence_score": cs,
            "decision": decision,
            "message": message,
            "metrics": {
                "retrieval_relevance": cr,
                "answer_relevance": v3_result["metrics"]["answer_relevance"],
                "faithfulness": rq
            }
        })
        return result

if __name__ == "__main__":
    pipeline = V4Pipeline()
    response = pipeline.invoke("What is the placement percentage?")
    print("Decision:", response["decision"])
    print("Answer:", response["answer"])
    print("Confidence Score:", response["confidence_score"])
