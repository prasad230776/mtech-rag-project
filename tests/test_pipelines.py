import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.versions.v0_basic_rag import V0Pipeline
from src.versions.v1_fallback_rag import V1Pipeline
from src.versions.v2_reranked_rag import V2Pipeline
from src.versions.v3_verified_rag import V3Pipeline
from src.versions.v4_escalated_rag import V4Pipeline
from src.versions.v5_intelligent_rag import V5Pipeline

def test_pipeline_versions():
    test_query_simple = "What placement packages are reported at SEAT?"
    test_query_compound = "Where is SEAT located and what placement packages are reported at SEAT?"
    test_query_irrelevant = "Who won the FIFA world cup 2022?"

    print("====================================================")
    print("Testing Pipeline Version 0 (Basic RAG)")
    print("====================================================")
    v0 = V0Pipeline()
    print("Invoking V0 pipeline...")
    res0 = v0.invoke(test_query_simple)
    print("V0 invocation finished!")
    print("Answer:", res0["answer"])
    print("Sources:", res0["sources"])

    print("\n====================================================")
    print("Testing Pipeline Version 1 (Web Fallback)")
    print("====================================================")
    v1 = V1Pipeline()
    print("Query:", test_query_irrelevant)
    print("Invoking V1 pipeline...")
    res1 = v1.invoke(test_query_irrelevant)
    print("V1 invocation finished!")
    print("Answer:", res1["answer"])
    print("Fallback Used:", res1.get("web_fallback_used"))
    print("Sources:", res1["sources"])

    print("\n====================================================")
    print("Testing Pipeline Version 2 (Reranking)")
    print("====================================================")
    v2 = V2Pipeline()
    print("Invoking V2 pipeline...")
    res2 = v2.invoke(test_query_simple)
    print("V2 invocation finished!")
    print("Answer:", res2["answer"])
    print("Sources:", res2["sources"])

    print("\n====================================================")
    print("Testing Pipeline Version 3 (Claim Verification)")
    print("====================================================")
    v3 = V3Pipeline()
    print("Invoking V3 pipeline...")
    res3 = v3.invoke(test_query_simple)
    print("V3 invocation finished!")
    print("Answer:", res3["answer"])
    print("Claims Extracted:", res3["claims"])
    print("Claims Verified:", res3["verified_claims"])
    print("Faithfulness:", res3["metrics"]["faithfulness"])

    print("\n====================================================")
    print("Testing Pipeline Version 4 (Confidence & Escalation)")
    print("====================================================")
    v4 = V4Pipeline()
    # Test with low confidence query
    print("Invoking V4 pipeline (Irrelevant query)...")
    res4_low = v4.invoke(test_query_irrelevant, threshold=0.70)
    print("V4 Irrelevant invocation finished!")
    print("Query (Irrelevant):", test_query_irrelevant)
    print("Decision:", res4_low["decision"])
    print("Answer:", res4_low["answer"])
    print("Confidence:", res4_low["confidence_score"])
    
    # Test with high confidence query
    print("Invoking V4 pipeline (Relevant query)...")
    res4_high = v4.invoke(test_query_simple, threshold=0.70)
    print("V4 Relevant invocation finished!")
    print("Query (Relevant):", test_query_simple)
    print("Decision:", res4_high["decision"])
    print("Answer:", res4_high["answer"])
    print("Confidence:", res4_high["confidence_score"])

    print("\n====================================================")
    print("Testing Pipeline Version 5 (Query Intelligence)")
    print("====================================================")
    v5 = V5Pipeline()
    print("Invoking V5 pipeline...")
    res5 = v5.invoke(test_query_compound)
    print("V5 invocation finished!")
    print("Query:", test_query_compound)
    print("Answer:", res5["answer"])
    print("Intent type:", res5.get("intent_type"))
    print("Sub-queries:", res5.get("sub_queries_processed"))
    print("Decision:", res5["decision"])
    print("Confidence Score:", res5["confidence_score"])

if __name__ == "__main__":
    test_pipeline_versions()
