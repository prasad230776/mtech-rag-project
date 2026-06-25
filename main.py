from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import shutil
from pathlib import Path
import os
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))
from config import RAW_DIR, ESCALATION_THRESHOLD
from src.versions.v0_basic_rag import V0Pipeline
from src.versions.v1_fallback_rag import V1Pipeline
from src.versions.v2_reranked_rag import V2Pipeline
from src.versions.v3_verified_rag import V3Pipeline
from src.versions.v4_escalated_rag import V4Pipeline
from src.versions.v5_intelligent_rag import V5Pipeline
from src.ingestion.prepare_data import main as run_prepare_data
from src.ingestion.create_chroma import create_chroma_db

app = FastAPI(
    title="Multi-Stage Hallucination Mitigation RAG API",
    description="Backend API orchestrating baseline V0 to intelligent V5 RAG pipelines.",
    version="1.0.0"
)

# Initialize pipelines lazily
pipelines = {}

def get_pipeline(version: str):
    v = version.lower()
    if v not in pipelines:
        if v == "v0":
            pipelines[v] = V0Pipeline()
        elif v == "v1":
            pipelines[v] = V1Pipeline()
        elif v == "v2":
            pipelines[v] = V2Pipeline()
        elif v == "v3":
            pipelines[v] = V3Pipeline()
        elif v == "v4":
            pipelines[v] = V4Pipeline()
        elif v == "v5":
            pipelines[v] = V5Pipeline()
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported version: {version}")
    return pipelines[v]

class QueryRequest(BaseModel):
    question: str
    version: str = "v5"
    escalation_threshold: float = ESCALATION_THRESHOLD

@app.post("/api/v1/ask")
def ask_question(req: QueryRequest):
    try:
        pipeline = get_pipeline(req.version)
        
        # Invoke matching pipeline
        if req.version.lower() in ["v4", "v5"]:
            result = pipeline.invoke(req.question, threshold=req.escalation_threshold)
        else:
            result = pipeline.invoke(req.question)
            
        # Re-format standard output dictionary
        return {
            "answer": result.get("answer"),
            "raw_answer": result.get("raw_answer", result.get("answer")),
            "version": req.version,
            "confidence_score": result.get("confidence_score", 1.0),
            "decision": result.get("decision", "ACCEPT"),
            "message": result.get("message", "Generated answer successfully."),
            "sources": result.get("sources", []),
            "intent_type": result.get("intent_type", "SINGLE"),
            "metrics": result.get("metrics", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/ingest")
async def ingest_files(
    pdfs: list[UploadFile] = File(None),
    csvs: list[UploadFile] = File(None)
):
    pdf_dir = RAW_DIR / "pdfs"
    csv_dir = RAW_DIR / "csvs"
    
    # Ensure dirs exist
    pdf_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    saved_pdfs = []
    saved_csvs = []
    
    # Save PDF files
    if pdfs:
        for pdf in pdfs:
            if not pdf.filename.endswith(".pdf"):
                raise HTTPException(status_code=400, detail="Only PDF files are allowed in pdfs field")
            file_path = pdf_dir / pdf.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(pdf.file, buffer)
            saved_pdfs.append(pdf.filename)
            
    # Save CSV files
    if csvs:
        for csv in csvs:
            if not csv.filename.endswith(".csv"):
                raise HTTPException(status_code=400, detail="Only CSV files are allowed in csvs field")
            file_path = csv_dir / csv.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(csv.file, buffer)
            saved_csvs.append(csv.filename)
            
    # Trigger parsing and database creation
    try:
        print("Starting data ingestion scripts...")
        run_prepare_data()
        create_chroma_db()
        
        # Reset cached retrievers to pick up new documents
        from src.utilities import retriever
        retriever._bm25_retriever = None
        
        return {
            "status": "success",
            "message": "Ingestion and Vector database creation finished.",
            "uploaded_pdfs": saved_pdfs,
            "uploaded_csvs": saved_csvs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline failed: {str(e)}")

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy"
    }
