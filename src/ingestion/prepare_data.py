import json
from pathlib import Path
import uuid
import sys
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import BASE_DIR, RAW_DIR, PROCESSED_DIR

# Paths
PDF_FOLDER = RAW_DIR / "pdfs"
CSV_FOLDER = RAW_DIR / "csvs"

# Ensure directories exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PDF_FOLDER.mkdir(parents=True, exist_ok=True)
CSV_FOLDER.mkdir(parents=True, exist_ok=True)

EXTRACTED_FILE = PROCESSED_DIR / "extracted_docs.json"
CLEANED_FILE = PROCESSED_DIR / "cleaned_docs.json"
CHUNKED_FILE = PROCESSED_DIR / "chunked_docs.json"


def clean_text(text: str) -> str:
    """Removes PDF extraction noise, URLs, empty lines, and page numbers."""
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if len(line) < 2:
            continue

        lower_line = line.lower()

        # Remove URL links
        if "www." in lower_line or "http://" in lower_line or "https://" in lower_line:
            continue

        # Remove page number indicators
        if lower_line.startswith("page") or lower_line.startswith("p. "):
            continue

        # Remove stand-alone digit lines (usually page numbers/indexes)
        if line.isdigit():
            continue

        cleaned_lines.append(line)

    return " ".join(cleaned_lines)


def extract_pdf_documents():
    """Extracts text from all PDFs in the PDF folder page-by-page."""
    pdf_documents = []
    pdf_files = list(PDF_FOLDER.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDFs in {PDF_FOLDER}")

    for pdf_path in pdf_files:
        print(f"Processing PDF: {pdf_path.name}")
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                if text.strip():
                    pdf_documents.append({
                        "text": text,
                        "metadata": {
                            "doc_type": "pdf",
                            "source": pdf_path.name,
                            "page": page_num + 1,
                        }
                    })
        except Exception as e:
            print(f"Error reading PDF {pdf_path.name}: {str(e)}")

    return pdf_documents


def load_faq_documents():
    """Loads all CSV files in the CSV folder and parses rows as FAQ documents."""
    faq_documents = []
    csv_files = list(CSV_FOLDER.glob("*.csv"))

    print(f"Found {len(csv_files)} CSVs in {CSV_FOLDER}")

    for csv_path in csv_files:
        print(f"Processing CSV: {csv_path.name}")
        try:
            df = pd.read_csv(csv_path)
            # Ensure required columns exist
            if "question" not in df.columns or "answer" not in df.columns:
                print(f"Skipping CSV {csv_path.name} (missing 'question' or 'answer' column)")
                continue

            for idx, row in df.iterrows():
                question = str(row["question"]).strip()
                answer = str(row["answer"]).strip()
                category = str(row["category"]).strip() if "category" in df.columns else "general"

                faq_text = f"Question: {question}\nAnswer: {answer}"

                faq_documents.append({
                    "text": faq_text,
                    "metadata": {
                        "doc_type": "faq",
                        "faq_id": idx,
                        "category": category,
                        "source": csv_path.name,
                    }
                })
        except Exception as e:
            print(f"Error reading CSV {csv_path.name}: {str(e)}")

    return faq_documents


def clean_documents(documents):
    """Filters document texts through clean_text."""
    cleaned_docs = []
    for doc in documents:
        cleaned_text = clean_text(doc["text"])
        if cleaned_text.strip():
            cleaned_docs.append({
                "text": cleaned_text,
                "metadata": doc["metadata"]
            })
    return cleaned_docs


def process_pdf_documents(pdf_docs):
    """Recursively chunks PDF documents to preserve semantic context."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    processed_pdfs = []

    for doc in pdf_docs:
        chunks = splitter.split_text(doc["text"])
        for idx, chunk in enumerate(chunks):
            metadata = doc["metadata"].copy()
            metadata["chunk_id"] = str(uuid.uuid4())
            metadata["chunk_index"] = idx
            processed_pdfs.append({
                "text": chunk,
                "metadata": metadata
            })

    return processed_pdfs


def process_faq_documents(faq_docs):
    """Saves FAQ documents as individual chunks without recursive segmenting."""
    processed_faqs = []
    for doc in faq_docs:
        metadata = doc["metadata"].copy()
        metadata["chunk_id"] = str(uuid.uuid4())
        metadata["chunk_index"] = 0
        processed_faqs.append({
            "text": doc["text"],
            "metadata": metadata
        })

    return processed_faqs


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    print("\n--- STEP 1: Extract PDFs ---")
    pdf_docs = extract_pdf_documents()

    print("\n--- STEP 2: Load FAQs ---")
    faq_docs = load_faq_documents()

    all_docs = pdf_docs + faq_docs
    if not all_docs:
        print("Warning: No documents found to process. Please place PDFs in data/raw/pdfs and CSVs in data/raw/csvs.")
        return

    print(f"\nTotal extracted docs: {len(all_docs)}")
    save_json(all_docs, EXTRACTED_FILE)

    print("\n--- STEP 3: Cleaning ---")
    cleaned_docs = clean_documents(all_docs)
    save_json(cleaned_docs, CLEANED_FILE)

    print("\n--- STEP 4: PDF Chunking ---")
    cleaned_pdf_docs = [doc for doc in cleaned_docs if doc["metadata"]["doc_type"] == "pdf"]
    processed_pdfs = process_pdf_documents(cleaned_pdf_docs)
    print(f"Generated {len(processed_pdfs)} PDF chunks")

    print("\n--- STEP 5: FAQ Processing ---")
    cleaned_faq_docs = [doc for doc in cleaned_docs if doc["metadata"]["doc_type"] == "faq"]
    processed_faqs = process_faq_documents(cleaned_faq_docs)
    print(f"Generated {len(processed_faqs)} FAQ chunks")

    chunked_docs = processed_pdfs + processed_faqs
    save_json(chunked_docs, CHUNKED_FILE)
    print(f"\nSuccessfully saved all {len(chunked_docs)} chunks to {CHUNKED_FILE}")


if __name__ == "__main__":
    main()
