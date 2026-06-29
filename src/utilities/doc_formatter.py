def format_docs(docs):
    """Formats retrieved LangChain documents by concatenating their page contents with source metadata."""
    return "\n\n".join(f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}" for doc in docs)
