def format_docs(docs):
    """Formats retrieved LangChain documents by concatenating their page contents."""
    return "\n\n".join(doc.page_content for doc in docs)
