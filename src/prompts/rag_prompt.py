from langchain_core.prompts import ChatPromptTemplate

def get_rag_prompt() -> ChatPromptTemplate:
    """Returns the baseline RAG generation prompt template."""
    return ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer the user question based ONLY on the provided context.
If the answer cannot be found in the context, output exactly "Information not found". Do not use external knowledge or fabricate details.

Context:
{context}

Question:
{question}

Answer:"""
    )

def get_validation_prompt() -> ChatPromptTemplate:
    """Returns the context validation prompt template used by Qwen to judge chunk relevance."""
    return ChatPromptTemplate.from_template(
        """Determine if the provided context is relevant and contains enough information to help answer the user question.
Reply with a single word: "RELEVANT" if the context contains helpful facts related to the question, or "IRRELEVANT" if it does not.

Context:
{context}

Question:
{question}

Relevance judgment (RELEVANT or IRRELEVANT):"""
    )

def get_claim_extraction_prompt() -> ChatPromptTemplate:
    """Returns the prompt template to extract atomic claims from a generated response."""
    return ChatPromptTemplate.from_template(
        """Decompose the given response text into a list of standalone, factual claims. 
Each claim must represent one simple, verifiable assertion from the response.
List each claim on a new line prefixed with a dash (-). Write nothing else.

Response:
{response}

Claims list:"""
    )

def get_claim_verification_prompt() -> ChatPromptTemplate:
    """Returns the prompt template to verify a single claim against the retrieved context."""
    return ChatPromptTemplate.from_template(
        """Evaluate whether the given claim is supported by the provided context. 
Classify the claim into exactly one of three categories:
- [SUPPORTED]: The context directly supports the claim.
- [CONTRADICTED]: The context directly contradicts the claim.
- [UNSUPPORTED]: The context does not provide enough evidence to support or contradict the claim.

Context:
{context}

Claim:
{claim}

Verdict:"""
    )

def get_query_intelligence_prompt() -> ChatPromptTemplate:
    """Returns the prompt template to classify query intent and decompose multi-intent queries."""
    return ChatPromptTemplate.from_template(
        """Analyze the user's input query.
1. Classify the query as either "SINGLE" (one question or one intent) or "MULTI" (contains multiple sub-questions or compound intents).
2. If the query is "MULTI", decompose it into a clean list of sub-questions (one per line, prefixed with a dash).
3. If the query is "SINGLE", just return the original question.

Your response must be in JSON format matching this structure:
{{
  "intent_type": "SINGLE" or "MULTI",
  "sub_queries": ["sub_query_1", "sub_query_2"]
}}

Query:
{question}

JSON response:"""
    )
