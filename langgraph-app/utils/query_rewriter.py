from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from datetime import datetime

### Get today's date
today = datetime.today().strftime("%Y-%m-%d")
current_year = datetime.today().year

verbose = True

query_rewrite_prompt = PromptTemplate.from_template("""
<ROLE>
You are a medical QIP retrieval specialist. Rewrite the user's query for optimal semantic search and retrieval performance.
Today's year is {current_year}.
</ROLE>

<STRATEGIES>
1. Preserve action phrases: Keep instructional language ("show me", "list", "are there") when present.
2. Expand terms with QIP context awareness:
    - Add synonyms ONLY for existing terms that are already present in the query (example, "fall" → "fall prevention").
    - Never introduce new medical QIP concepts or time concepts not mentioned in the original query.
3. **Strict time handling**
    - **Only process time range when original query contains clear time indicators**
    - When present: resolve to years using {current_year} and list all years.
    - **STRICTLY DO NOT add time references/context unless explicitly stated**
4. Keep the rewritten query concise and directly usable for retrieval.
5. **Mandatory term unification**
    - **When any of these appear: "posters" or "projects" or "titles" or "QIPs" → use "posters/projects/titles/QIPs"**
    - **Apply even if only one term is present in original query**
</STRATEGIES>

<INPUT>
Original query: {query}
</INPUT>

<INSTRUCTIONS>
- Output ONLY the rewritten query, no explanations.
- Never include notes, instructions, or other text.
- **Always unify "posters/projects/titles/QIPs" when any format term appears**
- **NEVER add any time context unless it is clearly stated.**
- Use natural language (no bullet points/JSON).
- Expand terms ONLY when the root term exists in the original query.
- Never add constraints (e.g. time range) that are not mentioned in the originals query.
- No new information.

""")
 
### ---------------------------------------------------------------------------------
relevance_score_prompt = PromptTemplate.from_template("""
You are a medical QI (Quality Improvement) specialist evaluating document relevance for a clinical or operational query. Judge strictly.

### Query:
{query}

### Document (excerpt):
{document}

### Scoring Criteria:
1. **Topical Match** (Does it address the query's core subject?)
2. **Actionability** (Does it provide usable insights for QI, e.g., protocols, evidence, or case studies?)
3. **Specificity** (Does it directly answer the query vs. generic information?)

### Score:
- 0.0-0.2: Irrelevant/no medical/QI content
- 0.3-0.5: Tangential (e.g., same specialty but unrelated to query)
- 0.6-0.7: Relevant but lacks actionable details
- 0.8-0.9: Highly relevant with actionable QI value
- 1.0: Perfect (direct evidence/guidance for the query)

Return ONLY the score (e.g., "0.7") with no other text.
""")


#####################################################################
# def rewrite_query(original_query: str, model_name: str = "llama3.1") -> str:
def rewrite_query(original_query: str, llm: ChatOllama) -> str:
    
    """
    Rewrite the query to include synonyms and related medical terms for better retrieval
    
    Args:
        original_query: The user's original query
        model_name: The name of the Ollama model to use (default: llama3.1)
        
    Returns:
        str: The expanded query with additional medical terminology
    """
    # Use a faster model for query rewriting since this is a lightweight task
    # llm = ChatOllama(model=model_name, temperature=0.1)
    
    # Get rewritten query
    rewrite_response = llm.invoke(
        query_rewrite_prompt.format(query=original_query, current_year=current_year)
    )
    
    rewritten_query = rewrite_response.content.strip()
    
    # Print for transparency
    print(f"\n🔄 Query Rewriting:")
    print(f"Original: \"{original_query}\"")
    print(f"Rewritten: \"{rewritten_query}\"\n")
    
    return rewritten_query 

######################################################################

def score_relevance(query: str, document, llm: ChatOllama) -> float:
    """
    Score the relevance of a document to a query on a scale of 0-1
    
    Args:
        query: The user's query
        document: The document to evaluate (Document object with page_content)
        llm: The ChatOllama model instance to use for scoring
        
    Returns:
        float: Relevance score between 0-1
    """

    ### Get document content (full)
    doc_content = document.page_content  # Use full document content without truncation
    
    # Get relevance score
    score_response = llm.invoke(
        relevance_score_prompt.format(
            query=query,
            document=doc_content
        )
    )
    
    ### Parse response to get score
    try:

        score = float(score_response.content.strip())

        score = float(score)

        score = max(0.0, min(1.0, score))
        
    except Exception as e:
        print(f"⚠️ Error parsing relevance score: {e}")
        ### Fallback if parsing fails
        print(f'TYPE: {type(score_response.content)}')
        print(f"⚠️ Failed to parse relevance score: {score_response.content}")
        score = 0.0  # Default to 0.0
    
    return score


def filter_documents_by_relevance(query: str, documents, llm: ChatOllama, threshold: float = 0.6):
    """
    Filter documents by their relevance score
    
    Args:
        query: The user's query
        documents: List of documents to filter
        llm: The ChatOllama model instance to use for scoring
        threshold: Minimum relevance score to keep (default: 0.6)
        
    Returns:
        list: Filtered and ranked list of documents with scores
    """
    
    ### Score all documents
    scored_docs = []
    print('='*100)
    print(f'Query: {query}')
    print(f"\n📊 Relevance Scoring (threshold: {threshold}):")
    
    for i, doc in enumerate(documents):
        score = score_relevance(query, doc, llm)
        doc.metadata["relevance_score"] = score
        scored_docs.append((doc, score))
        
        # Get document ID following Weaviate's standard format
        doc_id = "unknown"
        
        # First try to get UUID directly from document object
        if hasattr(doc, "uuid"):
            doc_id = doc.uuid
        # If not found, check metadata for standard Weaviate ID field
        elif hasattr(doc, "metadata") and "uuid" in doc.metadata:
            doc_id = doc.metadata["uuid"]
        
        print(f"  Document {i+1} (ID: {doc_id}): Score {score:.2f} {'✅' if score >= threshold else '❌'} > > > Year: {doc.metadata.get('year', 'Unknown')} | Section: {doc.metadata.get('section', 'Unknown')} | Title: {doc.metadata.get('title', 'Unknown')}")
    
    # Filter and sort by score
    filtered_docs = [doc for doc, score in scored_docs if score >= threshold]
    filtered_docs.sort(key=lambda d: d.metadata["relevance_score"], reverse=True)
    
    print(f"  Kept {len(filtered_docs)}/{len(documents)} documents that met the relevance threshold.")

    return filtered_docs