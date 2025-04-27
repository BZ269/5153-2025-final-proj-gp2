import os
import pandas as pd
from typing import Annotated, Literal, Sequence, Optional, Any, Union
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.errors import GraphRecursionError

import weaviate
from weaviate.classes.query import Filter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_core.tools import tool
from ast import literal_eval
import re
import traceback
from utils.query_rewriter import rewrite_query, filter_documents_by_relevance
from utils.metadata_filters import MetadataFilterBuilder

from dotenv import load_dotenv
import os
from datetime import datetime
import warnings
from pathlib import Path

from config import MODELS, MODEL_TEMPERATURES

from langsmith import traceable

load_dotenv(dotenv_path=Path(__file__).parents[1] / '.env')

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

verbose = True
verbose_2 = False

# ----------------------------------------
#  Initialize Models, Weaviate and Vector Store
# ----------------------------------------
embedding_model = OllamaEmbeddings(model=MODELS["embeddings"])

llm = ChatOllama(model=MODELS["rag_generator"], temperature=MODEL_TEMPERATURES["rag_generator"])
llm_rewriter = ChatOllama(model=MODELS["llama3"], temperature=MODEL_TEMPERATURES["llama3"])

exported_llm = llm

client = weaviate.connect_to_local(port=8081)

vectorstore = WeaviateVectorStore(
    client=client,
    index_name="MedicalQIDocument_Poster_Chunks",
    embedding=embedding_model,
    text_key="content"
)

exported_client = client

# ----------------------------------------
# Define Section List and Classifier
# ----------------------------------------
SECTION_LIST = [
    "TITLE OF PROJECT", "BACKGROUND", "INTRODUCTION", "MISSION STATEMENT", "ANALYSIS OF PROBLEM",
    "ROOT CAUSE ANALYSIS", "METHODOLOGY", "INTERVENTIONS / INITIATIVES",
    "RESULTS", "OUTCOME", "DISCUSSION", "SUSTAINABILITY AND SPREAD", "CONCLUSION"
    ]

# ----------------------------------------
# Define LangGraph State Schema
# ----------------------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "Chat history"]
    metadata_filter: Optional[Any]  

# ----------------------------------------
# Define Tools
# ----------------------------------------
@tool
def retriever_tool(
    query: str = "The user's full query/request.",
    state: AgentState = AgentState(messages=[], metadata_filter=None),
    llm_classifier: ChatOllama = None,
    k: int = 15,
    relevance_threshold: float = 0.6
    ) -> str:

    """
    Semantic search over medical QI documents. 
    Pass the full user message in both `query` and `state.messages` 
    so it can be reused for classification, reranking, etc.
    """

    section_classifier_prompt = PromptTemplate.from_template("""
    You are a classifier that maps user questions to relevant QI document sections.

    Given this list of section names:
    {section_list}

    IMPORTANT MAPPING RULES:
    - Questions about specific QI projects or initiatives should include "TITLE OF PROJECT"
    - Questions about problems or issues should include "ANALYSIS OF PROBLEM" and "ROOT CAUSE ANALYSIS"
    - Questions about methods should include "METHODOLOGY"
    - Questions about results or impact should include "RESULTS" and "OUTCOME"
    - Questions asking to list all projects related to a topic should include "TITLE OF PROJECT"

    Examples:
    - Question: "What QIPs were done on patient safety?" → ["TITLE OF PROJECT"]
    - Question: "What caused medication errors?" → ["ANALYSIS OF PROBLEM", "ROOT CAUSE ANALYSIS"]
    - Question: "How was the fall prevention program implemented?" → ["METHODOLOGY", "INTERVENTIONS / INITIATIVES"]
    - Question: "What were the outcomes of fall reduction efforts?" → ["RESULTS", "OUTCOME"]

    Return a Python list of **exact section strings** (case-sensitive) that match the user's question intent.
    ALWAYS include "TITLE OF PROJECT" if the question is asking about specific QI projects or initiatives or posters.
    Do not include any explanations. Only return a Python list of strings.

    Question:
    "{question}"

    Matching Sections:
    """)

    prompt = section_classifier_prompt.format(
        question=query,
        section_list=", ".join(SECTION_LIST)
    )
    
    raw_msg = llm_classifier.invoke(prompt)
    raw = raw_msg.content

    try:
        list_match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not list_match:
            raise ValueError("No list found in LLM output.")

        extracted = list_match.group(0)
        predicted = literal_eval(extracted)
        if not isinstance(predicted, list):
            raise ValueError("Parsed result is not a list.")

    except Exception as e:
        print(f"\n⚠️ Failed to parse LLM section prediction: {e}")
        predicted = SECTION_LIST

    print(f"\n📌 LLM selected sections: {predicted}")

    metadata_dict, _, clean_query = MetadataFilterBuilder.parse_metadata_from_query(
        query=query,
        client=client,
        llm=llm,
        collection_name="MedicalQIDocument_Poster_Chunks"
    )

    print(f"\n📌 Metadata filters: {metadata_dict}")
    print(f"\n📌 Clean query: {clean_query}")

    metadata_filter_from_query = None
    
    if metadata_dict:
        year_filter = None
        hospital_filter = None
        
        for field, value in metadata_dict.items():
            if field == "year":
                if isinstance(value, list):
                    for val in value:
                        try:
                            f_val = Filter.by_property("year").equal(str(val))
                            year_filter = f_val if year_filter is None else year_filter | f_val
                        except:
                            pass
                else:
                    try:
                        year_filter = Filter.by_property("year").equal(str(value))
                    except:
                        pass
            
            elif field == "hospital":
                if isinstance(value, list):
                    for val in value:
                        f_val = Filter.by_property("hospital").equal(val)
                        hospital_filter = f_val if hospital_filter is None else hospital_filter | f_val
                else:
                    hospital_filter = Filter.by_property("hospital").equal(value)
        
        metadata_filter_from_query = {
            "year_filter": year_filter,
            "hospital_filter": hospital_filter
        }

    section_filter = None 
    for sec in predicted:
        f = Filter.by_property("section").equal(sec)
        section_filter = f if section_filter is None else section_filter | f

    if verbose:
        print(f'\n📌 section_filter: {section_filter}') 

    metadata_filter_from_state = state.get("metadata_filter", None)

    if verbose:
        print(f'\n📌 metadata_filter_from_state: {metadata_filter_from_state}')
        print(f'\n📌 metadata_filter_from_query: {metadata_filter_from_query}')
        print(f'\n📌 section_filter: {section_filter}')

    year_filter = None
    if metadata_filter_from_state and "year_filter" in metadata_filter_from_state:
        year_filter = metadata_filter_from_state["year_filter"]
        if verbose:
            print(f'\n📌 ACTIVE FILTER contains year_filter: {year_filter}') 

    if metadata_filter_from_query and "year_filter" in metadata_filter_from_query:
        year_filter = year_filter | metadata_filter_from_query["year_filter"] if year_filter else metadata_filter_from_query["year_filter"]
        if verbose:
            print(f'\n📌 QUERY FILTER contains year_filter: {metadata_filter_from_query["year_filter"]}') 

    hospital_filter = None
    if metadata_filter_from_state and "hospital_filter" in metadata_filter_from_state:
        hospital_filter = metadata_filter_from_state["hospital_filter"]
        if verbose:
            print(f'\n📌 ACTIVE FILTER contains hospital_filter: {hospital_filter}') 

    if metadata_filter_from_query and "hospital_filter" in metadata_filter_from_query:
        hospital_filter = hospital_filter | metadata_filter_from_query["hospital_filter"] if hospital_filter else metadata_filter_from_query["hospital_filter"]
        if verbose:
            print(f'\n📌 QUERY FILTER contains hospital_filter: {metadata_filter_from_query["hospital_filter"]}') 
    
    print(f'\n📌 year_filter: {year_filter}')
    print(f'\n📌 hospital_filter: {hospital_filter}')

    final_filter = None
    if year_filter:
        final_filter = year_filter
    if hospital_filter:
        final_filter = final_filter & hospital_filter if final_filter else hospital_filter
    if section_filter:
        final_filter = final_filter & section_filter if final_filter else section_filter

    print(f'\n📌 final_filter: {final_filter}')

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": k,
            "alpha": 0.5,
            "score_threshold": 0.2,
            "filters": final_filter
        }
    )

    docs = retriever.invoke(clean_query)

    num_docs_retrieved = len(docs)
    if verbose:
        print(f"\n📌 Number of documents retrieved: {num_docs_retrieved}")

    if verbose: 
        print('='*100)

        results_with_scores = vectorstore.similarity_search_with_score(clean_query, k=k, filters=final_filter)
        print(f"\n📄 Number of results with scores: {len(results_with_scores)}")

        score_map = {result.page_content: score for result, score in results_with_scores}
        
        for doc in docs:
            if doc.page_content in score_map:
                doc.metadata['similarity_score'] = score_map[doc.page_content]
                if verbose_2:
                    print(f"Chunk: {doc}, Score: {score_map[doc.page_content]}")
    
    if verbose:
        print('='*100)
        print(f"\n📄 Retrieved Documents ({len(docs)} total):")
        for i, doc in enumerate(docs):
            print(f"\nDocument {i+1}:")
            print(f"Similarity Score (from retrieval, not used for grading): {doc.metadata.get('similarity_score')}")
            print(f"Title: {doc.metadata.get('title')}")
            print(f"Section: {doc.metadata.get('section')}")
            print(f"Year: {doc.metadata.get('year')}")
            print(f"Hospital: {doc.metadata.get('hospital')}")
            print(f"Project Code: {doc.metadata.get('project_code')}")
            if verbose_2:
                print(f"Content: {doc.page_content}")  

    relevant_docs = filter_documents_by_relevance(clean_query,
                                                 documents=docs, 
                                                 threshold=relevance_threshold,
                                                 llm=llm)

    project_codes = [doc.metadata.get('project_code') for doc in relevant_docs if doc.metadata.get('project_code')]

    global unique_project_codes
    unique_project_codes = list(set(project_codes))

    if verbose:
        print(f"\n📌 Project codes of relevant chunks: {unique_project_codes}")

    if verbose:
        print('-'*100)
        print(f"\n📄 Relevant Documents ({len(relevant_docs)} total):")
        for i, doc in enumerate(relevant_docs):
            print(f"\nDocument {i+1}:")
            print(f"Similarity Score (from retrieval, not used for grading): {doc.metadata.get('similarity_score')}")
            print(f"Relevance Scorer Output (from relevance scorer): {doc.metadata.get('relevance_score')}")
            print(f"Title: {doc.metadata.get('title')}")
            print(f"Section: {doc.metadata.get('section')}")
            print(f"Year: {doc.metadata.get('year')}")
            print(f"Hospital: {doc.metadata.get('hospital')}")
            print(f"Project Code: {doc.metadata.get('project_code')}")
            if verbose_2:
                print(f"Content: {doc.page_content}")

    return  clean_query, "\n\n".join(f"Source: {doc.metadata.get('source')}\nSection: {doc.metadata.get('section')}\nContent: {doc.page_content}" for doc in relevant_docs)


@traceable(tags=["agent:agentic_rag", "function:is_medical_qi_query"])
def is_medical_qi_query(state: AgentState) -> str:
    """
    Use LLM to determine if the query is related to medical quality improvement projects.
    Returns 'retrieve' if related, 'not_relevant' otherwise.
    """
    messages = state["messages"]
    if not messages:
        return "not_relevant"
    
    last_message = messages[-1]
    if not isinstance(last_message, HumanMessage):
        return "not_relevant"
    
    query = last_message.content
    
    model = ChatOllama(model=MODELS["llama3"], temperature=MODEL_TEMPERATURES["llama3"])
    
    prompt = f"""
    Analyze the following query to determine if it relates to Medical Quality Improvement Projects (QIPs). 
    QIPs involve systematic efforts to improve healthcare quality, including:

    [INCLUDE]
    - Clinical process optimization/redesign
    - Patient safety initiatives
    - Healthcare outcome measurement/analysis
    - Clinical audit processes
    - Error reduction strategies
    - Healthcare protocol development/implementation
    - Care coordination improvements
    - Compliance with clinical guidelines/standards
    - Data-driven performance improvement
    - Resource utilization optimization

    [EXCLUDE]
    - General medical questions
    - Personal health advice
    - Disease treatment specifics
    - Pharmacological information
    - Non-quality related research
    - Basic hospital operations
    - Individual clinical cases

    Evaluate based on these criteria:

    Question: {query}

    Is this query fundamentally about IMPROVING SYSTEMS/PROCESSES in healthcare quality? 
    Answer ONLY with lowercase 'yes' or 'no' without punctuation.
    """
    
    response = model.invoke(prompt)
    
    response_text = response.content.lower().strip()
    print(f"\nRelevance check for '{query}': {response_text}")
    
    if 'yes' in response_text:
        return "retrieve"
    else:
        return "not_relevant"


def handle_not_relevant(state: AgentState) -> AgentState:
    """
    Handle queries that are not related to medical QI projects.
    """
    response = AIMessage(content="I'm specialized in medical quality improvement projects. Please ask a question related to QIPs, hospital initiatives, patient safety, or similar healthcare improvement topics.")
    
    return {
        "messages": state["messages"] + [response],
        "metadata_filter": state.get("metadata_filter", {})
    }


@traceable(tags=["agent:agentic_rag", "function:custom_retriever"])
def custom_retriever(state: AgentState, llm: ChatOllama, relevance_threshold: float = 0.6) -> AgentState:
    """
    Wrapper for retriever_tool that properly handles state.
    """
    messages = state["messages"]
    last_message = messages[-1]
    query = last_message.content if isinstance(last_message, HumanMessage) else "quality improvement projects"
    
    clean_query, retrieval_results = retriever_tool.run({
        "query": query, 
        "state": state, 
        "llm_classifier": llm, 
        "relevance_threshold": relevance_threshold
    })
    
    if verbose:
        print(f"\n📌 Retrieval results >>>>> {retrieval_results}") 

    clean_query_message = HumanMessage(content=clean_query)

    return {
        "messages": state["messages"] + [clean_query_message, AIMessage(content=retrieval_results)],
        "metadata_filter": state.get("metadata_filter", {})
    }

# ----------------------------------------
# Relevance Grader
# ----------------------------------------

@traceable(tags=["agent:agentic_rag", "function:grade_documents"])
def grade_documents(state: AgentState, llm: ChatOllama) -> Literal["generate", "rewrite"]:
    print("\n--- CHECKING RELEVANCE (of combined retrieved documents) ---")

    class GradeOutput(BaseModel):
        binary_score: str = Field(description="Relevance score: 'yes' or 'no'")

    model = llm.with_structured_output(GradeOutput) 

    prompt = PromptTemplate(
        template="""You are an expert document grader assessing whether the retrieved documents can collectively answer the user's question. Consider that:
        1. Documents may contain partial information that contributes to a complete answer
        2. Some relevance may be implied rather than explicit
        3. The context may contain supporting evidence even if not directly answering

        **Evaluation Guidelines:**
        - Answer 'yes' if the documents contain information that could help answer the question, even partially
        - Answer 'yes' if the documents provide context or supporting evidence relevant to the question
        - Answer 'no' ONLY if the documents are completely unrelated or provide no value

        **Decision:** Strictly provide a single word ('yes' or 'no') based on your assessment:

        **User Question:**
        {question}

        **Documents:**
        {context}
        
        """,
            input_variables=["context", "question"],
        )

    chain = prompt | model

    last_human_message = None
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            last_human_message = message
            break
    
    if last_human_message is None:
        question = state["messages"][0].content
    else:
        question = last_human_message.content
    
    context = state["messages"][-1].content
    result = chain.invoke({"question": question, "context": context})

    print(f"RELEVANCE SCORE: {result.binary_score}")
    return "generate" if result.binary_score == "yes" else "rewrite"
    

@traceable(tags=["agent:agentic_rag", "function:rewrite"])
def rewrite(state: AgentState, llm: ChatOllama) -> dict:
    print("--- REWRITING QUESTION ---")
    
    last_human_message = None
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            last_human_message = message
            break
    
    if last_human_message is None:
        question = state["messages"][0].content
    else:
        question = last_human_message.content

    rewritten = rewrite_query(question, llm) 

    print(f"Rewritten query: {rewritten}")
    return {"messages": [HumanMessage(content=rewritten)]}

# ----------------------------------------
# Generate Node
# ----------------------------------------

@traceable(tags=["agent:agentic_rag", "function:generate"])
def generate(state: AgentState, llm: ChatOllama) -> dict:
    print ('='*100)
    print("\n--- GENERATING FINAL ANSWER ---")

    if verbose:
        print(f"Using LLM >>> {llm.model}")

    last_human_message = None
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            last_human_message = message
            break
    
    last_human_message = next((msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), None)
    
    clean_query = last_human_message.content if last_human_message else "quality improvement projects"
    
    if verbose:
        print(f"Clean query (from STATE) >>> {clean_query}") 

    docs = state["messages"][-1].content

    if not docs.strip():
        return {"messages": [AIMessage(content="No relevant content was retrieved.")]}

    prompt = PromptTemplate.from_template("""
    You are a medical QIP analysis assistant synthesizing information from retrieved documents.

    **Response Format Requirements:**
    1. FIRST provide a concise natural language answer directly addressing the question
    2. THEN present detailed project information in the specified format:


    PROJECT TITLE: [Full title of the project, including poster/project code (e.g. SHM_RM001)]
    - [Key point 1]
    - [Key point 2]
    - [Additional relevant information]

    3. If there are multiple projects, present each one separately in this format
    4. If the answer is not found in the context, say "No relevant projects found in current documents."

    --------------------
    Context:
    {context}

    --------------------
    Question: {question}
    Answer:
    """)
    
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": docs, "question": clean_query})
    
    return {"messages": [AIMessage(content=response)]}

# ----------------------------------------
# Update the graph  
# ----------------------------------------

graph = StateGraph(AgentState)

graph.add_node("check_relevance", lambda state: state)
graph.add_node("retrieve", lambda state: custom_retriever(state, llm, relevance_threshold=0.6))
graph.add_node("not_relevant", handle_not_relevant)

graph.add_node("grade_documents", lambda state: state)
graph.add_node("generate", lambda state: generate(state, llm))
graph.add_node("rewrite", lambda state: rewrite(state, llm_rewriter))

graph.set_entry_point("check_relevance")

graph.add_conditional_edges(
    "check_relevance",
    is_medical_qi_query,
    {
        "retrieve": "retrieve",
        "not_relevant": "not_relevant"
    }
    )

graph.add_edge("retrieve", "grade_documents")

graph.add_conditional_edges(
    "grade_documents",
    lambda state: grade_documents(state, llm),
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)

graph.add_edge("rewrite", "retrieve")

graph.add_edge("generate", END)
graph.add_edge("not_relevant", END)

compiled_graph = graph.compile()
