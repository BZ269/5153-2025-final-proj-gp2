# %%
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

### Load environment variables from .env file
from dotenv import load_dotenv
import os
from datetime import datetime
import warnings
from pathlib import Path

# Import config for models and settings
from config import MODELS, MODEL_TEMPERATURES

from langsmith import traceable

# Load the .env file from the project root directory (parent of the agents folder)
load_dotenv(dotenv_path=Path(__file__).parents[1] / '.env')


# Suppress Pydantic deprecation warnings from the Ollama package
# These warnings are related to accessing model_fields on instances rather than classes
# Will be fixed in future Ollama package updates
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress ResourceWarnings about unclosed socket connections
# These occur because the Ollama client doesn't properly close TCP socket connections
# after communicating with the Ollama server
warnings.filterwarnings("ignore", category=ResourceWarning)
# Load the .env file from the specified path

# load_dotenv("/Users/royyeo/env_files/.env")  #bookmark

verbose = True
verbose_2 = False

# ### Access the variables  #bookmark
# langchain_api_key = os.getenv("LANGCHAIN_API_KEY")   
# langchain_tracing_v2 = os.getenv("LANGCHAIN_TRACING_V2")
# langchain_project = os.getenv("LANGCHAIN_PROJECT")

# %%


# ----------------------------------------
#  Initialize Models, Weaviate and Vector Store
# ----------------------------------------
### bookmark

# Initialize embeddings and vector store
embedding_model = OllamaEmbeddings(model=MODELS["embeddings"])

# Initialize LLMs
llm = ChatOllama(model=MODELS["rag_generator"], temperature=MODEL_TEMPERATURES["rag_generator"])
llm_rewriter = ChatOllama(model=MODELS["llama3"], temperature=MODEL_TEMPERATURES["llama3"])

exported_llm = llm

## ----------------------------------------
# ### Connect to Weaviate #bookmark

# client = weaviate.connect_to_local(port=8081)

# vectorstore = WeaviateVectorStore(
#     client=client,
#     index_name="MedicalQIDocument_Poster_Chunks",
#     embedding=embedding_model,
#     text_key="content"
# )

# ### Connect to Weaviate with Docker configuration
# client = weaviate.connect_to_local(
#     host="localhost",  # Docker host
#     port=8081,         # REST API port
#     grpc_port=50052,   # gRPC port from docker-compose
#     skip_init_checks=True  # Skip gRPC health check if needed
#     )

# Connect to Weaviate
client = weaviate.connect_to_local(port=8081)

vectorstore = WeaviateVectorStore(
    client=client,
    index_name="MedicalQIDocument_Poster_Chunks",
    embedding=embedding_model,
    text_key="content"
)

### Expore the client for reuse (eg. streamlit_app_v2.py, poster_qa.py)
exported_client = client

# ----------------------------------------
# 3. Define Section List and Classifier
# ----------------------------------------

SECTION_LIST = [
    "TITLE OF PROJECT", "BACKGROUND", "INTRODUCTION", "MISSION STATEMENT", "ANALYSIS OF PROBLEM",
    "ROOT CAUSE ANALYSIS", "METHODOLOGY", "INTERVENTIONS / INITIATIVES",
    "RESULTS", "OUTCOME", "DISCUSSION", "SUSTAINABILITY AND SPREAD", "CONCLUSION"
    ]



# ----------------------------------------
# 1. Define LangGraph State Schema
# ----------------------------------------

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "Chat history"]
    metadata_filter: Optional[Any]  

    
# ----------------------------------------
# 4. Define Tools
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

    ### Define prompt for section classifier
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


    ## Format the prompt
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


    ###---------------------------------------------
    ### Extract metadata filter from query using LLM
    ###---------------------------------------------
    metadata_dict, _, clean_query = MetadataFilterBuilder.parse_metadata_from_query(
        query=query,
        client=client,
        llm=llm,
        collection_name="MedicalQIDocument_Poster_Chunks"
    )

    # print(f"\n📌 LLM selected sections: {section_list}")
    print(f"\n📌 Metadata filters: {metadata_dict}")
    print(f"\n📌 Clean query: {clean_query}")

    ### Build metadata filter from metadata_dict
    metadata_filter_from_query = None
    

    if metadata_dict:
        # Initialize separate filters for year and hospital
        year_filter = None
        hospital_filter = None
        
        for field, value in metadata_dict.items():
            if field == "year":
                if isinstance(value, list):
                    # Handle multiple years with OR logic
                    for val in value:
                        try:
                            f_val = Filter.by_property("year").equal(str(val))
                            year_filter = f_val if year_filter is None else year_filter | f_val
                        except:
                            pass
                else:
                    # Handle single year
                    try:
                        year_filter = Filter.by_property("year").equal(str(value))
                    except:
                        pass
            
            elif field == "hospital":
                if isinstance(value, list):
                    # Handle multiple hospitals with OR logic
                    for val in value:
                        f_val = Filter.by_property("hospital").equal(val)
                        hospital_filter = f_val if hospital_filter is None else hospital_filter | f_val
                else:
                    # Handle single hospital
                    hospital_filter = Filter.by_property("hospital").equal(value)
        
        # Store filters separately instead of combining them
        metadata_filter_from_query = {
            "year_filter": year_filter,
            "hospital_filter": hospital_filter
        }

    ###---------------------------------------------
    ### Build section filter
    section_filter = None 
    for sec in predicted:
        f = Filter.by_property("section").equal(sec)
        section_filter = f if section_filter is None else section_filter | f

    ###---------------------------------------------
    ### Get metadata filter from state if available
    if verbose:
        print(f'\n📌 section_filter: {section_filter}') 


    ###---------------------------------------------
    ### Get metadata filter from state if available
    metadata_filter_from_state = state.get("metadata_filter", None)

    if verbose:
        print(f'\n📌 metadata_filter_from_state: {metadata_filter_from_state}')
        print(f'\n📌 metadata_filter_from_query: {metadata_filter_from_query}')
        print(f'\n📌 section_filter: {section_filter}')



    ### ------------------------------------------------------------
    ### Combine metadata filters for each field using OR logic
    ### Get year filter from state and query if available
    year_filter = None
    if metadata_filter_from_state and "year_filter" in metadata_filter_from_state:
        year_filter = metadata_filter_from_state["year_filter"]
        if verbose:
            print(f'\n📌 ACTIVE FILTER contains year_filter: {year_filter}') 

    if metadata_filter_from_query and "year_filter" in metadata_filter_from_query:
        year_filter = year_filter | metadata_filter_from_query["year_filter"] if year_filter else metadata_filter_from_query["year_filter"]
        if verbose:
            print(f'\n📌 QUERY FILTER contains year_filter: {metadata_filter_from_query["year_filter"]}') 

    ### Get hospital filter from state and query if available
    hospital_filter = None
    if metadata_filter_from_state and "hospital_filter" in metadata_filter_from_state:
        hospital_filter = metadata_filter_from_state["hospital_filter"]
        if verbose:
            print(f'\n📌 ACTIVE FILTER contains hospital_filter: {hospital_filter}') 

    if metadata_filter_from_query and "hospital_filter" in metadata_filter_from_query:
        hospital_filter = hospital_filter | metadata_filter_from_query["hospital_filter"] if hospital_filter else metadata_filter_from_query["hospital_filter"]
        if verbose:
            print(f'\n📌 QUERY FILTER contains hospital_filter: {metadata_filter_from_query["hospital_filter"]}') 
    
    ### Print year_filter and hospital_filter
    print(f'\n📌 year_filter: {year_filter}')
    print(f'\n📌 hospital_filter: {hospital_filter}')

    ### Combine all fields using AND logic
    final_filter = None
    if year_filter:
        final_filter = year_filter
    if hospital_filter:
        final_filter = final_filter & hospital_filter if final_filter else hospital_filter
    if section_filter:
        final_filter = final_filter & section_filter if final_filter else section_filter

    print(f'\n📌 final_filter: {final_filter}')

    ######################################################
    ### Configure retriever with final_filter
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": k,
            "alpha": 0.5,
            # "score_threshold": 0.5, 
            "score_threshold": 0.2, # Set this to 0.0 or low value to retrieve all results that pass the final_filter, i.e. no semantic filtering here.
            "filters": final_filter
        }
    )

    ### Perform retrieval
    docs = retriever.invoke(clean_query) # query excluded the metadata filters values


    ### Get the number of doc retrieved.
    num_docs_retrieved = len(docs)
    if verbose:
        print(f"\n📌 Number of documents retrieved: {num_docs_retrieved}")

    if verbose: 
        print('='*100)

        results_with_scores = vectorstore.similarity_search_with_score(clean_query, k=k, filters=final_filter)
        print(f"\n📄 Number of results with scores: {len(results_with_scores)}")

        ### Create a mapping of document content to scores for easy lookup
        score_map = {result.page_content: score for result, score in results_with_scores}
        
        ### Inject scores into each document's metadata
        for doc in docs:
            if doc.page_content in score_map:
                doc.metadata['similarity_score'] = score_map[doc.page_content]
                if verbose_2:
                    print(f"Chunk: {doc}, Score: {score_map[doc.page_content]}")
    
    if verbose:
        ### Print retrieved documents for debugging
        print('='*100)
        print(f"\n📄 Retrieved Documents ({len(docs)} total):")
        for i, doc in enumerate(docs):
            print(f"\nDocument {i+1}:")
            print(f"Similarity Score (from retrieval, not used for grading): {doc.metadata.get('similarity_score')}")
            # print(f"Source: {doc.metadata.get('source')}")
            print(f"Title: {doc.metadata.get('title')}")
            print(f"Section: {doc.metadata.get('section')}")
            print(f"Year: {doc.metadata.get('year')}")
            print(f"Hospital: {doc.metadata.get('hospital')}")
            print(f"Project Code: {doc.metadata.get('project_code')}")
            if verbose_2:
                print(f"Content: {doc.page_content}")  

    ### Filter documents by relevance 
    relevant_docs = filter_documents_by_relevance(clean_query,  # query is clean, exclude the metadata filters values
                                                  documents=docs, 
                                                  threshold=relevance_threshold,
                                                  llm=llm)

    ### Extract project codes from relevant docs 
    project_codes = [doc.metadata.get('project_code') for doc in relevant_docs if doc.metadata.get('project_code')]

    ### Update the global variable (for recording purpose)
    global unique_project_codes
    unique_project_codes = list(set(project_codes))  # Remove duplicates

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
    # Get the last user message
    messages = state["messages"]
    if not messages:
        return "not_relevant"
    
    last_message = messages[-1]
    if not isinstance(last_message, HumanMessage):
        return "not_relevant"
    
    query = last_message.content
    
    # Use llama3 model from config
    model = ChatOllama(model=MODELS["llama3"], temperature=MODEL_TEMPERATURES["llama3"])
    
    ### ---------------------------------------------------------
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
    
    ############################################################
    response = model.invoke(prompt)
    
    # Check if the response indicates relevance
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
    # Get the query from the last message
    messages = state["messages"]
    last_message = messages[-1]
    query = last_message.content if isinstance(last_message, HumanMessage) else "quality improvement projects"
    
    ### Call the retriever tool with llm as llm_classifier
    clean_query, retrieval_results = retriever_tool.run({
        "query": query, 
        "state": state, 
        "llm_classifier": llm, 
        "relevance_threshold": relevance_threshold
    })
    
    if verbose:
        print(f"\n📌 Retrieval results >>>>> {retrieval_results}") 

    ### Create a new message with the clean_query
    clean_query_message = HumanMessage(content=clean_query)

    ## Return the updated state
    return {
        "messages": state["messages"] + [clean_query_message, AIMessage(content=retrieval_results)],
        "metadata_filter": state.get("metadata_filter", {})
    }


# ----------------------------------------
# 6. Relevance Grader
# ----------------------------------------

@traceable(tags=["agent:agentic_rag", "function:grade_documents"])
def grade_documents(state: AgentState, llm: ChatOllama) -> Literal["generate", "rewrite"]:
    print("\n--- CHECKING RELEVANCE (of combined retrieved documents) ---")

    class GradeOutput(BaseModel):
        binary_score: str = Field(description="Relevance score: 'yes' or 'no'")

    ### Configures the LLM to return its output in the structured format defined by GradeOutput
    model = llm.with_structured_output(GradeOutput) 


    ### ----------------------------------------------------------------------------
    # prompt = PromptTemplate(
    #     template="""You are an expert document grader assessing whether the retrieved documents can collectively answer the user's question. Consider that:
    #     1. Documents may contain partial information that contributes to a complete answer
    #     2. Some relevance may be implied rather than explicit
    #     3. The context may contain supporting evidence even if not directly answering

    #     **Documents:**
    #     {context}

    #     **User Question:**
    #     {question}

    #     **Evaluation Guidelines:**
    #     - Answer 'yes' if the documents contain information that could help answer the question, even partially
    #     - Answer 'yes' if the documents provide context or supporting evidence relevant to the question
    #     - Answer 'no' ONLY if the documents are completely unrelated or provide no value

    #     **Mandatory Respnse Format:**
    #     - You must answer "Relevance score: 'yes' or 'no'".
        
    #     **Decision:** Provide a single word ('yes' or 'no') based on your assessment:
    #     """,
    #         input_variables=["context", "question"],
    #     )

    ### -----------------------------------------------------------------------------
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

    ### -----------------------------------------------------------------------------
    chain = prompt | model

    # Find the last human message in the conversation
    last_human_message = None
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            last_human_message = message
            break
    
    if last_human_message is None:
        # Fallback to the first message if no human message found (shouldn't happen)
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
    
    # Find the last human message in the conversation
    last_human_message = None
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            last_human_message = message
            break
    
    if last_human_message is None:
        # Fallback to the first message if no human message found (shouldn't happen)
        question = state["messages"][0].content
    else:
        question = last_human_message.content


    ### Rewrite the query
    rewritten = rewrite_query(question, llm) 

    ### Print the rewritten query
    print(f"Rewritten query: {rewritten}")
    return {"messages": [HumanMessage(content=rewritten)]}


# ----------------------------------------
# 8. Generate Node
# ----------------------------------------

@traceable(tags=["agent:agentic_rag", "function:generate"])
def generate(state: AgentState, llm: ChatOllama) -> dict:
    print ('='*100)
    print("\n--- GENERATING FINAL ANSWER ---")

    if verbose:
        print(f"Using LLM >>> {llm.model}")

    # Find the last human message in the conversation
    last_human_message = None
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            last_human_message = message
            break
    

    ### Find the last human message in the conversation
    last_human_message = next((msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), None)
    
    ### Use the last human message as the clean_query
    clean_query = last_human_message.content if last_human_message else "quality improvement projects"
    
    if verbose:
        print(f"Clean query (from STATE) >>> {clean_query}") 


    ### Get the documents from the state["messages"]
    docs = state["messages"][-1].content

    if not docs.strip():
        return {"messages": [AIMessage(content="No relevant content was retrieved.")]}


    ### --------------------------------------------------------
    # prompt = PromptTemplate.from_template("""
    # You are a helpful assistant helping with medical quality improvement (QI) document analysis.

    # Use the following context from QI documents to answer the user's question. Always ground your answer in the provided content.

    # IMPORTANT FORMATTING INSTRUCTIONS:
    # 1. ALWAYS identify and list the project titles first, extracted from each source document.
    # 2. For each project, format your response as:

    #     PROJECT: [Project Code] | [Full title of the project] | [Year of the project]
    #     - [Key point 1]
    #     - [Key point 2]
    #     - [Additional relevant information]

    # 3. If there are multiple projects, present each one separately in this format.
    # 4. Say "The information is not available in the current documents." after verifying all these:
    #     - No project titles relate to query theme.
    #     - No inferred connections exist.
    #     - No plausible deliverables match request.
    # --------------------
    # Context:
    # {context}

    # --------------------
    # Question: {question}
    # Answer:
    # """)
    
    ### --------------------------------------------------------
    ### NEW
    # prompt = PromptTemplate.from_template("""
    # You are a medical QIP analysis assistant synthesizing information from retrieved documents.

    # **Response Format Requirements:**
    # 1. FIRST provide a concise natural language answer directly addressing the question
    # 2. THEN present detailed project information in the specified format:

    # PROJECT: [Code] | [Title] | [Year]
    # - [Key point 1]
    # - [Key point 2]
    # - [Additional details]

    # **Content Rules:**
    # 1. **Natural language answer must:**
    # - Be 5 sentences maximum, be information dense, but accurate to retrieved documents.
    # - Explicitly state if projects exist/do not exist
    # - Highlight the most relevant theme matching the query
    # 2. **Project listings must:**
    # - Appear ONLY after the natural language summary
    # - Include all available projects matching the query theme
    # - Preserve the exact PROJECT: | | formatting
    # 3. **When no matches exist:**
    # - State "No relevant projects found in current documents."
    # - Do NOT show project formatting

    # **Example Output Structure:**
    # [Natural language answer summarizing findings]

    # [PROJECT: formatted details...]

    # **Current Context:**
    # {context}

    # **Question:** {question}
    # **Answer:**
    # """)
    
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
    # response = rag_chain.invoke({"context": docs, "question": question})
    response = rag_chain.invoke({"context": docs, "question": clean_query})
    
    return {"messages": [AIMessage(content=response)]}


# ----------------------------------------
# Update the graph  
# ----------------------------------------

graph = StateGraph(AgentState)

### Add the nodes
graph.add_node("check_relevance", lambda state: state)  # Just passes state through
graph.add_node("retrieve", lambda state: custom_retriever(state, llm, relevance_threshold=0.6))
graph.add_node("not_relevant", handle_not_relevant)

### Add the new nodes
graph.add_node("grade_documents", lambda state: state)  # Just passes state through
graph.add_node("generate", lambda state: generate(state, llm))  # Final answer generation
graph.add_node("rewrite", lambda state: rewrite(state, llm_rewriter))  # Query rewriting

### - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
### Set the entry point
graph.set_entry_point("check_relevance") # previous
# graph.set_entry_point("rewrite") # new

# Add edge
# graph.add_edge("rewrite", "check_relevance")

### Add the conditional edges
graph.add_conditional_edges(
    "check_relevance",
    is_medical_qi_query,
    {
        "retrieve": "retrieve",
        "not_relevant": "not_relevant"
    }
    )

### Add edge from retrieve to grade_documents
graph.add_edge("retrieve", "grade_documents")

### Add CONDITIONAL edges from grade_documents based on relevance grading
graph.add_conditional_edges(
    "grade_documents",
    lambda state: grade_documents(state, llm),
    {
        "generate": "generate",  # If documents are relevant, generate answer
        "rewrite": "rewrite"     # If documents aren't relevant, rewrite query
    }
)

#### Add edge from rewrite back to retrieve to try again with rewritten query
graph.add_edge("rewrite", "retrieve")

### Add the final edges
graph.add_edge("generate", END)  # End after generating response
graph.add_edge("not_relevant", END)

# Compile the graph
compiled_graph = graph.compile()


# # %%
# # ### Visualize graph
# # ################################################################################
# # #                                Visual Graph                                 #
# # ################################################################################

# # image_data = compiled_graph.get_graph().draw_mermaid_png(timeout=30)

# # with open("langgraph_rag_graph_v4.png", mode="wb") as f:
# #     f.write(image_data)


# # %%
# ######################################################################
# #                        Run the RAG pipeline                        #
# ######################################################################

# ### Main execution code for direct script running
# @traceable(name="Run QIP Agent")
# def run_rag(query: str, metadata_filter: Optional[Any] = None) -> str:
#     """
#     Run the RAG pipeline with the given query and optional metadata filter.
    
#     Args:
#         query: The user's query/question
#         metadata_filter: Optional Weaviate filter for metadata filtering
        
#     Returns:
#         The generated response from the RAG pipeline
#     """
#     ### Initialize the agent state with the query
#     state = AgentState(
#         messages=[HumanMessage(content=query)],
#         metadata_filter=metadata_filter,
#     )
    
#     ### Run the compiled graph with the initial state
#     result = compiled_graph.invoke(state)
    
#     ### Get the final response from the last message
#     if result["messages"]:
#         return result["messages"][-1].content
#     return "No response generated"

# # %% #de3
# ### Simple command-line interface when script is run directly
# if __name__ == "__main__":
#     print("✅ Medical QI RAG Agent Ready")
#     print("💡 Type 'exit' or 'quit' to end the session")
    
#     while True:
#         query = input("🔎 Ask a QIP-related question: ")
#         if query.lower() in ["exit", "quit"]:
#             break
            
#         try:
#             ## Measure response time
#             start_time = datetime.now()
#             response = run_rag(query)
#             end_time = datetime.now()
#             elapsed = (end_time - start_time).total_seconds()
            
#             # Print the response with timing info
#             print(f"\n🧠 Answer (generated in {elapsed:.2f} seconds):\n")
#             print(response)
#             print("\n" + "-" * 50 + "\n")
            
#         except Exception as e:
#             print(f"\n❌ Error: {str(e)}")
#             print("Full error details:")
            
#             traceback.print_exc()  # Print full traceback including line numbers
#             print("\n" + "-" * 50 + "\n")



# # %%
# ########################################################################
# #                      Run RAG on specific query                       #
# ########################################################################
# # #### Run RAG on specific query "Give me posters on falls"
# # # query = "Give me posters on falls"
# # query = "What metrics were commonly used to evaluate fall prevention interventions?"

# # metadata_filter = None  # No specific metadata filters

# # try:
# #     print("\n🔎 Running RAG for query:", query)
# #     start_time = datetime.now()
# #     response = run_rag(query, metadata_filter)
# #     end_time = datetime.now()
# #     elapsed = (end_time - start_time).total_seconds()
    
# #     print(f"\n🧠 Answer (generated in {elapsed:.2f} seconds):\n")
# #     print(response)
# #     print("\n" + "-" * 50 + "\n")
    
# # except Exception as e:
# #     print(f"\n❌ Error processing query: {str(e)}")
# #     print("Full error details:")
# #     traceback.print_exc()
# #     print("\n" + "-" * 50 + "\n")


# # %%

# ###################################################################################
# #                     Print all Full Poster chunks in vectorstore                 #
# ###################################################################################
# def print_all_chunks(client, limit=1000, collection_name="MedicalQIDocument_Poster_Chunks"):
#     """
#     Print all chunks in the MedicalQIDocument collection with their metadata
    
#     Args:
#         client: The Weaviate client instance
#         limit: Maximum number of chunks to retrieve (default: 1000)
#     """
#     ### Get the collection
#     collection = client.collections.get(collection_name)
    
#     collection_index_metadata_map = {
#         'MedicalQIDocument_Poster_Chunks': [
#             'source',
#             'section',
#             'content',
#             'year',
#             'project_code',
#             'hospital',
#             'title'
#         ],
#         'MedicalQIDocument_Poster_Full': [
#             'source',
#             'content',
#             'year',
#             'project_code',
#             'hospital',
#             'title'
#         ]
#     }

    
#     results = collection.query.fetch_objects(
#         return_properties=collection_index_metadata_map[collection_name],
#         limit=limit
#     )
    
#     ### Print header
#     print(f"\n📚 All Chunks in Vectorstore (showing first {limit}):")
    
#     # Print each chunk with metadata
#     for i, chunk in enumerate(results.objects):
#         print(f"\n🧩 Chunk {i+1}")
#         print(f"🆔 Document ID: {chunk.uuid}")  # Added document ID
#         print(f"📁 Source: {chunk.properties.get('source', 'N/A')}")
#         print(f"📌 Section: {chunk.properties.get('section', 'N/A')}")
#         # print(f"📄 Page: {chunk.properties.get('page', 'N/A')}")
#         print(f"📅 Year: {chunk.properties.get('year', 'N/A')}")
#         print(f"🔢 Project Code: {chunk.properties.get('project_code', 'N/A')}")
#         print(f"🏥 Hospital: {chunk.properties.get('hospital', 'N/A')}")
#         print(f"📚 Title: {chunk.properties.get('title', 'N/A')}")
#         # print(f"📝 Content Preview: {chunk.properties.get('content', '')[:300]}...")
#         print(f"📝 Content: {chunk.properties.get('content', '')}")
        
    
#     print(f"\n✅ Printed {len(results.objects)} chunks")

# # ### Example usage: 
# ### #fr4
# # print_all_chunks(client, collection_name="MedicalQIDocument_Poster_Full")
# # print_all_chunks(client, collection_name="MedicalQIDocument_Poster_Chunks")



# # %%

# # ### #gt5
# # ########################################################################
# # #                            TEST QUERY SET                            #
# # ########################################################################
# # ### Run tests query set

# # # test_queries_df = pd.reaTake your time to think and reasond_csv("test_queries/test_queries.csv", header=0)
# # test_queries_df = pd.read_csv("test_queries/test_queries_with_ground_truths.csv", header=0)
# # # test_queries_df = pd.read_csv("test_queries/test_queries_sub_sub.csv", header=0)
# # print(test_queries_df)

# # answers = []
# # project_codes_list = [] 

# # for index, row in test_queries_df.iterrows():
# #     # Print test query number in a clear, obvious format
# #     print(f"\n ================ 🔢 TEST QUERY #{index + 1} ==========================")
# #     print("=" * 50)
    
# #     test_query = row["test_query"]
# #     print(f'Running test query: {test_query}')
    

# #     ### ---------------------------------------------
# #     ### run_rag() implementation

# #     # query = "Give me posters on falls"
# #     query = test_query
# #     metadata_filter = None  # No specific metadata filters

# #     try:
# #         print("\n🔎 Running RAG for query:", query)
# #         ### -----------------------------------
# #         ### Get the final response from the last message
# #         start_time = datetime.now()
# #         answer = run_rag(query, metadata_filter)
# #         end_time = datetime.now()
# #         elapsed = (end_time - start_time).total_seconds()
        
# #         # ### ----------------------------------- 
# #         # result = compiled_graph.invoke(AgentState(
# #         #     messages=[HumanMessage(content=query)],
# #         #     metadata_filter=metadata_filter,
# #         # ))
# #         # # Extract project codes from the intermediate state
# #         # project_codes = result.get("project_codes", [])
# #         # project_codes_str = ", ".join(project_codes) if project_codes else "None"
# #         # if verbose:
# #         #     print(f"\n📌 Project codes >>>>>>>>>> {project_codes_str}")
# #         ### ----------------------------------- 
# #         # Access and print the global unique_project_codes variable
# #         if 'unique_project_codes' in globals():
# #             print(f"\n📌 Project codes >>>>>>>>>> {unique_project_codes}")
# #             project_codes_str = unique_project_codes
# #         else:
# #             print("\n📌 No project codes found - unique_project_codes not defined")
# #             project_codes_str = "None"

# #         print(f"\n🧠 Answer (generated in {elapsed:.2f} seconds):\n")
# #         print(answer)
# #         print("\n" + "-" * 50 + "\n")
        
# #     except Exception as e:
# #         print(f"\n❌ Error processing query: {str(e)}")
# #         print("Full error details:")
# #         traceback.print_exc()
# #         answer = "RUN ERROR"
# #         project_codes_str = "RUN ERROR" 
# #         # answer = f"RUN ERROR: {str(e)}"
# #         print("\n" + "-" * 50 + "\n")


# #     ###############################################
# #     print(f'Test query {index + 1} : {test_query}')

# #     print("\n💬 Final Answer:\n")
# #     print(answer)
# #     answers.append(answer)
# #     project_codes_list.append(project_codes_str) 
# #     print("#" * 100)

# # ###----------------------------------------------------------------------
# # test_queries_df["answer"] = answers
# # test_queries_df["project_codes_retrieved"] = project_codes_list 

# # print(f'Answers for test queries: {test_queries_df}')
# # test_queries_df.to_csv("test_queries/test_queries_answers.csv", index=False)
# # print(f'Answers saved to test_queries/test_queries_answers.csv')

# # %%
