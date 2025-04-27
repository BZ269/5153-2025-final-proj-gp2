

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to evaluate the agentic_rag system using test queries from a CSV file.
This script will:
1. Load test queries from a CSV file
2. Run each query through the RAG system
3. Extract project codes
4. Record the final LLM output
5. Directly retrieve and capture chunks from vectorstore
6. Save results to a new CSV with individual chunk columns
"""

import os
import pandas as pd
import re
import time
from datetime import datetime
from typing import Optional, Any, List, Dict
import traceback
import argparse

# Import components from the agentic_rag module
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from agents.agentic_rag import compiled_graph, AgentState, retriever_tool, vectorstore, llm
from utils.query_rewriter import filter_documents_by_relevance  # Import the filter function
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate

# Configuration variables
DEBUG = True
DEFAULT_TEST_QUERIES_FILE = "test_queries_25apr11pm.csv"
DEFAULT_CHUNKS_TO_RETRIEVE = 15
DEFAULT_RELEVANCE_THRESHOLD = 0.6
DEFAULT_MAX_INDIVIDUAL_CHUNKS = 15

def debug_print(msg, important=False):
    """Print debug messages if debug mode is enabled"""
    if DEBUG:
        if important:
            print(f"\n{'='*20} DEBUG {'='*20}")
            print(msg)
            print(f"{'='*50}\n")
        else:
            print(f"[DEBUG] {msg}")

# Define the function to extract project codes from the text
def extract_project_codes(text: str) -> List[str]:
    """
    Extract project codes from text using regex patterns.
    """
    # Patterns to match project codes in various formats
    patterns = [
        r"SHM[_]?([A-Z]{2}\d{3})",  # SHM_XX000 format
        r"PROJECT TITLE:.*?SHM[_]?([A-Z]{2}\d{3})",  # PROJECT TITLE: with SHM code
        r"PROJECT:.*?SHM[_]?([A-Z]{2}\d{3})",  # PROJECT: with SHM code
        r"([A-Z]{2}\d{3})"  # Any XX000 format (fallback)
    ]
    
    found_codes = []
    
    # Try each pattern in order of specificity
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            # Extract the group with the code part
            code = match.group(1) if len(match.groups()) > 0 else match.group(0)
            
            # Clean up the code and add SHM_ prefix if missing
            if not code.startswith("SHM_") and "SHM" not in code:
                code = f"SHM_{code}"
            elif "SHM" in code and not code.startswith("SHM_"):
                code = code.replace("SHM", "SHM_")
            
            # Add to the list if not already present
            if code not in found_codes:
                found_codes.append(code)
    
    return found_codes

def directly_retrieve_documents(query: str, k: int = 15, relevance_threshold: float = 0.6) -> List[Dict]:
    """
    Directly retrieve and filter documents from vectorstore,
    replicating the exact process used in the retriever_tool function
    
    Args:
        query: The query to search for
        k: Number of documents to retrieve
        relevance_threshold: Threshold for relevance filtering
        
    Returns:
        List of dictionaries representing the documents
    """
    debug_print(f"Direct retrieval for query: '{query}' with k={k}")
    
    # Step 1: Perform initial retrieval with similarity search
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": k,
            "alpha": 0.5,
            "score_threshold": 0.2,  # Same as in agentic_rag.py
        }
    )
    docs = retriever.invoke(query)
    
    debug_print(f"Retrieved {len(docs)} initial documents")
    
    # Step 2: Apply relevance filtering (same as in agentic_rag.py)
    try:
        relevant_docs = filter_documents_by_relevance(
            query=query,
            documents=docs,
            llm=llm,
            threshold=relevance_threshold
        )
        debug_print(f"After relevance filtering: {len(relevant_docs)} documents", important=True)
    except Exception as e:
        debug_print(f"Error during relevance filtering: {str(e)}")
        relevant_docs = docs  # Fallback to unfiltered documents
    
    # Step 3: Convert to dictionary format
    chunks = []
    for i, doc in enumerate(relevant_docs):
        chunk_info = {
            "source": doc.metadata.get("source", "N/A"),
            "section": doc.metadata.get("section", "N/A"),
            "title": doc.metadata.get("title", "N/A"),
            "year": doc.metadata.get("year", "N/A"),
            "hospital": doc.metadata.get("hospital", "N/A"),
            "project_code": doc.metadata.get("project_code", "N/A"),
            "content": doc.page_content,
            "relevance_score": doc.metadata.get("relevance_score", "N/A"),
            "similarity_score": doc.metadata.get("similarity_score", "N/A")
        }
        chunks.append(chunk_info)
        
    return chunks

def run_rag_query(query: str, metadata_filter: Optional[Any] = None, k: int = 15, relevance_threshold: float = 0.6) -> Dict:
    """
    Run the RAG pipeline with the given query and capture both the response and project codes.
    
    Args:
        query: The user's query/question
        metadata_filter: Optional Weaviate filter for metadata filtering
        k: Number of chunks to retrieve
        relevance_threshold: Threshold for relevance filtering
        
    Returns:
        Dictionary with response, project codes, and retrieved chunks
    """
    debug_print(f"Running RAG query: '{query}'")
    
    # First get direct retrieval results
    chunks = directly_retrieve_documents(query, k=k, relevance_threshold=relevance_threshold)
    
    # Initialize the agent state with the query
    state = AgentState(
        messages=[HumanMessage(content=query)],
        metadata_filter=metadata_filter,
    )
    
    # Run the compiled graph with the initial state
    result = compiled_graph.invoke(state)
    
    # Get the final response from the last message
    response = "No response generated"
    if result["messages"]:
        response = result["messages"][-1].content
    
    # Log the message structure for debugging
    debug_print("Message structure in result:", important=True)
    for i, msg in enumerate(result["messages"]):
        msg_type = msg.__class__.__name__
        msg_content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        debug_print(f"Message {i}: {msg_type} - {msg_content_preview}")
    
    # Extract project codes from the response
    project_codes = extract_project_codes(response)
    
    return {
        "response": response,
        "project_codes": project_codes,
        "chunks": chunks
    }

def format_chunk_for_csv(chunk: Dict, index: int) -> str:
    """Format a chunk dictionary into a CSV-friendly string with clear delimiters."""
    parts = [
        f"CHUNK {index}",
        f"SECTION: {chunk.get('section', 'N/A')}",
        f"TITLE: {chunk.get('title', 'N/A')}",
        f"YEAR: {chunk.get('year', 'N/A')}",
        f"HOSPITAL: {chunk.get('hospital', 'N/A')}",
        f"PROJECT_CODE: {chunk.get('project_code', 'N/A')}",
        f"RELEVANCE: {chunk.get('relevance_score', 'N/A')}",
        f"CONTENT: {chunk.get('content', 'N/A')}"
    ]
    return " || ".join(parts)

def main():
    """Main function to run the evaluation."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate agentic_rag system with test queries from a CSV file.")
    parser.add_argument("--input", "-i", 
                      default=DEFAULT_TEST_QUERIES_FILE,
                      help=f"Input CSV filename in the rag_test_queries directory (default: {DEFAULT_TEST_QUERIES_FILE})")
    parser.add_argument("--k", "-k", 
                      default=DEFAULT_CHUNKS_TO_RETRIEVE, type=int,
                      help=f"Number of chunks to retrieve (default: {DEFAULT_CHUNKS_TO_RETRIEVE})")
    parser.add_argument("--relevance-threshold", "-r", 
                      default=DEFAULT_RELEVANCE_THRESHOLD, type=float,
                      help=f"Relevance threshold for filtering documents (default: {DEFAULT_RELEVANCE_THRESHOLD})")
    parser.add_argument("--max-individual-chunks", "-m", 
                      default=DEFAULT_MAX_INDIVIDUAL_CHUNKS, type=int,
                      help=f"Maximum number of individual chunks to include as separate columns (default: {DEFAULT_MAX_INDIVIDUAL_CHUNKS})")
    parser.add_argument("--debug", "-d", 
                      action="store_true",
                      help="Enable debug output")
    args = parser.parse_args()
    
    # Set debug mode based on argument
    global DEBUG
    DEBUG = args.debug
    
    # Define input and output file paths
    input_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                           "rag_test_queries", 
                           args.input)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                            "../results", 
                            f"rag_evaluation_results_{timestamp}.csv")
    
    # Create the results directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    print(f"Loading test queries from: {input_csv}")
    
    # Load the test queries
    try:
        test_queries_df = pd.read_csv(input_csv)
        print(f"Loaded {len(test_queries_df)} test queries")
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        return
    
    # Initialize results lists
    responses = []
    project_codes_list = []
    execution_times = []
    chunk_lists = []
    chunk_counts = []
    
    # Lists for individual chunks (limited to max_individual_chunks)
    max_chunks = args.max_individual_chunks
    individual_chunks = [[] for _ in range(max_chunks)]
    
    # Process each query in the dataframe
    for index, row in test_queries_df.iterrows():
        # Get the test query
        test_query = row["test_query"]
        print(f"\n================ 🔢 TEST QUERY #{index + 1}/{len(test_queries_df)} ====================")
        print(f"Query: {test_query}")
        
        # Run the query and time the execution
        start_time = datetime.now()
        
        try:
            result = run_rag_query(
                test_query, 
                k=args.k, 
                relevance_threshold=args.relevance_threshold
            )
            response = result["response"]
            project_codes = result["project_codes"]
            chunks = result["chunks"]
            
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()
            
            # Print results
            print(f"Project codes found: {', '.join(project_codes)}")
            print(f"Response generated in {elapsed:.2f} seconds")
            print(f"Retrieved {len(chunks)} chunks")
            
            # Add to results lists
            responses.append(response)
            project_codes_list.append(", ".join(project_codes))
            execution_times.append(elapsed)
            chunk_counts.append(len(chunks))
            
            # Format chunks for CSV storage using '||' as delimiter
            formatted_chunks = []
            for i, chunk in enumerate(chunks):
                formatted_chunks.append(format_chunk_for_csv(chunk, i+1))
            
            chunk_lists.append("||".join(formatted_chunks))
            
            # Store individual chunks in separate columns (limited to max_chunks)
            for i in range(max_chunks):
                if i < len(chunks):
                    chunk = chunks[i]
                    # Create a more structured format for individual chunk columns
                    metadata = {
                        "section": chunk.get("section", "N/A"),
                        "project_code": chunk.get("project_code", "N/A"),
                        "year": chunk.get("year", "N/A"),
                        "hospital": chunk.get("hospital", "N/A"),
                        "relevance": chunk.get("relevance_score", "N/A")
                    }
                    
                    # Format metadata as a compact string
                    metadata_str = ", ".join([f"{k}: {v}" for k, v in metadata.items() if v != "N/A"])
                    
                    # Combine metadata with content
                    chunk_text = f"[{metadata_str}] {chunk.get('content', '')}"
                    individual_chunks[i].append(chunk_text)
                else:
                    individual_chunks[i].append("")  # Empty for missing chunks
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            traceback.print_exc()
            
            # Add error info to results
            responses.append(f"ERROR: {str(e)}")
            project_codes_list.append("")
            execution_times.append(-1)
            chunk_lists.append("")
            chunk_counts.append(0)
            
            # Add empty values for individual chunks
            for i in range(max_chunks):
                individual_chunks[i].append("")
        
        # Add a separator for readability
        print("=" * 50)
    
    # Add results to the dataframe
    test_queries_df["llm_output"] = responses
    test_queries_df["project_codes_retrieved"] = project_codes_list
    test_queries_df["execution_time_seconds"] = execution_times
    test_queries_df["chunks_count"] = chunk_counts
    test_queries_df["retrieved_chunks"] = chunk_lists
    
    # Add individual chunk columns
    for i in range(max_chunks):
        test_queries_df[f"chunk_{i+1}"] = individual_chunks[i]
    
    # Save the results
    print(f"\nSaving results to: {output_csv}")
    test_queries_df.to_csv(output_csv, index=False)
    print("Evaluation complete!")

if __name__ == "__main__":
    main() 