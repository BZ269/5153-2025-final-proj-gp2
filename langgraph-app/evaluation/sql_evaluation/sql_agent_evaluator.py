#!/usr/bin/env python3
"""
SQL Agent Evaluator

This script reads queries from a CSV file, processes them through the SQL agent,
and records the results including the SQL query, answer, and response time.
"""

import os
import sys
import uuid
import csv
import time
import json
import traceback
from pathlib import Path
from datetime import datetime


# Add the parent directory to the path to import the agents module
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

# Import the SQL query agent modules but create a modified version
from agents.sql_query_agent import write_query, format_question, execute_query
from agents.sql_query_agent import generate_answer, clarify_with_user, format_input
from agents.sql_query_agent import format_response, State
from langgraph.graph import StateGraph, START, END

def create_custom_graph():
    """Create a graph without using MemorySaver to avoid checkpointer issues"""
    graph_builder = StateGraph(State)
    graph_builder.add_node('format_question', format_question)
    graph_builder.add_node('clarify_with_user', clarify_with_user)
    graph_builder.add_node('write_query', write_query)
    graph_builder.add_node('execute_query', execute_query)
    graph_builder.add_node('generate_answer', generate_answer)
    graph_builder.add_node('format_response', format_response)

    graph_builder.add_edge(START, 'format_question')
    graph_builder.add_conditional_edges('format_question', lambda state: state['route'], 
                                      {'write_query': 'write_query', 'clarify_with_user': 'clarify_with_user'})
    graph_builder.add_edge('write_query', 'execute_query')
    graph_builder.add_conditional_edges('execute_query', lambda state: state['route'], 
                                      {'write_query': 'write_query', 'generate_answer': 'generate_answer'})
    graph_builder.add_edge('generate_answer', 'format_response')
    graph_builder.add_edge('clarify_with_user', 'format_response')
    graph_builder.add_edge('format_response', END)
    
    # Compile without a checkpointer
    return graph_builder.compile()

def load_test_queries(csv_path):
    """Load test queries from a CSV file."""
    queries = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append(row['test_query'])
    return queries

def run_evaluation(input_csv, output_csv):
    """Run the evaluation on all test queries."""
    # Load the test queries
    queries = load_test_queries(input_csv)
    
    # Create the SQL agent graph without using MemorySaver
    sql_graph = create_custom_graph()
    
    # Prepare output file
    results = []
    headers = [
        'query_id', 
        'query_text', 
        'sql_query', 
        'query_result',
        'answer', 
        'response_time_seconds',
        'error_messages'
    ]
    
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each query
    for i, query_text in enumerate(queries, 1):
        print(f"Processing query {i}/{len(queries)}: {query_text}")
        
        # Time the execution
        start_time = time.time()
        
        # Format the input for the SQL agent
        formatted_input = format_input(query_text)
        
        # Execute the query through the graph with tracing
        try:
            # Set a unique thread_id for this run
            config = {
                "recursion_limit": 10,
            }
            result = sql_graph.invoke(formatted_input, config=config)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Extract data from the result
            sql_query = result.get('query', 'N/A')
            query_result = result.get('result', 'N/A')
            answer = result.get('answer', 'N/A')
            
            # Debug prints to verify we are getting the correct SQL query
            print("\n----- DEBUG INFO -----")
            print(f"SQL Query captured: {sql_query}")
            print(f"Query type: {type(sql_query)}")
            print(f"Query result type: {type(query_result)}")
            print("----------------------\n")
            
            # Get any error messages
            error_messages = []
            if 'error' in result and result['error']:
                for error_msg in result['error']:
                    if hasattr(error_msg, 'content'):
                        error_messages.append(error_msg.content)
                    else:
                        error_messages.append(str(error_msg))
            
            error_str = '; '.join(error_messages) if error_messages else 'None'
            
            # Store the results
            results.append({
                'query_id': i,
                'query_text': query_text,
                'sql_query': sql_query,
                'query_result': query_result,
                'answer': answer,
                'response_time_seconds': round(response_time, 2),
                'error_messages': error_str
            })
            
            # Save detailed debug information for this query
            debug_dir = os.path.join(os.path.dirname(output_csv), "debug")
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
                
            debug_file = os.path.join(debug_dir, f"query_{i}_details.json")
            try:
                with open(debug_file, 'w', encoding='utf-8') as f:
                    debug_data = {
                        'query_text': query_text,
                        'sql_query': sql_query,
                        'query_result': query_result,
                        'answer': answer,
                        'result_keys': list(result.keys())
                    }
                    json.dump(debug_data, f, indent=2)
                print(f"Saved debug info to {debug_file}")
            except Exception as e:
                print(f"Failed to save debug info: {str(e)}")
            
            print(f"Completed query {i}. Response time: {round(response_time, 2)} seconds")
            
        except Exception as e:
            # Handle any unexpected errors
            response_time = time.time() - start_time
            error_traceback = traceback.format_exc()
            print(f"Error processing query {i}: {str(e)}")
            print(error_traceback)
            
            results.append({
                'query_id': i,
                'query_text': query_text,
                'sql_query': 'Error',
                'query_result': 'Error',
                'answer': 'Error processing query',
                'response_time_seconds': round(response_time, 2),
                'error_messages': str(e) + '\n' + error_traceback
            })
        
        # Write results incrementally to avoid data loss
        try:
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(results)
        except Exception as e:
            print(f"Error writing to CSV: {str(e)}")
            # Write to a backup file if main file fails
            backup_file = output_csv.replace('.csv', '_backup.csv')
            with open(backup_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(results)
        
        print("-" * 50)

    # Generate a summary of the evaluation
    total_time = sum([r['response_time_seconds'] for r in results])
    avg_time = total_time / len(results) if results else 0
    
    print(f"\n=== Evaluation Summary ===")
    print(f"Total queries: {len(results)}")
    print(f"Total time: {round(total_time, 2)} seconds")
    print(f"Average time per query: {round(avg_time, 2)} seconds")
    print(f"Results saved to: {output_csv}")
    
    return results

if __name__ == "__main__":
    input_file = os.path.join(os.path.dirname(__file__), "test_queries.csv")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Change the output directory to be the main evaluation/results folder
    eval_results_dir = os.path.join(Path(__file__).resolve().parent.parent, "results")
    output_file = os.path.join(eval_results_dir, f"sql_evaluation_results_{timestamp}.csv")
    
    print(f"Starting SQL agent evaluation...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Create the results directory if it doesn't exist
    if not os.path.exists(eval_results_dir):
        os.makedirs(eval_results_dir)
    
    # Try a single query test first
    try:
        test_queries = load_test_queries(input_file)
        if test_queries:
            print(f"Running a test with the first query: '{test_queries[0]}'")
            test_input = format_input(test_queries[0])
            test_graph = create_custom_graph()
            test_result = test_graph.invoke(test_input)
            print("\n----- TEST QUERY RESULTS -----")
            print(f"Query text: {test_queries[0]}")
            print(f"SQL query: {test_result.get('query', 'N/A')}")
            print(f"Result keys: {list(test_result.keys())}")
            print("-----------------------------\n")
            print("Test successful! Running full evaluation...")
    except Exception as e:
        print(f"Test failed: {str(e)}")
        print(traceback.format_exc())
        print("Proceeding with full evaluation anyway...")
    
    results = run_evaluation(input_file, output_file)
    print(f"Evaluation complete. Results saved to {output_file}") 