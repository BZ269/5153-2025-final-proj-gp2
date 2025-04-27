# This file should be placed in the langgraph-aftershock-edrive-v2/evaluation/rag/ directory
# It contains the RAGAS evaluation framework for RAG systems

# Define the ground truth data file names (can be a list of files)
GROUND_TRUTH_FILES = [
    "qip_ground_truth_27apr2025_basellamaqwen7b.xlsx",
    "qip_ground_truth_27apr2025_basellamaqwen14b.xlsx",
    "qip_ground_truth_27apr2025_basellamaqwen32b.xlsx"
]

import pandas as pd
import numpy as np
from tabulate import tabulate
import csv
import re
import os
import json
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_core.outputs import ChatResult
from langchain_ollama import OllamaEmbeddings
import torch
import sys
from dotenv import load_dotenv
from pathlib import Path

# Add the parent directory to sys.path to import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import MODELS


# Load the .env file from the project root directory (parent of the evaluation folder)
load_dotenv(dotenv_path=Path(__file__).parents[2] / '.env')

# Model definitions: separate models for LLM calls and embeddings
llm_model_name = "qwen-2.5-custom-4096:latest"
embedding_model_name = MODELS["embeddings"]  # Use same model as in agentic_rag.py

# Initialize Ollama embeddings model matching the one used in agentic_rag.py
embedding_model = OllamaEmbeddings(model=embedding_model_name)

# Use the full path or proper relative path
def load_data(file_path):
    return pd.read_excel(os.path.join(os.path.dirname(__file__), file_path), index_col=0)

# Set up LangSmith tracing
callbacks = []
if os.environ.get("LANGSMITH_TRACING", "false").lower() == "true":
    if os.environ.get("LANGSMITH_API_KEY"):
        langsmith_project = os.environ.get("LANGSMITH_PROJECT", "ragas-eval-5153")
        from langchain_core.tracers import LangChainTracer
        callbacks.append(LangChainTracer(project_name=langsmith_project))
        print(f"LangSmith tracing enabled for project: {langsmith_project}")
    else:
        print("LangSmith tracing disabled. LANGSMITH_API_KEY not found.")

# Initiate ChatOllama model with qwen for LLM tasks
llm = ChatOllama(model=llm_model_name,
                 base_url="http://localhost:11434",
                 temperature=0,
                 callbacks=callbacks)

print(f"Using models - LLM: {llm_model_name}, Embeddings: {embedding_model_name}")

# Helper function for computing cosine similarity
def cosine_similarity(embedding1, embedding2):
    # Convert embeddings to tensors
    tensor1 = torch.tensor(embedding1) if not isinstance(embedding1, torch.Tensor) else embedding1
    tensor2 = torch.tensor(embedding2) if not isinstance(embedding2, torch.Tensor) else embedding2
    
    # Normalize the vectors to unit length
    tensor1 = tensor1 / tensor1.norm()
    tensor2 = tensor2 / tensor2.norm()
    
    # Compute cosine similarity
    return torch.dot(tensor1, tensor2).item()

########## PROCESS INPUT DATA ##########
def segregate_poster_filename(text):
    # Handle non-string inputs (like NaN/float values)
    if not isinstance(text, str):
        return []
        
    patterns = [
        r"SHM_\w+",
    ]
    matches = []
    for pattern in patterns:
        found = re.findall(pattern, text)
        matches.extend(found)
    return matches

# Get retrieved chunks
def get_retrieved_chunks(data:pd.DataFrame, row:str) -> list[str]:
    retrieved_chunks = []
    try:
        chunk_count = data.iloc[row]['chunks_count']
        if pd.isna(chunk_count):
            return []
            
        for i in range(int(chunk_count)):
            try:
                chunk = data.iloc[row][f'chunk_{i+1}']
                if isinstance(chunk, str):
                    retrieved_chunks.append(chunk)
            except (KeyError, IndexError, TypeError):
                # Skip this chunk if there's an issue
                continue
    except (KeyError, IndexError, TypeError) as e:
        print(f"Warning: Error retrieving chunks for row {row}: {e}")
    
    return retrieved_chunks

########## METRICS ##########
# Retrieved chunks vs. Ground truth -> Cosine similarity
def calculate_context_precision(retrieved_chunks: list[str], ground_truth: str) -> float:
    """
    Calculates context precision by measuring the extent to which each
    retrieved chunk is relevant to the ground truth answer. For each chunk
    in the retrieved context, it finds the maximum cosine similarity with the
    embedding of the sentences in the ground truth answer. The precision
    is the average of these maximum similarities.
    """
    if not retrieved_chunks:
        return 0.0

    # Split ground truth into sentences
    response_sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s", ground_truth)
    response_sentences = [s.strip() for s in response_sentences if s.strip()] # Clean up empty sentences

    mean_max_similarity_per_chunk = []
    for chunk in retrieved_chunks:
        max_similarity = 0.0
        try:
            chunk_embedding = embedding_model.embed_query(chunk)
            for sentence in response_sentences:
                sentence_embedding = embedding_model.embed_query(sentence)
                similarity_score = cosine_similarity(chunk_embedding, sentence_embedding)
                if similarity_score > max_similarity:
                    max_similarity = similarity_score
            mean_max_similarity_per_chunk.append(max_similarity)
        except Exception as e:
            print(f"Warning: Error calculating chunk embedding: {e}")
            mean_max_similarity_per_chunk.append(0.0)

    if not mean_max_similarity_per_chunk:
        return 0.0

    return np.mean(mean_max_similarity_per_chunk)
    
# Retrieved chunks vs. Ground truth -> Cosine similarity
def calculate_context_recall(retrieved_chunks: list[str], ground_truth: str) -> float:
    """
    Measures the extent to which the retrieved context captures the 
    essential information present in the ground truth answer.
    It assesses this by determining the maximum semantic similarity
    between each sentence of the ground truth and any of the retrieved chunks.
    The context recall is then the average of these maximum similarity scores
    across all ground truth sentences.
    """
    if not retrieved_chunks:
        return 0.0

    # Split ground truth into sentences
    response_sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s", ground_truth)
    response_sentences = [s.strip() for s in response_sentences if s.strip()] # Clean up empty sentences

    max_similarity_per_sentence = []
    for sentence in response_sentences:
        max_similarity = 0.0
        try:
            sentence_embedding = embedding_model.embed_query(sentence)
            for chunk in retrieved_chunks:
                chunk_embedding = embedding_model.embed_query(chunk)
                similarity_score = cosine_similarity(sentence_embedding, chunk_embedding)
                if similarity_score > max_similarity:
                    max_similarity = similarity_score
            max_similarity_per_sentence.append(max_similarity)
        except Exception as e:
            print(f"Warning: Error calculating sentence embedding: {e}")
            max_similarity_per_sentence.append(0.0)

    if not max_similarity_per_sentence:
        return 0.0

    return np.mean(max_similarity_per_sentence)
    
# Response vs. Retrieved chunks -> LLM score
def calculate_metrics_with_llm(llm_response: str, retrieved_chunks: list[str], query: str) -> tuple:
    """
    Evaluates both faithfulness and response relevancy in a single function 
    with fewer LLM calls.
    
    Returns:
        tuple: (faithfulness_score, response_relevancy_score)
    """
    if not retrieved_chunks:
        return 0.0, 0.0
        
    # If we're only calculating response_relevancy (with dummy chunk), use a simplified approach
    if len(retrieved_chunks) == 1 and retrieved_chunks[0] == "Dummy chunk to satisfy function":
        # Just generate questions for response relevancy
        prompt = f"""Generate three questions based on this response that would help evaluate if it addresses the query.
        
        QUERY: {query}
        RESPONSE: {llm_response}
        
        Format your answer as a valid JSON array of exactly 3 questions:
        ["Question 1", "Question 2", "Question 3"]
        """
        
        try:
            evaluation_result = llm.invoke(prompt)
            evaluation_str = evaluation_result.content.strip()
            
            # Try to extract a JSON array from the response
            import json
            json_match = re.search(r'\[.*\]', evaluation_str, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                try:
                    generated_questions = json.loads(json_str)
                    if isinstance(generated_questions, list):
                        # Ensure we have exactly 3 questions
                        if len(generated_questions) < 3:
                            generated_questions.extend([''] * (3 - len(generated_questions)))
                        elif len(generated_questions) > 3:
                            generated_questions = generated_questions[:3]
                    else:
                        # If not a list, create default questions
                        generated_questions = ["", "", ""]
                except:
                    # Default if JSON parsing fails
                    print("Warning: Failed to parse questions JSON. Creating default questions.")
                    # Try to extract questions with regex as fallback
                    question_matches = re.findall(r'"([^"]+)"', evaluation_str)
                    generated_questions = question_matches[:3] if question_matches else ["", "", ""]
                    if len(generated_questions) < 3:
                        generated_questions.extend([''] * (3 - len(generated_questions)))
            else:
                # If no JSON array found, try to extract questions with regex
                print("Warning: No JSON array found. Extracting questions with regex.")
                lines = evaluation_str.split('\n')
                generated_questions = []
                for line in lines:
                    if '?' in line and len(generated_questions) < 3:
                        question = line.strip()
                        generated_questions.append(question)
                
                if len(generated_questions) < 3:
                    generated_questions.extend([''] * (3 - len(generated_questions)))
            
            # Calculate response relevancy
            query_embedding = embedding_model.embed_query(query)
            question_embeddings = [embedding_model.embed_query(q) for q in generated_questions]
            similarity_scores = [
                cosine_similarity(query_embedding, q_embedding) 
                for q_embedding in question_embeddings
            ]
            response_relevancy = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
            
            return 0.0, response_relevancy
            
        except Exception as e:
            print(f"Error in response relevancy calculation: {e}")
            return 0.0, 0.0
    
    # For full evaluation with real chunks
    # Build a single prompt to handle all tasks at once
    # Allow up to 15 chunks for evaluation
    max_chunks = min(15, len(retrieved_chunks))
    consolidated_chunks = "\n\n".join([
        f"CHUNK {i+1}:\n{chunk.page_content if hasattr(chunk, 'page_content') else chunk}"
        for i, chunk in enumerate(retrieved_chunks[:max_chunks])
    ])
    
    # This prompt will ask the LLM to:
    # 1. Score faithfulness for all chunks
    # 2. Generate questions for relevancy assessment
    prompt = f"""You are evaluating an LLM response. Complete ALL these tasks:

TASK 1: Rate the faithfulness of this response against each context chunk on a scale of 0 to 1.
CONTEXT CHUNKS:
{consolidated_chunks}

RESPONSE:
{llm_response}

QUERY:
{query}

Provide JSON output with the following format:
{{
  "faithfulness_scores": [<score_for_chunk_1>, <score_for_chunk_2>, ...],
  "generated_questions": [
    "<question_1>",
    "<question_2>",
    "<question_3>"
  ]
}}

The questions should be based on the response content and help evaluate if the response properly addresses the original query.
"""

    # Make a single LLM call
    try:
        evaluation_result = llm.invoke(prompt)
        evaluation_str = evaluation_result.content.strip()
        
        # Extract JSON data
        import json
        
        # Try to find a JSON object in the response
        json_match = re.search(r'\{.*\}', evaluation_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                evaluation_data = json.loads(json_str)
                
                # Extract faithfulness scores
                faithfulness_scores = evaluation_data.get("faithfulness_scores", [])
                if not faithfulness_scores or len(faithfulness_scores) != max_chunks:
                    # Try to extract scores with regex if JSON doesn't have correct number
                    score_matches = re.findall(r'[0-9]\.[0-9]+|[0-9]', evaluation_str)
                    faithfulness_scores = [float(s) for s in score_matches[:max_chunks] if 0 <= float(s) <= 1]
                    
                    if len(faithfulness_scores) < max_chunks:
                        # If still not enough scores, fill with 0.5 (neutral)
                        faithfulness_scores.extend([0.5] * (max_chunks - len(faithfulness_scores)))
                
                # Extract generated questions
                generated_questions = evaluation_data.get("generated_questions", [])
                if not generated_questions or len(generated_questions) != 3:
                    # If no questions or wrong number in JSON, extract with regex
                    question_matches = []
                    lines = evaluation_str.split('\n')
                    for line in lines:
                        if '?' in line and len(question_matches) < 3:
                            question_matches.append(line.strip())
                    
                    generated_questions = question_matches
                    
                if len(generated_questions) < 3:
                    # Pad with empty questions if needed
                    generated_questions.extend([""] * (3 - len(generated_questions)))
                elif len(generated_questions) > 3:
                    # Truncate if too many
                    generated_questions = generated_questions[:3]
                
                # Calculate faithfulness score (average of scores for all chunks)
                faithfulness_score = np.mean(faithfulness_scores) if faithfulness_scores else 0.5
                
                # Calculate response relevancy using the generated questions
                query_embedding = embedding_model.embed_query(query)
                question_embeddings = [embedding_model.embed_query(q) for q in generated_questions]
                similarity_scores = [
                    cosine_similarity(query_embedding, q_embedding) 
                    for q_embedding in question_embeddings
                ]
                response_relevancy = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
                
                return faithfulness_score, response_relevancy
                
            except json.JSONDecodeError:
                print("Warning: Invalid JSON format in LLM response. Falling back to regex extraction.")
                # Fallback: extract scores with regex
                score_matches = re.findall(r'[0-9]\.[0-9]+|[0-9]', evaluation_str)
                faithfulness_scores = [float(s) for s in score_matches[:max_chunks] if 0 <= float(s) <= 1]
                if not faithfulness_scores:
                    faithfulness_scores = [0.5] * max_chunks
                
                # Extract questions for relevancy
                question_matches = []
                lines = evaluation_str.split('\n')
                for line in lines:
                    if '?' in line and len(question_matches) < 3:
                        question_matches.append(line.strip())
                
                if len(question_matches) < 3:
                    question_matches.extend([''] * (3 - len(question_matches)))
                
                # Calculate metrics
                faithfulness_score = np.mean(faithfulness_scores)
                
                query_embedding = embedding_model.embed_query(query)
                question_embeddings = [embedding_model.embed_query(q) for q in question_matches]
                similarity_scores = [
                    cosine_similarity(query_embedding, q_embedding) 
                    for q_embedding in question_embeddings
                ]
                response_relevancy = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
                
                return faithfulness_score, response_relevancy
        
        else:
            print("Warning: Could not extract JSON from LLM response. Using fallback extraction.")
            # Fallback: extract scores with regex
            score_matches = re.findall(r'[0-9]\.[0-9]+|[0-9]', evaluation_str)
            faithfulness_scores = [float(s) for s in score_matches[:max_chunks] if 0 <= float(s) <= 1]
            if not faithfulness_scores:
                faithfulness_scores = [0.5] * max_chunks
            
            # Extract questions for relevancy
            question_matches = []
            lines = evaluation_str.split('\n')
            for line in lines:
                if '?' in line and len(question_matches) < 3:
                    question_matches.append(line.strip())
            
            if len(question_matches) < 3:
                question_matches.extend([''] * (3 - len(question_matches)))
            
            # Calculate metrics
            faithfulness_score = np.mean(faithfulness_scores)
            
            query_embedding = embedding_model.embed_query(query)
            question_embeddings = [embedding_model.embed_query(q) for q in question_matches]
            similarity_scores = [
                cosine_similarity(query_embedding, q_embedding) 
                for q_embedding in question_embeddings
            ]
            response_relevancy = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
            
            return faithfulness_score, response_relevancy
    
    except Exception as e:
        print(f"Error in consolidated metrics calculation: {e}")
        return 0.5, 0.0  # Return neutral faithfulness and zero relevancy on error

# Response vs. Retrieved chunks -> LLM score (LEGACY FUNCTION - KEPT FOR REFERENCE)
def calculate_faithfulness(llm_response: str, retrieved_chunks: list[str]) -> float:
    """
    Legacy function - now calls the consolidated metrics function.
    """
    faithfulness, _ = calculate_metrics_with_llm(llm_response, retrieved_chunks, "")
    return faithfulness

# Query vs. Response -> Cosine similarity (LEGACY FUNCTION - KEPT FOR REFERENCE)
def calculate_response_relevancy(query: str, llm_response: str) -> float:
    """
    Legacy function - now calls the consolidated metrics function.
    """
    _, response_relevancy = calculate_metrics_with_llm(llm_response, ["Dummy chunk to satisfy function"], query)
    return response_relevancy

# Ground truth poster vs. Retrieved posters -> Cosine similarity
def calculate_retrieval_matches(expected_posters: list[str], retrieved_posters: list[str]) -> float:
    """
    Calculates the exact match between the expected posters and the retrieved posters.
    Poster filename only.
    """
    correct_matches = 0
    for expected_poster in expected_posters:
        if expected_poster in retrieved_posters:
            correct_matches += 1

    if not expected_posters:
        return 0.0

    return correct_matches / len(expected_posters)

# Response vs. Ground truth -> Cosine similarity
def calculate_response_similarity(llm_response: str, ground_truth: str,) -> float:
    """
    Calculates the cosine similarity between the LLM response and the ground truth answer.
    Higher score indicates better semantic similarity.
    """
    response_embedding = embedding_model.embed_query(llm_response)
    ground_truth_embedding = embedding_model.embed_query(ground_truth)
    similarity_score = cosine_similarity(response_embedding, ground_truth_embedding)
    return similarity_score

########## RUN EVALUATION ##########
def run_evaluation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Run the evaluation on the provided DataFrame and return the results as a new DataFrame.
    """
    results = []
    for num_row in range(len(data)):
        try:
            print(f"Evaluating example {num_row + 1}/{len(data)}...")
            
            query = data.iloc[num_row]['query']
            answerable = data.iloc[num_row]['answerable']
            ground_truth = data.iloc[num_row]['ground_truth']
            difficulty = data.iloc[num_row]['difficulty']
            llm_output = data.iloc[num_row]['llm_output']
            
            # Handle potential NaN values in project_codes and project_codes_retrieved
            try:
                project_codes = segregate_poster_filename(data.iloc[num_row]['project_codes'])
            except (KeyError, TypeError):
                project_codes = []
                
            try:
                project_codes_retrieved = segregate_poster_filename(data.iloc[num_row]['project_codes_retrieved'])
            except (KeyError, TypeError):
                project_codes_retrieved = []
            
            # Safely get retrieved chunks
            try:
                retrieved_chunks = get_retrieved_chunks(data, num_row)
            except Exception as e:
                print(f"Warning: Error getting retrieved chunks for row {num_row}: {e}")
                retrieved_chunks = []

            # Calculate metrics that don't use LLM calls
            context_precision = calculate_context_precision(retrieved_chunks, query)
            context_recall = calculate_context_recall(retrieved_chunks, ground_truth)
            response_similarity = calculate_response_similarity(llm_output, ground_truth)
            
            # Calculate LLM-based metrics with a single call
            faithfulness, response_relevancy = calculate_metrics_with_llm(llm_output, retrieved_chunks, query)
            
            # Calculate average score instead of total score and exclude poster_similarity
            metrics_count = 5  # context_precision, context_recall, faithfulness, response_relevancy, response_similarity
            average_score = (context_precision + context_recall + faithfulness + response_relevancy + response_similarity) / metrics_count
            
            results.append({
                "difficulty": difficulty,
                "answerable": answerable,
                "query": query,
                "context_precision": context_precision,
                "context_recall": context_recall,
                "faithfulness": faithfulness,
                "response_relevancy": response_relevancy,
                "response_similarity": response_similarity,
                "average_score": average_score,
            })
        except Exception as e:
            print(f"Error processing row {num_row}: {e}")
            # Add a placeholder result for the failed row
            results.append({
                "difficulty": "ERROR",
                "answerable": False,
                "query": f"Error processing row {num_row}: {e}",
                "context_precision": 0.0,
                "context_recall": 0.0,
                "faithfulness": 0.0,
                "response_relevancy": 0.0,
                "response_similarity": 0.0,
                "average_score": 0.0,
            })

    return pd.DataFrame(results)

# Print the results
def print_results(results_df: pd.DataFrame):
    print("--- Individual Evaluation Results ---")
    for index, row in results_df.iterrows():
        print(f"--- Evaluation Example {index + 1} ---")
        print(f"Difficulty: {row['difficulty']}")
        print(f"Answerable: {row['answerable']}")
        print(f"Query: {row['query']}")
        print(f"Context Precision: {row['context_precision']:.4f}")
        print(f"Context Recall: {row['context_recall']:.4f}")
        print(f"Faithfulness: {row['faithfulness']:.4f}")
        print(f"Response Relevancy: {row['response_relevancy']:.4f}")
        print(f"Response Similarity: {row['response_similarity']:.4f}")
        print(f"Average Score: {row['average_score']:.4f}")
        print()

    print("####################################################################################################\n")

    table_data = results_df[[
        'difficulty', 'answerable', 'query', 'context_precision', 'context_recall',
        'faithfulness', 'response_relevancy', 'response_similarity',
        'average_score'
    ]].values.tolist()

    headers = [
        "Difficulty", "Answerable",
        "Query", "Context Precision", "Context Recall", "Faithfulness", "Response Relevancy",
        "Response Similarity", "Average Score"
    ]
    print("--- Poster RAG Evaluation Results ---")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


# Export table data to CSV
def export_results(results_df: pd.DataFrame, input_filename="", base_filename="rag_ragas_evaluation_results", sheet_name="Results"):
    try:
        from datetime import datetime
        
        if results_df.empty:
            print("No results to export.")
            return
            
        # Get current date and time for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract model name suffix from input filename if it exists
        model_suffix = ""
        if input_filename:
            # First, get the filename without extension and directory
            base_name = os.path.basename(input_filename)
            file_name_without_ext = os.path.splitext(base_name)[0]
            
            # For filenames like qip_ground_truth_26apr2025_v2_basellamaqwen7b
            # Extract the model name which should be the last part after underscore
            parts = file_name_without_ext.split('_')
            if len(parts) > 0:
                model_suffix = f"_{parts[-1]}"
        
        # Create filename with timestamp and ensure it's in the same directory as this script
        output_filename = f"{base_filename}{model_suffix}_{timestamp}.xlsx"
        output_path = os.path.join(os.path.dirname(__file__), output_filename)
        
        # Save the results
        results_df.to_excel(output_path, sheet_name=sheet_name, index=False)
        print(f"Evaluation results successfully exported to '{output_path}'")
    except Exception as e:
        print(f"Error exporting results to Excel: {e}")

########## MAIN CHUNK ##########
if __name__ == "__main__":
    # Dictionary to store model summary results for final comparison
    model_results = {}
    
    if len(GROUND_TRUTH_FILES) == 0:
        # Interactive mode if no files are specified
        while True:
            file_name = input("Please enter the name of your Excel data file (.xlsx): ").strip()
            file_path = file_name  # Assume the file is in the current directory

            if file_path.lower().endswith(".xlsx"):
                try:
                    data = load_data(file_path)
                    print("File loaded successfully. Running evaluation...")
                    evaluation_results_df = run_evaluation(data)
                    print_results(evaluation_results_df)
                    print("Evaluation completed. Exporting results...")
                    export_results(evaluation_results_df, file_name)
                    
                    # Extract model name for summary
                    base_name = os.path.basename(file_path)
                    file_name_without_ext = os.path.splitext(base_name)[0]
                    parts = file_name_without_ext.split('_')
                    model_name = parts[-1] if len(parts) > 0 else "unknown"
                    
                    # Store average score for this model
                    model_results[model_name] = evaluation_results_df['average_score'].mean()
                    
                    break  # Exit the loop if the file is processed successfully
                except FileNotFoundError:
                    print(f"Error: File '{file_name}' not found in the current directory. Please check the name.")
                except Exception as e:
                    print(f"Error reading the Excel file '{file_name}': {e}. Please check the file.")
            else:
                print("Error: Invalid file type. Please enter the name of an .xlsx file.")
    else:
        # Process all specified files in the list
        print(f"Found {len(GROUND_TRUTH_FILES)} files to process:")
        for i, file_name in enumerate(GROUND_TRUTH_FILES):
            print(f"  {i+1}. {file_name}")
        
        # Process each file independently
        for i, file_path in enumerate(GROUND_TRUTH_FILES):
            try:
                print(f"\n{'='*80}")
                print(f"PROCESSING FILE {i+1}/{len(GROUND_TRUTH_FILES)}: {file_path}")
                print(f"{'='*80}")
                
                # Extract model name for display
                base_name = os.path.basename(file_path)
                file_name_without_ext = os.path.splitext(base_name)[0]
                parts = file_name_without_ext.split('_')
                model_name = parts[-1] if len(parts) > 0 else "unknown"
                
                print(f"Model: {model_name}")
                
                # Load and process this specific file
                data = load_data(file_path)
                print(f"File loaded successfully with {len(data)} examples. Running evaluation...")
                evaluation_results_df = run_evaluation(data)
                print_results(evaluation_results_df)
                
                # Store average score for this model
                model_results[model_name] = evaluation_results_df['average_score'].mean()
                
                # Output file for this specific model
                print(f"Evaluation for {model_name} completed. Exporting results...")
                export_results(evaluation_results_df, file_path)
                print(f"Completed processing file {i+1}/{len(GROUND_TRUTH_FILES)}\n")
                
            except FileNotFoundError:
                print(f"Error: File '{file_path}' not found. Skipping to next file.")
            except Exception as e:
                print(f"Error processing file '{file_path}': {e}. Skipping to next file.")
    
    # Print model comparison after all processing is complete
    if model_results:
        print("\n")
        print("="*80)
        print("MODEL COMPARISON - AVERAGE SCORES")
        print("="*80)
        
        # Sort models by score in descending order for ranking
        sorted_results = sorted(model_results.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate padding for model names
        max_name_length = max(len(model) for model in model_results.keys())
        
        # Print header
        print(f"{'Rank':<6}{'Model':<{max_name_length+4}}{'Average Score':<15}")
        print("-" * (6 + max_name_length + 4 + 15))
        
        # Print each model with its average score
        for rank, (model, score) in enumerate(sorted_results, 1):
            print(f"{rank:<6}{model:<{max_name_length+4}}{score:.6f}")
        
        print("\nNote: Higher scores indicate better performance.")
        print("="*80)