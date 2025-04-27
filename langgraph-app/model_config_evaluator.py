#!/usr/bin/env python3
"""
Automated Model Evaluation Script

This script automates the evaluation of three different model configurations:
1. Base models llama3.1 + other models using qwen-7b-custom-4096:latest
2. Base models llama3.1 + other models using qwen-14b-custom-4096:latest
3. Base models llama3.1 + other models using qwen-2.5-custom-4096:latest

For each configuration, it will:
1. Update the config.py file
2. Run the RAG evaluation
3. Run the SQL agent evaluation
4. Save results with appropriate identifiers

Use --test flag to run in test mode with reduced number of queries.
"""

import os
import sys
import subprocess
import time
import shutil
from datetime import datetime
import argparse

# Define model configurations to test
MODEL_CONFIGS = [
    {
        "name": "base_llama_rest_qwen7b",
        "description": "Base models llama3.1 + other models using qwen-7b-custom-4096:latest",
        "models": {
            "sql_question_gen": "qwen-7b-custom-4096:latest",
            "poster_qa": "qwen-7b-custom-4096:latest",
            "rag_classifier": "llama3.1",
            "rag_grader": "llama3.1",
            "rag_rewriter": "llama3.1",
            "rag_generator": "qwen-7b-custom-4096:latest",
            "embeddings": "qwen-2.5-custom-4096:latest",
            "sql_agent": "qwen-7b-custom-4096:latest",
            "llama3": "llama3.1",  # Used in agentic_rag_v2_roy.py for rewriter
        }
    },
    {
        "name": "base_llama_rest_qwen14b",
        "description": "Base models llama3.1 + other models using qwen-14b-custom-4096:latest",
        "models": {
            "sql_question_gen": "qwen-14b-custom-4096:latest",
            "poster_qa": "qwen-14b-custom-4096:latest",
            "rag_classifier": "llama3.1",
            "rag_grader": "llama3.1",
            "rag_rewriter": "llama3.1",
            "rag_generator": "qwen-14b-custom-4096:latest",
            "embeddings": "qwen-2.5-custom-4096:latest",
            "sql_agent": "qwen-14b-custom-4096:latest",
            "llama3": "llama3.1",  # Keep this as llama3.1 to avoid breaking rewriter
        }
    },
    {
        "name": "base_llama_rest_qwen",
        "description": "Base models llama3.1 + other models using qwen-2.5-custom-4096:latest",
        "models": {
            "sql_question_gen": "qwen-2.5-custom-4096:latest",
            "poster_qa": "qwen-2.5-custom-4096:latest",
            "rag_classifier": "llama3.1",
            "rag_grader": "llama3.1",
            "rag_rewriter": "llama3.1",
            "rag_generator": "qwen-2.5-custom-4096:latest",
            "embeddings": "qwen-2.5-custom-4096:latest",
            "sql_agent": "qwen-2.5-custom-4096:latest",
            "llama3": "llama3.1",  # Keep this as llama3.1 to avoid breaking rewriter
        }
    }
]

# Path to config file
CONFIG_FILE = "config.py"
# Backup of original config file
CONFIG_BACKUP = "config.py.backup"
# Paths to evaluation scripts
RAG_EVAL_SCRIPT = "evaluation/rag/generate_rag_results.py"
SQL_EVAL_SCRIPT = "evaluation/sql_evaluation/sql_agent_evaluator.py"

# Test mode flag
TEST_MODE = False

def backup_config():
    """Create a backup of the original config file"""
    print(f"Creating backup of original config file at {CONFIG_BACKUP}")
    shutil.copy2(CONFIG_FILE, CONFIG_BACKUP)

def restore_config():
    """Restore the original config file from backup"""
    if os.path.exists(CONFIG_BACKUP):
        print(f"Restoring original config file from {CONFIG_BACKUP}")
        shutil.copy2(CONFIG_BACKUP, CONFIG_FILE)
        os.remove(CONFIG_BACKUP)
    else:
        print("Warning: Config backup file not found!")

def update_config(model_config):
    """Update the config.py file with the given model configuration"""
    print(f"Updating config.py with {model_config['name']} configuration")
    
    # Read the current config file
    with open(CONFIG_FILE, 'r') as f:
        config_lines = f.readlines()
    
    # Find the MODELS dictionary and update it
    in_models_dict = False
    new_config_lines = []
    
    for line in config_lines:
        if "MODELS = {" in line:
            in_models_dict = True
            new_config_lines.append(line)
            continue
        
        if in_models_dict and "}" in line and ":" not in line:
            in_models_dict = False
            new_config_lines.append(line)
            continue
        
        if in_models_dict and ":" in line:
            # Extract the model name
            model_name = line.split(":")[0].strip().strip('"\'')
            if model_name in model_config['models']:
                # Replace the model value
                indent = line.split('"')[0]
                new_line = f'{indent}"{model_name}": "{model_config["models"][model_name]}",{line.split(",")[-1] if "," in line else ""}\n'
                new_config_lines.append(new_line)
                continue
        
        new_config_lines.append(line)
    
    # Write the updated config back to file
    with open(CONFIG_FILE, 'w') as f:
        f.writelines(new_config_lines)
    
    print(f"Config file updated successfully with {model_config['name']} configuration")

def run_rag_evaluation(model_config_name):
    """Run the RAG evaluation script"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_suffix = f"{model_config_name}_{timestamp}"
    
    print(f"\n{'='*50}")
    print(f"Starting RAG evaluation for {model_config_name} configuration")
    print(f"{'='*50}\n")
    
    cmd = [
        sys.executable,
        RAG_EVAL_SCRIPT,
        "--debug"
    ]
    
    # Add test mode parameters if enabled
    if TEST_MODE:
        # Create/update a temporary test queries file with only 2 queries
        try:
            original_queries_file = os.path.join(os.path.dirname(RAG_EVAL_SCRIPT), "rag_test_queries", "test_queries_25apr11pm.csv")
            test_queries_file = os.path.join(os.path.dirname(RAG_EVAL_SCRIPT), "rag_test_queries", "temp_test_queries.csv")
            
            # Read first 2 queries from original file
            import pandas as pd
            df = pd.read_csv(original_queries_file)
            test_df = df.head(2)  # Get only first 2 rows
            test_df.to_csv(test_queries_file, index=False)
            
            cmd.extend(["--input", "temp_test_queries.csv"])
            print("Running in TEST MODE with only 2 RAG queries")
        except Exception as e:
            print(f"Error creating test queries file: {str(e)}")
            # If we can't create the test file, just use the limit flag
            cmd.append("--limit")
            cmd.append("2")
    
    try:
        # Run the evaluation
        subprocess.run(cmd, check=True)
        
        # Rename the output file to include the model configuration name
        # Find the latest results file in the evaluation/results directory
        results_dir = os.path.join(os.path.dirname(RAG_EVAL_SCRIPT), "..", "results")
        latest_file = None
        latest_time = 0
        for file in os.listdir(results_dir):
            if file.startswith("rag_evaluation_results_"):
                file_path = os.path.join(results_dir, file)
                file_time = os.path.getctime(file_path)
                if file_time > latest_time:
                    latest_time = file_time
                    latest_file = file_path
        
        if latest_file:
            # Rename the file to include the model configuration
            mode_prefix = "test_" if TEST_MODE else ""
            new_name = os.path.join(results_dir, f"{mode_prefix}rag_evaluation_results_{model_config_name}_{timestamp}.csv")
            shutil.move(latest_file, new_name)
            print(f"RAG evaluation results saved to {new_name}")
        else:
            print("Warning: Could not find latest RAG evaluation results file")
            
        # Clean up temporary test file
        if TEST_MODE and os.path.exists(test_queries_file):
            os.remove(test_queries_file)
            
        return True
    except Exception as e:
        print(f"Error running RAG evaluation: {str(e)}")
        return False

def run_sql_evaluation(model_config_name):
    """Run the SQL agent evaluation script"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_suffix = f"{model_config_name}_{timestamp}"
    
    print(f"\n{'='*50}")
    print(f"Starting SQL agent evaluation for {model_config_name} configuration")
    print(f"{'='*50}\n")
    
    cmd = [
        sys.executable,
        SQL_EVAL_SCRIPT
    ]
    
    # Add test mode parameters if enabled
    if TEST_MODE:
        # Create/update a temporary test queries file with only 2 queries
        try:
            original_queries_file = os.path.join(os.path.dirname(SQL_EVAL_SCRIPT), "test_queries.csv")
            test_queries_file = os.path.join(os.path.dirname(SQL_EVAL_SCRIPT), "temp_test_queries.csv")
            
            # Read and copy first 2 lines
            with open(original_queries_file, 'r') as infile:
                lines = infile.readlines()
                header = lines[0]
                
                with open(test_queries_file, 'w') as outfile:
                    outfile.write(header)  # Write header
                    for i in range(1, min(3, len(lines))):  # Write 2 data rows or fewer if file has fewer rows
                        outfile.write(lines[i])
            
            # Use the temp file for evaluation
            cmd.extend(["--input", test_queries_file])
            print("Running in TEST MODE with only 2 SQL queries")
        except Exception as e:
            print(f"Error creating test queries file: {str(e)}")
    
    try:
        # Run the evaluation
        subprocess.run(cmd, check=True)
        
        # Rename the output file to include the model configuration name
        # Find the latest results file in the evaluation/results directory
        results_dir = os.path.join(os.path.dirname(SQL_EVAL_SCRIPT), "..", "results")
        latest_file = None
        latest_time = 0
        for file in os.listdir(results_dir):
            if file.startswith("sql_evaluation_results_"):
                file_path = os.path.join(results_dir, file)
                file_time = os.path.getctime(file_path)
                if file_time > latest_time:
                    latest_time = file_time
                    latest_file = file_path
        
        if latest_file:
            # Rename the file to include the model configuration
            mode_prefix = "test_" if TEST_MODE else ""
            new_name = os.path.join(results_dir, f"{mode_prefix}sql_evaluation_results_{model_config_name}_{timestamp}.csv")
            shutil.move(latest_file, new_name)
            print(f"SQL evaluation results saved to {new_name}")
        else:
            print("Warning: Could not find latest SQL evaluation results file")
            
        # Clean up temporary test file
        if TEST_MODE and os.path.exists(test_queries_file):
            os.remove(test_queries_file)
            
        return True
    except Exception as e:
        print(f"Error running SQL evaluation: {str(e)}")
        return False

def run_evaluations():
    """Run all evaluations for all model configurations"""
    results = {}
    
    # Create backup of original config
    backup_config()
    
    try:
        # Run evaluations for each model configuration
        for config in MODEL_CONFIGS:
            print(f"\n\n{'#'*80}")
            print(f"# Starting evaluation for {config['name']} configuration")
            print(f"# {config['description']}")
            print(f"{'#'*80}\n")
            
            # Update the config file
            update_config(config)
            
            # Wait for the config changes to take effect
            print("Waiting for configuration changes to take effect...")
            time.sleep(2)
            
            # Run RAG evaluation
            rag_success = run_rag_evaluation(config['name'])
            
            # Run SQL evaluation
            sql_success = run_sql_evaluation(config['name'])
            
            # Record results
            results[config['name']] = {
                "rag_evaluation": "Success" if rag_success else "Failed",
                "sql_evaluation": "Success" if sql_success else "Failed"
            }
            
            print(f"\nCompleted evaluation for {config['name']} configuration")
    finally:
        # Restore original config
        restore_config()
    
    # Print summary of results
    print("\n\n")
    print("="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    for config_name, result in results.items():
        print(f"\n{config_name}:")
        print(f"  RAG Evaluation: {result['rag_evaluation']}")
        print(f"  SQL Evaluation: {result['sql_evaluation']}")
    mode_text = "TEST MODE" if TEST_MODE else "FULL MODE"
    print(f"\nAll evaluations completed in {mode_text}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluations with different configurations")
    parser.add_argument("--config", type=int, choices=[0, 1, 2], help="Run only a specific config (0, 1, or 2)")
    parser.add_argument("--test", action="store_true", help="Run in test mode with limited queries")
    args = parser.parse_args()
    
    # Set test mode flag
    TEST_MODE = args.test
    if TEST_MODE:
        print("Running in TEST MODE - Only 2 queries will be run for each evaluation")
    
    # If a specific config is requested, run only that one
    if args.config is not None:
        # Create backup of original config
        backup_config()
        
        try:
            config = MODEL_CONFIGS[args.config]
            print(f"\n\n{'#'*80}")
            print(f"# Starting evaluation for {config['name']} configuration")
            print(f"# {config['description']}")
            print(f"{'#'*80}\n")
            
            # Update the config file
            update_config(config)
            
            # Wait for the config changes to take effect
            print("Waiting for configuration changes to take effect...")
            time.sleep(2)
            
            # Run RAG evaluation
            run_rag_evaluation(config['name'])
            
            # Run SQL evaluation
            run_sql_evaluation(config['name'])
            
            print(f"\nCompleted evaluation for {config['name']} configuration")
        finally:
            # Restore original config
            restore_config()
    else:
        # Run all configurations
        run_evaluations() 