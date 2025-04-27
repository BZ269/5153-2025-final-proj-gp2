# Model Evaluation Guide

This README provides instructions for running the automated model evaluation tools to compare different model configurations.

## Overview

The evaluation compares three different model configurations:

1. **Base Llama + Qwen 7B** - Base models using llama3.1 + other models using qwen-7b-custom-4096:latest
2. **Base Llama + Qwen 14B** - Base models using llama3.1 + other models using qwen-14b-custom-4096:latest
3. **Base Llama + Qwen 2.5** - Base models using llama3.1 + other models using qwen-2.5-custom-4096:latest

## Running the Evaluation

### Step 1: Run the Model Evaluation Script

The `model_config_evaluator.py` script automates the evaluation process. It will:
- Update the config.py file for each model configuration
- Run both RAG and SQL evaluations
- Save results with appropriate identifiers

```bash
# Run all three configurations
python model_config_evaluator.py

# OR run a specific configuration (0, 1, or 2)
python model_config_evaluator.py --config 0  # Base Llama + Qwen 7B
python model_config_evaluator.py --config 1  # Base Llama + Qwen 14B
python model_config_evaluator.py --config 2  # Base Llama + Qwen 2.5

# Run in test mode with limited queries
python model_config_evaluator.py --test
```

All results will be saved in the `evaluation/results` directory with timestamps and model configuration identifiers.

### Step 2: Analyze the Results

After running the evaluations, use the `run_results_analyzer.py` script to generate comparison metrics and visualizations:

```bash
python run_results_analyzer.py
```

This will:
- Load all result files from the `evaluation/results` directory
- Analyze performance metrics (latency, accuracy)
- Generate comparison plots
- Create a summary report with recommendations

## Evaluation Metrics

### RAG Evaluation Metrics:
- Execution time (latency)
- Project codes retrieved
- Success rate for finding relevant codes

### SQL Evaluation Metrics:
- Response time (latency)
- SQL query success rate
- Error frequency

## Interpreting Results

The summary report will include:
- Performance comparison across configurations
- Best configuration for latency
- Best configuration for accuracy
- Overall recommendation based on balanced scoring

## Notes

- The original configuration is backed up and restored after evaluation
- Each evaluation run is time-stamped for tracking
- For visualizing only the latest results, run the analyzer right after completing an evaluation
- Use the `--test` flag to run with reduced number of queries for quick testing 