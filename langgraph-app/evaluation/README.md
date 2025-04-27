# Evaluation Tools

This directory contains evaluation scripts and tools for benchmarking both RAG (Retrieval-Augmented Generation) and SQL query generation capabilities.

## Directory Structure

- `sql_evaluation/` - Tools for SQL query generation evaluation
- `rag/` - Tools for RAG system evaluation
- `results/` - Storage for evaluation results
- `langsmith/` - LangSmith integration utilities

## RAG Evaluation

### RAG Metrics Generator

Located in `rag/generate_rag_results.py`, this tool:
- Runs test queries through the RAG system
- Captures retrieved chunks and their metadata
- Records RAG system responses
- Extracts project codes from responses
- Exports results to CSV format for analysis

### RAGAS Evaluation

Located in `rag/rag_evaluation_ragas.py`, this tool:
- Implements custom evaluation metrics for RAG systems:
  - Context precision/recall
  - Faithfulness
  - Response relevancy
  - Response similarity
- Uses both embedding-based and LLM-based evaluation methods
- Produces detailed evaluation reports with performance metrics

## SQL Evaluation

### SQL Query Generator

Located in `sql_evaluation/sql_trim_langgraph.py`, this tool:
- Creates a trimmed LangGraph pipeline for SQL query generation
- Processes natural language questions into SQL queries
- Exports generated SQL queries to text files
- Designed for evaluation without execution overhead

### SQL Evaluator

Located in `sql_evaluation/sql_agent_evaluator.py`, this tool:
- Evaluates SQL query generation against test queries
- Records SQL query quality, execution time, and results
- Exports detailed evaluation results to CSV
- Supports debugging and error analysis

Additionally, `sql_evaluation/evaluation.py` provides:
- Evaluation against gold standard SQL queries
- Multiple metrics (exact match, partial match, execution accuracy)
- Detailed breakdown of SQL component accuracy (SELECT, WHERE, etc.)

## Usage

See individual README files in each subdirectory for specific usage instructions. 