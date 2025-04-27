"""
Central configuration file for the Medical QI Document Assistant
Contains model configurations for the application
"""

# Model configurations - preserving the exact models used in the original application
MODELS = {
    # Main application models
    "sql_question_gen": "qwen-2.5-custom-6000:latest",  # Used in streamlit_app_v2.py
    "poster_qa": "qwen-2.5-custom-4096:latest",        # Used in poster_qa.py
    
    # RAG agent models
    "rag_classifier": "llama3.1",                      # Used in agentic_rag_v2.py for classification
    "rag_grader": "llama3.1",                          # Used for document grading
    "rag_rewriter": "llama3.1",                        # Used for query rewriting
    "rag_generator": "qwen-2.5-custom-4096:latest",    # Used for response generation
    
    # Embeddings model
    "embeddings": "qwen-2.5-custom-4096:latest",       # Used for vectorstore
    
    # SQL agent model
    "sql_agent": "qwen-2.5-custom-6000:latest",        # Used in sql_query_agent.py
    
    # agentic_rag_v2_roy.py models
    #"qwen_large": "qwen2.5:14b-instruct",              # Used in agentic_rag_v2_roy.py for main LLM
    "llama3": "llama3.1",                              # Used in agentic_rag_v2_roy.py for rewriter
}

# Model parameters
MODEL_TEMPERATURES = {
    "sql_question_gen": 0.5,
    "poster_qa": 0.0,
    "rag_classifier": 0.0,
    "rag_grader": 0.0,
    "rag_rewriter": 0.0,
    "rag_generator": 0.0,
    "sql_agent": 0.5,
    
    # agentic_rag_v2_roy.py temperature settings
    "qwen_large": 0.1,
    "llama3": 0.1,
}

# Retrieval settings
RETRIEVAL_TOP_K = 15
RETRIEVAL_SCORE_THRESHOLD = 0.5
RETRIEVAL_ALPHA = 0.5

# SQL agent settings
SQL_MAX_RETRY_LIMIT = 3
SQL_MESSAGE_CONTEXT_LIMIT = 3 