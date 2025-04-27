# Medical QI Document Assistant

A Streamlit application that helps users access and analyze medical Quality Improvement (QI) projects through a conversational interface with dual functionality: poster retrieval and data generation.

## Overview

This application provides a conversational interface to:

1. **Poster Retrieval Mode**: Search and retrieve medical quality improvement posters from a vector database, with filtering capabilities by year and hospital.
2. **Data Generation Mode**: Run SQL queries against a hospital database to generate insights related to medical projects.

## Key Features

### Poster Retrieval System
- Semantic search over medical QI documents using LangGraph-based agent
- Filtering by year and hospital
- Follow-up question mode after retrieving relevant posters
- PDF downloads for poster documents
- Project code extraction and linking

### Data Generation System
- SQL query interface to hospital database
- Automated SQL question generation based on poster content
- Direct SQL query capability in data generation mode

### User Interface
- Chat-based interface with message history
- Mode switching between poster retrieval and data generation
- Sidebar with debug information and session state tracking
- Real-time response streaming

## Architecture

The application uses:
- **LangGraph**: For orchestrating the RAG workflow with multiple nodes (retriever, grader, rewriter, generator)
- **Weaviate**: Vector database for storing and retrieving document embeddings
- **Langchain**: For connecting components and managing the RAG pipeline
- **Ollama**: For local LLM integration
- **SQLite**: For hospital database storage and querying
- **Streamlit**: For the web interface

## Technical Components

- **RAG System**: Uses semantic search with section classification and metadata filtering
- **Follow-up System**: Processes follow-up questions about retrieved posters
- **SQL Query Agent**: Processes database queries and formats results
- **Session State Management**: Tracks conversation state, follow-up mode, and SQL questions

## Getting Started

1. Ensure you have all required dependencies installed
2. Start a local Weaviate instance on port 8081
3. Place poster PDFs in the `posters` directory
4. Make sure the SQLite database exists at `database/bt5153_gp.db`
5. Run the Streamlit application:
```
streamlit run streamlit_app_v2.py
```

## Debug Features

The application includes extensive debug logging and session state tracking to aid in development and troubleshooting. 