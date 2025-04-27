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
- Project code extraction and linking for easy reference

### Data Generation System
- SQL query interface to hospital database
- Automated SQL question generation based on poster content
- Direct SQL query capability in data generation mode
- Suggestions for relevant SQL queries based on retrieved posters

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

## Prerequisites

Before running the application, ensure you have:

1. **Python 3.9+** installed on your system
2. **Ollama** running locally with the required models (see config.py for model names)
3. **Weaviate** instance running on port 8081 (see setup instructions below)
4. The SQLite database file in the `database` directory

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies include:
- streamlit
- weaviate-client
- langchain
- langchain-ollama
- langgraph
- pdfplumber
- sqlalchemy

### 2. Set Up Weaviate

Make sure you have Docker installed, then:

```bash
# Navigate to the weaviate-rag-loader directory
cd ../weaviate-rag-loader

# Start Weaviate using docker-compose
docker-compose up -d
```

### 3. Process Posters

Ensure your poster PDFs are placed in the `posters` directory before running:

```bash
cd ../weaviate-rag-loader
python ollama_poster_chunker_by_sections.py
python ollama_poster_chunker_by_full_poster.py
```

### 4. Run the Application

```bash
cd ../langgraph-app
streamlit run app.py
```

The application will be available at http://localhost:8501

## Application Modes

The application has two main modes:

1. **Poster Retrieval Mode**: Default mode for searching and retrieving QI posters.
2. **Data Generation Mode**: For executing SQL queries against the hospital database.

Toggle between modes using the selector in the sidebar.

## Debug Features

The application includes extensive debug logging and session state tracking to aid in development and troubleshooting. Enable debug mode by setting `debug_mode = True` in app.py. 