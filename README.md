# Weaviate RAG and LangGraph Application System

This project implements a two-part integrated system:
1. A Retrieval Augmented Generation (RAG) system using Weaviate for processing and indexing medical poster PDFs
2. A Text-to-SQL system for querying a hospital database using natural language
3. A Streamlit application that integrates both components into a user-friendly conversational interface

## Datasets

### RAG Posters
The medical QI posters used in this project are publicly available from the Singapore Healthcare Management Congress:
[https://www.singaporehealthcaremanagement.sg/Pages/Poster-Exhibition-2022.aspx](https://www.singaporehealthcaremanagement.sg/Pages/Poster-Exhibition-2022.aspx)

### Database
The hospital database contains synthesized data (not real patient data) stored in a SQLite database file (.db). This database is used for the Text-to-SQL component to demonstrate how natural language queries can be converted to SQL and executed against structured data.

## Prerequisites

Before getting started, make sure you have the following installed:

- [Docker](https://www.docker.com/get-started) for running Weaviate
- [Ollama](https://ollama.com/download) for local language models
- Python 3.8+ with pip

Download the required Ollama models by running:

```bash
# Embedding model
ollama pull nomic-embed-text

# LLM options - pick one or more based on your hardware capabilities
ollama pull llama3.2       # For basic testing
ollama pull llama3.1       # 8B parameter model
ollama pull qwen2.5        # Base 7B parameter model
ollama pull qwen2.5:14b    # 14B parameter model
ollama pull qwen2.5:32b    # 32B parameter model (requires high-end GPU)
```

We will be running Weaviate and language models locally. We recommend using a modern computer with at least 8GB of RAM, preferably 16GB or more. For the larger models (qwen2.5:14b and qwen2.5:32b), a GPU with 16GB+ VRAM is recommended.

## Python Environment Setup

It's recommended to use a virtual environment for this project:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Project Structure

The project is organized into two main components:

```
5153 langgraph E drive v2/
├── weaviate-rag-loader/       # RAG component for processing and indexing PDFs
│   ├── data/                  # PDF files to be processed
│   ├── docker-compose.yml     # Weaviate configuration
│   ├── ollama_poster_chunker_by_sections.py
│   ├── ollama_poster_chunker_by_full_poster.py
│   └── table_loader.py        # Utility for loading tabular data from PDFs
│
└── langgraph-app/             # LangGraph application component
    ├── agents/                # Agent definitions and components
    ├── database/              # SQLite database for hospital data
    ├── evaluation/            # Tools for evaluating model performance
    ├── logs/                  # Application logs
    ├── posters/               # PDF posters for display in the app
    ├── utils/                 # Utility functions
    ├── requirements.txt       # Python dependencies
    ├── config.py              # Application configuration
    ├── model_config_evaluator.py  # Model evaluation tools
    └── streamlit_app_v2.py    # Main Streamlit application
```

## Step 1: Set up Weaviate RAG System

### 1.1 Create a Weaviate Database

A `docker-compose.yml` file is already provided in the `weaviate-rag-loader` directory. This configuration sets up a Weaviate instance with the necessary modules for text vectorization and generation using Ollama.

The docker-compose file includes:
- Weaviate running on port 8081
- Text2vec-ollama and generative-ollama modules enabled
- Persistent volume for data storage

To start the Weaviate instance, navigate to the `weaviate-rag-loader` directory and run:

```bash
cd weaviate-rag-loader
docker-compose up -d
```

This will start Weaviate in detached mode. You can check if it's running by accessing the Weaviate health endpoint at http://localhost:8081/v1/.well-known/ready

### 1.2 Process and Load Posters to Weaviate

The project includes two main Python scripts for processing PDF posters:

1. `ollama_poster_chunker_by_sections.py` - Splits posters into sections before embedding
2. `ollama_poster_chunker_by_full_poster.py` - Processes each poster as a whole document

To process the posters and load them to Weaviate, run:

```bash
cd weaviate-rag-loader
python ollama_poster_chunker_by_sections.py
```

or

```bash
python ollama_poster_chunker_by_full_poster.py
```

These scripts will:
1. Extract text from the PDF posters in the `data` directory
2. Process the text using Ollama models
3. Create chunks with metadata
4. Upload the chunks to Weaviate for vector search

## Step 2: Run the LangGraph Application

After setting up the Weaviate database and loading your data, you can run the LangGraph-based Streamlit application.

### 2.1 Application Overview

The `langgraph-app` is a Streamlit application that provides:

- **Poster Retrieval Mode**: Search and retrieve medical quality improvement posters from the Weaviate database
- **Data Generation Mode**: Run SQL queries against a hospital database to generate insights
- **Conversation Interface**: Chat-based interface with support for follow-up questions
- **LangGraph Agents**: Orchestrated workflow for retrieval, grading, rewriting, and generation

### 2.2 Running the Application

To run the Streamlit application:

```bash
cd langgraph-app
streamlit run streamlit_app_v2.py
```

The application will be available at http://localhost:8501 in your web browser.

### 2.3 Supported Models

The application can use various Ollama models for different quality and performance tradeoffs:

- **llama3.1**: Good balance of performance and resource usage
- **qwen2.5**: Base model, suitable for most systems
- **qwen2.5:14b**: Better quality responses, requires more RAM/VRAM
- **qwen2.5:32b**: Highest quality responses, requires powerful GPU

You can configure which model to use in the `config.py` file in the main folder.

### 2.4 Key Files in langgraph-app

- `streamlit_app_v2.py`: Main Streamlit application with UI and session management
- `config.py`: Configuration settings for the application
- `agents/`: Directory containing LangGraph agent definitions
- `utils/`: Utility functions for data processing and API interactions
- `database/`: SQLite database for hospital data queries
- `posters/`: PDF posters displayed in the application
- `model_config_evaluator.py`: Tools for evaluating model configurations

## For More Information

- For more information on using Weaviate, see the [official quickstart guide](https://weaviate.io/developers/weaviate/quickstart/local)
- To learn more about LangGraph, visit the [LangGraph documentation](https://github.com/langchain-ai/langgraph) 