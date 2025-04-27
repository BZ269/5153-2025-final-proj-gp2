import streamlit as st
import weaviate
from langchain_core.messages import HumanMessage, AIMessage
from agents.agentic_rag import compiled_graph, vectorstore
from agents.agentic_rag import AgentState
from weaviate.classes.query import Filter
from langgraph.errors import GraphRecursionError
import re
import os
from pathlib import Path
from utils.poster_qa import answer_poster_question, extract_project_codes
# Import SQLDatabase for the new feature
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from datetime import datetime
import json
# Import SQL query agent
from agents.sql_query_agent import create_graph, format_input

# Load environment variables from .env file
from dotenv import load_dotenv
import os

# Import config for models and settings
from config import (
    MODELS, 
    MODEL_TEMPERATURES,
    SQL_MAX_RETRY_LIMIT,
    SQL_MESSAGE_CONTEXT_LIMIT
)

# Debug mode flag - set to False to disable all debug output
debug_mode = False

# Load the .env file from the current directory
load_dotenv()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# New: Initialize agent state for LangGraph
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

# New: Track if we're in follow-up mode
if "follow_up_mode" not in st.session_state:
    st.session_state.follow_up_mode = False
    
# New: Store the last poster response for follow-up questions
if "last_poster_response" not in st.session_state:
    st.session_state.last_poster_response = ""

# New: Store SQL question suggestions
if "sql_questions" not in st.session_state:
    st.session_state.sql_questions = []
    
# New: Flag to track if the UI needs updating after SQL question generation
if "needs_rerun" not in st.session_state:
    st.session_state.needs_rerun = False

# New: Debug log for tracking activity
if "debug_log" not in st.session_state:
    st.session_state.debug_log = []

# New: Initialize app mode toggle (poster retrieval is default)
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "poster_retrieval"

# New: Initialize SQL query graph
if "sql_graph" not in st.session_state:
    st.session_state.sql_graph = create_graph()

# New: Store the message index that generated the SQL questions
if "sql_source_message_index" not in st.session_state:
    st.session_state.sql_source_message_index = -1

# New: Store the current SQL input for data generation mode
if "current_sql_input" not in st.session_state:
    st.session_state.current_sql_input = ""

# New: Initialize session state for first-time SQL query
if "is_first_sql_query" not in st.session_state:
    st.session_state.is_first_sql_query = False

# Connect to Weaviate to get metadata values
client = weaviate.connect_to_local(port=8081)

# Path to the SQLite database for hospital data
database_path = os.path.join("database", "bt5153_gp.db")

# Function to add to debug log
def add_debug_log(message):
    if debug_mode:
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.debug_log.append(f"[{timestamp}] {message}")
        if len(st.session_state.debug_log) > 100:  # Keep last 100 entries
            st.session_state.debug_log = st.session_state.debug_log[-100:]

# Wrapper for answer_poster_question with debug logging
def process_follow_up_question(question: str, poster_response: str) -> str:
    """Debug wrapper for answer_poster_question"""
    add_debug_log(f"Processing follow-up question: {question[:50]}...")
    add_debug_log(f"Using poster response of length: {len(poster_response)}")
    
    # Extract project codes for debugging
    project_codes = extract_project_codes(poster_response)
    add_debug_log(f"Found project codes: {project_codes}")
    
    try:
        # Call the original function
        response = answer_poster_question(question, poster_response)
        add_debug_log(f"Follow-up response generated: {len(response)} chars")
        return response
    except Exception as e:
        error_msg = f"Error in answer_poster_question: {str(e)}"
        add_debug_log(f"ERROR: {error_msg}")
        import traceback
        tb = traceback.format_exc()
        add_debug_log(tb)
        return f"Sorry, I encountered an error while answering your follow-up question: {str(e)}"

# Function to generate SQL questions based on response context
def generate_sql_questions(response_text):
    try:
        # Connect to the SQLite database
        db = SQLDatabase.from_uri(f'sqlite:///{database_path}')
        
        # Get table schema information
        table_info = db.get_table_info()
        
        # Extract table names and available columns for validation
        tables_and_columns = {}
        for line in table_info.split('\n'):
            if line.strip().startswith('CREATE TABLE'):
                current_table = line.strip().split('"')[1] if '"' in line else line.split('TABLE')[1].strip().strip('(')
                tables_and_columns[current_table] = []
            elif 'INTEGER' in line or 'VARCHAR' in line or 'TEXT' in line or 'DATE' in line or 'TIMESTAMP' in line or 'BOOLEAN' in line:
                if ',' in line:
                    col_name = line.strip().split()[0].strip('"')
                    if col_name and col_name != 'PRIMARY' and col_name != 'CONSTRAINT' and col_name != 'FOREIGN':
                        if current_table in tables_and_columns:
                            tables_and_columns[current_table].append(col_name)
        
        # Initialize the Qwen model for question generation
        model = ChatOllama(model=MODELS["sql_question_gen"], temperature=MODEL_TEMPERATURES["sql_question_gen"])
        
        # Create a prompt for the model with better guidance for relevant but answerable questions
        prompt = f"""
        Generate EXACTLY 3 SQL questions based on the AI response below. For each question:
        1. First, think through what tables and columns would be needed to answer this question
        2. Write valid SQL code that can actually be run against the database schema
        3. Then write a clear, simple question that the SQL would answer

        REQUIREMENTS:
        1. Questions MUST use ONLY the fields in the database schema
        2. Make questions relevant to the projects described in the AI response
        3. SQL code must be valid and executable against the provided schema
        4. Focus on SIMPLE queries that count, aggregate or compare basic data
        5. Do NOT reference data that isn't in the schema (like "fall rates" or "confidence levels")
        
        Here are the tables and columns available in the database:
        {json.dumps(tables_and_columns, indent=2)}
        
        AI RESPONSE ABOUT PROJECTS:
        {response_text}
        
        DATABASE SCHEMA:
        {table_info}
        
        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        Question 1:
        SQL code: [VALID SQL CODE USING ONLY TABLES AND COLUMNS IN THE SCHEMA]
        Question: [SIMPLE QUESTION THAT THE SQL ANSWERS]

        Question 2:
        SQL code: [VALID SQL CODE USING ONLY TABLES AND COLUMNS IN THE SCHEMA]
        Question: [SIMPLE QUESTION THAT THE SQL ANSWERS]

        Question 3:
        SQL code: [VALID SQL CODE USING ONLY TABLES AND COLUMNS IN THE SCHEMA]
        Question: [SIMPLE QUESTION THAT THE SQL ANSWERS]
        """
        
        # Generate questions with the model
        result = model.invoke(prompt)
        questions_text = result.content.strip()
        
        # Log raw output for debugging
        try:
            os.makedirs('logs', exist_ok=True)
            with open('logs/sql_raw_output.log', 'a') as f:
                f.write(f"--- {datetime.now()} ---\n")
                f.write(questions_text)
                f.write("\n\n")
        except:
            pass  # Silently ignore logging errors
        
        # Process the results - extract just the questions, not the SQL code
        if "No relevant database questions available" in questions_text:
            return []
        
        # Extract questions - look for lines that start with "Question: "
        questions = []
        lines = questions_text.split('\n')
        for line in lines:
            if line.strip().startswith("Question:"):
                # Extract the question part (after "Question: ")
                question = line.strip()[len("Question:"):].strip()
                if question and len(question) > 5:  # Basic validation
                    questions.append(question)
        
        # Return the questions (up to 3)
        return questions[:3]
    
    except Exception as e:
        add_debug_log(f"SQL question generation error: {str(e)}")
        return []

# Hardcoded unique values for filters
years = ['2021', '2022', '2023', '2024']
hospitals = [
    'SGH',   # Singapore General Hospital
    'CGH',   # Changi General Hospital 
    'KTPH',  # Khoo Teck Puat Hospital
    'NUH',   # National University Hospital
    'TTSH',  # Tan Tock Seng Hospital
    'SKCH',  # Sengkang Community Hospital
    'SKH',   # Sengkang Hospital
    'AH',    # Alexandra Hospital
    'NTFGH', # Ng Teng Fong General Hospital
    'KKH',   # KK Women's and Children's Hospital
    'IMH',   # Institute of Mental Health
    'NHCS',  # National Heart Centre Singapore
    'NCCS',  # National Cancer Centre Singapore
    'SHHQ',  # SingHealth HQ
    'OCH',   # Outram Community Hospital
]

# Function to extract project code and create clickable links
def process_project_links(content):
    # Regular expression to find project codes in titles
    # This pattern looks for various formats:
    # "SHM RM001", "SHM_RM001", "PE035", etc.
    patterns = [
        r"PROJECT TITLE: SHM[_ ]?([A-Z]{2}\d{3})",  # Standard format
        r"PROJECT TITLE:.*?SHM[_ ]?([A-Z]{2}\d{3})",  # With text between PROJECT TITLE: and SHM
        r"PROJECT TITLE:.*?([A-Z]{2}\d{3})"  # Just looking for the code pattern
    ]
    
    # Define the poster directory with absolute path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    poster_dir = os.path.join(base_dir, "posters")
    
    # List all files in the poster directory once
    if os.path.exists(poster_dir):
        poster_files = os.listdir(poster_dir)
        if debug_mode:
            st.sidebar.write("Available poster files:", ", ".join(poster_files))
    else:
        if debug_mode:
            st.sidebar.write(f"Poster directory not found: {poster_dir}")
        os.makedirs(poster_dir, exist_ok=True)
    
    # Process the content
    lines = content.split("\n")
    processed_lines = []
    found_codes = []
    poster_buttons = {}  # Will store project codes and file paths for buttons
    
    i = 0
    while i < len(lines):
        line = lines[i]
        processed_lines.append(line)
        
        # Check if this is a project title line
        project_code = None
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                project_code = match.group(1)  # Extract the project code
                break
                
        if project_code and project_code not in found_codes:
            found_codes.append(project_code)
            
            # Look for the corresponding PDF file
            # First check for exact matches with the poster code
            file_found = False
            for poster_file in poster_files:
                if project_code in poster_file:
                    # Store the full absolute path for later use
                    file_path = os.path.join(poster_dir, poster_file)
                    poster_buttons[project_code] = (file_path, poster_file)
                    
                    # Add a placeholder message - we'll replace this with buttons later
                    processed_lines.append(f"__POSTER_BUTTON__{project_code}__")
                    file_found = True
                    break
            
            # If no file found, add a message
            if not file_found:
                processed_lines.append(f"*Poster file for {project_code} not available*")
        
        i += 1
    
    # Show the codes found in the sidebar
    if found_codes and debug_mode:
        st.sidebar.write("Project codes found:", ", ".join(found_codes))
    
    # Convert to string and return both the processed text and the button info
    return "\n".join(processed_lines), poster_buttons

# Main chat interface with wider styling
st.title("Medical QI Document Assistant")

# Use custom CSS to improve visual appearance and fix header cutoff
st.markdown("""
<style>
    div.block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem;
        max-width: 1200px !important;
    }
    
    button.stButton>button {
        font-weight: bold !important;
    }
    
    h1 {
        margin-bottom: 1.5rem !important;
    }
    
    .filter-column {
        background-color: #f9f9f9;
        border-right: 1px solid #ddd;
        padding: 10px;
    }
    
    .sql-column {
        background-color: #f9f9f9;
        border-left: 1px solid #ddd;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Create a 3-column layout: Filters, Main chat, SQL questions
cols = st.columns([1, 3, 1])

# Left column for RAG filters (moved from sidebar)
with cols[0]:
    st.markdown('<div class="filter-column">', unsafe_allow_html=True)
    
    # Only show filters in poster retrieval mode
    if st.session_state.app_mode == "poster_retrieval":
        st.markdown("### Poster Retrieval Filters")
        
        # Year filter
        selected_years = st.multiselect(
            "Filter by Year",
            options=years,
            default=[],
            help="Select years to filter documents"
        )
        
        # Hospital filter
        selected_hospitals = st.multiselect(
            "Filter by Hospital",
            options=hospitals,
            default=[],
            help="Select hospitals to filter documents"
        )
    
    # Add a clear conversation button
    if st.button("Clear Conversation", type="primary"):
        st.session_state.messages = []
        st.session_state.agent_messages = []
        st.session_state.follow_up_mode = False
        st.session_state.last_poster_response = ""
        st.session_state.sql_questions = []
        st.rerun()  # Force a complete UI refresh
        
    # Add a button to exit follow-up mode in a prominent position
    if st.session_state.app_mode == "poster_retrieval" and st.session_state.follow_up_mode:
        st.markdown("---")
        st.write("**Currently in Follow-up Mode**")
        if st.button("Exit Follow-up Mode", key="exit_follow_up"):
            st.session_state.follow_up_mode = False
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main chat area in the middle column
with cols[1]:
    # Show follow-up mode indicator
    if st.session_state.follow_up_mode:
        st.info("📌 You are now in **Poster Follow-up Mode**. Ask anything about the returned posters.")
        # Add a small space for visual separation
        st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)

    # Add debug indicator to show pending SQL query (ADDED)
    if st.session_state.app_mode == "data_generation" and st.session_state.current_sql_input:
        st.warning(f"⚠️ Processing SQL query... (Query length: {len(st.session_state.current_sql_input)} chars)")
        add_debug_log(f"CRITICAL - At top of middle column: current_sql_input exists with {len(st.session_state.current_sql_input)} chars")
        add_debug_log(f"CRITICAL - is_first_sql_query = {st.session_state.is_first_sql_query}")

    # Display chat messages
    for message_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Process project links for assistant messages
            if message["role"] == "assistant":
                processed_text, poster_buttons = process_project_links(message["content"])
                
                # Split by poster button placeholders and render with buttons
                parts = processed_text.split("__POSTER_BUTTON__")
                for i, part in enumerate(parts):
                    if i > 0:
                        # Extract the project code
                        code_end = part.find("__")
                        if code_end > 0:
                            project_code = part[:code_end]
                            if project_code in poster_buttons:
                                file_path, file_name = poster_buttons[project_code]
                                # Read the file content
                                try:
                                    with open(file_path, "rb") as f:
                                        file_content = f.read()
                                    # Display a download button with a unique key
                                    unique_key = f"download_{message_idx}_{project_code}_{i}"
                                    st.download_button(
                                        label=f"📄 Download Poster PDF for {project_code}",
                                        data=file_content,
                                        file_name=file_name,
                                        mime="application/pdf",
                                        key=unique_key
                                    )
                                except Exception as e:
                                    st.error(f"Error reading file: {str(e)}")
                            
                            # Display the rest of the part (after the closing __)
                            if code_end + 2 < len(part):
                                st.markdown(part[code_end + 2:])
                    else:
                        # Display the first part as is
                        st.markdown(part)
            else:
                st.markdown(message["content"])

    # New: Display prompt for follow-up questions if we have received a poster response
    if st.session_state.follow_up_mode and extract_project_codes(st.session_state.last_poster_response):
        pass
    
    # NOW add auto-processing AFTER chat history is displayed
    # This ensures previous chat remains visible while processing
    if st.session_state.app_mode == "data_generation" and st.session_state.current_sql_input:
        add_debug_log("AUTO-PROCESSING: Found pending SQL query to process automatically")
        
        # Make sure SQL graph is initialized
        if "sql_graph" not in st.session_state or st.session_state.sql_graph is None:
            add_debug_log("CRITICAL - sql_graph not found or is None, creating now")
            from agents.sql_query_agent import create_graph
            try:
                st.session_state.sql_graph = create_graph()
                add_debug_log("Successfully created SQL graph")
            except Exception as graph_error:
                add_debug_log(f"CRITICAL ERROR creating SQL graph: {str(graph_error)}")
                st.error(f"Failed to initialize SQL processing engine: {str(graph_error)}")
                import traceback
                add_debug_log(traceback.format_exc())
                
        pending_sql_input = st.session_state.current_sql_input
        
        # Inspect the content of the SQL input
        add_debug_log(f"CRITICAL - SQL input content: {pending_sql_input[:100]}...")
        
        # Make sure we're definitely in data generation mode
        if st.session_state.app_mode != "data_generation":
            st.session_state.app_mode = "data_generation"
            add_debug_log("CRITICAL - Forced app_mode to data_generation")
        
        # Get the first query flag status and reset it immediately
        is_first_query = st.session_state.is_first_sql_query
        st.session_state.is_first_sql_query = False
        add_debug_log(f"First SQL query status: {is_first_query}")
        
        # Clear SQL input immediately to prevent reprocessing on the next rerun
        current_input_length = len(st.session_state.current_sql_input)
        st.session_state.current_sql_input = ""
        add_debug_log(f"Cleared current_sql_input (was {current_input_length} chars) to prevent reprocessing")
        
        # Show a progress indicator that the query is being processed
        st.info("⏳ SQL Query is being processed...")
        
        # Create database connection for verification
        try:
            # Verify database exists before running
            if os.path.exists(database_path):
                add_debug_log(f"Database found at: {database_path}")
                # Try to connect to verify database is accessible
                test_db = SQLDatabase.from_uri(f'sqlite:///{database_path}')
                test_info = test_db.get_table_info()
                add_debug_log(f"Database connection successful: {len(test_info.split('CREATE TABLE'))} tables found")
            else:
                add_debug_log(f"WARNING: Database not found: {database_path}")
                st.error(f"Database not found at {database_path}. SQL queries cannot be executed.")
        except Exception as e:
            add_debug_log(f"ERROR: Database connection test failed: {str(e)}")
            st.error(f"Database connection failed: {str(e)}")
        
        # Handle differently for the first query after a page load to avoid faded message
        if is_first_query:
            add_debug_log("Using special first-query handling to avoid faded message")
            
            # For the first query, create a new container outside the chat flow
            # This completely avoids the faded message issue for the first query
            first_query_container = st.container()
            with first_query_container:
                # Prominent processing indicator
                with st.spinner("Processing SQL query..."):
                    try:
                        add_debug_log(f"Processing first SQL query: {pending_sql_input[:100]}...")
                        
                        # Log SQL graph status
                        if "sql_graph" in st.session_state:
                            add_debug_log("SQL graph exists in session state")
                        else:
                            add_debug_log("WARNING: sql_graph not found in session state!")
                            st.session_state.sql_graph = create_graph()
                            add_debug_log("Created new SQL graph")
                        
                        add_debug_log("Starting SQL agent execution for first query...")
                        
                        # Process the query but don't stream (to avoid faded message issue)
                        response_text = ""
                        try:
                            # Format the input for the SQL graph
                            formatted_input = format_input(pending_sql_input)
                            add_debug_log(f"Formatted input created with {len(formatted_input)} elements")
                            
                            # Execute the query with additional debugging
                            for event in st.session_state.sql_graph.stream(formatted_input, config={"configurable": {"thread_id": "1"}}):
                                add_debug_log(f"Event received: {list(event.keys())}")
                                if "format_response" in event:
                                    for value in event.values():
                                        content = value["messages"].content
                                        response_text += content
                                        add_debug_log(f"Received first query response chunk: {len(content)} chars")
                            
                            # Check if we got a valid response
                            if not response_text:
                                add_debug_log("WARNING: No response text was generated from SQL agent!")
                                response_text = "Sorry, I couldn't process this SQL query. The database server might be unavailable or there was an error in the query."
                        except Exception as sql_exec_error:
                            error_detail = f"SQL execution error: {str(sql_exec_error)}"
                            add_debug_log(f"ERROR during SQL execution: {error_detail}")
                            import traceback
                            add_debug_log(f"Traceback: {traceback.format_exc()}")
                            response_text = f"An error occurred while executing the SQL query: {error_detail}"
                        
                        # Add the response to chat history when complete
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                        add_debug_log(f"First SQL query completed: {len(response_text)} chars - added to messages")
                        
                        # Force another rerun to properly display the completed response in chat history
                        st.session_state.needs_rerun = True
                        add_debug_log("Setting needs_rerun=True to display completed first query")
                        st.rerun()
                    except Exception as e:
                        error_msg = f"An error occurred in SQL agent: {str(e)}"
                        add_debug_log(f"ERROR in first query: {error_msg}")
                        
                        # Add stack trace to debug log
                        import traceback
                        stack_trace = traceback.format_exc()
                        add_debug_log(f"Stack trace: {stack_trace}")
                        
                        # Add error to messages
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        st.error(error_msg)
                        st.rerun()
        else:
            # Use the normal approach for subsequent queries
            # Create a custom container for SQL processing with styling
            sql_processing_container = st.container()
            with sql_processing_container:
                # Add a visual divider
                st.markdown("---")
                # Use columns to create an assistant-like message without using chat_message
                cols = st.columns([0.1, 0.9])
                with cols[0]:
                    st.image("https://api.dicebear.com/7.x/bottts/svg?seed=assistant", width=50)
                
                with cols[1]:
                    response_container = st.container()
                    with response_container:
                        response_placeholder = st.empty()
                        response_text = ""
                        response_placeholder.markdown("🔍 **Processing SQL query...**")
                
                        try:
                            add_debug_log(f"Auto-processing SQL input: {pending_sql_input[:100]}...")
                            
                            # Log SQL graph status
                            if "sql_graph" in st.session_state:
                                add_debug_log("SQL graph exists in session state")
                            else:
                                add_debug_log("WARNING: sql_graph not found in session state!")
                                st.session_state.sql_graph = create_graph()
                                add_debug_log("Created new SQL graph")
                            
                            add_debug_log("Starting SQL agent execution...")
                            
                            # Stream response from SQL query agent
                            for event in st.session_state.sql_graph.stream(format_input(pending_sql_input), config={"configurable": {"thread_id": "1"}}):
                                if "format_response" in event:
                                    for value in event.values():
                                        content = value["messages"].content
                                        response_text += content
                                        response_placeholder.markdown(response_text)
                                        add_debug_log(f"Received SQL agent response chunk: {len(content)} chars")
                            
                            # Add the response to chat history only after it's complete
                            st.session_state.messages.append({"role": "assistant", "content": response_text})
                            add_debug_log(f"SQL agent auto-response completed: {len(response_text)} chars - added to messages")
                            
                        except Exception as e:
                            error_msg = f"An error occurred in SQL agent: {str(e)}"
                            add_debug_log(f"ERROR in auto-processing: {error_msg}")
                            
                            # Add stack trace to debug log
                            import traceback
                            stack_trace = traceback.format_exc()
                            add_debug_log(f"Stack trace: {stack_trace}")
                            
                            response_placeholder.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                # Add a visual divider
                st.markdown("---")

# SQL questions in the right column
with cols[2]:
    st.markdown('<div class="sql-column">', unsafe_allow_html=True)
    
    # Display SQL questions if available
    if st.session_state.sql_questions:
        st.markdown("### 📊 Database Insights")
        st.markdown("**Database queries you can run:**")
        
        # Add debug info about current SQL questions
        add_debug_log(f"Rendering {len(st.session_state.sql_questions)} SQL questions")
        
        for i, question in enumerate(st.session_state.sql_questions):
            q_container = st.container(border=True)
            with q_container:
                st.markdown(f"**Q{i+1}:** {question}")
                #st.caption("This query explores relationships in the hospital data to provide insights relevant to the poster topics")
                btn_key = f"sql_q_{i}"
                add_debug_log(f"Creating SQL button with key '{btn_key}'")
                
                if st.button("📊 Run this query", key=btn_key, use_container_width=True):
                    # Add extensive debug logging for button click event
                    add_debug_log(f"SQL BUTTON CLICKED: Button {btn_key} for question {i+1}")
                    add_debug_log(f"Question text: {question[:50]}...")
                    
                    # Log session state details
                    add_debug_log(f"App mode: {st.session_state.app_mode}")
                    add_debug_log(f"Follow-up mode: {st.session_state.follow_up_mode}")
                    add_debug_log(f"SQL source message index: {st.session_state.sql_source_message_index}")
                    add_debug_log(f"Messages count: {len(st.session_state.messages)}")
                    
                    # When clicked, add this as a new user query
                    st.session_state.messages.append({"role": "user", "content": question})
                    add_debug_log(f"Added user message, new count: {len(st.session_state.messages)}")
                    
                    # Get the poster data from the stored source message index
                    poster_message = ""
                    if 0 <= st.session_state.sql_source_message_index < len(st.session_state.messages):
                        source_msg = st.session_state.messages[st.session_state.sql_source_message_index]
                        if source_msg["role"] == "assistant":
                            poster_message = source_msg["content"]
                            add_debug_log(f"Found source poster message at index {st.session_state.sql_source_message_index}, length: {len(poster_message)} chars")
                        else:
                            add_debug_log(f"WARNING: Source message at index {st.session_state.sql_source_message_index} is not from assistant, role is: {source_msg['role']}")
                    else:
                        add_debug_log(f"WARNING: Invalid source message index {st.session_state.sql_source_message_index}, messages length: {len(st.session_state.messages)}")
                    
                    # IMPORTANT: First disable follow-up mode if it's active
                    if st.session_state.follow_up_mode:
                        add_debug_log("Exiting follow-up mode to process SQL query")
                        st.session_state.follow_up_mode = False
                    
                    # Force app mode to data generation for this query
                    old_mode = st.session_state.app_mode
                    st.session_state.app_mode = "data_generation"
                    add_debug_log(f"Switching app mode from '{old_mode}' to 'data_generation'")
                    
                    # Store the combined input for the SQL agent to process on rerun
                    combined_input = f"Context from previous response:\n{poster_message}\n\nUser Question: {question}"
                    st.session_state.current_sql_input = combined_input
                    add_debug_log(f"Set current_sql_input with length: {len(combined_input)} chars")
                    
                    # Set the first SQL query flag to true BEFORE setting needs_rerun
                    st.session_state.is_first_sql_query = True
                    add_debug_log("Set is_first_sql_query=True for special first-query handling")
                    
                    # Add a visual indicator that we're about to rerun
                    st.write("Processing query... please wait.")
                    
                    # Force rerun to refresh the page - LAST STEP
                    st.session_state.needs_rerun = True
                    add_debug_log("Set needs_rerun=True, calling st.rerun() now...")
                    
                    # Force rerun to refresh the page
                    st.rerun()
    else:
        # Always show a placeholder to ensure column is visible
        placeholder = st.empty()
        placeholder.markdown("### Database Insights")
        placeholder.markdown("*DB questions will appear here*")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Chat input area (below both columns)
if prompt := st.chat_input("Ask about medical QI documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add debug log
    add_debug_log(f"User input: {prompt[:50]}...")

    # Process based on current app mode
    if st.session_state.app_mode == "poster_retrieval":
        # Handle follow-up questions if in follow-up mode
        if st.session_state.follow_up_mode:
            add_debug_log("Processing in follow-up mode")
            with st.spinner("Thinking about your follow-up question..."):
                try:
                    # Process the follow-up question
                    response = process_follow_up_question(prompt, st.session_state.last_poster_response)
                    add_debug_log(f"Follow-up response generated: {len(response)} chars")
                    
                    # Add assistant response to display chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Display the response
                    with st.chat_message("assistant"):
                        # Process project links in follow-up responses same as normal responses
                        processed_text, poster_buttons = process_project_links(response)
                        
                        # Split by poster button placeholders and render with buttons
                        parts = processed_text.split("__POSTER_BUTTON__")
                        for i, part in enumerate(parts):
                            if i > 0:
                                # Extract the project code
                                code_end = part.find("__")
                                if code_end > 0:
                                    project_code = part[:code_end]
                                    if project_code in poster_buttons:
                                        file_path, file_name = poster_buttons[project_code]
                                        # Read the file content
                                        try:
                                            with open(file_path, "rb") as f:
                                                file_content = f.read()
                                            # Display a download button with a unique key for follow-up response
                                            unique_key = f"followup_{project_code}_{i}"
                                            st.download_button(
                                                label=f"📄 Download Poster PDF for {project_code}",
                                                data=file_content,
                                                file_name=file_name,
                                                mime="application/pdf",
                                                key=unique_key
                                            )
                                        except Exception as e:
                                            st.error(f"Error reading file: {str(e)}")
                                    
                                    # Display the rest of the part (after the closing __)
                                    if code_end + 2 < len(part):
                                        st.markdown(part[code_end + 2:])
                            else:
                                # Display the first part as is
                                st.markdown(part)
                    
                    # Stay in follow-up mode after answering
                    st.session_state.follow_up_mode = True
                    
                except Exception as e:
                    error_msg = f"Error in follow-up mode: {str(e)}"
                    add_debug_log(f"ERROR: {error_msg}")
                    st.error(error_msg)
                    
        else:
            add_debug_log("Processing in normal poster retrieval mode")
            # Handle normal query flow with metadata filters
            # Prepare metadata filters
            metadata_filter = None
            active_filters = []
            
            if selected_years or selected_hospitals:
                year_filter = None
                hospital_filter = None
                
                if selected_years:
                    active_filters.append(f"Years: {', '.join(selected_years)}")
                    for year in selected_years:
                        f = Filter.by_property("year").equal(year)
                        year_filter = f if year_filter is None else year_filter | f
                
                if selected_hospitals:
                    active_filters.append(f"Hospitals: {', '.join(selected_hospitals)}")
                    for hospital in selected_hospitals:
                        f = Filter.by_property("hospital").equal(hospital)
                        hospital_filter = f if hospital_filter is None else hospital_filter | f
                
                if year_filter and hospital_filter:
                    metadata_filter = year_filter & hospital_filter
                else:
                    metadata_filter = year_filter or hospital_filter

            # Add active filters message if any are selected
            if active_filters:
                filter_message = "🔍 Active Filters:\n" + "\n".join(f"- {f}" for f in active_filters)
                st.session_state.messages.append({"role": "system", "content": filter_message})
                with st.chat_message("system"):
                    st.markdown(filter_message)

            # Add the new user message to the agent's messages
            st.session_state.agent_messages.append(HumanMessage(content=prompt))
            
            # Create initial state for the graph that includes the full conversation history
            initial_state = {
                "messages": st.session_state.agent_messages,
                "metadata_filter": metadata_filter
            }

            # Debugging output to verify state contents
            if debug_mode:
                st.sidebar.write("Debug: Metadata Filter Applied", "Yes" if metadata_filter else "No")
                st.sidebar.write("Debug: Conversation Length", len(st.session_state.agent_messages))

            # Run the agent with metadata filter
            try:
                with st.spinner("Thinking..."):
                    add_debug_log("Invoking LangGraph agent")
                    result = compiled_graph.invoke(initial_state, {"recursion_limit": 10})
                    
                    # Get the latest AI message from the result
                    response = result["messages"][-1].content
                    add_debug_log(f"Agent response received: {len(response)} chars")
                    
                    # Save the full updated message history back to session state
                    st.session_state.agent_messages = result["messages"]
                    
                    # Add assistant response to display chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        processed_text, poster_buttons = process_project_links(response)
                        
                        # Split by poster button placeholders and render with buttons
                        parts = processed_text.split("__POSTER_BUTTON__")
                        for i, part in enumerate(parts):
                            if i > 0:
                                # Extract the project code
                                code_end = part.find("__")
                                if code_end > 0:
                                    project_code = part[:code_end]
                                    if project_code in poster_buttons:
                                        file_path, file_name = poster_buttons[project_code]
                                        # Read the file content
                                        try:
                                            with open(file_path, "rb") as f:
                                                file_content = f.read()
                                            # Display a download button with a unique key for current response
                                            unique_key = f"current_response_{project_code}_{i}"
                                            st.download_button(
                                                label=f"📄 Download Poster PDF for {project_code}",
                                                data=file_content,
                                                file_name=file_name,
                                                mime="application/pdf",
                                                key=unique_key
                                            )
                                        except Exception as e:
                                            st.error(f"Error reading file: {str(e)}")
                                    
                                    # Display the rest of the part (after the closing __)
                                    if code_end + 2 < len(part):
                                        st.markdown(part[code_end + 2:])
                            else:
                                # Display the first part as is
                                st.markdown(part)
                    
                    # NEW FEATURE: Generate SQL questions based on the response
                    with st.spinner("Generating related database questions..."):
                        try:
                            # Check if the response is the "not relevant" message
                            not_relevant_message = "I'm specialized in medical quality improvement projects. Please ask a question related to QIPs"
                            if not_relevant_message in response:
                                add_debug_log("Skipping SQL question generation for irrelevant query response")
                                # Empty the SQL questions to ensure none are displayed
                                st.session_state.sql_questions = []
                            else:
                                # Verify database exists before running
                                if os.path.exists(database_path):
                                    add_debug_log(f"Database found: {database_path}")
                                else:
                                    add_debug_log(f"WARNING: Database not found: {database_path}")
                                
                                # Generate the SQL questions
                                add_debug_log("Starting SQL question generation")
                                sql_questions = generate_sql_questions(response)
                                add_debug_log(f"Generated {len(sql_questions)} SQL questions")
                                
                                # Store the current message index as the source for these SQL questions
                                # The -1 refers to the most recently added message (the assistant response)
                                st.session_state.sql_source_message_index = len(st.session_state.messages) - 1
                                
                                # For debugging, log output to text file
                                try:
                                    os.makedirs('logs', exist_ok=True)
                                    with open('logs/sql_questions.log', 'a') as f:
                                        f.write(f"--- {datetime.now()} ---\n")
                                        f.write(f"Response: {response[:200]}...\n")
                                        f.write(f"Questions found: {len(sql_questions)}\n")
                                        for q in sql_questions:
                                            f.write(f"  - {q}\n")
                                        f.write("\n\n")
                                except:
                                    pass  # Silently ignore logging errors
                                
                                # Debug the returned questions
                                for q in sql_questions:
                                    add_debug_log(f"SQL Question: {q}")
                                
                                # Update session state
                                st.session_state.sql_questions = sql_questions
                                
                                # Force a rerun to update the UI with the new questions
                                if sql_questions:
                                    add_debug_log("Setting needs_rerun flag for SQL questions")
                                    st.session_state.needs_rerun = True
                                
                        except Exception as e:
                            error_msg = f"Error in SQL question workflow: {str(e)}"
                            add_debug_log(f"ERROR: {error_msg}")
                            import traceback
                            add_debug_log(traceback.format_exc())
                            st.session_state.sql_questions = []
                    
                    # Check if the response contains project codes
                    if extract_project_codes(response):
                        # Store the poster response for follow-up questions
                        st.session_state.last_poster_response = response
                        # Switch to follow-up mode
                        st.session_state.follow_up_mode = True
                        add_debug_log("Enabled follow-up mode for poster questions")
            except GraphRecursionError as e:
                error_msg = "There are no relevant projects on your query. Try again with a different query."
                add_debug_log(f"ERROR: {error_msg}")
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.error(error_msg)
                    
                # Reset follow-up mode when no relevant projects found
                st.session_state.follow_up_mode = False
                # Clear SQL questions
                st.session_state.sql_questions = []
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                add_debug_log(f"ERROR: {error_msg}")
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.error(error_msg)
                    
    else:  # Data generation mode
        add_debug_log("Processing direct user input in data generation mode")
        
        # Check if we're coming from a SQL button click
        if st.session_state.current_sql_input:
            add_debug_log(f"WARNING: Found current_sql_input with length: {len(st.session_state.current_sql_input)} - this should be handled automatically")
            # Clear it since we're handling a direct user input now
            st.session_state.current_sql_input = ""
            add_debug_log("Cleared current_sql_input to avoid conflict with direct user input")
        
        # First, clear follow-up mode flag if it's somehow still set
        if st.session_state.follow_up_mode:
            add_debug_log("Clearing follow-up mode flag in data generation mode")
            st.session_state.follow_up_mode = False
        
        # Stream agent response using the SQL query agent
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_text = ""
            response_placeholder.markdown("Thinking...")
            
            try:
                # Use direct user input
                input_query = prompt
                add_debug_log(f"Direct SQL agent input from user: {input_query[:100]}...")
                
                # Log SQL graph status
                if "sql_graph" in st.session_state:
                    add_debug_log("SQL graph exists in session state")
                else:
                    add_debug_log("WARNING: sql_graph not found in session state!")
                    st.session_state.sql_graph = create_graph()
                    add_debug_log("Created new SQL graph")
                
                add_debug_log("Starting SQL agent execution for direct user input...")
                
                # Stream response from SQL query agent
                for event in st.session_state.sql_graph.stream(format_input(input_query), config={"configurable": {"thread_id": "1"}}):
                    if "format_response" in event:
                        for value in event.values():
                            content = value["messages"].content
                            response_text += content
                            response_placeholder.markdown(response_text)
                            add_debug_log(f"Received SQL agent response chunk: {len(content)} chars")
                
                # Add the response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                add_debug_log(f"SQL agent direct response completed: {len(response_text)} chars - added to messages")
                
            except Exception as e:
                error_msg = f"An error occurred in SQL agent: {str(e)}"
                add_debug_log(f"ERROR in direct input: {error_msg}")
                
                # Add stack trace to debug log
                import traceback
                stack_trace = traceback.format_exc()
                add_debug_log(f"Stack trace: {stack_trace}")
                
                response_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Check if we need to rerun for UI updates
if st.session_state.needs_rerun:
    add_debug_log("EXECUTING RERUN: needs_rerun was True, resetting to False...")
    st.session_state.needs_rerun = False
    
    # Log the current state before rerunning
    add_debug_log(f"Before rerun: App mode = {st.session_state.app_mode}")
    add_debug_log(f"Before rerun: current_sql_input length = {len(st.session_state.current_sql_input) if st.session_state.current_sql_input else 0}")
    add_debug_log(f"Before rerun: Messages count = {len(st.session_state.messages)}")
    
    # Actually rerun the app
    st.rerun()
else:
    add_debug_log("No rerun needed at end of script")

# Move debug section to sidebar
with st.sidebar:
    # Move app mode toggle back to sidebar
    st.title("App Mode")
    
    # Add styling for app mode toggle
    st.markdown("""
    <style>
    div[data-testid="stRadio"] {
        background-color: #f0f0f5;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    div[data-testid="stRadio"] > label {
        font-weight: bold;
        color: #1E3250;
    }
    </style>
    """, unsafe_allow_html=True)
    
    app_mode = st.radio(
        "Select Mode:",
        options=["Poster retrieval", "Data generation"],
        index=0 if st.session_state.app_mode == "poster_retrieval" else 1,
        help="Switch between poster search and database query modes",
        key="app_mode_radio"  # Added a unique key
    )
    
    # Update session state when mode changes
    if app_mode == "Poster retrieval" and st.session_state.app_mode != "poster_retrieval":
        st.session_state.app_mode = "poster_retrieval"
        st.session_state.needs_rerun = True
    elif app_mode == "Data generation" and st.session_state.app_mode != "data_generation":
        st.session_state.app_mode = "data_generation"
        st.session_state.needs_rerun = True
    
    # Debug information section - only show when debug_mode is True
    if debug_mode:
        st.title("Debug Information")
        
        with st.expander("Current Session State", expanded=True):
            st.write(f"- App mode: {st.session_state.app_mode}")
            st.write(f"- Follow-up mode: {st.session_state.follow_up_mode}")
            st.write(f"- Messages: {len(st.session_state.messages)}")
            st.write(f"- SQL questions: {len(st.session_state.sql_questions)}")
            st.write(f"- needs_rerun: {st.session_state.needs_rerun}")
            st.write(f"- current_sql_input length: {len(st.session_state.current_sql_input) if st.session_state.current_sql_input else 0}")
            st.write(f"- is_first_sql_query: {st.session_state.is_first_sql_query}")
            
            # Add button to show all session state variables
            if st.button("Show All Session State", key="show_session_state"):
                st.write("**All Session State Variables:**")
                for key, value in st.session_state.items():
                    if isinstance(value, str):
                        display_value = f"{value[:50]}..." if len(value) > 50 else value
                    elif isinstance(value, list):
                        display_value = f"List with {len(value)} items"
                    else:
                        display_value = str(value)
                    st.write(f"- {key}: {display_value}")
        
        if st.session_state.sql_questions:
            with st.expander("SQL Questions"):
                for i, q in enumerate(st.session_state.sql_questions):
                    st.write(f"{i+1}. {q}")
                
        # More prominent activity log
        with st.expander("🔍 Recent Activity Log", expanded=True):
            st.markdown("<div style='height: 200px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: #f9f9f9;'>", unsafe_allow_html=True)
            
            if "debug_log" not in st.session_state:
                st.session_state.debug_log = []
                
            # Show more log entries
            for log in st.session_state.debug_log[-20:]:  # Show last 20 log entries
                # Highlight important messages
                if "ERROR" in log or "WARNING" in log:
                    st.markdown(f"<span style='color: red;'>{log}</span>", unsafe_allow_html=True)
                elif "BUTTON CLICKED" in log:
                    st.markdown(f"<span style='color: blue; font-weight: bold;'>{log}</span>", unsafe_allow_html=True)
                elif "Set" in log or "Switch" in log:
                    st.markdown(f"<span style='color: green;'>{log}</span>", unsafe_allow_html=True)
                else:
                    st.write(log)
                    
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add a button to clear just the debug log
            if st.button("Clear Debug Log", key="clear_debug"):
                st.session_state.debug_log = []
                st.rerun()

# Check if we need to rerun for UI updates
