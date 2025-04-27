import re
import weaviate
from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from weaviate.classes.query import Filter

# Import config for models
from config import MODELS, MODEL_TEMPERATURES

# Regular expression patterns to extract project codes
PROJECT_CODE_PATTERNS = [
    r"PROJECT TITLE: SHM[_ ]?([A-Z]{2}\d{3})",  # Standard format
    r"PROJECT TITLE:.*?SHM[_ ]?([A-Z]{2}\d{3})",  # With text between PROJECT TITLE: and SHM
    r"PROJECT TITLE:.*?([A-Z]{2}\d{3})"  # Just looking for the code pattern
]

def extract_project_codes(text: str) -> List[str]:
    """
    Extract project codes from response text using the defined patterns.
    
    Args:
        text: The text containing project codes to extract
        
    Returns:
        List of extracted project codes
    """
    found_codes = []
    
    # First try with the most specific patterns
    for pattern in PROJECT_CODE_PATTERNS:
        matches = re.finditer(pattern, text)
        for match in matches:
            code = match.group(1)  # Extract the project code
            if code and code not in found_codes:
                found_codes.append(code)
    
    # If no codes found with standard patterns, try to find any possible codes
    # with a more generic pattern for SHM projects
    if not found_codes:
        # Look for codes in format like SHM_XX000 or SHM XX000 anywhere in text
        generic_patterns = [
            r"SHM[_ ]?([A-Z]{2}\d{3})",  # SHM followed by 2 letters and 3 digits
            r"([A-Z]{2}\d{3})"           # Any 2 letters followed by 3 digits (fallback)
        ]
        
        for pattern in generic_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                code = match.group(1)
                if code and code not in found_codes:
                    found_codes.append(code)
                    
    print(f"Extracted project codes: {found_codes}")
    return found_codes

def retrieve_full_poster_data(project_codes: List[str]) -> str:
    """
    Retrieve full poster data from Weaviate for the given project codes.
    
    Args:
        project_codes: List of project codes to retrieve data for
        
    Returns:
        String containing the combined full poster data
    """
    if not project_codes:
        return "No project codes found to retrieve data for."
    
    # Connect to Weaviate
    client = weaviate.connect_to_local(port=8081)
    
    # Build filter to get documents matching any of the project codes
    project_filter = None
    for code in project_codes:
        # Try both "project_code" and "code" as possible field names
        code_filter = Filter.by_property("project_code").equal(code)
        # Use only one filter for simplicity
        project_filter = code_filter if project_filter is None else project_filter | code_filter
    
    # Execute the query to retrieve full poster data
    try:
        # Try both collection names
        try:
            result = (
                client.collections
                .get("MedicalQIDocument_Poster_Full")
                .query
                .fetch_objects(
                    filters=project_filter,
                    limit=20
                )
            )
        except Exception as e:
            # Fallback to Chunks collection if Full doesn't exist
            result = (
                client.collections
                .get("MedicalQIDocument_Poster_Chunks")
                .query
                .fetch_objects(
                    filters=project_filter,
                    limit=20
                )
            )
        
        # Combine the results into a single text for context
        if not result.objects:
            return "No poster data found for the provided project codes."
        
        all_poster_data = []
        for obj in result.objects:
            props = obj.properties
            poster_data = f"PROJECT CODE: {props.get('project_code', props.get('code', 'Unknown'))}\n"
            
            # Try different possible field names for title and content
            title = props.get('title', props.get('name', 'Unknown'))
            content = props.get('content', props.get('text', ''))
            
            poster_data += f"PROJECT TITLE: {title}\n"
            poster_data += f"CONTENT: {content}\n\n"
            
            # Also include any section information if available
            if 'section' in props:
                poster_data += f"SECTION: {props.get('section')}\n\n"
                
            all_poster_data.append(poster_data)
        
        return "\n".join(all_poster_data)
    
    except Exception as e:
        print(f"Error retrieving poster data: {str(e)}")
        return f"Error retrieving poster data: {str(e)}"

def answer_poster_question(question: str, poster_response: str) -> str:
    """
    Answer follow-up questions about posters using LLM.
    
    Args:
        question: The user's follow-up question
        poster_response: The original response containing project information
        
    Returns:
        Generated answer to the follow-up question
    """
    # Extract project codes from the original response
    project_codes = extract_project_codes(poster_response)
    
    if not project_codes:
        return "I couldn't identify any specific projects to provide more information about. Could you please ask about a specific project or topic?"
    
    # Retrieve full poster data for the identified projects
    full_poster_data = retrieve_full_poster_data(project_codes)
    
    # Fallback to using original response if Weaviate retrieval failed
    if "Error retrieving poster data" in full_poster_data:
        print("Falling back to original response as context due to retrieval error")
        full_poster_data = poster_response
    
    # Create a prompt for the LLM
    prompt = PromptTemplate.from_template("""
You are a medical QI (Quality Improvement) document assistant helping with follow-up questions about medical posters.

Use the following context from medical QI posters to answer the user's specific question. 
Ground your answer in the provided content and be specific.

If the answer is not found in the context, say "I don't have enough information to answer that question about these posters."

PROJECT CODES: {project_codes}

--------------------
Context:
{context}

--------------------
Question: {question}
Answer:
""")

    # Initialize the LLM
    model = ChatOllama(model=MODELS["poster_qa"], temperature=MODEL_TEMPERATURES["poster_qa"])
    qa_chain = prompt | model | StrOutputParser()

    # Generate the answer
    try:
        return qa_chain.invoke({
            "context": full_poster_data, 
            "question": question,
            "project_codes": ", ".join(project_codes)
        })
    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        # Ultra fallback - just respond with a generic message
        return f"I encountered an issue while processing your question about the posters. The projects I found were: {', '.join(project_codes)}. Could you try asking a more specific question about one of these projects?" 