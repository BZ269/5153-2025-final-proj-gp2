import os
import re
from typing import List, Dict, Any
from datetime import datetime
from langchain_core.documents import Document
from langchain.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

from table_loader import TableBasedQIPDFLoader

SECTION_HEADERS = [
    'HOSPITAL',  # Singapore hospital codes
    'YEAR',      # Year QIP was conducted
    'BACKGROUND',
    'INTRODUCTION', 
    'MISSION STATEMENT',
    'ANALYSIS OF PROBLEM',  
    'ROOT CAUSE ANALYSIS',
    'METHODOLOGY', 
    'INTERVENTIONS / INITIATIVES',
    'RESULTS',
    'OUTCOME', 
    'DISCUSSION', 
    'SUSTAINABILITY AND SPREAD',
    'CONCLUSION'
    # UNKNOWN --> add everything else
]

# hospital_codes = [
#     'SGH',   # Singapore General Hospital
#     'CGH',   # Changi General Hospital 
#     'KTPH',  # Khoo Teck Puat Hospital
#     'NUH',   # National University Hospital
#     'TTSH',  # Tan Tock Seng Hospital
#     'SKCH',  # Sengkang Community Hospital
#     'SKH',   # Sengkang Hospital
#     'AH',    # Alexandra Hospital
#     'NTFGH', # Ng Teng Fong General Hospital
#     'KKH',   # KK Women's and Children's Hospital
#     'IMH',   # Institute of Mental Health
#     'NHCS',  # National Heart Centre Singapore
#     'NCCS',  # National Cancer Centre Singapore
#     'SHHQ',  # SingHealth HQ
#     'OCH',   # Outram Community Hospital
# ]




### ---------------------------------------------------------------------------
### METADATA EXTRACTION FUNCTION ###
def extract_metadata(filename: str, full_text: str, pdf_path: str = None) -> Dict[str, Any]:
    """Extract metadata from filename and content"""
    metadata = {
        "source": filename,
        "processing_timestamp": datetime.now().isoformat()
    }
    
    try:
        # Extract from filename
        parts = filename.replace(".pdf", "").split(" - ")
        
        # Extract project code
        raw_project_code = parts[0].strip()
        code_match = re.match(r"^SHM_[A-Z]{2,3}\d{3}", raw_project_code, re.I)
        metadata["project_code"] = code_match.group(0).upper() if code_match else raw_project_code
        
        ### Extract hospital code
        underscore_split = filename.replace(".pdf", "").split("_")
        hospital_code = re.match(r"^(\w+)", underscore_split[2].strip()).group(1).upper() if len(underscore_split) >= 3 else "Unknown"
        metadata["hospital"] = hospital_code
        
        ### Extract title
        if len(parts) > 2:
            filename_title = parts[2].strip().replace("_", " ").title()
            metadata["title"] = re.sub(r"^Shm\s+\w+\d*\s+\w+\s+-\s+", "", filename_title, flags=re.I).strip(" -")
        else:
            metadata["title"] = "Unknown Title"
        
        # Try to extract year from PDF metadata if path is provided
        if pdf_path:
            print(f"pdf_path: {pdf_path}") 
            try:
                pdf_loader = PyPDFLoader(pdf_path)
                pdf_docs = pdf_loader.load()
                if pdf_docs and "moddate" in pdf_docs[0].metadata:
                    # Extract year from moddate (format: D:YYYYMMDD...)
                    mod_date = pdf_docs[0].metadata["moddate"]
                    metadata["year"] = mod_date[0:4]  # Extract YYYY part from moddate
                    return metadata
            except Exception as e:
                print(f"Warning: Unable to extract year from PDF metadata: {str(e)}")
                
            
    except Exception as e:
        print(f"Warning: Error extracting metadata: {str(e)}")
        metadata.update({
            "project_code": "UNKNOWN",
            "hospital": "UNKNOWN",
            "title": filename.replace(".pdf", "").replace("_", " ").title(),
            "year": "UNKNOWN"
        })
    
    return metadata







########################################################################
### STEP 1: COLUMN TEXT EXTRACTION ONLY ###
def extract_column_text(pdf_path: str) -> str:
    loader = TableBasedQIPDFLoader(pdf_path)
    docs = loader.load()
    
    column_text = ""
    for doc in docs:
        if "# COLUMN CONTENT" in doc.page_content:
            column_text += doc.page_content.split("# COLUMN CONTENT", 1)[-1]
    
    return column_text.strip()

### STEP 2: LLAMA3 STRUCTURED OUTPUT ###
def get_structured_qi_output(column_text: str, chat_model: ChatOllama) -> str:
    template = PromptTemplate.from_template("""
You are a medical QI document formatter. Your job is to reformat the raw PDF content into structured sections.

Please organize the content into the following sections in this exact order:

{headers}

For each section:

- YOU MUST use markdown headers with double hash signs, like this: "## SECTION NAME"
- Always use uppercase for section names, e.g., "## BACKGROUND" not "## Background"
- Never use any other format like **bold** for section names
- Place all relevant content under its matching section.
- If content fits multiple sections, you may duplicate it — do not summarize by referencing other sections.
- Never say things like "(as mentioned above)" or "(see Introduction)".
- Each section must contain full, standalone content, even if repeated in another section.

Here is the raw PDF content:
----------------------------
{content}
""")

    chain: RunnableSequence = (
        template
        | chat_model
        | StrOutputParser()
    )
    
    # Filter out HOSPITAL and YEAR from headers since we handle those separately
    filtered_headers = [h for h in SECTION_HEADERS if h not in ['HOSPITAL', 'YEAR']]
    
    return chain.invoke({
        "content": column_text,
        "headers": "\n".join(filtered_headers)
    })

#########################################################################################
### Define function to inject metadata into content of chunk (before embedding into vector store)
def create_chunk_with_metadata(content: str, metadata: Dict[str, Any]) -> Document:
    """
    Create a document chunk with metadata injected into the content.
    
    Args:
        content: The content of the chunk.
        metadata: The metadata to include.
        
    Returns:
        Document: The chunk with metadata injected into the content.
    """
    # Format metadata into a string
    metadata_str = "\n".join([f"{key}: {value}" for key, value in metadata.items()])
    
    # Combine metadata and content
    combined_content = f"Metadata:\n{metadata_str}\n\nContent:\n{content}"
    
    # Create a Document object
    return Document(page_content=combined_content, metadata=metadata)





### def create_poster_chunk(structured_output: str, pdf_filename: str) -> Document: 
def create_poster_chunk(structured_output: str, pdf_filename: str, pdf_path: str) -> Document: 
    """Create a single document chunk for the entire poster with metadata"""
    
    # Extract metadata using the new function
    # metadata = extract_metadata(pdf_filename, structured_output, os.path.join(".", pdf_filename))
    metadata = extract_metadata(pdf_filename, structured_output, pdf_path)
    
    # Create a single document with all content
    doc = Document(
        page_content=structured_output,
        metadata=metadata
        )
    
    ### inject metadata into content of chunk
    doc = create_chunk_with_metadata(structured_output, metadata)
    
    
    return doc


### FULL POSTER CHUNKER
def process_posters_as_whole_chunks(directory=os.path.join("..", "data"), chat_model: ChatOllama = None) -> List[Document]:
    
    if chat_model is None:
        chat_model = ChatOllama(model="llama3.1") # if no chat model is provided, default to llama3.1
        
    all_posters = []
    
    # pdf_files = [f for f in os.listdir(directory) if f.endswith(".pdf")]
    ### Get full paths to PDF files in data directory
    pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pdf")]


    # for pdf in pdf_files:
    for pdf_path in pdf_files:
        
        pdf_filename = os.path.basename(pdf_path) 
        
        print(f"\n📄 Processing: {pdf_filename}")
        
        # Step 1: Extract raw text
        raw_text = extract_column_text(os.path.join(directory, pdf_filename))
        
        # Step 2: Get structured format from LLM
        print("🧠 Sending to ChatOllama...")
        structured_output = get_structured_qi_output(raw_text, chat_model)
        
        # Create the poster chunk
        poster_chunk = create_poster_chunk(structured_output, pdf_filename, pdf_path)
        all_posters.append(poster_chunk)
        print(f"✅ Created poster chunk from {pdf_filename}")

    return all_posters


### Example usage:
if __name__ == "__main__":
    # Initialize ChatOllama model
    chat_model = ChatOllama(model="llama3.1")
    
    full_poster_chunks = process_posters_as_whole_chunks(chat_model=chat_model)
    print(f"\n📦 Total poster chunks: {len(full_poster_chunks)}")
    
    # Print a sample of the first poster
    if full_poster_chunks:
        sample = full_poster_chunks[0]
        print(f"\n--- Sample Poster Chunk ---")
        print(f"Project Code: {sample.metadata['project_code']}")
        print(f"Source: {sample.metadata['source']}")
        print(f"Title: {sample.metadata['title']}")
        print(f"Hospital: {sample.metadata['hospital']}")
        print(f"Year: {sample.metadata['year']}")
        print(f"Content Preview:\n{sample.page_content[:300]}...")

# Uploading to Weaviate

from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_ollama import OllamaEmbeddings
import weaviate
import sys

def upload_to_weaviate(chunks, index_name, client, llm=None):
    """
    Upload chunks to Weaviate with validation and error handling
    
    Args:
        chunks: List of document chunks to upload
        index_name: Name of the Weaviate collection
        llm: OllamaEmbeddings instance (if None, will create a default one)
        port: Weaviate server port (default: 8081)
    """


    ### Check for empty or invalid chunks
    if not chunks:
        print("❌ No chunks provided. Please check your document processing.")
        return None
    
    #### Validate chunks and print diagnostics
    print(f"🔍 Analyzing {len(chunks)} chunks...")
    
    ### Check for attributes
    valid_chunks = []
    invalid_chunks = []
    
    for i, chunk in enumerate(chunks):
        if not hasattr(chunk, 'page_content') or not chunk.page_content.strip():
            invalid_chunks.append((i, chunk))
        else:
            valid_chunks.append(chunk)
    
    # Print invalid chunk information
    if invalid_chunks:
        print(f"⚠️ Found {len(invalid_chunks)} invalid chunks:")
        for idx, chunk in invalid_chunks:
            print(f"  - Chunk #{idx}:")
            print(f"    Type: {type(chunk)}")
            print(f"    Content: {repr(chunk)}")
            print(f"    Attributes: {dir(chunk)}")
            print()
    
    # Check if we have any valid chunks to proceed
    if not valid_chunks:
        print("❌ No valid chunks to upload. Please check your document processing.")
        return None
    
    print(f"✓ Found {len(valid_chunks)} valid chunks to upload")
    
    # Step 3: Set up embeddings if not provided
    if llm is None:
        print("📦 Setting up default embedding model...")
        try:
            llm = OllamaEmbeddings(model="qwen-2.5-custom-4096:latest")
            
            # Quick test to ensure embeddings work
            test_embed = llm.embed_query("Test embedding")
            print(f"✓ Embedding test successful (vector dimension: {len(test_embed)})")
        except Exception as e:
            print(f"❌ Error with embedding model: {e}")
            print("   Please check if Ollama is running and the model is available")
            return None

    # Step 4: Clear existing index if it exists
    try:
        client.collections.delete(index_name)
        print(f"🗑️ Deleted existing '{index_name}' collection")
    except Exception:
        # Collection might not exist, which is fine
        pass

    # Step 5: Upload valid chunks to Weaviate
    print(f"📤 Uploading {len(valid_chunks)} chunks to Weaviate collection '{index_name}'...")
    
    try:
        vectorstore = WeaviateVectorStore.from_documents(
            documents=valid_chunks,
            embedding=llm,
            client=client,
            index_name=index_name,
            text_key="content"
        )
        print(f"✅ Successfully uploaded {len(valid_chunks)} chunks to Weaviate!")
        return vectorstore
    except Exception as e:
        print(f"❌ Error during upload: {e}")
        print(f"   Traceback: {sys.exc_info()[2]}")
        return None

#####################################################################
### Example usage
# Example usage
if __name__ == "__main__":
    # This line assumes final_chunks is defined elsewhere
    # Replace with your actual chunks or pass them as parameters
    if 'final_chunks' and 'full_poster_chunks' in globals():   
        #upload_to_weaviate(final_chunks, "MedicalQIDocument_Poster_Chunks")     
        upload_to_weaviate(full_poster_chunks, "MedicalQIDocument_Poster_Full")
    else:
        print("❌ 'final_chunks' variable not found. Please define your chunks first.")