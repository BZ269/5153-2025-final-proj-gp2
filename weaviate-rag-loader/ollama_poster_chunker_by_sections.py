import os
import re
from typing import List, Dict, Any
from datetime import datetime
from langchain_core.documents import Document
from langchain.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader 
from langchain_core.runnables import RunnableSequence

from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_ollama import OllamaEmbeddings
import weaviate
import sys

from table_loader import TableBasedQIPDFLoader  # assume your class is saved here

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

### STEP 1: COLUMN TEXT EXTRACTION ONLY ###
def extract_column_text(pdf_path: str) -> str:
    loader = TableBasedQIPDFLoader(pdf_path)
    docs = loader.load()
    
    column_text = ""
    for doc in docs:
        if "# COLUMN CONTENT" in doc.page_content:
            column_text += doc.page_content.split("# COLUMN CONTENT", 1)[-1]
    
    return column_text.strip()


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
        
        ### Extract project code
        raw_project_code = parts[0].strip()
        code_match = re.match(r"^SHM_[A-Z]{2,3}\d{3}", raw_project_code, re.I)
        metadata["project_code"] = code_match.group(0).upper() if code_match else raw_project_code
        
        ### Extract hospital code
        underscore_split = filename.replace(".pdf", "").split("_")
        hospital_code = re.match(r"^(\w+)", underscore_split[2].strip()).group(1).upper() if len(underscore_split) >= 3 else "Unknown"
        metadata["hospital"] = hospital_code
        
        ### Extract title
        if len(parts) > 1:
            filename_title = parts[1].strip().replace("_", " ").title()
            metadata["title"] = re.sub(r"^Shm\s+\w+\d*\s+\w+\s+-\s+", "", filename_title, flags=re.I).strip(" -")
        else:
            metadata["title"] = "Unknown Title"
        
        # Try to extract year from PDF metadata if path is provided
        if pdf_path:
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


##################################################################
### STEP 2: LLAMA3 STRUCTURED OUTPUT ###
# def get_structured_qi_output(column_text: str) -> str: # to deprecate
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




#########################################################################################
### STEP 3: CHUNK BY HEADERS + METADATA ###
# def split_into_chunks(structured_output: str, pdf_filename: str) -> List[Document]:

#     """
#     This function splits the structured output into chunks based on the headers.
#     """

#     docs = []
#     current_section = None
#     buffer = []

#     ### Extract hospital, year, and project code for metadata (to deprecate)
#     hospital = extract_hospital_from_title(pdf_filename)
#     year = get_project_year()
#     project_code = extract_project_code(pdf_filename)
    

#     ### Add HOSPITAL and YEAR chunks first # DO NOT DELETE THIS, JUST
#     docs.append(Document(
#         page_content=hospital,
#         metadata={
#             "project_code": project_code,
#             "source": pdf_filename,
#             "section": "HOSPITAL",
#             "hospital": hospital,
#             "year": year
#         }
#     ))
    
#     docs.append(Document(
#         page_content=year,
#         metadata={
#             "project_code": project_code,
#             "source": pdf_filename,
#             "section": "YEAR",
#             "hospital": hospital,
#             "year": year
#         }
#     ))

#     lines = structured_output.splitlines()
#     for line in lines:
#         line_strip = line.strip()
#         if line_strip.startswith("## "):
#             # Save previous section
#             if current_section and buffer:
#                 docs.append(Document(
#                     page_content="\n".join(buffer).strip(),
#                     metadata={
#                         "project_code": project_code,
#                         "source": pdf_filename,
#                         "section": current_section,
#                         "hospital": hospital,
#                         "year": year
#                     }
#                 ))
#                 buffer = []

#             current_section = line_strip.replace("## ", "").strip().upper()
#         else:
#             buffer.append(line)
    
#     # Last section
#     if current_section and buffer:
#         docs.append(Document(
#             page_content="\n".join(buffer).strip(),
#             metadata={
#                 "project_code": project_code,
#                 "source": pdf_filename,
#                 "section": current_section,
#                 "hospital": hospital,
#                 "year": year
#             }
#         ))

#     return docs

### -------------------------------------------------------------------------

### CHUNK BY HEADERS + METADATA ###
def split_into_chunks(structured_output: str, pdf_filename: str, 
                      full_text: str, pdf_path: str = None,
                      inject_metadata_flag: bool = True
                      ) -> List[Document]:
    docs = []
    current_section = None
    buffer = []
    
    ### Extract metadata first
    base_metadata = extract_metadata(pdf_filename, full_text, pdf_path)

    lines = structured_output.splitlines()
    for line in lines:
        line_strip = line.strip()
        if line_strip.startswith("## "):
            # Save previous section
            if current_section and buffer:
                # Create metadata dictionary with base metadata plus section
                metadata = {**base_metadata, "section": current_section}
                
                if inject_metadata_flag: ### Inject metadata into content of chunk
                    docs.append(create_chunk_with_metadata("\n".join(buffer).strip(), metadata))
                else: ### Don't inject metadata into content of chunk
                    docs.append(Document(
                        page_content="\n".join(buffer).strip(),
                        metadata=metadata
                    ))
                buffer = []

            current_section = line_strip.replace("## ", "").strip().upper()
        else:
            buffer.append(line)
        
    ### Handle final chunk creation with optional metadata injection.
    if current_section and buffer:
        metadata = {**base_metadata, "section": current_section}
        if inject_metadata_flag: ### Inject metadata into content of chunk
            docs.append(create_chunk_with_metadata("\n".join(buffer).strip(), metadata))
        
        else: ### Don't inject metadata into content of chunk
            docs.append(Document(
                page_content="\n".join(buffer).strip(),
                metadata=metadata
            ))
        
    return docs



##############################################################################
### FULL RUNNER ###
### TO DEPRECATE
# def process_all_pdfs(directory=".") -> List[Document]: 
#     all_docs = []
#     pdf_files = [f for f in os.listdir(directory) if f.endswith(".pdf")]

#     for pdf in pdf_files:
#         print(f"\n📄 Processing: {pdf}")
        
#         # Extract metadata
#         hospital = extract_hospital_from_title(pdf)
#         year = get_project_year()
#         project_code = extract_project_code(pdf)
        
#         # Create TITLE OF PROJECT chunk from filename
#         title_chunk = Document(
#             page_content=pdf.replace(".pdf", "").replace("_", " ").strip(),
#             metadata={
#                 "project_code": project_code,
#                 "source": pdf,
#                 "section": "TITLE OF PROJECT",
#                 "hospital": hospital,
#                 "year": year
#             }
#         )
#         all_docs.append(title_chunk)

#         # Extract rest of the structured sections
#         raw_text = extract_column_text(pdf)
#         print("🧠 Sending to ChatOllama...")
#         structured = get_structured_qi_output(raw_text)
#         print("✂️ Splitting into chunks...")
#         chunks = split_into_chunks(structured, pdf)
#         all_docs.extend(chunks)
#         print(f"✅ {1 + len(chunks)} chunks created from {pdf}")

#     return all_docs

# def process_all_pdfs(directory="data") -> List[Document]:
def process_all_pdfs(directory=os.path.join(".", "data"), chat_model: ChatOllama = None) -> List[Document]:

    """
    This function processes all PDF files in the given directory and returns a list of Document objects.
    It uses the ChatOllama model to get the structured output and then splits it into chunks.
    Uses get_structured_qi_output() to get the structured output.
    Uses split_into_chunks() to split the structured output into chunks.
    """
    if chat_model is None:
        chat_model = ChatOllama(model="qwen-2.5-custom-4096:latest") # if no chat model is provided, default to llama3.1
        
    all_docs = []
    
    # Get full paths to PDF files in data directory
    pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pdf")]
    
    for pdf_path in pdf_files:
        ### Get just the filename for metadata
        pdf_filename = os.path.basename(pdf_path)
        print(f"\n📄 Processing: {pdf_filename}")
        
        ### Extract raw text first since we need it for metadata
        raw_text = extract_column_text(pdf_path)
        
        ### Extract metadata and create TITLE OF PROJECT chunk
        base_metadata = extract_metadata(pdf_filename, raw_text, pdf_path)

        # title_chunk = Document(
        #     page_content=pdf_filename.replace(".pdf", "").replace("_", " ").strip(),
        #     metadata={**base_metadata, "section": "TITLE OF PROJECT"}
        # )

        title_chunk = create_chunk_with_metadata(
            pdf_filename.replace(".pdf", "").replace("_", " ").strip(),
            {**base_metadata, "section": "TITLE OF PROJECT"}
            )

        all_docs.append(title_chunk)

        # Process the content
        print("🧠 Sending to ChatOllama...")
        # structured = get_structured_qi_output(raw_text) # to deprecate
        structured_output = get_structured_qi_output(raw_text, chat_model)

        print("✂️ Splitting into chunks...")
        # chunks = split_into_chunks(structured, pdf_filename, raw_text) # to deprecate
        chunks = split_into_chunks(structured_output, pdf_filename, raw_text, pdf_path, inject_metadata_flag=True)
        all_docs.extend(chunks)
        print(f"✅ {1 + len(chunks)} chunks created from {pdf_filename}")

    return all_docs

#####################################################################
# Example usage:
if __name__ == "__main__":
    final_chunks = process_all_pdfs()
    print(f"\n📦 Total chunks: {len(final_chunks)}")
    for i, doc in enumerate(final_chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Project Code: {doc.metadata['project_code']}")
        print(f"Section: {doc.metadata['section']}")
        print(f"Source: {doc.metadata['source']}")
        print(f"Hospital: {doc.metadata['hospital']}")
        print(f"Year: {doc.metadata['year']}")
        # print(doc.page_content[:300] + "...")
        print(doc.page_content)  


#####################################################################
### Uploading to Weaviate

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

### Example usage
# Example usage
if __name__ == "__main__":
    # This line assumes final_chunks is defined elsewhere
    # Replace with your actual chunks or pass them as parameters
    if 'final_chunks' and 'full_poster_chunks' in globals():
        upload_to_weaviate(final_chunks, "MedicalQIDocument_Poster_Chunks")
        #upload_to_weaviate(full_poster_chunks, "MedicalQIDocument_Poster_Full")
    else:
        print("❌ 'final_chunks' variable not found. Please define your chunks first.")