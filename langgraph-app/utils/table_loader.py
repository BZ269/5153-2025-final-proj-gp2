#!/usr/bin/env python3
# Table-Based Medical QI PDF Extractor
# Uses table extraction as the primary method for identifying sections

from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
import pdfplumber
import re
from typing import List, Dict, Any, Optional, Tuple
import os
from collections import defaultdict

class TableBasedQIPDFLoader(BaseLoader):
    """
    A specialized LangChain loader for medical QI PDFs that uses table extraction
    as the primary method for identifying sections and content.
    
    This loader produces three optimized views:
    1. TABLE CONTENT: Raw tables extracted from the document
    2. DOCUMENT SECTIONS: Clean sections parsed from table content
    3. COLUMN CONTENT: Left/right column separation for spatial layout
    """
    
    def __init__(self, file_path: str):
        """Initialize with a file path."""
        self.file_path = file_path
        
        # Common section headers in medical QI documents
        self.qi_section_patterns = [
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
        ]
    
    def load(self) -> List[Document]:
        """
        Load PDF and return a list of Document objects with structured content.
        """
        documents = []
        
        with pdfplumber.open(self.file_path) as pdf:
            metadata = {
                "source": self.file_path,
                "file_path": self.file_path,
                "total_pages": len(pdf.pages)
            }
            
            # Add PDF metadata if available
            if hasattr(pdf, "metadata") and pdf.metadata:
                for k, v in pdf.metadata.items():
                    # Clean up keys that start with '/'
                    clean_key = k[1:] if isinstance(k, str) and k.startswith('/') else k
                    metadata[clean_key] = v
            
            # Process each page
            for i, page in enumerate(pdf.pages):
                # Page-specific metadata
                page_metadata = {
                    **metadata,
                    "page": i,
                    "page_number": i + 1,
                }
                
                # Extract tables first - this is our primary extraction method
                tables = page.extract_tables()
                tables_content, section_info = self._extract_tables_with_sections(tables)
                
                # Add section information to metadata
                if section_info:
                    page_metadata["sections_found"] = section_info
                
                # Create sections content directly from table content
                sections_content = self._extract_sections_from_tables(tables)
                
                # Extract columns as a separate view
                left_column, right_column = self._extract_columns(page)
                
                # Create structured content with clear separation between views
                structured_content = self._combine_content_views(
                    tables_content, 
                    sections_content,
                    left_column,
                    right_column
                )
                
                # Create document with the formatted text
                document = Document(
                    page_content=structured_content,
                    metadata=page_metadata
                )
                documents.append(document)
        
        return documents
    
    def _extract_tables_with_sections(self, tables) -> Tuple[str, List[str]]:
        """
        Extract tables and identify sections within them.
        Returns table content and a list of section headers found.
        """
        result_parts = []
        sections_found = []
        
        if tables:
            result_parts.append("# TABLE CONTENT")
            for table_idx, table in enumerate(tables):
                table_content = []
                for row in table:
                    # Clean row (remove None values)
                    clean_row = [str(cell).strip() if cell is not None else "" for cell in row]
                    # Skip empty rows
                    if any(cell for cell in clean_row):
                        row_text = " | ".join(clean_row)
                        table_content.append(row_text)
                        
                        # Look for section headers in this row
                        row_upper = row_text.upper()
                        for pattern in self.qi_section_patterns:
                            if pattern in row_upper:
                                if pattern not in sections_found:
                                    sections_found.append(pattern)
                
                if table_content:
                    result_parts.append(f"## TABLE {table_idx + 1}")
                    result_parts.append("\n".join(table_content))
        
        return "\n\n".join(result_parts) if result_parts else "", sections_found
    
    def _extract_sections_from_tables(self, tables) -> str:
        """
        Extract sections directly from tables, which preserves the
        correct order and association of content with headers.
        """
        if not tables:
            return ""
            
        result_parts = ["# DOCUMENT SECTIONS"]
        current_section = None
        section_content = {}
        
        # Process all tables to find sections and their content
        for table in tables:
            if not table:
                continue
                
            for row in table:
                if not row or all(cell is None for cell in row):
                    continue
                
                # Join row cells
                row_text = " ".join(str(cell).strip() if cell is not None else "" for cell in row)
                if not row_text.strip():
                    continue
                
                # Check if this row contains a section header
                is_section = False
                for pattern in self.qi_section_patterns:
                    if pattern in row_text.upper():
                        current_section = pattern
                        is_section = True
                        if current_section not in section_content:
                            section_content[current_section] = []
                        break
                
                # If not a section header and we have a current section, add to content
                if not is_section and current_section:
                    # Don't add the section name again to the content
                    if current_section not in row_text.upper():
                        # Filter out chart data patterns
                        if not self._is_chart_data(row_text):
                            section_content[current_section].append(row_text)
        
        # Format and add each section
        for section, content in section_content.items():
            if content:  # Only add sections that have content
                result_parts.append(f"\n## {section}")
                result_parts.append("\n".join(content))
        
        return "\n\n".join(result_parts)
    
    def _extract_columns(self, page) -> tuple:
        """Extract left and right columns content."""
        # Get page dimensions for column detection
        width = page.width
        midpoint = width / 2
        
        # Extract all words with their positions
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True
        )
        
        if not words:
            return "", ""
        
        # Separate words into left and right columns
        left_words = [w for w in words if w['x0'] < midpoint]
        right_words = [w for w in words if w['x0'] >= midpoint]
        
        # Group by lines
        left_lines = defaultdict(list)
        for word in left_words:
            y_key = round(word['top'] / 5) * 5
            left_lines[y_key].append(word)
        
        right_lines = defaultdict(list)
        for word in right_words:
            y_key = round(word['top'] / 5) * 5
            right_lines[y_key].append(word)
        
        # Format columns
        left_column = []
        for y, line_words in sorted(left_lines.items()):
            sorted_words = sorted(line_words, key=lambda w: w['x0'])
            line_text = " ".join(w['text'] for w in sorted_words)
            left_column.append(line_text)
        
        right_column = []
        for y, line_words in sorted(right_lines.items()):
            sorted_words = sorted(line_words, key=lambda w: w['x0'])
            line_text = " ".join(w['text'] for w in sorted_words)
            right_column.append(line_text)
        
        return "\n".join(left_column), "\n".join(right_column)
    
    def _is_chart_data(self, text: str) -> bool:
        """Determine if text line is likely chart data."""
        if not text.strip():
            return False
            
        # Count different character types
        digits = sum(1 for c in text if c.isdigit())
        symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
        total = len(text.strip())
        
        # If mostly numbers and symbols, likely chart data
        if (digits + symbols) / total > 0.5:
            return True
        
        # Check for common chart patterns
        chart_patterns = [
            r'\d+%',                             # Percentage values
            r'\d+\.\d+',                         # Decimal numbers
            r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec', # Month abbreviations
            r'Series \d',                        # Series labels
            r'^\s*\d+\s*$',                      # Just a number 
            r'\d+-\d+',                          # Ranges like 10-20
        ]
        
        if any(re.search(pattern, text) for pattern in chart_patterns) and total < 30:
            # Only consider it chart data if the line is relatively short
            return True
            
        return False
    
    def _combine_content_views(self, tables_content, sections_content, left_column, right_column) -> str:
        """Combine the different content views with clear separation."""
        result_parts = []
        
        # Add table content if available
        if tables_content:
            result_parts.append(tables_content)
        
        # Add sections content if available
        if sections_content:
            result_parts.append(sections_content)
        
        # Add column content if available
        if left_column or right_column:
            column_parts = ["# COLUMN CONTENT"]
            
            if left_column:
                column_parts.append("## LEFT COLUMN")
                column_parts.append(left_column)
            
            if right_column:
                column_parts.append("## RIGHT COLUMN")
                column_parts.append(right_column)
            
            result_parts.append("\n\n".join(column_parts))
        
        # Join all parts with clear separation
        return "\n\n\n".join(result_parts)

def inspect_medical_qi_pdf(pdf_path):
    """
    Load and inspect a medical QI PDF using the table-based loader.
    """
    print(f"Examining medical QI PDF: {pdf_path}")
    
    # Load the PDF with our specialized loader
    loader = TableBasedQIPDFLoader(pdf_path)
    documents = loader.load()
    
    # Display basic metadata
    print(f"\n{'='*50}")
    print(f"DOCUMENT SUMMARY")
    print(f"{'='*50}")
    print(f"Total pages extracted: {len(documents)}")
    if documents:
        for key, value in documents[0].metadata.items():
            if key != 'page_content':
                print(f"{key}: {value}")
    
    # Process each page document
    for i, doc in enumerate(documents):
        print(f"\n{'='*50}")
        print(f"PAGE {i + 1}")
        print(f"{'='*50}")
        
        # Print metadata
        print(f"METADATA:")
        print(f"{'-'*50}")
        for key, value in doc.metadata.items():
            if key != 'page_content':
                print(f"{key}: {value}")
        
        # Print content
        print(f"\nEXTRACTED CONTENT:")
        print(f"{'-'*50}")
        print(doc.page_content)
    
    print(f"\n{'='*50}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*50}")
    print(f"Total pages processed: {len(documents)}")
    print(f"Extraction method: Table-Based Medical QI Document Extractor")
    print(f"This extractor provides three optimized views:")
    print(f"1. TABLE CONTENT - Raw tables extracted from the document")
    print(f"2. DOCUMENT SECTIONS - Clean sections parsed directly from table content")
    print(f"3. COLUMN CONTENT - Left/right column separation for spatial layout")
    print(f"The structured output is optimized for RAG systems with reduced duplication")

if __name__ == "__main__":
    # Find PDFs in current directory
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in the current directory!")
    else:
        # Process the first PDF found
        pdf_path = pdf_files[0]
        inspect_medical_qi_pdf(pdf_path)