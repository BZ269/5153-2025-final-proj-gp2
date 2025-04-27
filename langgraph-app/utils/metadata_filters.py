from typing import Dict, Any, List, Union, Optional, Tuple
from weaviate.classes.query import Filter
import re
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
import json
import ast

verbose = True

class MetadataFilterBuilder:
    """Flexible builder for constructing Weaviate metadata filters."""

    ### ----------------------------------------------------------------------
    @staticmethod
    def build_filter(
        filter_dict: Dict[str, Any] = None,
        section_list: List[str] = None,
        client: Any = None  
        ) -> Optional[Filter]:
        """
        Build a Weaviate filter with type conversion.
        """
        # Define expected types for metadata fields
        field_types = {
            "year": str,
            "section": str,
            "hospital": str,
            "project_code": str,
            "title": str
        }

        combined_filter = None
        
        # Process section list filter
        if section_list and len(section_list) > 0:
            section_filter = None
            for section in section_list:
                f = Filter.by_property("section").equal(section)
                section_filter = f if section_filter is None else section_filter | f
            combined_filter = section_filter
                
        # Process other metadata filters
        if filter_dict and len(filter_dict) > 0:
            metadata_filter = None
            for field, value in filter_dict.items():
                # Skip if property type is not defined
                if field not in field_types:
                    continue
                    
                f = None
                
                # Convert value to expected type
                expected_type = field_types[field]
                try:
                    if isinstance(value, list):
                        converted_values = [str(v) if expected_type == str else expected_type(v) for v in value]
                        value_filter = None
                        for val in converted_values:
                            f_val = Filter.by_property(field).equal(val)
                            value_filter = f_val if value_filter is None else value_filter | f_val
                        f = value_filter
                    else:
                        converted_value = str(value) if expected_type == str else expected_type(value)
                        f = Filter.by_property(field).equal(converted_value)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Failed to convert {field} value {value} to {expected_type}: {e}")
                    continue
                
                # Combine with AND logic
                if f:
                    metadata_filter = f if metadata_filter is None else metadata_filter & f
            
            # Combine metadata filter with section filter
            if combined_filter is None:
                combined_filter = metadata_filter
            else:
                combined_filter = combined_filter & metadata_filter
                
        return combined_filter
    
    ### ----------------------------------------------------------------------
    @staticmethod
    def parse_metadata_from_query(
        query: str, 
        client: any,
        llm_classifier = None,
        llm = None,
        collection_name: str = "MedicalQIDocument_Poster_Chunks",
    ) -> Tuple[Dict[str, Any], List[str], str]:
        """
        Parse metadata filter conditions from a natural language query.
        
        Args:
            query: The user's query
            llm_classifier: Optional LLM classifier for section inference
            
        Returns:
            Tuple of (
                metadata_filters: Dict of metadata filters,
                section_list: List of sections to filter on,
                clean_query: Query with metadata filter commands removed
            )
        """
        ### Handle explicit format filters (backward compatibility)
        metadata_filters = {}
        
        ### Extract metadata using LLM if the query has potential filters
        metadata_filters = MetadataFilterBuilder._extract_metadata_from_query(query, client, llm, collection_name) 

            
        if verbose:
            ### Clean the query by removing metadata filter commands
            print(f"\n\n📌  Metadata filters extraced from query > > > {metadata_filters}") 
        

        clean_query = query
        for field, values in metadata_filters.items():
            if values:  # Only proceed if there are values to clean
                for value in values:
                    # Create patterns to match the value in different formats
                    patterns = [
                        f"\\b{re.escape(value)}\\b",  # standalone value
                        f"'{re.escape(value)}'",      # value in single quotes
                        f"\"{re.escape(value)}\""     # value in double quotes
                    ]
                    
                    for pattern in patterns:
                        clean_query = re.sub(pattern, "", clean_query, flags=re.IGNORECASE)

        ### Get sections using the provided classifier if available
        section_list = []
        if llm_classifier and clean_query.strip():
            section_list = llm_classifier(clean_query.strip())
        
        ### Clean up the query
        clean_query = clean_query.strip()
        
        return metadata_filters, section_list, clean_query 


    ###########################################################
    @staticmethod
    def get_unique_metadata_values(client, collection_name: str = "MedicalQIDocument_Poster_Chunks") -> Dict[str, List[str]]:
        """
        Retrieve unique values for all metadata fields from the specified Weaviate collection.
        
        Args:
            client: Weaviate client instance
            collection_name: Name of the Weaviate collection/index to query 
                             (default: "MedicalQIDocument_Poster_Chunks")
            
        Returns:
            Dictionary mapping metadata fields to their unique values
        """
        # List of metadata fields to aggregate over
        metadata_fields = ["year", "project_code", "hospital", "section"]
        
        # Initialize empty dictionary to store unique values
        unique_values_dict = {}
        
        # Loop through metadata fields and aggregate unique values
        for field in metadata_fields:
            response = client.collections.get(collection_name).aggregate.over_all(group_by=field)
            unique_values = [group.grouped_by.value for group in response.groups]
            unique_values_dict[field] = unique_values
            
        return unique_values_dict

    ###########################################################
    @staticmethod
    def _extract_metadata_from_query(query: str, client: Any, llm: ChatOllama, collection_name: str = "MedicalQIDocument_Poster_Chunks") -> Dict[str, Any]:
        """
        Extract metadata filters from query using LLM and unique metadata values from Weaviate.
        
        Args:
            query: The user's query
            client: Weaviate client instance to get unique metadata values
            llm: ChatOllama instance for metadata extraction
            
        Returns:
            Dictionary of extracted metadata filters
        """
        ### First get unique metadata values from Weaviate
        unique_metadata_values = MetadataFilterBuilder.get_unique_metadata_values(client, collection_name)
        
 
        ### ----------------------------------------------------------------
        ### Remove section, section will be handled in section_filter, not here.

        prompt_template = PromptTemplate.from_template("""
            <ROLE>
            You are a precise metadata extraction assistant for a medical QIP database. Your task is to parse the user's query and extract relevant metadata fields.
            </ROLE>

            <DATABASE METADATA SCHEMA>
            ONLY these fields exist in our database:
            - year: {years} (must be exact match)
            - hospital: {hospitals} (must be exact match)
            </DATABASE METADATA SCHEMA>

            <RULES>
            1. Extract ONLY year or hospital if explicitly mentioned in query
            2. Never invent fields not in the schema above
            3. Return empty JSON if no metadata matches
            4. Values must exactly match provided options
            5. Output must be valid JSON parsable by json.loads()
            </RULES>

            <INPUT QUERY>
            {query}
            </INPUT QUERY>

            <OUTPUT FORMAT>
            {{ 
            "year": [<array of matching years>],
            "hospital": [<array of matching hospitals>]
            }}
            </OUTPUT FORMAT>

            Extracted Metadata:
            """)


        ### =================================================================
        prompt = prompt_template.format(
            years=", ".join(unique_metadata_values.get("year", [])),
            sections=", ".join(unique_metadata_values.get("section", [])),
            hospitals=", ".join(unique_metadata_values.get("hospital", [])),
            query=query
        )

        ### ------------------------------------------------------------
        def extract_json_from_llm_response(content: str) -> dict:
            """
            Safely extract a JSON dictionary from an LLM string response.
            """
            try:
                # Try to extract JSON block using regex
                json_match = re.search(r'\{.*?\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    return json.loads(json_str)  # Use json.loads for safety
                else:
                    print("⚠️ No JSON object found in content.")
                    return {}
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON decoding error: {e}")
                return {}


        # Invoke LLM to extract metadata
        try:
            response = llm.invoke(prompt)
            content = response.content
            print(f'type(content) > > >  {type(content)}')
            print(f"\n📌 Metadata LLM extractor's response: {content}")
            content_metadata_dict = extract_json_from_llm_response(content)
            print(f'content_metadata_dict > > >  {content_metadata_dict}')
            print(f'type(content_metadata_dict) > > >  {type(content_metadata_dict)}')
            
            return content_metadata_dict

        except Exception as e:
            print(f"Error extracting metadata: {e}")
            return {}




