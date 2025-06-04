import streamlit as st
import logging
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from Features.azure_setup import initialize_azure_connections as init_azure, load_config
import io
import zipfile
from azure.storage.blob import BlobServiceClient
from azure.ai.formrecognizer import DocumentAnalysisClient
from openai import AzureOpenAI
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_COLOR_INDEX


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define user roles and credentials
USERS = {
    "Admin1": {"password": "Admin1", "role": "admin"},
    "User1": {"password": "User1", "role": "user"}
}

def login():
    """Handle user login"""
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username in USERS and password == USERS[username]["password"]:
            st.session_state.logged_in = True
            st.session_state.user_role = USERS[username]["role"]
            st.session_state.username = username
            st.session_state.page = 'home'
            st.success("Login successful!")
        else:
            st.error("Invalid username or password")

def logout():
    """Handle user logout"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.logged_in = False
    st.session_state.page = 'login'

def initialize_azure_connections() -> Tuple[Dict[str, Any], bool]:
    """Initialize connections to Azure services"""
    try:
        azure_services, retry_success = init_azure()
        azure_connected = True
        logger.info("All Azure services initialized successfully")
        
        if 'container_name' not in azure_services:
            config = load_config()
            azure_services['container_name'] = config['azure_storage_container_name']
        
        if retry_success:
            logger.info("Azure services connection successful after retry!")
        
        return azure_services, azure_connected
    except Exception as e:
        logger.error(f"Failed to initialize Azure services: {str(e)}")
        return None, False

def process_document(
    document_analysis_client: DocumentAnalysisClient,
    content: bytes,
    filename: str
) -> str:
    """
    Process a document using Azure's Document Analysis
    
    Args:
        document_analysis_client: Azure Document Analysis client
        content: Document content in bytes
        filename: Name of the file being processed
    
    Returns:
        Extracted text from the document
    """
    try:
        import os
        import subprocess
        import tempfile
        import time
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            file_extension = os.path.splitext(filename)[1].lower()
            
            if file_extension == '.docx':
                logger.info(f"Converting DOCX to PDF: {filename}")
                
                # Save the Word document to temporary file
                temp_docx = os.path.join(temp_dir, 'temp.docx')
                with open(temp_docx, 'wb') as f:
                    f.write(content)
                
                # Convert to PDF using unoconv
                temp_pdf = os.path.join(temp_dir, 'temp.pdf')
                
                # Start LibreOffice in headless mode
                subprocess.run(['soffice', '--headless', '--convert-to', 'pdf', '--outdir', temp_dir, temp_docx], 
                             check=True, capture_output=True)
                
                # Wait for the conversion to complete
                max_wait = 30  # Maximum wait time in seconds
                wait_time = 0
                while not os.path.exists(temp_pdf) and wait_time < max_wait:
                    time.sleep(1)
                    wait_time += 1
                
                if not os.path.exists(temp_pdf):
                    raise Exception("PDF conversion timed out")
                
                # Read the PDF content
                with open(temp_pdf, 'rb') as f:
                    content = f.read()
                
                logger.info(f"Successfully converted DOCX to PDF: {filename}")
            
            # Process with Document Intelligence
            with io.BytesIO(content) as doc_stream:
                poller = document_analysis_client.begin_analyze_document(
                    "prebuilt-document", doc_stream)
                result = poller.result()
                
                # Extract text with page numbers and metadata
                extracted_text = ""
                for page_num, page in enumerate(result.pages, 1):
                    extracted_text += f"\n----- Page {page_num} -----\n"
                    for line in page.lines:
                        extracted_text += line.content + "\n"
                    
                logger.info(f"Successfully processed document: {filename}")
                return extracted_text
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Error converting document {filename}: {e.stderr.decode()}")
        raise Exception(f"Error converting document: {e.stderr.decode()}")
    except Exception as e:
        logger.error(f"Error processing document {filename}: {str(e)}")
        raise



def process_uploaded_files(
    uploaded_files: List[io.BytesIO],
    blob_service_client: BlobServiceClient,
    container_name: str,
    document_analysis_client: DocumentAnalysisClient
) -> List[Dict[str, Any]]:
    """
    Process only the newly uploaded files
    
    Args:
        uploaded_files: List of uploaded file objects
        blob_service_client: Azure Blob Storage client
        container_name: Name of the blob container
        document_analysis_client: Azure Document Analysis client
    
    Returns:
        List of processed document results
    """
    processed_results = []
    
    try:
        container_client = blob_service_client.get_container_client(container_name)
        
        for file in uploaded_files:
            try:
                # Upload to blob storage
                blob_client = container_client.get_blob_client(file.name)
                blob_client.upload_blob(file.getvalue(), overwrite=True)
                
                # Process the uploaded file
                content = file.getvalue()
                extracted_text = process_document(
                    document_analysis_client,
                    content,
                    file.name
                )
                
                processed_results.append({
                    "filename": file.name,
                    "content": extracted_text,
                    "status": "success"
                })
                
                logger.info(f"Successfully processed: {file.name}")
                
            except Exception as e:
                logger.error(f"Error processing file {file.name}: {str(e)}")
                processed_results.append({
                    "filename": file.name,
                    "content": None,
                    "status": "error",
                    "error": str(e)
                })
                continue
        
        return processed_results
        
    except Exception as e:
        logger.error(f"Error in file processing: {str(e)}")
        raise

def get_storage_info(
    blob_service_client: BlobServiceClient,
    container_name: str
) -> Dict[str, Any]:
    """
    Get detailed information about files in blob storage
    
    Args:
        blob_service_client: Azure Blob Storage client
        container_name: Name of the blob container
    
    Returns:
        Dictionary containing storage information
    """
    try:
        container_client = blob_service_client.get_container_client(container_name)
        blobs = list(container_client.list_blobs())
        
        storage_info = {
            "total_files": len(blobs),
            "files": [],
            "total_size": 0
        }
        
        for blob in blobs:
            file_info = {
                "name": blob.name,
                "size": blob.size,
                "created": blob.creation_time,
                "last_modified": blob.last_modified,
                "content_type": blob.content_settings.content_type
            }
            storage_info["files"].append(file_info)
            storage_info["total_size"] += blob.size
            
        return storage_info
        
    except Exception as e:
        logger.error(f"Error getting storage info: {str(e)}")
        return {
            "total_files": 0,
            "files": [],
            "total_size": 0,
            "error": str(e)
        }

def clear_storage(
    blob_service_client: BlobServiceClient,
    container_name: str
) -> Tuple[bool, str]:
    """
    Clear all files from blob storage
    
    Args:
        blob_service_client: Azure Blob Storage client
        container_name: Name of the blob container
    
    Returns:
        Tuple of (success_status, message)
    """
    try:
        container_client = blob_service_client.get_container_client(container_name)
        blobs = container_client.list_blobs()
        
        deleted_count = 0
        for blob in blobs:
            container_client.delete_blob(blob.name)
            deleted_count += 1
            
        return True, f"Successfully deleted {deleted_count} files"
        
    except Exception as e:
        logger.error(f"Error clearing storage: {str(e)}")
        return False, f"Error clearing storage: {str(e)}"

def delete_specific_files(
    blob_service_client: BlobServiceClient,
    container_name: str,
    file_names: List[str]
) -> Tuple[bool, str]:
    """
    Delete specific files from blob storage
    
    Args:
        blob_service_client: Azure Blob Storage client
        container_name: Name of the blob container
        file_names: List of file names to delete
    
    Returns:
        Tuple of (success_status, message)
    """
    try:
        container_client = blob_service_client.get_container_client(container_name)
        
        deleted_count = 0
        failed_files = []
        
        for file_name in file_names:
            try:
                container_client.delete_blob(file_name)
                deleted_count += 1
            except Exception as e:
                failed_files.append(file_name)
                logger.error(f"Error deleting {file_name}: {str(e)}")
        
        if failed_files:
            return False, f"Deleted {deleted_count} files, but failed to delete: {', '.join(failed_files)}"
        return True, f"Successfully deleted {deleted_count} files"
        
    except Exception as e:
        logger.error(f"Error deleting files: {str(e)}")
        return False, f"Error deleting files: {str(e)}"

def build_knowledge_base(
    processed_files: List[str],
    blob_service_client: BlobServiceClient,
    document_analysis_client: DocumentAnalysisClient,
    container_name: str
) -> Dict[str, Any]:
    """
    Build a knowledge base from specified processed files
    
    Args:
        processed_files: List of filenames to include
        blob_service_client: Azure Blob Storage client
        document_analysis_client: Azure Document Analysis client
        container_name: Name of the blob container
    
    Returns:
        Dictionary containing processed documents and their metadata
    """
    try:
        knowledge_base = {
            "documents": [],
            "last_updated": datetime.now().isoformat(),
            "total_documents": 0
        }
        
        container_client = blob_service_client.get_container_client(container_name)
        
        for filename in processed_files:
            try:
                # Download document
                blob_client = container_client.get_blob_client(filename)
                content = blob_client.download_blob().readall()
                
                # Process document
                extracted_text = process_document(
                    document_analysis_client,
                    content,
                    filename
                )
                
                # Add to knowledge base
                knowledge_base["documents"].append({
                    "filename": filename,
                    "content": extracted_text,
                    "metadata": {
                        "processed_date": datetime.now().isoformat()
                    }
                })
                
                logger.info(f"Added {filename} to knowledge base")
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                continue
        
        knowledge_base["total_documents"] = len(knowledge_base["documents"])
        return knowledge_base
        
    except Exception as e:
        logger.error(f"Error building knowledge base: {str(e)}")
        raise

def generate_documentation(
    openai_client: 'AzureOpenAI',
    knowledge_base: Dict[str, Any],
    topic: str,
    config: Dict[str, str]
) -> Dict[str, Any]:
    """
    Generate documentation using GPT-4 based on the knowledge base and topic.

    Args:
        openai_client: Azure OpenAI client
        knowledge_base: Processed knowledge base
        topic: Search topic
        config: Configuration dictionary

    Returns:
        Generated documentation in structured format.
    """
    try:
        # Prepare the context for GPT-4
        relevant_content = []
        for doc in knowledge_base["documents"]:
            # Basic relevance check
            if topic.lower() in doc["content"].lower():
                relevant_content.append({
                    "filename": doc["filename"],
                    "content": doc["content"][:2000]  # Limit content to avoid token limits
                })

        if not relevant_content:
            logger.warning(f"No relevant content found for topic: {topic}")
            return None

        # Create a more focused prompt to ensure valid JSON response
        prompt = f"""You are a technical documentation expert. Create comprehensive documentation about: {topic}

Based on the following source documents, generate a JSON response with the exact structure shown below.

Source documents:
{json.dumps([{"filename": doc["filename"], "content_preview": doc["content"][:500]} for doc in relevant_content], indent=2)}

IMPORTANT: Respond ONLY with valid JSON in exactly this format:
{{
    "title": "Documentation for {topic}",
    "toc": [
        "1. Executive Summary",
        "2. Overview", 
        "3. Key Information",
        "4. Technical Details",
        "5. Best Practices",
        "6. References"
    ],
    "executive_summary": "Brief summary of the documentation content",
    "sections": [
        {{
            "title": "Overview",
            "content": "Main content for this section",
            "subsections": [
                {{
                    "title": "Key Points",
                    "content": "Subsection content"
                }}
            ]
        }}
    ],
    "references": ["Source document names"],
    "metadata": {{
        "generated_date": "{datetime.now().isoformat()}",
        "topic": "{topic}",
        "sources_used": {len(relevant_content)}
    }}
}}

Do not include any text before or after the JSON. Ensure all strings are properly escaped."""

        # Make the API call with better error handling
        try:
            logger.info("Making OpenAI API call...")
            response = openai_client.chat.completions.create(
                model=config.get("openai_model", "gpt-4"),
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a technical documentation expert. Always respond with valid JSON only. Never include explanatory text before or after the JSON."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=3000
            )

            logger.info("OpenAI API call successful")

        except Exception as api_error:
            logger.error(f"OpenAI API call failed: {str(api_error)}")
            raise Exception(f"OpenAI API error: {str(api_error)}")

        # Get the response content
        response_content = response.choices[0].message.content

        # Log the raw response for debugging
        logger.info(f"Raw OpenAI response length: {len(response_content) if response_content else 0}")
        logger.info(f"Raw OpenAI response preview: {response_content[:200] if response_content else 'None'}...")

        # Validate response content
        if not response_content:
            logger.error("Empty response from OpenAI")
            raise Exception("Empty response from OpenAI API")

        if response_content.strip() == "":
            logger.error("Whitespace-only response from OpenAI")
            raise Exception("Invalid response from OpenAI API")

        # Clean the response content
        cleaned_content = response_content.strip()

        # Try to extract JSON from the response
        json_pattern = r'\{.*\}'
        json_match = re.search(json_pattern, cleaned_content, re.DOTALL)

        if json_match:
            json_str = json_match.group()
            logger.info(f"Extracted JSON string length: {len(json_str)}")
        else:
            logger.warning("No JSON pattern found in response, using full content")
            json_str = cleaned_content

        # Parse the JSON with detailed error handling
        try:
            documentation = json.loads(json_str)
            logger.info("Successfully parsed JSON response")

        except json.JSONDecodeError as json_error:
            logger.error(f"JSON parsing failed: {str(json_error)}")
            logger.error(f"Problematic JSON: {json_str[:500]}...")

            # Create fallback documentation structure
            logger.info("Creating fallback documentation structure")
            documentation = {
                "title": f"Documentation for {topic}",
                "toc": [
                    "1. Executive Summary",
                    "2. Content Overview", 
                    "3. Source Information",
                    "4. References"
                ],
                "executive_summary": f"This documentation covers information about {topic} based on the provided source materials.",
                "sections": [
                    {
                        "title": "Content Overview",
                        "content": cleaned_content[:1000] + "..." if len(cleaned_content) > 1000 else cleaned_content,
                        "subsections": [
                            {
                                "title": "Source Information",
                                "content": f"Information extracted from {len(relevant_content)} source document(s)."
                            }
                        ]
                    }
                ],
                "references": [doc["filename"] for doc in relevant_content],
                "metadata": {
                    "generated_date": datetime.now().isoformat(),
                    "topic": topic,
                    "note": "Fallback structure due to JSON parsing error"
                }
            }

        # Validate and complete the documentation structure
        if not isinstance(documentation, dict):
            raise Exception("Documentation is not a dictionary")

        # Ensure required fields are present
        required_fields = ["title", "toc", "executive_summary", "sections", "references", "metadata"]
        for field in required_fields:
            if field not in documentation:
                logger.warning(f"Missing field '{field}', adding default")
                if field == "title":
                    documentation[field] = f"Documentation for {topic}"
                elif field == "toc":
                    documentation[field] = ["1. Overview", "2. Content Overview", "3. References"]
                elif field == "executive_summary":
                    documentation[field] = f"This document provides information about {topic}."
                elif field == "sections":
                    documentation[field] = []
                elif field == "references":
                    documentation[field] = [doc["filename"] for doc in relevant_content]
                elif field == "metadata":
                    documentation[field] = {
                        "generated_date": datetime.now().isoformat(),
                        "topic": topic,
                        "sources_used": len(relevant_content)
                    }

        logger.info("Documentation generation completed successfully")
        return documentation

    except Exception as e:
        logger.error(f"Error generating documentation: {str(e)}")

        # Create fallback documentation structure
        fallback_doc = {
            "title": f"Documentation for {topic}",
            "toc": ["1. Overview", "2. Error Information"],
            "executive_summary": f"An error occurred while generating documentation for {topic}.",
            "sections": [
                {
                    "title": "Error Information",
                    "content": f"An error occurred while generating documentation: {str(e)}",
                    "subsections": []
                }
            ],
            "references": [],
            "metadata": {
                "generated_date": datetime.now().isoformat(),
                "topic": topic
            }
        }
        logger.info("Returning emergency fallback documentation structure")
        return fallback_doc



def export_documentation(
    documentation: Dict[str, Any],
    format: str = "docx"
) -> bytes:
    """
    Export the documentation in the specified format
    
    Args:
        documentation: Generated documentation dictionary
        format: Export format ("docx" or "json")
    
    Returns:
        Formatted documentation as bytes
    """
    try:
        if format.lower() == "json":
            return json.dumps(documentation, indent=2).encode('utf-8')
        
        elif format.lower() == "docx":
            doc = Document()
            
            # Add title
            doc.add_heading(documentation["title"], 0)
            
            # Add metadata
            doc.add_paragraph(f"Generated: {documentation['metadata']['generated_date']}")
            doc.add_paragraph(f"Topic: {documentation['metadata']['topic']}")
            doc.add_paragraph(f"Sources Used: {documentation['metadata']['sources_used']}")
            
            # Add table of contents
            doc.add_heading("Table of Contents", 1)
            for item in documentation["toc"]:
                # Use standard paragraph style with indentation
                p = doc.add_paragraph(item)
                p.paragraph_format.left_indent = Inches(0.25)
            
            # Add executive summary
            doc.add_heading("Executive Summary", 1)
            doc.add_paragraph(documentation["executive_summary"])
            
            # Add sections and subsections
            for section in documentation["sections"]:
                doc.add_heading(section["title"], 1)
                doc.add_paragraph(section["content"])
                
                for subsection in section.get("subsections", []):
                    doc.add_heading(subsection["title"], 2)
                    doc.add_paragraph(subsection["content"])
            
            # Add references
            doc.add_heading("References", 1)
            for ref in documentation["references"]:
                p = doc.add_paragraph(ref)
                p.style = 'List Bullet'

            # Line break for signature
            line_length = 50  
            doc.add_paragraph('_' * line_length)
            # Add a paragraph with highlighting
            p1 = doc.add_paragraph()
            run1 = p1.add_run("I am proud to be designated a ")
            run2 = p1.add_run("US Healthcare Fundamentals Specialist")
            run2.font.highlight_color = WD_COLOR_INDEX.YELLOW  # Highlight this part
            run3 = p1.add_run("\nthrough the ")
            run4 = p1.add_run("Cognizant US Healthcare Essentials")
            run4.font.highlight_color = WD_COLOR_INDEX.YELLOW  # Highlight this part
            run5 = p1.add_run(" Learning Program")

            # Add another paragraph with highlighting
            p2 = doc.add_paragraph()
            run6 = p2.add_run("Get your designation today and ")
            run7 = p2.add_run("#showwhatyouknow")
            run7.font.highlight_color = WD_COLOR_INDEX.YELLOW  # Highlight this part
        
            # Save to bytes
            doc_bytes = io.BytesIO()
            doc.save(doc_bytes)
            doc_bytes.seek(0)
            
            return doc_bytes.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
    except Exception as e:
        logger.error(f"Error exporting documentation: {str(e)}")
        raise

def get_file_list(
    blob_service_client: BlobServiceClient,
    container_name: str
) -> List[str]:
    """Get list of files in blob storage"""
    try:
        container_client = blob_service_client.get_container_client(container_name)
        blobs = container_client.list_blobs()
        return [blob.name for blob in blobs]
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        return []
dlp-ai-tool-1  | 2025-06-04 13:32:41,926 - Features.main_functions - ERROR - Error exporting documentation: 'sources_used' 
dlp-ai-tool-1  | 2025-06-04 13:32:41,928 - __main__ - ERROR - Documentation generation error: 'sources_used'

        
        return fallback_doc
