Error generating documentation: Expecting value: line 1 column 1 (char 0)

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
    openai_client: AzureOpenAI,
    knowledge_base: Dict[str, Any],
    topic: str,
    config: Dict[str, str]
) -> Dict[str, Any]:
    """
    Generate documentation using GPT-4 based on the knowledge base and topic
    
    Args:
        openai_client: Azure OpenAI client
        knowledge_base: Processed knowledge base
        topic: Search topic
        config: Configuration dictionary
    
    Returns:
        Generated documentation in structured format
    """
    try:
        # Prepare the context for GPT-4
        relevant_content = []
        for doc in knowledge_base["documents"]:
            # Basic relevance check
            if topic.lower() in doc["content"].lower():
                relevant_content.append({
                    "filename": doc["filename"],
                    "content": doc["content"]
                })
        
        if not relevant_content:
            logger.warning(f"No relevant content found for topic: {topic}")
            return None
            
        prompt = f"""
        Based on the following documents, create comprehensive documentation about: {topic}
        
        Source documents:
        {json.dumps(relevant_content, indent=2)}
        
        Generate detailed technical documentation that includes:
        1. A clear numbered table of contents with a single tab after the number
        2. Executive summary
        3. Detailed main content with proper sections and subsections numbered after respective table of contents allocation
        4. References to source documents
        5. Any relevant examples, code snippets, or technical specifications
        6. Best practices and recommendations if applicable
        
        Format the response as a JSON with this structure:
        {{
            "title": "Documentation title",
            "toc": ["list of sections and subsections"],
            "executive_summary": "summary text",
            "sections": [
                {{
                    "title": "section title",
                    "content": "section content",
                    "subsections": [
                        {{
                            "title": "subsection title",
                            "content": "subsection content"
                        }}
                    ]
                }}
            ],
            "references": ["list of document references"],
            "metadata": {{
                "generated_date": "current date",
                "topic": "search topic",
                "sources_used": "number of sources"
            }}
        }}
        """
        
        response = openai_client.chat.completions.create(
            model=config["openai_model"],
            messages=[
                {"role": "system", "content": "You are a technical documentation expert skilled in creating comprehensive and well-structured documentation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        documentation = json.loads(response.choices[0].message.content)
        documentation["metadata"]["sources_used"] = len(relevant_content)
        documentation["metadata"]["generated_date"] = datetime.now().isoformat()
        
        return documentation
        
    except Exception as e:
        logger.error(f"Error generating documentation: {str(e)}")
        raise

# Continuing from the export_documentation function:

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

