


dlp-ai-tool-1  | 2025-06-03 21:46:57,727 - __main__ - ERROR - Documentation generation error: Expecting value: line 1 column 1 (char 0)


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
        1. A clear table of contents
        2. Executive summary
        3. Detailed main content with proper sections and subsections
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

import json
import logging
import time
from typing import Dict, Any, Tuple, Optional
from azure.storage.blob import BlobServiceClient
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AzureServiceConfiguration:
    """Class to handle Azure service configuration"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, str]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path) as config_file:
                config = json.load(config_file)
                self._validate_config(config)
                return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in configuration file: {self.config_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
            
    def _validate_config(self, config: Dict[str, str]) -> None:
        """Validate required configuration parameters"""
        required_params = [
            "openai_endpoint",
            "openai_key",
            "openai_model",
            "openai_api_version",
            "azure_doc_intelligence_endpoint",
            "azure_doc_intelligence_key",
            "azure_storage_account_name",
            "azure_storage_account_key",
            "azure_storage_container_name"
        ]
        
        missing_params = [param for param in required_params if param not in config]
        if missing_params:
            raise ValueError(f"Missing required configuration parameters: {', '.join(missing_params)}")

def retry_with_backoff(func: callable, max_retries: int = 5, initial_backoff: int = 1) -> Tuple[Any, bool]:
    """
    Execute a function with exponential backoff retry logic
    
    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial backoff time in seconds
    
    Returns:
        Tuple containing the function result and whether it was a retry success
    """
    retries = 0
    while retries < max_retries:
        try:
            result = func()
            return result, (retries > 0)
        except Exception as e:
            retries += 1
            if retries == max_retries:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
            wait_time = initial_backoff * (2 ** (retries - 1))
            logger.warning(f"Attempt {retries} failed. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

class AzureServicesManager:
    """Class to manage Azure service connections"""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.services: Dict[str, Any] = {}
        
    def initialize_blob_storage(self) -> BlobServiceClient:
        """Initialize Azure Blob Storage client"""
        try:
            connection_string = (
                f"DefaultEndpointsProtocol=https;"
                f"AccountName={self.config['azure_storage_account_name']};"
                f"AccountKey={self.config['azure_storage_account_key']};"
                f"EndpointSuffix=core.windows.net"
            )
            
            client = BlobServiceClient.from_connection_string(connection_string)
            
            # Ensure container exists
            container_name = self.config['azure_storage_container_name']
            try:
                client.create_container(container_name)
                logger.info(f"Created new container: {container_name}")
            except ResourceExistsError:
                logger.info(f"Container already exists: {container_name}")
            
            return client
            
        except Exception as e:
            logger.error(f"Failed to initialize Blob Storage: {str(e)}")
            raise
            
    def initialize_document_intelligence(self) -> DocumentAnalysisClient:
        """Initialize Azure Document Intelligence client"""
        try:
            return DocumentAnalysisClient(
                endpoint=self.config["azure_doc_intelligence_endpoint"],
                credential=AzureKeyCredential(self.config["azure_doc_intelligence_key"])
            )
        except Exception as e:
            logger.error(f"Failed to initialize Document Intelligence: {str(e)}")
            raise
            
    def initialize_openai(self) -> AzureOpenAI:
        """Initialize Azure OpenAI client"""
        try:
            return AzureOpenAI(
                api_key=self.config["openai_key"],
                api_version=self.config["openai_api_version"],
                azure_endpoint=self.config["openai_endpoint"]
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {str(e)}")
            raise
            
    def initialize_all_services(self) -> Tuple[Dict[str, Any], bool]:
        """
        Initialize all Azure services
        
        Returns:
            Tuple containing services dictionary and retry success status
        """
        retry_success = False
        
        try:
            # Initialize Blob Storage with retry
            def init_blob():
                client = self.initialize_blob_storage()
                self.services["blob_service_client"] = client
                self.services["container_name"] = self.config['azure_storage_container_name']
                return client
            
            _, blob_retry = retry_with_backoff(init_blob)
            retry_success = retry_success or blob_retry
            
            # Initialize Document Intelligence with retry
            def init_doc_intelligence():
                client = self.initialize_document_intelligence()
                self.services["document_analysis_client"] = client
                return client
            
            _, doc_retry = retry_with_backoff(init_doc_intelligence)
            retry_success = retry_success or doc_retry
            
            # Initialize OpenAI with retry
            def init_openai():
                client = self.initialize_openai()
                self.services["openai_client"] = client
                return client
            
            _, openai_retry = retry_with_backoff(init_openai)
            retry_success = retry_success or openai_retry
            
            return self.services, retry_success
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure services: {str(e)}")
            raise

def initialize_azure_connections() -> Tuple[Dict[str, Any], bool]:
    """
    Main function to initialize all Azure connections
    
    Returns:
        Tuple containing services dictionary and connection status
    """
    try:
        # Load and validate configuration
        config_manager = AzureServiceConfiguration()
        config = config_manager.load_config()
        
        # Initialize services
        services_manager = AzureServicesManager(config)
        services, retry_success = services_manager.initialize_all_services()
        
        logger.info("Successfully initialized all Azure services")
        if retry_success:
            logger.info("Some services required retry attempts but were successful")
            
        return services, True
        
    except Exception as e:
        logger.error(f"Failed to initialize Azure connections: {str(e)}")
        return None, False

def test_azure_connections(services: Dict[str, Any]) -> Dict[str, bool]:
    """
    Test all Azure service connections
    
    Args:
        services: Dictionary of Azure service clients
    
    Returns:
        Dictionary of service status
    """
    status = {}
    
    try:
        # Test Blob Storage
        blob_client = services.get("blob_service_client")
        if blob_client:
            container_name = services.get("container_name")
            container_client = blob_client.get_container_client(container_name)
            container_client.get_container_properties()
            status["blob_storage"] = True
        
        # Test Document Intelligence
        doc_client = services.get("document_analysis_client")
        if doc_client:
            # Simple test - just verify the client exists
            status["document_intelligence"] = doc_client is not None
        
        # Test OpenAI
        openai_client = services.get("openai_client")
        if openai_client:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            status["openai"] = True
            
    except Exception as e:
        logger.error(f"Error testing connections: {str(e)}")
        
    return status


def load_config() -> Dict[str, str]:
    """Wrapper function to load configuration"""
    config_manager = AzureServiceConfiguration()
    return config_manager.load_config()

import streamlit as st 
from PIL import Image
import logging
from typing import Optional, Dict, Any, List, Tuple
from Features.azure_setup import initialize_azure_connections, test_azure_connections, load_config
from Features.search_interface import create_search_interface
from Features.main_functions import (
    process_uploaded_files,
    get_storage_info,
    clear_storage,
    delete_specific_files,
    build_knowledge_base,
    generate_documentation,
    export_documentation,
    login,
    logout
)
import json
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_css(file_name: str) -> None:
    """Load custom CSS styles"""
    try:
        with open(f"static/{file_name}") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error loading CSS file: {str(e)}")

def initialize_session_state() -> None:
    """Initialize Streamlit session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'azure_initialized' not in st.session_state:
        st.session_state.azure_initialized = False
    if 'azure_services' not in st.session_state:
        st.session_state.azure_services = None
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    if 'generated_docs' not in st.session_state:
        st.session_state.generated_docs = {}

def display_header() -> None:
    """Display application header and title"""
    st.markdown('<div class="title-section">', unsafe_allow_html=True)
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    try:
        logo = Image.open("Trizetto_logo.png")
        st.image(logo, width=400)
    except FileNotFoundError:
        st.error("Logo file not found. Please check the file path.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="title">DLP AI Tool</h1>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="line"></div>', unsafe_allow_html=True)

def check_azure_services() -> bool:
    """Check and initialize Azure services if needed"""
    if not st.session_state.azure_initialized:
        with st.spinner("Initializing Azure services..."):
            services, connected = initialize_azure_connections()
            if connected:
                st.session_state.azure_services = services
                st.session_state.azure_initialized = True
                
                # Test connections
                status = test_azure_connections(services)
                if all(status.values()):
                    st.success("Successfully connected to all Azure services!")
                else:
                    for service, is_connected in status.items():
                        if not is_connected:
                            st.warning(f"Warning: {service} connection issues detected")
            else:
                st.error("Failed to initialize Azure services. Please check your configuration.")
                return False
    return True


def display_storage_management() -> None:
    """Display storage management interface in sidebar"""
    st.sidebar.markdown("### Storage Management")
    
    # Get storage info
    storage_info = get_storage_info(
        st.session_state.azure_services['blob_service_client'],
        st.session_state.azure_services['container_name']
    )
    
    # Display storage statistics
    st.sidebar.markdown(f"**Total Files:** {storage_info['total_files']}")
    st.sidebar.markdown(f"**Total Size:** {storage_info['total_size'] / 1024 / 1024:.2f} MB")
    
    # View files button
    if st.sidebar.button("View All Files"):
        st.sidebar.markdown("#### Files in Storage:")
        for file in storage_info['files']:
            col1, col2 = st.sidebar.columns([3, 1])
            col1.markdown(f"- {file['name']}")
            if col2.button("Delete", key=f"del_{file['name']}"):
                success, message = delete_specific_files(
                    st.session_state.azure_services['blob_service_client'],
                    st.session_state.azure_services['container_name'],
                    [file['name']]
                )
                if success:
                    st.sidebar.success(message)
                else:
                    st.sidebar.error(message)
                st.rerun()
    
    # Clear storage button with confirmation
    if st.sidebar.button("Clear All Storage"):
        if st.sidebar.button("Confirm Clear All"):
            success, message = clear_storage(
                st.session_state.azure_services['blob_service_client'],
                st.session_state.azure_services['container_name']
            )
            if success:
                st.sidebar.success(message)
                st.session_state.processed_files = []
            else:
                st.sidebar.error(message)
            st.rerun()

def display_file_management() -> None:
    """Handle file upload and management"""
    st.sidebar.title("Document Management")
    
    # File uploader
    uploaded_files = st.sidebar.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload documents to be processed for documentation generation"
    )
    
    if uploaded_files:
        with st.spinner("Processing uploaded files..."):
            processed_results = process_uploaded_files(
                uploaded_files,
                st.session_state.azure_services['blob_service_client'],
                st.session_state.azure_services['container_name'],
                st.session_state.azure_services['document_analysis_client']
            )
            
            for result in processed_results:
                if result['status'] == 'success':
                    if result['filename'] not in st.session_state.processed_files:
                        st.session_state.processed_files.append(result['filename'])
                    st.sidebar.success(f"Processed: {result['filename']}")
                else:
                    st.sidebar.error(f"Error processing {result['filename']}: {result['error']}")
    
    # Display processed files
    if st.session_state.processed_files:
        st.sidebar.markdown("### Recently Processed Documents")
        for file in st.session_state.processed_files:
            st.sidebar.markdown(f"- {file}")

def display_search_history() -> None:
    """Display search history in the sidebar"""
    if st.session_state.search_history:
        st.sidebar.markdown("### Recent Searches")
        for search in reversed(st.session_state.search_history[-5:]):  # Show last 5 searches
            st.sidebar.markdown(f"- {search['query']} ({search['timestamp']})")

def handle_documentation_generation(
    search_query: str,
    export_format: str,
    file_list: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    """Handle the documentation generation process"""
    try:
        with st.spinner("Generating documentation..."):
            # Load config
            config = load_config()
            
            # Build knowledge base from specified files
            knowledge_base = build_knowledge_base(
                file_list or st.session_state.processed_files,
                st.session_state.azure_services['blob_service_client'],
                st.session_state.azure_services['document_analysis_client'],
                st.session_state.azure_services['container_name']
            )
            
            # Generate documentation
            documentation = generate_documentation(
                st.session_state.azure_services['openai_client'],
                knowledge_base,
                search_query,
                config
            )
            
            if documentation:
                # Store in session state
                doc_id = f"doc_{int(time.time())}"
                st.session_state.generated_docs[doc_id] = documentation
                
                # Add to search history
                st.session_state.search_history.append({
                    "query": search_query,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "doc_id": doc_id
                })
                
                # Handle export
                output = export_documentation(documentation, export_format)
                st.download_button(
                    f"Download {export_format.upper()}",
                    output,
                    file_name=f"documentation_{doc_id}.{export_format}",
                    mime="application/octet-stream"
                )
                
                return documentation
            else:
                st.warning("No relevant documentation could be generated for your query.")
                return None
                
    except Exception as e:
        st.error(f"Error generating documentation: {str(e)}")
        logger.error(f"Documentation generation error: {str(e)}")
        return None

def display_documentation_preview(documentation: Dict[str, Any]) -> None:
    """Display a preview of the generated documentation"""
    with st.expander("Documentation Preview", expanded=True):
        # Title and metadata
        st.title(documentation["title"])
        # st.markdown("### Metadata")
        # for key, value in documentation["metadata"].items():
        #     st.markdown(f"**{key}:** {value}")
        
        # Table of Contents
        st.markdown("### Table of Contents")
        for index, item in enumerate(documentation["toc"], start=1):
            st.markdown(f"{index}. {item}")
        
        # Executive Summary
        st.markdown("### Executive Summary")
        st.markdown(documentation["executive_summary"])
        
        # Main Sections
        for section in documentation["sections"]:
            st.markdown(f"### {section['title']}")
            st.markdown(section["content"])
            
            # Subsections
            for subsection in section.get("subsections", []):
                st.markdown(f"#### {subsection['title']}")
                st.markdown(subsection["content"])
                
                if subsection.get("references"):
                    st.markdown("*References:*")
                    for ref in subsection["references"]:
                        st.markdown(f"- {ref}")
        
        # Recommendations
        if documentation.get("recommendations"):
            st.markdown("### Recommendations")
            for rec in documentation["recommendations"]:
                st.markdown(f"- {rec}")
        
         # References
        st.markdown("### References")
        for ref in documentation["references"]:
            st.markdown(f"- {ref}")

        st.divider()

        # Cognizant Signature
        st.markdown("""
        I am proud to be designated as a *US Healthcare Fundamentals Specialist*<br>
        through the <mark>Cognizant US Healthcare Essential Learning Program</mark>.
        """, unsafe_allow_html=True)

        st.markdown("""
        Get your designation today and <mark>#showwhatyouknow</mark>.
        """, unsafe_allow_html=True)

def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="DLP AI Tool",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load CSS and initialize session state
    load_css("styles.css")
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Handle login/logout
    if not st.session_state.logged_in:
        login()
    else:
        
        # Check Azure services
        if not check_azure_services():
            return
        
        # Display file management sidebar
        display_file_management()
        
        # Display storage management
        display_storage_management()

        # Add logout button to sidebar
        if st.sidebar.button("Logout"):
            logout()
            st.rerun()
        
        # Display search history
        display_search_history()
        
        # Create search interface
        search_query, export_format = create_search_interface(st)
        
        # Handle search and documentation generation
        if search_query and st.button("Generate Documentation"):
            if not st.session_state.processed_files:
                st.warning("Please upload some documents first.")
                return
                
            documentation = handle_documentation_generation(
                search_query,
                export_format,
                file_list=st.session_state.processed_files
            )
            
            if documentation:
                display_documentation_preview(documentation)
        
        # Help text
        st.markdown("---")
        st.markdown("""
            ### How to use this tool:
            1. Upload your documents using the sidebar
            2. View and manage stored files in the Storage Management section
            3. Enter a topic in the search box
            4. Choose your preferred export format
            5. Click "Generate Documentation"
            6. Download the generated documentation
        """)

if __name__ == "__main__":
    main()


