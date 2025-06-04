import re  # Add this import at the top of your file

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
                    "sources_used": len(relevant_content),
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
                    documentation[field] = {}

        # Ensure metadata has all required fields
        if "metadata" not in documentation:
            documentation["metadata"] = {}
        
        # **FIX: Ensure all required metadata fields are present**
        documentation["metadata"].update({
            "generated_date": documentation["metadata"].get("generated_date", datetime.now().isoformat()),
            "topic": documentation["metadata"].get("topic", topic),
            "sources_used": documentation["metadata"].get("sources_used", len(relevant_content))
        })

        logger.info("Documentation generation completed successfully")
        return documentation

    except Exception as e:
        logger.error(f"Error generating documentation: {str(e)}")

        # Create emergency fallback documentation structure
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
                "topic": topic,
                "sources_used": 0,
                "error": str(e)
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
            title = documentation.get("title", "Documentation")
            doc.add_heading(title, 0)
            
            # **FIX: Safe access to metadata with defaults**
            metadata = documentation.get("metadata", {})
            generated_date = metadata.get("generated_date", "Unknown")
            topic = metadata.get("topic", "Unknown")
            sources_used = metadata.get("sources_used", 0)
            
            # Add metadata
            doc.add_paragraph(f"Generated: {generated_date}")
            doc.add_paragraph(f"Topic: {topic}")
            doc.add_paragraph(f"Sources Used: {sources_used}")
            
            # Add table of contents
            toc = documentation.get("toc", [])
            if toc:
                doc.add_heading("Table of Contents", 1)
                for item in toc:
                    # Use standard paragraph style with indentation
                    p = doc.add_paragraph(str(item))
                    p.paragraph_format.left_indent = Inches(0.25)
            
            # Add executive summary
            executive_summary = documentation.get("executive_summary", "No executive summary available.")
            doc.add_heading("Executive Summary", 1)
            doc.add_paragraph(executive_summary)
            
            # Add sections and subsections
            sections = documentation.get("sections", [])
            for section in sections:
                if isinstance(section, dict):
                    section_title = section.get("title", "Untitled Section")
                    section_content = section.get("content", "No content available.")
                    
                    doc.add_heading(section_title, 1)
                    doc.add_paragraph(section_content)
                    
                    # Add subsections
                    subsections = section.get("subsections", [])
                    for subsection in subsections:
                        if isinstance(subsection, dict):
                            subsection_title = subsection.get("title", "Untitled Subsection")
                            subsection_content = subsection.get("content", "No content available.")
                            
                            doc.add_heading(subsection_title, 2)
                            doc.add_paragraph(subsection_content)
            
            # Add references
            references = documentation.get("references", [])
            if references:
                doc.add_heading("References", 1)
                for ref in references:
                    p = doc.add_paragraph(str(ref))
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
        logger.error(f"Documentation structure: {documentation}")
        raise
