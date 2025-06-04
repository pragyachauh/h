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
        import re
        
        # Look for JSON pattern
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
                                "title": "Source Materials",
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
        
        # Ensure required fields exist
        required_fields = ["title", "toc", "executive_summary", "sections", "references", "metadata"]
        for field in required_fields:
            if field not in documentation:
                logger.warning(f"Missing field '{field}', adding default")
                if field == "title":
                    documentation[field] = f"Documentation for {topic}"
                elif field == "toc":
                    documentation[field] = ["1. Overview", "2. Content", "3. References"]
                elif field == "executive_summary":
                    documentation[field] = f"This document provides information about {topic}."
                elif field == "sections":
                    documentation[field] = []
                elif field == "references":
                    documentation[field] = [doc["filename"] for doc in relevant_content]
                elif field == "metadata":
                    documentation[field] = {}
        
        # Update metadata
        if "metadata" not in documentation:
            documentation["metadata"] = {}
        
        documentation["metadata"].update({
            "sources_used": len(relevant_content),
            "generated_date": datetime.now().isoformat(),
            "topic": topic
        })
        
        logger.info("Documentation generation completed successfully")
        return documentation
        
    except Exception as e:
        logger.error(f"Error generating documentation: {str(e)}")
        
        # Create emergency fallback documentation
        fallback_doc = {
            "title": f"Documentation for {topic}",
            "toc": ["1. Overview", "2. Error Information"],
            "executive_summary": f"An error occurred while generating documentation for {topic}.",
            "sections": [
                {
                    "title": "Error Information",
                    "content": f"Documentation generation failed with error: {str(e)}",
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
        
        return fallback_doc
