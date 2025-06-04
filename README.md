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
        
        # **FIX: Add proper validation and error handling for JSON parsing**
        response_content = response.choices[0].message.content
        
        # Check if response is empty or None
        if not response_content or response_content.strip() == "":
            logger.error("Empty response from OpenAI API")
            raise Exception("Empty response from OpenAI API")
        
        # Log the response for debugging
        logger.info(f"OpenAI Response: {response_content[:200]}...")  # Log first 200 chars
        
        try:
            # Attempt to parse JSON
            documentation = json.loads(response_content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.error(f"Response content: {response_content}")
            
            # Try to extract JSON from response if it's wrapped in other text
            import re
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                try:
                    documentation = json.loads(json_match.group())
                    logger.info("Successfully extracted JSON from response")
                except json.JSONDecodeError:
                    logger.error("Even extracted JSON is invalid")
                    raise Exception("Invalid JSON response from OpenAI")
            else:
                # Create a fallback response structure
                logger.warning("Creating fallback documentation structure")
                documentation = {
                    "title": f"Documentation for {topic}",
                    "toc": ["Overview", "Content"],
                    "executive_summary": response_content[:500] + "...",
                    "sections": [
                        {
                            "title": "Generated Content",
                            "content": response_content,
                            "subsections": []
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
        
        # Add missing metadata if not present
        if "metadata" not in documentation:
            documentation["metadata"] = {}
        
        documentation["metadata"]["sources_used"] = len(relevant_content)
        documentation["metadata"]["generated_date"] = datetime.now().isoformat()
        documentation["metadata"]["topic"] = topic
        
        return documentation
        
    except Exception as e:
        logger.error(f"Error generating documentation: {str(e)}")
        raise

