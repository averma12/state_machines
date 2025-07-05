"""
Centralized sync LLM client for making all LLM calls through Portkey
All LLM interactions should go through this module
"""

import os
import json
import re
import requests
from typing import Dict, List, Any, Optional, Union
from uuid import uuid4
from datetime import datetime
from pathlib import Path
from loguru import logger
from config import settings

# Configure loguru logger
logger.add("logs/llm_client.log", rotation="10 MB", retention="7 days", level="INFO")

class LLMResponse:
    """Standardized response from LLM calls"""
    def __init__(self, content: str, message_id: str = None, trace_id: str = None, raw_response: Dict[str, Any] = None):
        self.content = content
        self.message_id = message_id or f"msg_{uuid4()}"
        self.trace_id = trace_id or str(uuid4())
        self.raw_response = raw_response or {}

class LLMClient:
    """Client for interacting with LLM providers through Portkey."""
    
    def __init__(self):
        # Load settings from environment variables or config
        self.portkey_api_key = settings.PORTKEY_API_KEY
        self.portkey_config_id = settings.PORTKEY_CONFIG_ID
        self.default_model = settings.DEFAULT_LLM_MODEL
        self.environment = settings.ENVIRONMENT
        self.api_base_url = "https://api.portkey.ai/v1/chat/completions"
        
        # Fallback settings
        self.enable_fallback_model = os.getenv("ENABLE_FALLBACK_MODEL", "false").lower() == "true"
        self.fallback_model = os.getenv("FALLBACK_MODEL", "gpt-3.5-turbo")
        self.fallback_error_message = os.getenv("FALLBACK_ERROR_MESSAGE", "Sorry, I'm having trouble processing your request.")
        
        logger.info("LLM client initialized with Portkey integration")
    
    def generate_portkey_headers(
        self, 
        prompt_name: Optional[str], 
        user: str, 
        gen_id: Optional[str] = None,
        stream: bool = False
    ) -> Dict[str, str]:
        """
        Generate headers for Portkey API requests
        
        Args:
            prompt_name: Name of the prompt for tracking
            user: User identifier
            gen_id: Generation ID for tracing
            stream: Whether to stream the response
            
        Returns:
            Dictionary of headers for Portkey API
        """
        if gen_id is None:
            gen_id = str(uuid4())
            
        logger.debug(f"Using trace ID: {gen_id}")
            
        # Base headers for Portkey
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream' if stream else 'application/json',
            "x-portkey-api-key": self.portkey_api_key,
            "x-portkey-trace-id": gen_id,
            "x-portkey-config": self.portkey_config_id
        }
        
        # Add metadata if prompt name is provided
        if prompt_name:
            headers["x-portkey-metadata"] = json.dumps({
                "_environment": self.environment,
                "_prompt": prompt_name,
                "_user": user
            })
        else:
            headers["x-portkey-metadata"] = json.dumps({
                "_environment": self.environment,
                "_user": user
            })
        logger.debug(f"Portkey headers: {headers}")
            
        return headers
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        prompt_name: Optional[str] = None,
        user: Optional[str] = "user",
        gen_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Call LLM chat completion API through Portkey
        
        Args:
            messages: List of message dictionaries with role and content
            model: LLM model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            prompt_name: Name of prompt for tracking
            user: User identifier
            gen_id: Generation ID for tracing
            
        Returns:
            Response text from LLM
        """
        model = model or self.default_model
        
        # Generate headers
        headers = self.generate_portkey_headers(
            prompt_name=prompt_name,
            user=user,
            gen_id=gen_id
        )
        
        # Prepare request parameters
        params = {
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        
        # Add max_tokens if provided
        if max_tokens:
            params["max_tokens"] = max_tokens
            
        logger.debug(f"Calling LLM with model: {model}, prompt: {prompt_name}")
        
        # Make sync request
        try:
            response = requests.post(
                url=self.api_base_url,
                headers=headers,
                json=params,
                timeout=180.0
            )
            
            # Raise for status
            response.raise_for_status()
            
            # Parse response
            response_json = response.json()
            response_text = response_json["choices"][0]["message"]["content"]
            
            logger.debug(f"Received response from LLM")
            
            return response_text
            
        except requests.exceptions.HTTPError as e:
            logger.exception(f"HTTP error from Portkey API: {e.response.status_code} - {e.response.text}")
            # Try a fallback if configured
            if self.enable_fallback_model and model != self.fallback_model:
                logger.info(f"Attempting fallback to {self.fallback_model}")
                params["model"] = self.fallback_model
                fallback_response = requests.post(
                    url=self.api_base_url,
                    headers=headers,
                    json=params,
                    timeout=180.0
                )
                fallback_response.raise_for_status()
                fallback_json = fallback_response.json()
                return fallback_json["choices"][0]["message"]["content"]
            raise
        except Exception as e:
            logger.error(f"Error calling Portkey API: {str(e)}")
            raise

# Create module-level instance
llm_client = LLMClient()

def call_llm_simple(
    system_message: str,
    user_message: str,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = 4096,
    prompt_name: Optional[str] = None,
    user: Optional[str] = "user",
    conversation_id: Optional[str] = None,
    node_name: Optional[str] = None,
    **kwargs
) -> LLMResponse:
    """
    Simple LLM call: takes a system prompt and a user message (both strings), 
    builds the message structure, and calls the LLM.

    Args:
        system_message: The system prompt string
        user_message: The user message string
        model: LLM model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        prompt_name: Name of prompt for tracking
        user: User identifier
        conversation_id: Optional conversation ID for logging
        node_name: Optional node name for logging

    Returns:
        LLMResponse object with content and metadata
    """
    logger.info(f"Node name: {node_name}")
    logger.info(f"Conversation ID: {conversation_id}")
    logger.info(f"Prompt name: {prompt_name}")
    
    # Generate trace ID
    trace_id = str(uuid4())
    
    # Generate message ID
    message_id = f"ai_{uuid4()}"
    
    # Prepare messages
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    logger.info(f"Calling LLM with model: {model}, prompt: {prompt_name}")
    logger.info(f"API Key: {llm_client.portkey_api_key}")
    logger.info(f"Config ID: {llm_client.portkey_config_id}")
    
    try:
        # Make sync LLM call
        response_text = llm_client.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_name=prompt_name or node_name,
            user=user,
            gen_id=trace_id,
            **kwargs
        )
        
        return LLMResponse(
            content=response_text,
            message_id=message_id,
            trace_id=trace_id
        )
        
    except Exception as e:
        logger.exception(f"Error in LLM call: {str(e)}")
        
        # If there's a fallback mechanism configured, you can handle it here
        if llm_client.fallback_error_message:
            return LLMResponse(
                content=llm_client.fallback_error_message,
                message_id=message_id,
                trace_id=trace_id
            )
        raise

def parse_llm_response(llm_response: str) -> Dict[str, Any]:
    """
    Parse JSON from an LLM response with multiple fallback options.

    This function attempts to extract valid JSON from an LLM response that may
    contain additional text or formatting issues. The parsing strategy is as follows:

    1. Attempt to parse the entire response as JSON.
    2. If that fails, use a regex to extract a JSON-like substring.
    3. If no valid JSON is found, return a generic error dictionary that can be
       used in any context, ensuring a consistent response format.

    Returns:
        A dictionary containing either the parsed JSON data or a generic error structure.
    """
    try:
        # First attempt: directly parse the entire response as JSON.
        return json.loads(llm_response)
    except json.JSONDecodeError as direct_error:
        # If direct parsing fails, try extracting a JSON substring using regex.
        json_match = re.search(r'(\{.*\})', llm_response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError as regex_error:
                # If the extracted substring is still not valid JSON, return a generic error.
                return {
                    "status": "error",
                    "message": "Extracted content is not valid JSON. Retry  again",
                    "raw_response": llm_response,
                    "error": str(regex_error)
                }
        else:
            # Fallback: No JSON-like structure found in the response.
            return {
                "status": "error",
                "message": "No JSON content found in the response. Retry again",
                "raw_response": llm_response,
                "error": str(direct_error)
            }
        
def parse_json_array(llm_response: str) -> List[Any]:
    """
    Parse JSON array from LLM response.
    Returns empty list if parsing fails.
    """
    try:
        # Try direct parsing
        result = json.loads(llm_response)
        if isinstance(result, list):
            return result
        else:
            return []  # Not an array
    except json.JSONDecodeError:
        # Try to extract array with regex
        array_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
        if array_match:
            try:
                return json.loads(array_match.group(0))
            except:
                return []
        return []