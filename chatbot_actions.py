# actions/chatbot_actions.py
"""
Burr actions for the chatbot application.
Each action represents a step in the conversation flow.
"""

from typing import Dict, Tuple
from datetime import datetime
from burr.core import State, action, ApplicationContext

# Import your LLM client
from llm_client import call_llm_simple, LLMResponse


@action(reads=["chat_history"], writes=["current_message", "chat_history"])
def process_user_message(state: State, user_message: str, __context: ApplicationContext) -> Tuple[Dict, State]:
    """
    Process incoming user message and add to chat history.
    This is the entry point for each user interaction.
    
    Args:
        state: Current conversation state containing chat history
        user_message: The message sent by the user
        __context: Burr context containing app metadata (conversation_id, user_id)
        
    Returns:
        Tuple of (result dict, updated state)
        - result: Contains the user message and conversation ID for tracking
        - state: Updated with current_message and appended chat_history
    """
    # Create chat item following OpenAI format
    chat_item = {
        "role": "user",
        "content": user_message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Log the processing
    print(f"Processing user message for conversation {__context.app_id}")
    
    # Return result (for tracking) and updated state
    return {
        "user_message": user_message,
        "conversation_id": __context.app_id
    }, state.update(
        current_message=user_message  # Store current message for next actions
    ).append(
        chat_history=chat_item  # Append to chat history list
    )


@action(reads=["current_message"], writes=["is_safe"])
def safety_check(state: State) -> Tuple[Dict, State]:
    """
    Check if the user's message is safe to process.
    In production, this would use a content moderation API.
    
    Args:
        state: Current state with user's message
        
    Returns:
        Tuple of (safety result, updated state)
        - result: Contains safety check outcome
        - state: Updated with is_safe boolean
    """
    # Get current message from state
    current_message = state["current_message"]
    
    # Simple safety check (in production, use proper content moderation)
    unsafe_keywords = ["hack", "exploit", "illegal", "harmful"]
    is_safe = not any(keyword in current_message.lower() for keyword in unsafe_keywords)
    
    print(f"Safety check result: {'SAFE' if is_safe else 'UNSAFE'}")
    
    return {
        "is_safe": is_safe,
        "message": current_message
    }, state.update(is_safe=is_safe)


@action(reads=["chat_history", "current_message"], writes=["ai_response", "chat_history"])
def generate_ai_response(state: State, __context: ApplicationContext) -> Tuple[Dict, State]:
    """
    Generate AI response using the LLM client with full conversation context.
    
    This action:
    1. Retrieves conversation history from state
    2. Formats it as context for the LLM
    3. Calls the LLM with appropriate prompts
    4. Updates state with the response
    
    Args:
        state: Current state with chat history and current message
        __context: Burr context for tracking metadata
        
    Returns:
        Tuple of (response data, updated state)
        - result: Contains AI response and tracking IDs
        - state: Updated with ai_response and appended chat_history
    """
    # Get conversation context
    chat_history = state.get("chat_history", [])
    current_message = state["current_message"]
    
    # Build context from chat history (last 10 messages for context window)
    recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
    
    # Format conversation history for system prompt
    conversation_context = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}" 
        for msg in recent_history
    ])
    
    # Create system prompt with conversation context
    system_prompt = f"""You are a helpful AI assistant. 
    
Previous conversation:
{conversation_context}

Please provide a helpful, accurate, and contextual response to the user's latest message.
Remember the conversation history and maintain consistency."""
    
    # Call LLM using your client
    llm_response: LLMResponse = call_llm_simple(
        system_message=system_prompt,
        user_message=current_message,
        temperature=0.7,
        max_tokens=1000,
        prompt_name="chat_response",
        user=__context.partition_key,  # User ID
        conversation_id=__context.app_id,  # Conversation ID
        node_name="generate_ai_response"
    )
    
    # Create AI message for chat history
    ai_message = {
        "role": "assistant",
        "content": llm_response.content,
        "timestamp": datetime.utcnow().isoformat(),
        "message_id": llm_response.message_id,
        "trace_id": llm_response.trace_id
    }
    
    print(f"Generated AI response with trace_id: {llm_response.trace_id}")
    
    return {
        "ai_response": llm_response.content,
        "message_id": llm_response.message_id,
        "trace_id": llm_response.trace_id
    }, state.update(
        ai_response=llm_response.content
    ).append(
        chat_history=ai_message
    )


@action(reads=[], writes=["ai_response", "chat_history"])
def unsafe_response(state: State) -> Tuple[Dict, State]:
    """
    Generate a safe response when unsafe content is detected.
    
    This action is triggered when the safety check fails.
    It provides a standard response without calling the LLM.
    
    Args:
        state: Current state
        
    Returns:
        Tuple of (response data, updated state)
        - result: Contains the safety response
        - state: Updated with ai_response and appended chat_history
    """
    # Standard unsafe content response
    response = "I'm sorry, but I cannot process that request as it may contain inappropriate content. Please rephrase your message."
    
    # Create message for history
    ai_message = {
        "role": "assistant",
        "content": response,
        "timestamp": datetime.utcnow().isoformat(),
        "is_safety_response": True
    }
    
    return {
        "ai_response": response,
        "is_safety_response": True
    }, state.update(
        ai_response=response
    ).append(
        chat_history=ai_message
    )