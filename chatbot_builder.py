# graph/chatbot_builder.py
"""
Burr application builder for the chatbot.
Defines the state machine graph and application configuration.
"""

import os
import uuid
from typing import Tuple, Optional, Any
from burr.core import ApplicationBuilder, when
from burr.tracking import LocalTrackingClient

# Import custom components
from mongodb_persister import MongoDBPersister
from chatbot_actions import (
    process_user_message,
    safety_check,
    generate_ai_response,
    unsafe_response
)


def create_chatbot_application(
    user_id: str,
    conversation_id: Optional[str] = None,
    mongodb_uri: Optional[str] = None
) -> Tuple[Any, str, MongoDBPersister]:
    """
    Create a Burr chatbot application with MongoDB persistence and tracking.
    
    This function sets up the complete state machine for the chatbot, including:
    - Actions (nodes in the graph)
    - Transitions (edges with conditions)
    - State persistence (MongoDB)
    - Tracking (local UI)
    - Initial state
    
    Args:
        user_id: Unique user identifier (used as partition_key)
        conversation_id: Optional conversation ID (generates new if not provided)
        mongodb_uri: MongoDB connection string (uses env var if not provided)
        
    Returns:
        Tuple of (Burr application, conversation_id, persister)
    """
    # Generate conversation ID if not provided
    if conversation_id is None:
        conversation_id = str(uuid.uuid4())
        print(f"Creating new conversation: {conversation_id}")
    else:
        print(f"Resuming conversation: {conversation_id}")
    
    # Get MongoDB URI from environment or parameter
    mongo_uri = mongodb_uri or os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MongoDB URI not provided")
    
    # Create MongoDB persister
    persister = MongoDBPersister(
        connection_string=mongo_uri,
        database="chatbot_db",
        collection="conversation_states"
    )
    
    # Create local tracker for debugging
    # This creates a folder ~/.burr with tracking data
    tracker = LocalTrackingClient(
        project="chatbot_app"  # Groups related runs in the UI
    )
    
    # Build the Burr application
    app = (
        ApplicationBuilder()
        # Register all actions (nodes in the state machine)
        .with_actions(
            process_user_message=process_user_message,
            safety_check=safety_check,
            generate_ai_response=generate_ai_response,
            unsafe_response=unsafe_response
        )
        # Define state transitions (edges in the state machine)
        # Format: (from_action, to_action, optional_condition)
        .with_transitions(
            # User message always goes to safety check
            ("process_user_message", "safety_check"),
            
            # Safety check branches based on is_safe value
            ("safety_check", "generate_ai_response", when(is_safe=True)),
            ("safety_check", "unsafe_response", when(is_safe=False)),
            
            # Both response types loop back to process user message
            # This creates the conversation loop
            (["generate_ai_response", "unsafe_response"], "process_user_message")
        )
        # Set identifiers for persistence
        .with_identifiers(
            app_id=conversation_id,     # Unique conversation ID
            partition_key=user_id       # Groups conversations by user
        )
        # Initialize from persisted state if it exists
        .initialize_from(
            persister,
            resume_at_next_action=True,  # Continue from where we left off
            default_state={              # Initial state for new conversations
                "chat_history": [],      # Empty conversation history
                "current_message": "",   # No current message
                "ai_response": "",       # No AI response yet
                "is_safe": True         # Default to safe
            },
            default_entrypoint="process_user_message"  # Where to start
        )
        # Enable state persistence after each action
        .with_state_persister(persister)
        # Enable tracking for debugging
        .with_tracker(tracker)
        # Set entry point for new runs
        .with_entrypoint("process_user_message")
        # Build the application
        .build()
    )
    
    return app, conversation_id, persister


def visualize_chatbot_graph(output_path: str = "chatbot_graph.png"):
    """
    Create a visual representation of the chatbot state machine.
    
    Args:
        output_path: Where to save the graph image
    """
    # Create a dummy app just for visualization
    app = (
        ApplicationBuilder()
        .with_actions(
            process_user_message=process_user_message,
            safety_check=safety_check,
            generate_ai_response=generate_ai_response,
            unsafe_response=unsafe_response
        )
        .with_transitions(
            ("process_user_message", "safety_check"),
            ("safety_check", "generate_ai_response", when(is_safe=True)),
            ("safety_check", "unsafe_response", when(is_safe=False)),
            (["generate_ai_response", "unsafe_response"], "process_user_message")
        )
        .with_state(chat_history=[])
        .with_entrypoint("process_user_message")
        .build()
    )
    
    # Generate visualization
    app.visualize(
        output_file_path=output_path,
        include_conditions=True,
        format="png",
        view=False  # Don't auto-open
    )
    print(f"Graph saved to {output_path}")