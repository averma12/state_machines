# api/chatbot_api.py
"""
FastAPI endpoints for the chatbot application.
Provides REST API for chat interactions and conversation management.
"""

import os
from typing import Optional, List
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import chatbot components
from chatbot_builder import create_chatbot_application
from mongodb_persister import MongoDBPersister


# Create FastAPI app
app = FastAPI(
    title="Burr Chatbot API",
    description="Stateful chatbot with conversation persistence",
    version="1.0.0"
)

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# REQUEST/RESPONSE MODELS
# ============================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    user_id: str = Field(..., description="Unique user identifier")
    message: str = Field(..., description="User's message")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID to continue")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="AI's response")
    conversation_id: str = Field(..., description="Conversation ID for future requests")
    message_id: str = Field(..., description="Unique message ID")
    trace_id: str = Field(..., description="Trace ID for debugging")


class ConversationMetadata(BaseModel):
    """Metadata about a conversation."""
    conversation_id: str
    message_count: int
    created_at: datetime
    last_updated: datetime
    last_message: str
    status: str


class UserConversations(BaseModel):
    """List of conversations for a user."""
    user_id: str
    conversations: List[ConversationMetadata]
    total_count: int


class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    mongodb_connected: bool
    timestamp: datetime


# ============================================
# ENDPOINTS
# ============================================

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """
    Health check endpoint to verify API and database connectivity.
    """
    try:
        # Test MongoDB connection
        persister = MongoDBPersister(
            connection_string=os.getenv("MONGODB_URI"),
            database="chatbot_db",
            collection="conversation_states"
        )
        # Try to list collections as a connection test
        _ = persister.db.list_collection_names()
        persister.cleanup()
        
        return HealthCheck(
            status="healthy",
            mongodb_connected=True,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        return HealthCheck(
            status="unhealthy",
            mongodb_connected=False,
            timestamp=datetime.utcnow()
        )


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint for sending messages and receiving AI responses.
    
    This endpoint:
    1. Creates or resumes a conversation based on conversation_id
    2. Processes the user message through the Burr state machine
    3. Returns the AI response with tracking metadata
    
    Args:
        request: Chat request with user_id, message, and optional conversation_id
        
    Returns:
        Chat response with AI message and metadata
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        # Create or resume chatbot application
        app, conversation_id, _ = create_chatbot_application(
            user_id=request.user_id,
            conversation_id=request.conversation_id
        )
        
        # Run the chatbot with user message
        # The state machine will execute:
        # 1. process_user_message
        # 2. safety_check
        # 3. generate_ai_response OR unsafe_response
        action_taken, result, state = app.run(
            halt_after=["generate_ai_response", "unsafe_response"],
            inputs={"user_message": request.message}
        )
        
        # Extract response metadata from the result
        # The result contains data from the last action that ran
        response_data = {
            "response": state["ai_response"],
            "conversation_id": conversation_id,
            "message_id": result.get("message_id", ""),
            "trace_id": result.get("trace_id", "")
        }
        
        return ChatResponse(**response_data)
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{user_id}", response_model=UserConversations)
async def get_user_conversations(
    user_id: str,
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0)
):
    """
    Get all conversations for a user with pagination.
    
    Args:
        user_id: User identifier
        limit: Maximum number of conversations to return
        offset: Number of conversations to skip
        
    Returns:
        List of conversation metadata
        
    Raises:
        HTTPException: If retrieval fails
    """
    try:
        # Create persister to query MongoDB
        persister = MongoDBPersister(
            connection_string=os.getenv("MONGODB_URI"),
            database="chatbot_db",
            collection="conversation_states"
        )
        
        # Get all conversation IDs for this user
        all_conversation_ids = persister.list_app_ids(partition_key=user_id)
        total_count = len(all_conversation_ids)
        
        # Apply pagination
        paginated_ids = all_conversation_ids[offset:offset + limit]
        
        # Get metadata for each conversation
        conversations = []
        for conv_id in paginated_ids:
            metadata = persister.get_conversation_metadata(conv_id)
            if metadata:
                # Load the actual state to get chat history
                state_data = persister.load(
                    partition_key=user_id,
                    app_id=conv_id
                )
                
                if state_data:
                    chat_history = state_data["state"].get("chat_history", [])
                    conversations.append(ConversationMetadata(
                        conversation_id=conv_id,
                        message_count=len(chat_history),
                        created_at=metadata["created_at"],
                        last_updated=metadata["last_updated"],
                        last_message=chat_history[-1]["content"] if chat_history else "",
                        status=metadata["status"]
                    ))
        
        # Clean up connection
        persister.cleanup()
        
        return UserConversations(
            user_id=user_id,
            conversations=conversations,
            total_count=total_count
        )
        
    except Exception as e:
        print(f"Error getting conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/{conversation_id}/history")
async def get_conversation_history(
    conversation_id: str,
    user_id: str = Query(..., description="User ID for authorization")
):
    """
    Get the full chat history for a specific conversation.
    
    Args:
        conversation_id: Conversation identifier
        user_id: User ID (for authorization check)
        
    Returns:
        Complete chat history
        
    Raises:
        HTTPException: If conversation not found or unauthorized
    """
    try:
        persister = MongoDBPersister(
            connection_string=os.getenv("MONGODB_URI"),
            database="chatbot_db",
            collection="conversation_states"
        )
        
        # Load the conversation state
        state_data = persister.load(
            partition_key=user_id,
            app_id=conversation_id
        )
        
        if not state_data:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
        # Verify the conversation belongs to the user
        if state_data["partition_key"] != user_id:
            raise HTTPException(status_code=403, detail="Unauthorized")
            
        # Extract chat history
        chat_history = state_data["state"].get("chat_history", [])
        
        # Clean up connection
        persister.cleanup()
        
        return {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "chat_history": chat_history,
            "message_count": len(chat_history)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversation/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user_id: str = Query(..., description="User ID for authorization")
):
    """
    Delete a conversation and all its history.
    
    Args:
        conversation_id: Conversation to delete
        user_id: User ID (for authorization check)
        
    Returns:
        Deletion confirmation
        
    Raises:
        HTTPException: If deletion fails or unauthorized
    """
    try:
        persister = MongoDBPersister(
            connection_string=os.getenv("MONGODB_URI"),
            database="chatbot_db",
            collection="conversation_states"
        )
        
        # Verify conversation exists and belongs to user
        state_data = persister.load(
            partition_key=user_id,
            app_id=conversation_id
        )
        
        if not state_data:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
        if state_data["partition_key"] != user_id:
            raise HTTPException(status_code=403, detail="Unauthorized")
        
        # Delete the conversation
        deleted_count = persister.delete_conversation(conversation_id)
        
        # Clean up connection
        persister.cleanup()
        
        return {
            "message": "Conversation deleted successfully",
            "conversation_id": conversation_id,
            "documents_deleted": deleted_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# STARTUP/SHUTDOWN EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    print("Chatbot API starting up...")
    # Verify MongoDB connection
    try:
        persister = MongoDBPersister(
            connection_string=os.getenv("MONGODB_URI"),
            database="chatbot_db",
            collection="conversation_states"
        )
        persister.cleanup()
        print("MongoDB connection verified")
    except Exception as e:
        print(f"Warning: MongoDB connection failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    print("Chatbot API shutting down...")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True  # Enable auto-reload for development
    )