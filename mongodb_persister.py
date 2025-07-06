# persistence/mongodb_persister.py
"""
MongoDB persister for Burr state management.
Handles saving and loading conversation states from MongoDB.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pymongo import MongoClient
from burr.core import State
from burr.core.persistence import BaseStatePersister


class MongoDBPersister(BaseStatePersister):
    """
    Custom MongoDB persister for Burr state.
    Implements the BaseStatePersister interface to save/load state from MongoDB.
    
    This persister stores each state transition as a document in MongoDB,
    allowing for full conversation history and state recovery.
    """
    
    def __init__(self, connection_string: str, database: str, collection: str):
        """
        Initialize MongoDB connection.
        
        Args:
            connection_string: MongoDB connection string (e.g., mongodb+srv://...)
            database: Database name
            collection: Collection name for storing states
        """
        # Create MongoDB client
        self.client = MongoClient(connection_string)
        # Select database
        self.db = self.client[database]
        # Select collection
        self.collection = self.db[collection]
        
        # Create indexes for efficient querying
        self._create_indexes()
        
    def _create_indexes(self):
        """Create indexes for efficient querying."""
        # Index for loading latest state by app_id
        self.collection.create_index([("app_id", 1), ("sequence_id", -1)])
        # Index for querying by partition_key (user_id)
        self.collection.create_index("partition_key")
        # Index for timestamp-based queries
        self.collection.create_index("created_at")
        
    def save(
        self,
        partition_key: Optional[str],
        app_id: str,
        sequence_id: int,
        position: str,
        state: State,
        status: str,
        **kwargs
    ):
        """
        Save state to MongoDB.
        Called automatically by Burr after each action execution.
        
        Args:
            partition_key: User ID in our case - groups conversations by user
            app_id: Conversation ID - unique identifier for each conversation
            sequence_id: Step number in the conversation (auto-incremented)
            position: Name of the action that just ran
            state: Current state object to persist
            status: Status of the action (completed, failed, etc.)
        """
        # Create document to save
        document = {
            "_id": f"{app_id}:{sequence_id}",  # Composite key for uniqueness
            "partition_key": partition_key,      # User ID
            "app_id": app_id,                   # Conversation ID
            "sequence_id": sequence_id,          # Step number
            "position": position,                # Last action name
            "state": state.get_all(),           # Convert state to dict
            "status": status,                    # Action status
            "created_at": datetime.utcnow(),    # Timestamp
            **kwargs                            # Any additional metadata
        }
        
        # Upsert document (insert or update if exists)
        self.collection.replace_one(
            {"_id": document["_id"]},
            document,
            upsert=True
        )
        
    def load(
        self,
        partition_key: Optional[str],
        app_id: str,
        sequence_id: Optional[int] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Load state from MongoDB.
        Called when initializing an app to restore previous state.
        
        Args:
            partition_key: User ID
            app_id: Conversation ID
            sequence_id: Specific step to load (None = latest)
            
        Returns:
            Dict with state data or None if not found
        """
        if sequence_id is not None:
            # Load specific sequence
            doc = self.collection.find_one({"_id": f"{app_id}:{sequence_id}"})
        else:
            # Load latest sequence for this app_id
            doc = self.collection.find_one(
                {"app_id": app_id},
                sort=[("sequence_id", -1)]  # Sort by sequence_id descending
            )
            
        if doc:
            return {
                "partition_key": doc["partition_key"],
                "app_id": doc["app_id"],
                "sequence_id": doc["sequence_id"],
                "position": doc["position"],
                "state": State(doc["state"]),  # Reconstruct State object
                "status": doc["status"]
            }
        return None
    
    def list_app_ids(self, partition_key: Optional[str] = None, **kwargs) -> List[str]:
        """
        List all conversation IDs, optionally filtered by user.
        
        Args:
            partition_key: User ID to filter by (optional)
            
        Returns:
            List of conversation IDs
        """
        query = {}
        if partition_key:
            query["partition_key"] = partition_key
            
        # Get distinct app_ids
        return self.collection.distinct("app_id", query)
    
    def get_conversation_metadata(self, app_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about a conversation.
        
        Args:
            app_id: Conversation ID
            
        Returns:
            Dictionary with conversation metadata or None
        """
        # Get latest state
        latest = self.collection.find_one(
            {"app_id": app_id},
            sort=[("sequence_id", -1)]
        )
        
        if not latest:
            return None
            
        # Get first message
        first = self.collection.find_one(
            {"app_id": app_id},
            sort=[("sequence_id", 1)]
        )
        
        # Count total steps
        total_steps = self.collection.count_documents({"app_id": app_id})
        
        return {
            "app_id": app_id,
            "partition_key": latest["partition_key"],
            "total_steps": total_steps,
            "created_at": first["created_at"] if first else None,
            "last_updated": latest["created_at"],
            "last_position": latest["position"],
            "status": latest["status"]
        }
    
    def delete_conversation(self, app_id: str) -> int:
        """
        Delete all states for a conversation.
        
        Args:
            app_id: Conversation ID to delete
            
        Returns:
            Number of documents deleted
        """
        result = self.collection.delete_many({"app_id": app_id})
        return result.deleted_count
    
    def cleanup(self):
        """Close MongoDB connection."""
        self.client.close()
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup connection."""
        self.cleanup()