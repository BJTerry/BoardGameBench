from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional


@dataclass
class MatchStateData:
    """Represents the state of a match that can be saved to and loaded from the database."""

    # Common fields across all game types
    turn: int  # The current turn number (0 for initial state)
    current_player_id: int  # ID of the player whose turn it is
    timestamp: datetime  # When this state was recorded

    # Game-specific state (serialized by the Game implementation)
    game_state: Dict[str, Any]  # Game-specific state that can be deserialized by the Game

    # Optional metadata
    metadata: Optional[Dict[str, Any]] = None  # Additional information (e.g., visible_state, history)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this MatchStateData instance to a dictionary suitable for database storage.
        
        Handles datetime serialization by converting to ISO format string.

        Returns:
            A dictionary representation of this match state
        """
        result = {
            "turn": self.turn,
            "current_player_id": self.current_player_id,
            "timestamp": self.timestamp.isoformat(),
            "game_state": self.game_state
        }
        
        if self.metadata is not None:
            result["metadata"] = self.metadata
            
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MatchStateData':
        """
        Create a MatchStateData instance from a dictionary.
        
        Handles datetime deserialization from ISO format string.

        Args:
            data: Dictionary containing the match state data

        Returns:
            A new MatchStateData instance
        """
        # Handle timestamp as string (from database) or datetime object (from memory)
        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
            
        return cls(
            turn=data["turn"],
            current_player_id=data["current_player_id"],
            timestamp=timestamp,
            game_state=data["game_state"],
            metadata=data.get("metadata")
        )
