from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class GameView:
    """What a player can see of the game state.
    
    Attributes:
        visible_state: Dictionary of game state visible to this player
        valid_moves: List of legal moves available to this player
        is_terminal: Whether the game has ended
        winner: Player ID of winner if game is over, else None
        history: List of previous moves and their results
        move_format_instructions: How moves should be formatted
        rules_explanation: Explanation of game rules
        error_message: Last error message if any
    """
    visible_state: Dict[str, Any]
    valid_moves: List[Any]
    is_terminal: bool
    winner: Optional[int] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    move_format_instructions: Optional[str] = None
    rules_explanation: Optional[str] = None
    error_message: Optional[str] = None
