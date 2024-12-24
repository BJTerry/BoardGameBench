from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class GameView:
    """What a player can see of the game state."""
    visible_state: Dict[str, Any]
    valid_moves: List[Any]
    is_terminal: bool
    winner: Optional[int] = None
    history: List[Dict[str, Any]] = None  # Add this line
