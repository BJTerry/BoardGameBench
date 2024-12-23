from abc import ABC, abstractmethod
from typing import Any, Tuple
from .utils import GameView

class Game(ABC):
    @abstractmethod
    def get_rules_explanation(self) -> str:
        """Return a clear explanation of the game rules."""
        pass
    
    @abstractmethod
    def get_move_format_instructions(self) -> str:
        """Explain how moves should be formatted in responses."""
        pass

    @abstractmethod
    def get_initial_state(self) -> Any:
        """Return the initial state of the game."""
        pass
    
    @abstractmethod
    def get_player_view(self, state: Any, player_id: int) -> 'GameView':
        """Return what this player can see of the current state."""
        pass
    
    @abstractmethod
    def parse_move(self, move_str: str) -> Any:
        """Parse move from LLM response string."""
        pass

    @abstractmethod
    def validate_move(self, state: Any, player_id: int, move: Any) -> Tuple[bool, str]:
        """Returns (is_valid, explanation)."""
        pass
    
    @abstractmethod
    def apply_move(self, state: Any, player_id: int, move: Any) -> Any:
        """Apply move to state and return new state."""
        pass
