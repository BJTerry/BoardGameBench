from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from bgbench.game import Game
from bgbench.game_view import GameView

@dataclass
class NimState:
    remaining: int
    current_player: int

    def to_dict(self) -> dict:
        return {
            "remaining": self.remaining,
            "current_player": self.current_player,
        }


@dataclass
class NimMove:
    """Represents a move in Nim game."""
    count: int
    
    def to_dict(self) -> dict:
        """Convert move to dictionary for serialization."""
        return {"count": self.count}
    
    def __str__(self) -> str:
        return str(self.count)

class NimGame(Game[NimState, NimMove]):
    def get_rules_explanation(self) -> str:
        return (
            f"We are playing Nim. There are {self.starting_count} objects in a pile. "
            f"Players take turns removing 1 to {self.max_take} objects. "
            "The player who takes the last object wins. "
        )
    
    def get_move_format_instructions(self) -> str:
        return (
            "On your turn, respond with only a number indicating how many objects "
            "you want to take. For example: '2' to take 2 objects."
        )
    
    def get_initial_state(self) -> NimState:
        return NimState(
            remaining=self.starting_count,
            current_player=0
        )
    
    def get_player_view(self, state: NimState, player_id: int, 
                       history: Optional[List[Dict[str, Any]]] = None) -> GameView:
        valid_moves = list(range(1, min(self.max_take, state.remaining) + 1))
        return GameView(
            move_format_instructions=self.get_move_format_instructions(),
            rules_explanation=self.get_rules_explanation(),
            visible_state={"remaining": state.remaining},
            valid_moves=valid_moves,
            is_terminal=state.remaining == 0,
            winner=state.current_player if state.remaining == 0 else None
        )
    
    def parse_move(self, move_str: str) -> Optional[NimMove]:
        """Parse move from LLM response string.
        
        Args:
            move_str: The raw string from the LLM containing a number
            
        Returns:
            NimMove object if valid, None if parsing failed
        """
        try:
            numbers = [int(s) for s in move_str.split() if s.isdigit()]
            if not numbers:
                return None
            return NimMove(count=numbers[0])
        except (ValueError, IndexError):
            return None
    
    def validate_move(self, state: NimState, player_id: int, move: NimMove) -> Tuple[bool, str]:
        """Validate if a move is legal in the current state.
        
        Args:
            state: Current game state
            player_id: ID of player making the move
            move: The NimMove to validate
            
        Returns:
            Tuple of (is_valid, explanation_string)
        """
        if state.current_player != player_id:
            return False, "It's not your turn."
        if not isinstance(move, NimMove):
            return False, "Invalid move type"
        if move.count < 1 or move.count > self.max_take:
            return False, f"You must take between 1 and {self.max_take} objects."
        if move.count > state.remaining:
            return False, f"There are only {state.remaining} objects remaining."
        return True, ""
    
    def apply_move(self, state: NimState, player_id: int, move: NimMove) -> NimState:
        """Apply move to state and return new state.
        
        Args:
            state: Current game state
            player_id: ID of player making the move
            move: The NimMove to apply
            
        Returns:
            New game state after applying the move
            
        Raises:
            ValueError: If the move is invalid
        """
        valid, reason = self.validate_move(state, player_id, move)
        if not valid:
            raise ValueError(reason)
            
        return NimState(
            remaining=state.remaining - move.count,
            current_player=1 - player_id
        )

    def get_current_player(self, state: NimState) -> int:
        return state.current_player

    def get_next_state(self, state: NimState, move: NimMove) -> NimState:
        """Return the next state after applying the move.
        
        Args:
            state: Current game state
            move: The NimMove to apply
            
        Returns:
            New game state
        """
        return NimState(
            remaining=state.remaining - move.count,
            current_player=1 - state.current_player
        )

    def __init__(self, starting_count: int = 12, max_take: int = 3):
        self.starting_count = starting_count
        self.max_take = max_take
