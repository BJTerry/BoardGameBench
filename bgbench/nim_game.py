from dataclasses import dataclass
from typing import List, Optional, Tuple
from .game import Game
from .utils import GameView

@dataclass
class NimState:
    remaining: int
    current_player: int

class NimGame(Game):
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
    
    def get_player_view(self, state: NimState, player_id: int) -> GameView:
        valid_moves = list(range(1, min(self.max_take, state.remaining) + 1))
        return GameView(
            visible_state={"remaining": state.remaining},
            valid_moves=valid_moves,
            is_terminal=state.remaining == 0,
            winner=state.current_player if state.remaining == 0 else None
        )
    
    def parse_move(self, move_str: str) -> Optional[int]:
        try:
            numbers = [int(s) for s in move_str.split() if s.isdigit()]
            return numbers[0] if numbers else None
        except:
            return None
    
    def validate_move(self, state: NimState, player_id: int, move: int) -> Tuple[bool, str]:
        if state.current_player != player_id:
            return False, "It's not your turn."
        if not isinstance(move, int):
            return False, "Move must be a number."
        if move < 1 or move > self.max_take:
            return False, f"You must take between 1 and {self.max_take} objects."
        if move > state.remaining:
            return False, f"There are only {state.remaining} objects remaining."
        return True, ""
    
    def apply_move(self, state: NimState, player_id: int, move: int) -> NimState:
        return NimState(
            remaining=state.remaining - move,
            current_player=1 - player_id
        )
    def __init__(self, starting_count: int = 12, max_take: int = 3):
        self.starting_count = starting_count
        self.max_take = max_take
        
