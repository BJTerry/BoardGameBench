from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set
from bgbench.game import Game
from bgbench.game_view import GameView, PromptStyle
import random

@dataclass
class ColumnState:
    """Represents the state of a single column."""
    player_positions: Dict[int, int]  # player_id -> position
    max_height: int  # Maximum height of this column
    is_claimed: bool = False
    claimed_by: Optional[int] = None
    
    def to_dict(self) -> dict:
        return {
            "player_positions": self.player_positions,
            "max_height": self.max_height,
            "is_claimed": self.is_claimed,
            "claimed_by": self.claimed_by
        }

@dataclass
class CantStopState:
    """Represents the current state of a Can't Stop game."""
    columns: Dict[int, ColumnState]  # number -> ColumnState
    current_player: int
    temp_positions: Dict[int, int]  # column -> position (white cubes)
    active_columns: Set[int]  # Columns currently in use this turn
    
    def to_dict(self) -> dict:
        return {
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
            "current_player": self.current_player,
            "temp_positions": self.temp_positions,
            "active_columns": list(self.active_columns)
        }

@dataclass
class CantStopMove:
    """Represents a move in Can't Stop."""
    dice: List[int]  # The four dice values
    combinations: List[Tuple[int, int]]  # The chosen combinations
    stop: bool  # Whether to end the turn
    
    def to_dict(self) -> dict:
        return {
            "dice": self.dice,
            "combinations": self.combinations,
            "stop": self.stop
        }

class CantStopGame(Game[CantStopState, CantStopMove]):
    """Implementation of Can't Stop game."""
    
    # Column heights in the original game
    COLUMN_HEIGHTS = {
        2: 3, 3: 5, 4: 7, 5: 9, 6: 11,
        7: 13, 8: 11, 9: 9, 10: 7, 11: 5, 12: 3
    }
    
    def __init__(self):
        super().__init__()
    
    def get_initial_state(self) -> CantStopState:
        """Return the initial state of the game."""
        columns = {
            num: ColumnState(
                player_positions={},
                max_height=self.COLUMN_HEIGHTS[num]
            )
            for num in range(2, 13)
        }
        return CantStopState(
            columns=columns,
            current_player=0,
            temp_positions={},
            active_columns=set()
        )
    
    def _get_possible_combinations(self, dice: List[int]) -> List[List[Tuple[int, int]]]:
        """Get all possible dice combinations."""
        if len(dice) != 4:
            return []
            
        results = []
        # Try all possible pairings
        pairs = [
            [(dice[0] + dice[1], dice[2] + dice[3]),
             (dice[0] + dice[2], dice[1] + dice[3]),
             (dice[0] + dice[3], dice[1] + dice[2])]
        ]
        return pairs[0]
    
    def parse_move(self, move_str: str) -> Optional[CantStopMove]:
        """Parse a move string."""
        try:
            parts = move_str.strip().split()
            if not parts:
                return None
                
            if parts[0].lower() == "stop":
                return CantStopMove([], [], True)
                
            # Parse dice and combinations
            dice = [int(d) for d in parts[0:4]]
            if len(dice) != 4:
                return None
                
            combinations = []
            combo_parts = parts[4:]
            for i in range(0, len(combo_parts), 2):
                if i + 1 >= len(combo_parts):
                    break
                combinations.append((int(combo_parts[i]), int(combo_parts[i + 1])))
            
            return CantStopMove(dice, combinations, False)
        except (ValueError, IndexError):
            return None
    
    def validate_move(self, state: CantStopState, player_id: int, move: CantStopMove) -> Tuple[bool, str]:
        """Validate if a move is legal."""
        if self.get_current_player(state) != player_id:
            return False, "Not your turn"
            
        if move.stop:
            return True, ""
            
        # Validate dice combinations
        possible_combos = self._get_possible_combinations(move.dice)
        if not all(combo in possible_combos for combo in move.combinations):
            return False, "Invalid dice combinations"
            
        # Check if columns are available
        for combo in move.combinations:
            for num in combo:
                if state.columns[num].is_claimed:
                    return False, f"Column {num} is already claimed"
                    
        return True, ""
    
    def apply_move(self, state: CantStopState, player_id: int, move: CantStopMove) -> CantStopState:
        """Apply move to state and return new state."""
        new_state = CantStopState(
            columns={k: ColumnState(
                player_positions=v.player_positions.copy(),
                max_height=v.max_height,
                is_claimed=v.is_claimed,
                claimed_by=v.claimed_by
            ) for k, v in state.columns.items()},
            current_player=state.current_player,
            temp_positions=state.temp_positions.copy(),
            active_columns=state.active_columns.copy()
        )
        
        if move.stop:
            # Convert temporary positions to permanent ones
            for col, pos in new_state.temp_positions.items():
                if col in new_state.columns:
                    new_state.columns[col].player_positions[player_id] = pos
                    # Check if column is claimed
                    if pos >= new_state.columns[col].max_height:
                        new_state.columns[col].is_claimed = True
                        new_state.columns[col].claimed_by = player_id
            
            # Clear temporary state
            new_state.temp_positions = {}
            new_state.active_columns = set()
            new_state.current_player = 1 - player_id
            return new_state
            
        # Process the combinations
        for combo in move.combinations:
            for num in combo:
                if num not in new_state.active_columns:
                    if len(new_state.active_columns) >= 3:
                        continue
                    new_state.active_columns.add(num)
                
                current_pos = new_state.temp_positions.get(num, 0)
                if num in new_state.columns[num].player_positions:
                    current_pos = max(current_pos, 
                                   new_state.columns[num].player_positions[player_id] + 1)
                new_state.temp_positions[num] = min(
                    current_pos + 1,
                    new_state.columns[num].max_height
                )
        
        return new_state
    
    def get_current_player(self, state: CantStopState) -> int:
        """Return the ID of the player whose turn it is."""
        return state.current_player
    
    def get_player_view(self, state: CantStopState, player_id: int,
                       history: Optional[List[Dict[str, Any]]] = None,
                       prompt_style: PromptStyle = PromptStyle.HEADER) -> GameView:
        """Return what this player can see of the current state."""
        visible_state = {
            "columns": {
                num: {
                    "max_height": col.max_height,
                    "claimed": col.is_claimed,
                    "claimed_by": col.claimed_by,
                    "your_position": col.player_positions.get(player_id, 0),
                    "opponent_position": col.player_positions.get(1 - player_id, 0),
                    "temp_position": state.temp_positions.get(num, 0) if player_id == state.current_player else 0
                }
                for num, col in state.columns.items()
            },
            "active_columns": list(state.active_columns),
            "your_turn": player_id == state.current_player
        }
        
        # Check win condition
        winner = None
        is_terminal = False
        claimed_columns = sum(1 for col in state.columns.values() 
                            if col.is_claimed and col.claimed_by == player_id)
        if claimed_columns >= 3:
            winner = player_id
            is_terminal = True
            
        return GameView(
            rules_explanation=self.get_rules_explanation(),
            visible_state=visible_state,
            valid_moves=[],  # Dice-based game, so no fixed move list
            is_terminal=is_terminal,
            winner=winner,
            history=history if history else [],
            move_format_instructions=self.get_move_format_instructions(),
            prompt_style=prompt_style
        )
    
    def get_move_format_instructions(self) -> str:
        return (
            "To make a move, enter:\n"
            "1. Four dice values (1-6)\n"
            "2. Your chosen combinations as pairs of sums\n"
            "Example: '2 3 4 6 5 7' (using 2+3=5 and 4+6=10)\n"
            "Or enter 'stop' to end your turn and keep progress"
        )
    
    def get_rules_explanation(self) -> str:
        return (
            "Can't Stop is a push-your-luck dice game where players try to claim three columns.\n"
            "On your turn:\n"
            "1. Roll 4 dice and make two pairs to advance in those columns\n"
            "2. You can use up to 3 different columns per turn\n"
            "3. After each roll, choose to continue or stop and keep progress\n"
            "4. If you can't use any dice combinations, you lose all progress\n"
            "5. Claim a column by reaching its top space\n"
            "6. First player to claim 3 columns wins"
        )
    
    def get_next_state(self, state: CantStopState, move: CantStopMove) -> CantStopState:
        """Return the next state after applying the move."""
        return self.apply_move(state, self.get_current_player(state), move)
