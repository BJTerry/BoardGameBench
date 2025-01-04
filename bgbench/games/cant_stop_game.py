from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set
from bgbench.game import Game
from bgbench.game_view import GameView, PromptStyle
import random
from copy import deepcopy

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
    current_dice: List[int]  # Current dice values (4 dice)
    awaiting_selection: bool  # True if waiting for dice selection, False if waiting for stop/roll decision
    
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
    action: str  # Either 'select' for dice selection, 'stop' to end turn, or 'roll' to continue
    selections: List[int]  # Indices of selected dice (0-3) when action is 'select'
    
    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "selections": self.selections
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
        # Roll initial dice
        initial_dice = [random.randint(1, 6) for _ in range(4)]
        return CantStopState(
            columns=columns,
            current_player=0,
            temp_positions={},
            active_columns=set(),
            current_dice=initial_dice,
            awaiting_selection=True
        )
    
    def _get_possible_combinations(self, dice: List[int]) -> List[Tuple[int, int]]:
        """Get all possible dice combinations."""
        if len(dice) != 4:
            return []
            
        # Try all possible pairings
        return [
            (dice[0] + dice[1], dice[2] + dice[3]),
            (dice[0] + dice[2], dice[1] + dice[3]),
            (dice[0] + dice[3], dice[1] + dice[2])
        ]

    def _validate_column_sums(self, state: CantStopState, sum1: int, sum2: int) -> Tuple[bool, str]:
        """Validate if the given column sums are possible and available."""
        # Check if sums are possible with current dice
        possible_combinations = self._get_possible_combinations(state.current_dice)
        possible_sums = {sum1, sum2}
        valid_pairs = [(s1, s2) for s1, s2 in possible_combinations 
                      if {s1, s2} == possible_sums]
        
        if not valid_pairs:
            return False, "Selected columns must be possible sums of dice pairs"
            
        # Check if columns can be used
        can_use_sum1 = (not state.columns[sum1].is_claimed and
                       (sum1 in state.active_columns or 
                        len(state.active_columns) < 3))
        can_use_sum2 = (not state.columns[sum2].is_claimed and
                       (sum2 in state.active_columns or 
                        len(state.active_columns) < 3))
        
        if not (can_use_sum1 or can_use_sum2):
            return False, "Selected columns must be available for progress"
            
        return True, ""

    def _has_valid_move(self, state: CantStopState) -> bool:
        """Check if any valid moves are possible with current dice."""
        dice = state.current_dice
        combinations = self._get_possible_combinations(dice)
        
        for sum1, sum2 in combinations:
            # Check if sums are valid column numbers (2-12)
            if not (2 <= sum1 <= 12 and 2 <= sum2 <= 12):
                continue
                
            # Check if either sum can be used
            can_use_sum1 = (sum1 in state.columns and 
                           not state.columns[sum1].is_claimed and
                           (sum1 in state.active_columns or 
                            len(state.active_columns) < 3))
            can_use_sum2 = (sum2 in state.columns and 
                           not state.columns[sum2].is_claimed and
                           (sum2 in state.active_columns or 
                            len(state.active_columns) < 3))
            
            if can_use_sum1 or can_use_sum2:
                return True
        return False
    
    def parse_move(self, move_str: str) -> Optional[CantStopMove]:
        """Parse a move string."""
        try:
            parts = move_str.strip().lower().split()
            if not parts:
                return None
                
            if parts[0] in ["stop", "roll"]:
                return CantStopMove(parts[0], [])
                
            if parts[0] == "select":
                # Parse column selections (2-12)
                selections = [int(i) for i in parts[1:]]
                if len(selections) != 2:
                    return None
                if not all(2 <= i <= 12 for i in selections):
                    return None
                return CantStopMove("select", selections)
                
            return None
        except (ValueError, IndexError):
            return None
    
    def validate_move(self, state: CantStopState, player_id: int, move: CantStopMove) -> Tuple[bool, str]:
        """Validate if a move is legal."""
        if self.get_current_player(state) != player_id:
            return False, "Not your turn"
            
        if state.awaiting_selection:
            if move.action != "select":
                return False, "Must select dice now"
                
            # Validate the selected column sums
            sum1, sum2 = move.selections
            valid, msg = self._validate_column_sums(state, sum1, sum2)
            if not valid:
                return False, msg
        else:
            if move.action not in ["stop", "roll"]:
                return False, "Must choose to stop or roll"
                
        return True, ""
    
    def apply_move(self, state: CantStopState, player_id: int, move: CantStopMove) -> CantStopState:
        """Apply move to state and return new state."""
        new_state = deepcopy(state)
        
        if state.awaiting_selection:
            # Process column selection
            sum1, sum2 = move.selections
            
            # Validate and use selected sums
            valid, _ = self._validate_column_sums(state, sum1, sum2)
            if valid and (sum1 in new_state.columns and 
                         not new_state.columns[sum1].is_claimed and
                         (sum1 in new_state.active_columns or 
                          len(new_state.active_columns) < 3)):
                current_pos = new_state.temp_positions.get(sum1, 0)
                new_state.temp_positions[sum1] = min(
                    current_pos + 1,
                    new_state.columns[sum1].max_height
                )
                new_state.active_columns.add(sum1)
            
            if valid and (sum2 in new_state.columns and 
                         not new_state.columns[sum2].is_claimed and
                         (sum2 in new_state.active_columns or 
                          len(new_state.active_columns) < 3)):
                current_pos = new_state.temp_positions.get(sum2, 0)
                new_state.temp_positions[sum2] = min(
                    current_pos + 1,
                    new_state.columns[sum2].max_height
                )
                new_state.active_columns.add(sum2)
            
            # Switch to stop/roll decision
            new_state.awaiting_selection = False
        
        elif move.action == "stop":
            # Convert temporary positions to permanent ones
            for col, pos in new_state.temp_positions.items():
                if col in new_state.columns:
                    new_state.columns[col].player_positions[player_id] = pos
                    # Check if column is claimed
                    if pos >= new_state.columns[col].max_height:
                        new_state.columns[col].is_claimed = True
                        new_state.columns[col].claimed_by = player_id
            
            # Reset for next player
            new_state.temp_positions = {}
            new_state.active_columns = set()
            new_state.current_player = 1 - player_id
            new_state.current_dice = [random.randint(1, 6) for _ in range(4)]
            new_state.awaiting_selection = True
        
        elif move.action == "roll":
            # Roll new dice
            new_state.current_dice = [random.randint(1, 6) for _ in range(4)]
            
            # Check if the player busted
            if not self._has_valid_move(new_state):
                # Current player busts - lose all progress and switch players
                new_state = deepcopy(state)  # Start fresh to ensure clean state
                new_state.temp_positions.clear()  # Explicitly clear temporary positions
                new_state.active_columns.clear()  # Explicitly clear active columns
                new_state.current_player = 1 - state.current_player
                new_state.current_dice = [random.randint(1, 6) for _ in range(4)]
                new_state.awaiting_selection = True
            else:
                new_state.awaiting_selection = True
        
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
                    "temp_progress_position": state.temp_positions.get(num, 0) if player_id == state.current_player else 0
                }
                for num, col in state.columns.items()
            },
            "active_columns": list(state.active_columns),
            "your_turn": player_id == state.current_player,
            "current_dice": state.current_dice if player_id == state.current_player else None,
            "action_required": "select two column sums" if state.awaiting_selection else "decide to roll or stop"
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
            "Game Flow:\n"
            "1. System rolls 4 dice\n"
            "2. Select 2 sums with 'select X Y' (X, Y are column numbers 2-12)\n"
            "3. Choose to 'roll' or 'stop'\n"
            "Example moves:\n"
            "- 'select 7 8' to advance in columns 7 and 8\n"
            "- 'roll' to continue turn\n"
            "- 'stop' to end turn and keep progress"
        )
    
    def get_rules_explanation(self) -> str:
        return (
            "Can't Stop is a push-your-luck dice game where players try to claim three columns.\n"
            "On your turn:\n"
            "1. Roll 4 dice and choose two sums from any combination of dice pairs\n"
            "   Example: With dice 3,4,2,5 you could choose sums (7,8), (5,10), or (9,6)\n"
            "2. You can use up to 3 different columns per turn\n"
            "3. After each roll, choose to continue or stop and keep progress\n"
            "4. If you can't use any dice combinations, you lose all progress\n"
            "5. Claim a column by reaching its top space\n"
            "6. First player to claim 3 columns wins"
        )
    
    def get_next_state(self, state: CantStopState, move: CantStopMove) -> CantStopState:
        """Return the next state after applying the move."""
        return self.apply_move(state, self.get_current_player(state), move)
