from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set
from bgbench.game import Game
from bgbench.match.view import MatchView, PromptStyle
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
            "player_positions": {str(k): v for k, v in self.player_positions.items()},
            "max_height": self.max_height,
            "is_claimed": self.is_claimed,
            "claimed_by": self.claimed_by,
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
            "active_columns": list(self.active_columns),
        }


@dataclass
class CantStopMove:
    """Represents a move in Can't Stop."""

    action: str  # Either 'select' for dice selection, 'stop' to end turn, or 'roll' to continue
    selections: List[int]  # Indices of selected dice (0-3) when action is 'select'

    def to_dict(self) -> dict:
        return {"action": self.action, "selections": self.selections}


class CantStopGame(Game[CantStopState, CantStopMove]):
    """Implementation of Can't Stop game."""

    # Column heights in the original game
    COLUMN_HEIGHTS = {
        2: 3,
        3: 5,
        4: 7,
        5: 9,
        6: 11,
        7: 13,
        8: 11,
        9: 9,
        10: 7,
        11: 5,
        12: 3,
    }

    def __init__(self):
        super().__init__()

    def get_initial_state(self) -> CantStopState:
        """Return the initial state of the game."""
        columns = {
            num: ColumnState(player_positions={}, max_height=self.COLUMN_HEIGHTS[num])
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
            awaiting_selection=True,
        )

    def _get_possible_combinations(self, dice: List[int]) -> List[Tuple[int, int]]:
        """Get all possible dice combinations."""
        if len(dice) != 4:
            return []

        # Try all possible pairings
        return [
            (dice[0] + dice[1], dice[2] + dice[3]),
            (dice[0] + dice[2], dice[1] + dice[3]),
            (dice[0] + dice[3], dice[1] + dice[2]),
        ]

    def _validate_column_sums(
        self, state: CantStopState, sum1: int, sum2: int
    ) -> Tuple[bool, str]:
        """Validate if the given column sums are possible and available."""
        # Check if sums are possible with current dice
        possible_combinations = self._get_possible_combinations(state.current_dice)
        possible_sums = {sum1, sum2}
        valid_pairs = [
            (s1, s2) for s1, s2 in possible_combinations if {s1, s2} == possible_sums
        ]

        if not valid_pairs:
            return False, "Selected columns must be possible sums of dice pairs"

        # Check if columns can be used
        can_use_sum1 = not state.columns[sum1].is_claimed and (
            sum1 in state.active_columns or len(state.active_columns) < 3
        )
        can_use_sum2 = not state.columns[sum2].is_claimed and (
            sum2 in state.active_columns or len(state.active_columns) < 3
        )

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
            can_use_sum1 = (
                sum1 in state.columns
                and not state.columns[sum1].is_claimed
                and (sum1 in state.active_columns or len(state.active_columns) < 3)
            )
            can_use_sum2 = (
                sum2 in state.columns
                and not state.columns[sum2].is_claimed
                and (sum2 in state.active_columns or len(state.active_columns) < 3)
            )

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

    def validate_move(
        self, state: CantStopState, player_id: int, move: CantStopMove
    ) -> Tuple[bool, str]:
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

    def _reset_for_next_player(
        self, state: CantStopState, next_player: int
    ) -> CantStopState:
        """Helper method to reset the state for the next player."""
        new_state = deepcopy(state)
        new_state.temp_positions.clear()
        new_state.active_columns.clear()
        new_state.current_player = next_player
        new_state.current_dice = [random.randint(1, 6) for _ in range(4)]
        new_state.awaiting_selection = True
        return new_state

    def apply_move(
        self, state: CantStopState, player_id: int, move: CantStopMove
    ) -> CantStopState:
        """Apply move to state and return new state."""
        new_state = deepcopy(state)

        if state.awaiting_selection:
            # Process column selection
            sum1, sum2 = move.selections

            # Validate and use selected sums
            valid, _ = self._validate_column_sums(state, sum1, sum2)
            if valid and (
                sum1 in new_state.columns
                and not new_state.columns[sum1].is_claimed
                and (
                    sum1 in new_state.active_columns
                    or len(new_state.active_columns) < 3
                )
            ):
                # Start from player's current position in the column if it exists
                player_pos = new_state.columns[sum1].player_positions.get(player_id, 0)
                current_pos = new_state.temp_positions.get(sum1, player_pos)
                new_state.temp_positions[sum1] = min(
                    current_pos + 1, new_state.columns[sum1].max_height
                )
                new_state.active_columns.add(sum1)

            if valid and (
                sum2 in new_state.columns
                and not new_state.columns[sum2].is_claimed
                and (
                    sum2 in new_state.active_columns
                    or len(new_state.active_columns) < 3
                )
            ):
                # Start from player's current position in the column if it exists
                player_pos = new_state.columns[sum2].player_positions.get(player_id, 0)
                current_pos = new_state.temp_positions.get(sum2, player_pos)
                new_state.temp_positions[sum2] = min(
                    current_pos + 1, new_state.columns[sum2].max_height
                )
                new_state.active_columns.add(sum2)

            # Switch to stop/roll decision and clear dice
            new_state.awaiting_selection = False
            new_state.current_dice = []

        elif move.action == "stop":
            # Convert temporary positions to permanent ones
            for col, pos in state.temp_positions.items():
                if col in new_state.columns:
                    new_state.columns[col].player_positions[player_id] = pos
                    # Check if column is claimed
                    if pos >= new_state.columns[col].max_height:
                        new_state.columns[col].is_claimed = True
                        new_state.columns[col].claimed_by = player_id

            # Reset for next player
            new_state = self._reset_for_next_player(new_state, 1 - player_id)

        elif move.action == "roll":
            # Roll new dice
            new_state.current_dice = [random.randint(1, 6) for _ in range(4)]

            # Check if the player busted
            if not self._has_valid_move(new_state):
                # Current player busts - lose all progress and switch players
                new_state.temp_positions.clear()
                new_state.active_columns.clear()
                new_state = self._reset_for_next_player(
                    new_state, 1 - new_state.current_player
                )
            else:
                new_state.awaiting_selection = True

        return new_state

    def get_current_player(self, state: CantStopState) -> int:
        """Return the ID of the player whose turn it is."""
        return state.current_player

    def get_player_view(
        self,
        state: CantStopState,
        player_id: int,
        history: Optional[List[Dict[str, Any]]] = None,
        prompt_style: PromptStyle = PromptStyle.HEADER,
    ) -> MatchView:
        """Return what this player can see of the current state."""
        visible_state = {
            "columns": {
                num: {
                    "max_height": col.max_height,
                    "claimed": col.is_claimed,
                    "claimed_by": col.claimed_by,
                    "your_position": col.player_positions.get(player_id, 0),
                    "opponent_position": col.player_positions.get(1 - player_id, 0),
                    "temp_progress_position": state.temp_positions.get(num, 0)
                    if player_id == state.current_player
                    else 0,
                }
                for num, col in state.columns.items()
            },
            "active_columns": list(state.active_columns),
            "your_turn": player_id == state.current_player,
            "current_dice": state.current_dice
            if player_id == state.current_player
            else None,
            "action_required": "select two column sums"
            if state.awaiting_selection
            else "decide to roll or stop",
        }

        # Check win condition
        winner = None
        is_terminal = False
        claimed_columns = sum(
            1
            for col in state.columns.values()
            if col.is_claimed and col.claimed_by == player_id
        )
        if claimed_columns >= 3:
            winner = player_id
            is_terminal = True

        return MatchView(
            rules_explanation=self.get_rules_explanation(),
            visible_state=visible_state,
            valid_moves=[],  # Dice-based game, so no fixed move list
            is_terminal=is_terminal,
            winner=winner,
            history=history if history else [],
            move_format_instructions=self.get_move_format_instructions(),
            prompt_style=prompt_style,
        )

    def get_move_format_instructions(self) -> str:
        return (
            "When action_required is 'select two column sums', you must choose two sums\n"
            "that can be created from the available dice using the format 'select X Y'\n"
            "When action_required is 'decide to roll or stop', respond with just 'roll' or 'stop'"
        )

    def get_rules_explanation(self) -> str:
        return (
            "Can't Stop is a push-your-luck dice game where players compete to claim three columns.\n\n"
            "OBJECTIVE:\n"
            "- Be the first player to claim three different columns by reaching their top spaces\n\n"
            "GAMEPLAY:\n"
            "1. On your turn, roll 4 dice and pair them to create two sums (2-12)\n"
            "   Example 1: With dice [3,4,2,5] you could make:\n"
            "   - (3+4=7) and (2+5=7) to advance twice in column 7\n"
            "   - (3+2=5) and (4+5=9) to advance in columns 5 and 9\n"
            "   - (3+5=8) and (2+4=6) to advance in columns 8 and 6\n\n"
            "   Example 2: With dice [6,6,1,4] you could make:\n"
            "   - (6+6=12) and (1+4=5) to advance in columns 12 and 5\n"
            "   - (6+1=7) and (6+4=10) to advance in columns 7 and 10\n\n"
            "2. Column Rules:\n"
            "   - You can work on up to 3 different columns per turn (shown as 'active_columns')\n"
            "   - Progress in columns is temporary until you choose to stop\n"
            "     (shown as 'temp_progress_position' - starts from your current progress)\n"
            "   - Your permanent position in each column is shown as 'your_position'\n"
            "   - Your opponent's position is shown as 'opponent_position'\n"
            "   - Each column has a different height (2:3, 7:13, 12:3 spaces)\n"
            "     (shown as 'max_height' for each column)\n\n"
            "3. After each advance, you must decide:\n"
            "   - STOP: Keep all progress made this turn (convert temp_progress to permanent progress)\n"
            "   - ROLL AGAIN: Risk everything for more progress\n\n"
            "4. Busting:\n"
            "   - If you can't use any dice combinations to advance in your active columns\n"
            "   - Or if no valid combinations exist for available columns\n"
            "   - You lose ALL progress made this turn (all temp_progress is lost)\n\n"
            "5. Winning:\n"
            "   - Claim a column by reaching its top space (when your_position = max_height)\n"
            "   - When a column is claimed, 'claimed' becomes True and 'claimed_by' shows who owns it\n"
            "   - First to claim ANY THREE columns wins the game\n\n"
            "GAME STATE EXPLANATION:\n"
            "- 'columns': Shows all columns (2-12) with their current status\n"
            "- 'active_columns': Columns you're currently working on this turn (max 3)\n"
            "- 'your_turn': Indicates if it's your turn to play\n"
            "- 'current_dice': The current dice values when selecting columns\n"
            "- 'action_required': What you need to do next ('select two column sums' or 'decide to roll or stop')"
        )

    def is_terminal(self, state: CantStopState) -> bool:
        return self.get_winner(state) is not None

    def get_winner(self, state: CantStopState) -> Optional[int]:
        for player_id in [0, 1]:
            claimed_columns = sum(
                1
                for col in state.columns.values()
                if col.is_claimed and col.claimed_by == player_id
            )
            if claimed_columns >= 3:
                return player_id
        return None

    def get_next_state(self, state: CantStopState, move: CantStopMove) -> CantStopState:
        """Return the next state after applying the move."""
        return self.apply_move(state, self.get_current_player(state), move)
        
    def serialize_state(self, state: CantStopState) -> Dict[str, Any]:
        """Serialize the game state into a JSON-compatible dictionary.

        This method ensures that all game-specific state is properly serialized
        into a format that can be stored in the database and later deserialized.

        Args:
            state: The CantStopState to serialize

        Returns:
            A JSON-compatible dictionary representing the game state
        """
        return {
            "columns": {str(k): v.to_dict() for k, v in state.columns.items()},
            "current_player": state.current_player,
            "temp_positions": {str(k): v for k, v in state.temp_positions.items()},
            "active_columns": sorted(list(state.active_columns)),
            "current_dice": state.current_dice,
            "awaiting_selection": state.awaiting_selection
        }
        
    def deserialize_state(self, state_data: Dict[str, Any]) -> CantStopState:
        """Deserialize state data into a CantStopState object.
        
        Args:
            state_data: Dictionary containing serialized state data from serialize_state
            
        Returns:
            Deserialized CantStopState object
        """
        # Reconstruct columns
        columns = {}
        for col_num, col_data in state_data["columns"].items():
            col_num = int(col_num)  # Convert string keys back to integers
            # Convert player_positions keys from strings to integers
            player_positions = {int(k): v for k, v in col_data["player_positions"].items()}
            columns[col_num] = ColumnState(
                player_positions=player_positions,
                max_height=col_data["max_height"],
                is_claimed=col_data["is_claimed"],
                claimed_by=col_data["claimed_by"]
            )
        
        # Reconstruct active columns as a set
        active_columns = set(state_data["active_columns"])
        
        # Convert temp_positions keys from strings to integers
        temp_positions = {int(k): v for k, v in state_data["temp_positions"].items()}
        
        # Return the reconstructed state
        return CantStopState(
            columns=columns,
            current_player=state_data["current_player"],
            temp_positions=temp_positions,
            active_columns=active_columns,
            current_dice=state_data["current_dice"],
            awaiting_selection=state_data["awaiting_selection"]
        )
