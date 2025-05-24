import random
import copy
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

from bgbench.game import Game
from bgbench.match.view import MatchView, PromptStyle

# Constants for board representation
HOME_BOARD_SIZE = 6
TOTAL_POINTS = 24
CHECKERS_PER_PLAYER = 15

# Player constants
WHITE = 0  # Moves clockwise (from 24 to 1)
BLACK = 1  # Moves counter-clockwise (from 1 to 24)

@dataclass
class BackgammonState:
    """Represents the state of a backgammon game.
    
    The board is represented as a list of 24 points, where:
    - Positive values represent WHITE checkers (count)
    - Negative values represent BLACK checkers (count)
    - 0 represents an empty point
    
    Points are indexed from 1 to 24, with index 0 unused to align with standard numbering.
    """
    # The board representation: points 1-24 (using index 0 as a dummy to align with standard numbering)
    points: List[int] = field(default_factory=lambda: [0] * (TOTAL_POINTS + 1))
    
    # Checkers on the bar (hit pieces)
    bar: List[int] = field(default_factory=lambda: [0, 0])  # [WHITE count, BLACK count]
    
    # Checkers that have been borne off (removed from the board)
    borne_off: List[int] = field(default_factory=lambda: [0, 0])  # [WHITE count, BLACK count]
    
    # Current player (WHITE = 0, BLACK = 1)
    current_player: int = WHITE
    
    # Dice rolled
    dice: List[int] = field(default_factory=list)
    
    # Flag to determine if a double was rolled
    is_double: bool = False
    
    # Dice already used in the current turn
    used_dice: List[int] = field(default_factory=list)
    
    # Move history for the entire game
    move_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Current doubling cube value (1, 2, 4, 8, 16, 32, 64)
    cube_value: int = 1
    
    # Player who has the doubling cube (-1 if centered, 0 for WHITE, 1 for BLACK)
    cube_owner: int = -1
    
    def roll_dice(self) -> List[int]:
        """Roll two dice and update the state."""
        die1 = random.randint(1, 6)
        die2 = random.randint(1, 6)
        
        self.dice = [die1, die2]
        self.is_double = die1 == die2
        
        # If doubles, each die can be used twice
        if self.is_double:
            self.dice = [die1, die1, die1, die1]
        
        self.used_dice = []
        return self.dice
    
    def get_player_points(self, player: int) -> Dict[int, int]:
        """Return a dict mapping point indices to checker counts for a player."""
        result = {}
        for i in range(1, TOTAL_POINTS + 1):
            count = self.points[i]
            # For WHITE, positive values represent checkers
            # For BLACK, negative values represent checkers
            if (player == WHITE and count > 0) or (player == BLACK and count < 0):
                result[i] = abs(count)
        return result
    
    def is_point_open(self, point: int, player: int) -> bool:
        """Check if a point is open for a player to land on."""
        # Points outside the board range are not valid
        if point < 1 or point > TOTAL_POINTS:
            return False
        
        # For WHITE, a point is blocked if it has 2+ BLACK checkers (negative value)
        # For BLACK, a point is blocked if it has 2+ WHITE checkers (positive value)
        if player == WHITE:
            return self.points[point] >= -1  # Open if empty or has 1 or less BLACK checkers
        else:
            return self.points[point] <= 1   # Open if empty or has 1 or less WHITE checkers
    
    def can_bear_off(self, player: int) -> bool:
        """Check if a player can bear off checkers."""
        # A player can bear off if all their checkers are in their home board or have been borne off
        
        # First, check if player has checkers on the bar - must re-enter them first
        if self.bar[player] > 0:
            return False
        
        # Define home board points for each player
        home_board_points = range(1, HOME_BOARD_SIZE + 1) if player == BLACK else range(TOTAL_POINTS - HOME_BOARD_SIZE + 1, TOTAL_POINTS + 1)
        
        # Check if all checkers are in home board or already borne off
        for point in range(1, TOTAL_POINTS + 1):
            # Skip points in home board
            if point in home_board_points:
                continue
                
            # Check if there are any checkers outside home board
            if player == WHITE and self.points[point] > 0:
                return False
            if player == BLACK and self.points[point] < 0:
                return False
        
        return True
    
    def is_game_over(self) -> bool:
        """Check if the game is over (a player has borne off all checkers)."""
        return self.borne_off[WHITE] == CHECKERS_PER_PLAYER or self.borne_off[BLACK] == CHECKERS_PER_PLAYER
    
    def get_winner(self) -> Optional[int]:
        """Return the winner of the game, or None if the game is not over."""
        if not self.is_game_over():
            return None
        
        return WHITE if self.borne_off[WHITE] == CHECKERS_PER_PLAYER else BLACK
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to a dictionary for serialization."""
        return {
            "points": self.points,
            "bar": self.bar,
            "borne_off": self.borne_off,
            "current_player": self.current_player,
            "dice": self.dice,
            "is_double": self.is_double,
            "used_dice": self.used_dice,
            "move_history": self.move_history,
            "cube_value": self.cube_value,
            "cube_owner": self.cube_owner,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackgammonState":
        """Create a state from a dictionary."""
        state = cls()
        state.points = data.get("points", [0] * (TOTAL_POINTS + 1))
        state.bar = data.get("bar", [0, 0])
        state.borne_off = data.get("borne_off", [0, 0])
        state.current_player = data.get("current_player", WHITE)
        state.dice = data.get("dice", [])
        state.is_double = data.get("is_double", False)
        state.used_dice = data.get("used_dice", [])
        state.move_history = data.get("move_history", [])
        state.cube_value = data.get("cube_value", 1)
        state.cube_owner = data.get("cube_owner", -1)
        return state


@dataclass
class BackgammonMove:
    """Represents a move in backgammon."""
    # From point (1-24, or 0 for bar)
    from_point: int
    
    # To point (1-24, or 0 for bearing off)
    to_point: int
    
    # Die used for this move
    die: int
    
    # True if this is a double move
    is_double_move: bool = False
    
    def __str__(self) -> str:
        """String representation of the move."""
        from_str = "bar" if self.from_point == 0 else str(self.from_point)
        to_str = "off" if self.to_point == 0 else str(self.to_point)
        return f"{from_str}-{to_str}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert move to a dictionary for serialization."""
        return {
            "from_point": self.from_point,
            "to_point": self.to_point,
            "die": self.die,
            "is_double_move": self.is_double_move,
        }


class BackgammonGame(Game[BackgammonState, BackgammonMove]):
    """Implementation of backgammon."""
    
    def __init__(self):
        super().__init__()
    
    def get_initial_state(self) -> BackgammonState:
        """Return the initial state of the game with the starting checker positions."""
        state = BackgammonState()
        
        # Set up the initial board position
        # WHITE: 2 on point 24, 5 on point 13, 3 on point 8, 5 on point 6
        # BLACK: 2 on point 1, 5 on point 12, 3 on point 17, 5 on point 19
        
        # WHITE checkers (positive values)
        state.points[24] = 2  # 2 WHITE checkers on point 24
        state.points[13] = 5  # 5 WHITE checkers on point 13
        state.points[8] = 3   # 3 WHITE checkers on point 8
        state.points[6] = 5   # 5 WHITE checkers on point 6
        
        # BLACK checkers (negative values)
        state.points[1] = -2  # 2 BLACK checkers on point 1
        state.points[12] = -5 # 5 BLACK checkers on point 12
        state.points[17] = -3 # 3 BLACK checkers on point 17
        state.points[19] = -5 # 5 BLACK checkers on point 19
        
        # Roll initial dice to determine first player
        die1 = random.randint(1, 6)
        die2 = random.randint(1, 6)
        
        if die1 > die2:
            state.current_player = WHITE
            state.dice = [die1, die2]
        elif die2 > die1:
            state.current_player = BLACK
            state.dice = [die1, die2]
        else:
            # If tie, re-roll until different values are obtained
            while die1 == die2:
                die1 = random.randint(1, 6)
                die2 = random.randint(1, 6)
            
            state.current_player = WHITE if die1 > die2 else BLACK
            state.dice = [die1, die2]
        
        state.is_double = False  # Initial roll cannot be a double for gameplay
        
        return state
    
    def parse_move(self, move_str: str) -> Optional[BackgammonMove]:
        """Parse move from LLM response string."""
        move_str = move_str.strip().lower()
        
        # Handle special case for doubles
        if "double" in move_str:
            return None  # Doubling is handled separately
        
        # Match move patterns like "bar-24" or "13-7" or "6-off"
        try:
            parts = move_str.split("-")
            if len(parts) != 2:
                return None
            
            # Parse from_point
            if parts[0] == "bar":
                from_point = 0
            else:
                from_point = int(parts[0])
            
            # Parse to_point
            if parts[1] == "off":
                to_point = 0
            else:
                to_point = int(parts[1])
            
            # For now, die value needs to be inferred by the game logic since it's not in the notation
            # We'll set it to None and fill it in during move validation
            return BackgammonMove(from_point=from_point, to_point=to_point, die=0)
        except (ValueError, IndexError):
            return None
    
    def validate_move(self, state: BackgammonState, player_id: int, move: BackgammonMove) -> Tuple[bool, str]:
        """Validate if a move is legal in the current state."""
        # Check if it's the player's turn
        if player_id != state.current_player:
            return False, "It's not your turn."

        # Check if there are dice to use
        if not state.dice:
            return False, "You must roll the dice first."

        # Get from and to points
        from_point = move.from_point
        to_point = move.to_point

        # When a die is provided in the move, use it directly for validation
        if move.die > 0 and move.die in state.dice:
            die_value = move.die
        else:
            # Calculate the expected die value based on the move
            if player_id == WHITE:
                # WHITE moves from higher to lower points
                die_value = from_point - to_point if to_point > 0 else from_point
            else:  # BLACK
                # BLACK moves from lower to higher points
                die_value = to_point - from_point if to_point > 0 else 25 - from_point

            # Check if the calculated die value matches an available die
            if die_value not in state.dice:
                return False, f"No die with value {die_value} available."

            # Update the move with the correct die value
            move.die = die_value
        
        # Rules for moving from the bar
        if state.bar[player_id] > 0:
            # Must move from bar first
            if from_point != 0:
                return False, "You must move checkers from the bar first."

            # For WHITE, entering from bar means moving to 25-die
            # For BLACK, entering from bar means moving to die
            entry_point = 25 - move.die if player_id == WHITE else move.die

            if to_point != entry_point:
                return False, f"From bar, you must move to point {entry_point}."

            # Check if the entry point is open
            if not state.is_point_open(entry_point, player_id):
                return False, f"Point {entry_point} is blocked."

            return True, "Valid move from bar."
        
        # Regular moves (not from bar)
        
        # Check that from_point has player's checkers
        if player_id == WHITE and (from_point == 0 or state.points[from_point] <= 0):
            return False, f"No WHITE checkers on point {from_point}."
        if player_id == BLACK and (from_point == 0 or state.points[from_point] >= 0):
            return False, f"No BLACK checkers on point {from_point}."
        
        # Bearing off
        if to_point == 0:
            # Check if player can bear off
            if not state.can_bear_off(player_id):
                return False, "All checkers must be in your home board to bear off."

            # For WHITE, home board is points 19-24
            # For BLACK, home board is points 1-6
            if player_id == WHITE:
                if from_point < 19 or from_point > 24:
                    return False, "Bearing off is only allowed from points 19-24 for WHITE."

                highest_point = max([p for p in range(19, 25) if state.points[p] > 0], default=0)
                if from_point < highest_point and move.die > from_point:
                    return False, "You must move the furthest checker when bearing off with a higher roll."

                # Check if die value matches distance
                if move.die != from_point and move.die > from_point:
                    return False, f"Cannot bear off from point {from_point} with die {move.die}."
            else:  # BLACK
                if from_point < 1 or from_point > 6:
                    return False, "Bearing off is only allowed from points 1-6 for BLACK."

                lowest_point = min([p for p in range(1, 7) if state.points[p] < 0], default=7)
                if from_point > lowest_point and move.die > (7 - from_point):
                    return False, "You must move the furthest checker when bearing off with a higher roll."

                # Check if die value matches distance
                if move.die != 7 - from_point and move.die > 7 - from_point:
                    return False, f"Cannot bear off from point {from_point} with die {move.die}."

            return True, "Valid bearing off move."
        
        # Regular moves to another point
        
        # Check if the target point is open
        if not state.is_point_open(to_point, player_id):
            return False, f"Point {to_point} is blocked."
        
        # All checks passed
        return True, "Valid move."
    
    def apply_move(self, state: BackgammonState, player_id: int, move: BackgammonMove) -> BackgammonState:
        """Apply move to state and return new state."""
        # Validate the move
        is_valid, explanation = self.validate_move(state, player_id, move)
        if not is_valid:
            raise ValueError(f"Invalid move: {explanation}")
        
        # Create a new state
        new_state = copy.deepcopy(state)
        
        # Update the board based on the move
        from_point = move.from_point
        to_point = move.to_point
        
        # Moving from the bar
        if from_point == 0:
            new_state.bar[player_id] -= 1
            
            if to_point > 0:
                # Check if hitting an opponent's blot
                if player_id == WHITE and new_state.points[to_point] == -1:
                    # Hit a single BLACK checker
                    new_state.points[to_point] = 1
                    new_state.bar[BLACK] += 1
                elif player_id == BLACK and new_state.points[to_point] == 1:
                    # Hit a single WHITE checker
                    new_state.points[to_point] = -1
                    new_state.bar[WHITE] += 1
                else:
                    # Add checker to the point
                    if player_id == WHITE:
                        new_state.points[to_point] += 1
                    else:
                        new_state.points[to_point] -= 1
        
        # Bearing off
        elif to_point == 0:
            # Remove checker from board
            if player_id == WHITE:
                new_state.points[from_point] -= 1
            else:
                new_state.points[from_point] += 1

            # Add to borne off count
            new_state.borne_off[player_id] += 1

            # For debugging
            # print(f"Bearing off: {player_id} from {from_point}, borne off now: {new_state.borne_off[player_id]}")
        
        # Regular move
        else:
            # Remove checker from starting point
            if player_id == WHITE:
                new_state.points[from_point] -= 1
            else:
                new_state.points[from_point] += 1
            
            # Check if hitting an opponent's blot
            if player_id == WHITE and new_state.points[to_point] == -1:
                # Hit a single BLACK checker
                new_state.points[to_point] = 1
                new_state.bar[BLACK] += 1
            elif player_id == BLACK and new_state.points[to_point] == 1:
                # Hit a single WHITE checker
                new_state.points[to_point] = 1
                new_state.bar[WHITE] += 1
            else:
                # Add checker to the target point
                if player_id == WHITE:
                    new_state.points[to_point] += 1
                else:
                    new_state.points[to_point] -= 1
        
        # Mark the die as used
        new_state.used_dice.append(move.die)
        new_state.dice.remove(move.die)
        
        # Add move to history
        new_state.move_history.append(move.to_dict())
        
        # If no more dice, switch player and roll new dice
        if not new_state.dice:
            new_state.current_player = 1 - player_id
            new_state.dice = []
            new_state.used_dice = []
            # The next player will roll dice when their turn begins
        
        return new_state
    
    def get_player_view(
        self,
        state: BackgammonState,
        player_id: int,
        history: Optional[List[Dict[str, Any]]] = None,
        prompt_style: PromptStyle = PromptStyle.HEADER,
    ) -> MatchView:
        """Return what this player can see of the current state."""
        # In backgammon, both players can see the entire board
        
        # Create a board representation for display
        board_str = self._board_to_string(state, player_id)
        
        # Get dice information
        dice_str = f"Dice: {state.dice}" if state.dice else "No dice rolled yet"
        
        # Create a dictionary with all the information for the player
        visible_state = {
            "board": board_str,
            "your_color": "White" if player_id == WHITE else "Black",
            "current_player": "White" if state.current_player == WHITE else "Black",
            "dice": dice_str,
            "your_checkers_on_bar": state.bar[player_id],
            "opponent_checkers_on_bar": state.bar[1 - player_id],
            "your_checkers_borne_off": state.borne_off[player_id],
            "opponent_checkers_borne_off": state.borne_off[1 - player_id],
            "doubling_cube": state.cube_value,
            "cube_owner": "Centered" if state.cube_owner == -1 else (
                "White" if state.cube_owner == WHITE else "Black"
            ),
        }
        
        # Get the valid moves
        valid_moves = self._get_valid_moves_str(state, player_id)
        
        # Create the match view
        return MatchView(
            visible_state=visible_state,
            valid_moves=valid_moves,
            is_terminal=state.is_game_over(),
            winner=state.get_winner(),
            history=history if history is not None else [],
            move_format_instructions=self.get_move_format_instructions(),
            rules_explanation=self.get_rules_explanation(),
            prompt_style=prompt_style,
        )
    
    def get_move_format_instructions(self) -> str:
        """Return instructions for formatting moves."""
        return """
To make a move, specify the starting point and ending point using the following format:
- For regular moves: "from-to" (e.g., "13-7" moves a checker from point 13 to point 7)
- For moves from the bar: "bar-to" (e.g., "bar-19")
- For bearing off: "from-off" (e.g., "6-off")

You must play as many dice as possible. If you can't use all dice, you must use the higher die if possible.
"""
    
    def get_rules_explanation(self) -> str:
        """Return a string explaining the rules of backgammon."""
        return """
Backgammon is a two-player board game played on a board with 24 triangular points.

Setup:
- Each player has 15 checkers.
- White moves clockwise from points 24 to 1.
- Black moves counterclockwise from points 1 to 24.
- Initial setup: White has 2 on 24-point, 5 on 13-point, 3 on 8-point, 5 on 6-point.
  Black has 2 on 1-point, 5 on 12-point, 3 on 17-point, 5 on 19-point.

Game Flow:
1. Players roll dice to determine moves.
2. A player must move checkers according to the dice values.
3. Checkers move to lower-numbered points for White, higher-numbered points for Black.
4. Players can only move to open points (empty or with only one opponent checker).
5. If a point has 2+ opponent checkers, it's blocked.
6. When landing on a point with one opponent checker, that checker is "hit" and placed on the bar.
7. A player with checkers on the bar must re-enter them before making other moves.
   - White re-enters from the 25 - die value point
   - Black re-enters from the die value point
8. When all of a player's checkers are in their home board, they can "bear off" (remove checkers).
9. The first player to bear off all 15 checkers wins.

Board Representation:
- Positive numbers represent White checkers.
- Negative numbers represent Black checkers.
- The bar holds hit checkers that must re-enter.
- Home boards: points 1-6 for Black, points 19-24 for White.
"""
    
    def get_current_player(self, state: BackgammonState) -> int:
        """Return the ID of the player whose turn it is."""
        return state.current_player
    
    def is_terminal(self, state: BackgammonState) -> bool:
        """Return True if the game has ended."""
        return state.is_game_over()
    
    def get_winner(self, state: BackgammonState) -> Optional[int]:
        """Return the ID of the winner if the game has ended, otherwise None."""
        return state.get_winner()
    
    def get_next_state(self, state: BackgammonState, move: BackgammonMove) -> BackgammonState:
        """Return the next state after applying the move."""
        return self.apply_move(state, state.current_player, move)
    
    def serialize_state(self, state: BackgammonState) -> Dict[str, Any]:
        """Serialize the game state into a JSON-compatible dictionary."""
        return state.to_dict()
    
    def deserialize_state(self, state_data: Dict[str, Any]) -> BackgammonState:
        """Deserialize a dictionary into a game state object."""
        return BackgammonState.from_dict(state_data)
    
    # Helper methods
    
    def _board_to_string(self, state: BackgammonState, player_id: int) -> str:
        """Convert the board to a string representation."""
        result = "   13  14  15  16  17  18    |    19  20  21  22  23  24\n"
        result += "  +--------------------------------------------------+\n"
        
        # Top row (points 13-18 and 19-24)
        top_row = "  |"
        for point in range(13, 19):
            checkers = state.points[point]
            if checkers > 0:
                top_row += f" W{checkers:1} " if checkers < 10 else f"W{checkers:2} "
            elif checkers < 0:
                top_row += f" B{abs(checkers):1} " if abs(checkers) < 10 else f"B{abs(checkers):2} "
            else:
                top_row += "    "
        
        top_row += "|"
        
        for point in range(19, 25):
            checkers = state.points[point]
            if checkers > 0:
                top_row += f" W{checkers:1} " if checkers < 10 else f"W{checkers:2} "
            elif checkers < 0:
                top_row += f" B{abs(checkers):1} " if abs(checkers) < 10 else f"B{abs(checkers):2} "
            else:
                top_row += "    "
        
        top_row += "|\n"
        result += top_row
        
        # Middle section with bar
        bar_str = f"BAR: White: {state.bar[WHITE]}, Black: {state.bar[BLACK]}"
        off_str = f"OFF: White: {state.borne_off[WHITE]}, Black: {state.borne_off[BLACK]}"
        
        result += f"  |{bar_str.center(49)}|\n"
        result += f"  |{off_str.center(49)}|\n"
        
        # Bottom row (points 12-7 and 6-1)
        bottom_row = "  |"
        for point in range(12, 6, -1):
            checkers = state.points[point]
            if checkers > 0:
                bottom_row += f" W{checkers:1} " if checkers < 10 else f"W{checkers:2} "
            elif checkers < 0:
                bottom_row += f" B{abs(checkers):1} " if abs(checkers) < 10 else f"B{abs(checkers):2} "
            else:
                bottom_row += "    "
        
        bottom_row += "|"
        
        for point in range(6, 0, -1):
            checkers = state.points[point]
            if checkers > 0:
                bottom_row += f" W{checkers:1} " if checkers < 10 else f"W{checkers:2} "
            elif checkers < 0:
                bottom_row += f" B{abs(checkers):1} " if abs(checkers) < 10 else f"B{abs(checkers):2} "
            else:
                bottom_row += "    "
        
        bottom_row += "|\n"
        result += bottom_row
        
        result += "  +--------------------------------------------------+\n"
        result += "    12  11  10   9   8   7    |     6   5   4   3   2   1\n"
        
        return result
    
    def _get_valid_moves_str(self, state: BackgammonState, player_id: int) -> List[str]:
        """Get a list of valid moves as strings."""
        valid_moves = []
        
        # If not the player's turn or no dice, no valid moves
        if player_id != state.current_player or not state.dice:
            return valid_moves
        
        # Check if player has checkers on the bar
        if state.bar[player_id] > 0:
            # Player must enter checkers from the bar first
            for die in state.dice:
                entry_point = 25 - die if player_id == WHITE else die
                if state.is_point_open(entry_point, player_id):
                    valid_moves.append(f"bar-{entry_point}")
            return valid_moves
        
        # Regular moves
        for point in range(1, TOTAL_POINTS + 1):
            # Check if the point has player's checkers
            has_checkers = (player_id == WHITE and state.points[point] > 0) or (
                player_id == BLACK and state.points[point] < 0
            )
            
            if has_checkers:
                for die in state.dice:
                    # Calculate target point
                    if player_id == WHITE:
                        target = point - die
                    else:
                        target = point + die
                    
                    # Bearing off
                    if state.can_bear_off(player_id):
                        if (player_id == WHITE and target <= 0) or (player_id == BLACK and target > TOTAL_POINTS):
                            # Check bearing off rules
                            if player_id == WHITE:
                                highest_point = max([p for p in range(19, 25) if state.points[p] > 0], default=0)
                                if point == highest_point or die == point:
                                    valid_moves.append(f"{point}-off")
                            else:  # BLACK
                                lowest_point = min([p for p in range(1, 7) if state.points[p] < 0], default=7)
                                if point == lowest_point or die == (TOTAL_POINTS + 1 - point):
                                    valid_moves.append(f"{point}-off")
                    
                    # Regular move to another point
                    if 1 <= target <= TOTAL_POINTS and state.is_point_open(target, player_id):
                        valid_moves.append(f"{point}-{target}")
        
        return valid_moves