import copy
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Set
from bgbench.game import Game
from bgbench.match.view import MatchView, PromptStyle

# Constants for board representation
EMPTY = None
DARK = 0  # First player
LIGHT = 1  # Second player
BOARD_SIZE = 8
DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]

@dataclass
class ReversiMove:
    """Represents a move in Reversi."""
    row: Optional[int]  # None if passing
    col: Optional[int]  # None if passing
    
    def __str__(self) -> str:
        if self.row is None or self.col is None:
            return "PASS"
        return f"{chr(self.col + 97)}{self.row + 1}"  # Convert to algebraic notation (a1, b2, etc.)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert move to a dictionary for serialization."""
        return {
            "row": self.row,
            "col": self.col
        }

@dataclass
class ReversiState:
    """Represents the state of a Reversi game."""
    board: List[List[Optional[int]]] = field(
        default_factory=lambda: [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    )
    current_player: int = DARK  # Dark moves first
    pass_count: int = 0  # Number of consecutive passes
    move_history: List[Tuple[int, int, int]] = field(default_factory=list)  # [(player, row, col), ...]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to a dictionary for serialization."""
        return {
            "board": self.board,
            "current_player": self.current_player,
            "pass_count": self.pass_count,
            "move_history": self.move_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReversiState":
        """Create a state from a dictionary."""
        return cls(
            board=data["board"],
            current_player=data["current_player"],
            pass_count=data["pass_count"],
            move_history=data["move_history"]
        )


class ReversiGame(Game[ReversiState, ReversiMove]):
    """Implementation of Reversi (Othello) game."""
    
    def __init__(self):
        super().__init__()
    
    def get_initial_state(self) -> ReversiState:
        """Return the initial state of the game with the four starting pieces."""
        state = ReversiState()
        # Place the four starting pieces
        state.board[3][3] = LIGHT
        state.board[3][4] = DARK
        state.board[4][3] = DARK
        state.board[4][4] = LIGHT
        return state
    
    def get_player_view(
        self,
        state: ReversiState,
        player_id: int,
        history: Optional[List[Dict[str, Any]]] = None,
        prompt_style: PromptStyle = PromptStyle.HEADER,
    ) -> MatchView:
        """Return what this player can see of the current state."""
        # In Reversi, both players can see the entire board
        valid_moves = self.get_valid_moves(state, player_id)
        
        # Create a string representation of the board
        board_str = self._board_to_string(state.board)
        
        # Count discs
        dark_count, light_count = self._count_discs(state.board)
        
        # Create a dictionary with all the information for the player
        visible_state = {
            "board": board_str,
            "your_color": "Dark" if player_id == DARK else "Light",
            "current_player": "Dark" if state.current_player == DARK else "Light",
            "dark_discs": dark_count,
            "light_discs": light_count,
        }
        
        # Create the match view
        valid_moves_list = [str(ReversiMove(row, col)) for row, col in valid_moves] if player_id == state.current_player else []
        return MatchView(
            visible_state=visible_state,
            valid_moves=valid_moves_list, # Keep for internal use if needed
            is_terminal=self.is_terminal(state),
            winner=self.get_winner(state),
            history=history if history is not None else [],
            move_format_instructions=self.get_move_format_instructions(),
            rules_explanation=self.get_rules_explanation(),
            prompt_style=prompt_style
        )
    
    def parse_move(self, move_str: str) -> Optional[ReversiMove]:
        """Parse move from LLM response string."""
        move_str = move_str.strip().lower()
        
        # Handle pass move
        if move_str == "pass":
            return ReversiMove(None, None)
        
        # Parse algebraic notation (e.g., "e4")
        if len(move_str) == 2 and 'a' <= move_str[0] <= 'h' and '1' <= move_str[1] <= '8':
            col = ord(move_str[0]) - ord('a')
            row = int(move_str[1]) - 1
            return ReversiMove(row, col)
        
        return None
    
    def validate_move(
        self, state: ReversiState, player_id: int, move: ReversiMove
    ) -> Tuple[bool, str]:
        """Validate if a move is legal in the current state."""
        # Check if it's the player's turn
        if player_id != state.current_player:
            return False, "It's not your turn."
        
        # Handle pass move
        if move.row is None or move.col is None:
            # Pass is only valid if there are no valid moves
            valid_moves = self.get_valid_moves(state, player_id)
            if valid_moves:
                return False, "You have valid moves available. You cannot pass."
            return True, "Pass is valid as there are no valid moves."
        
        # Check if the position is on the board
        if not (0 <= move.row < BOARD_SIZE and 0 <= move.col < BOARD_SIZE):
            return False, "Position is outside the board."
        
        # Check if the position is empty
        if state.board[move.row][move.col] is not EMPTY:
            return False, "Position is already occupied."
        
        # Check if the move would flip any opponent's discs
        if not self._would_flip_any(state.board, move.row, move.col, player_id):
            return False, "Move must flip at least one opponent's disc."
        
        return True, "Valid move."
    
    def get_current_player(self, state: ReversiState) -> int:
        """Return the ID of the player whose turn it is."""
        return state.current_player
    
    def apply_move(self, state: ReversiState, player_id: int, move: ReversiMove) -> ReversiState:
        """Apply move to state and return new state."""
        # Validate the move
        is_valid, explanation = self.validate_move(state, player_id, move)
        if not is_valid:
            raise ValueError(f"Invalid move: {explanation}")
        
        # Create a new state
        new_state = copy.deepcopy(state)
        
        # Handle pass move
        if move.row is None or move.col is None:
            new_state.pass_count += 1
            new_state.move_history.append((player_id, -1, -1))  # -1, -1 represents a pass
        else:
            # Place the disc and flip opponent's discs
            new_state.board[move.row][move.col] = player_id
            self._flip_discs(new_state.board, move.row, move.col, player_id)
            new_state.pass_count = 0
            new_state.move_history.append((player_id, move.row, move.col))
        
        # Determine the next player
        opponent_id = 1 - player_id
        new_state.current_player = opponent_id

        # Check if the opponent (now current player) has valid moves
        opponent_has_moves = bool(self.get_valid_moves(new_state, new_state.current_player))

        if not opponent_has_moves:
            # Opponent must pass, increment pass count
            new_state.pass_count += 1
            # Add a pass marker to history for the opponent
            new_state.move_history.append((opponent_id, -1, -1))
            # Switch turn back to the original player
            new_state.current_player = player_id

            # Check if the original player also has no moves now
            original_player_has_moves = bool(self.get_valid_moves(new_state, new_state.current_player))
            if not original_player_has_moves:
                # If original player also has no moves, they must pass too.
                # Increment pass count again. The game ends.
                # Add a pass marker to history for the original player
                # Check if the last move wasn't already a pass for this player to avoid double counting passes
                # when a player explicitly passes and the opponent also has no moves.
                if not (move.row is None or move.col is None):
                    new_state.pass_count += 1
                    new_state.move_history.append((player_id, -1, -1))
                # The current player doesn't strictly matter now, but leave it as the original player
        # else: # Opponent has moves
            # If the original move was not a pass, the pass count was reset earlier.
            # If the original move was an explicit pass, pass_count is already 1.
            # Current player remains opponent_id.

        return new_state

    def is_terminal(self, state: ReversiState) -> bool:
        """Return True if the game has ended."""
        # Condition 1: The board is full.
        is_board_full = all(
            state.board[r][c] is not EMPTY 
            for r in range(BOARD_SIZE) 
            for c in range(BOARD_SIZE)
        )
        if is_board_full:
            return True
            
        # Condition 2: Neither player has any valid moves.
        # Note: The pass_count is updated by apply_move when passes occur.
        # The fundamental condition for the game ending (besides a full board)
        # is that no player can make a move.
        dark_has_moves = bool(self.get_valid_moves(state, DARK))
        light_has_moves = bool(self.get_valid_moves(state, LIGHT))
        
        if not dark_has_moves and not light_has_moves:
            return True
            
        return False

    def get_winner(self, state: ReversiState) -> Optional[int]:
        """Return the ID of the winner if the game has ended, otherwise None."""
        if not self.is_terminal(state):
            return None
        
        # Count discs
        dark_count, light_count = self._count_discs(state.board)
        
        if dark_count > light_count:
            return DARK
        elif light_count > dark_count:
            return LIGHT
        else:
            return None  # Draw
    
    def get_next_state(self, state: ReversiState, move: ReversiMove) -> ReversiState:
        """Return the next state after applying the move."""
        return self.apply_move(state, state.current_player, move)
    
    def serialize_state(self, state: ReversiState) -> Dict[str, Any]:
        """Serialize the game state into a JSON-compatible dictionary."""
        return state.to_dict()
    
    def deserialize_state(self, state_data: Dict[str, Any]) -> ReversiState:
        """Deserialize a dictionary into a game state object."""
        return ReversiState.from_dict(state_data)
    
    def get_rules_explanation(self) -> str:
        """Return a string explaining the rules of Reversi."""
        return """
Reversi (also known as Othello) is a strategy board game played on an 8×8 grid.

Rules:
1. The game starts with 4 discs placed in the center: 2 dark and 2 light.
2. Dark moves first.
3. Players take turns placing one disc of their color on the board.
4. A valid move must outflank at least one opponent's disc.
5. Outflanking means placing a disc such that one or more opponent's discs are in a straight line (horizontal, vertical, or diagonal) between the newly placed disc and another disc of your color.
6. After placing a disc, all outflanked opponent discs are flipped to your color.
7. If a player cannot make a valid move, they must pass.
8. The game ends when the board is full or neither player can make a valid move.
9. The player with the most discs of their color on the board wins.

Board Representation:
- 'X' represents Dark discs.
- 'O' represents Light discs.
- '-' represents an empty square.
"""
    
    def get_move_format_instructions(self) -> str:
        """Return instructions for formatting moves."""
        return """
To make a move, specify the position using algebraic notation:
- A letter (a-h) for the column
- A number (1-8) for the row
- Example: "e4" places a disc at column e, row 4

If you have no valid moves, the game will automatically pass for you.
"""
    
    # Helper methods
    
    def get_valid_moves(self, state: ReversiState, player_id: int) -> Set[Tuple[int, int]]:
        """Return a set of valid move positions for the player."""
        valid_moves = set()
        
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if state.board[row][col] is EMPTY and self._would_flip_any(state.board, row, col, player_id):
                    valid_moves.add((row, col))
        
        return valid_moves
    
    def _would_flip_any(self, board: List[List[Optional[int]]], row: int, col: int, player_id: int) -> bool:
        """Check if placing a disc at (row, col) would flip any opponent's discs."""
        opponent = 1 - player_id
        
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            found_opponent_in_direction = False 
            # Check if we have at least one opponent's disc in this direction
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == opponent:
                found_opponent_in_direction = True 
                r += dr
                c += dc
                # Continue in this direction
                while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                    current_piece = board[r][c] 
                    if current_piece is EMPTY:
                        break
                    if current_piece == player_id:
                        # Only return true if we found at least one opponent disc
                        # and then found our own disc in this direction
                        if found_opponent_in_direction: 
                           return True  # Found a disc to flip
                        else: 
                           break # Found own piece without opponent in between
                    # Continue in this direction (implicitly opponent piece)
                    r += dr
                    c += dc
                # Reached end of board or broke loop for this direction

        return False
    
    def _get_flipped_positions(self, board: List[List[Optional[int]]], row: int, col: int, player_id: int) -> List[Tuple[int, int]]:
        """Return a list of positions that would be flipped by placing a disc at (row, col)."""
        opponent = 1 - player_id
        flipped = []
        
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            to_flip = []
            
            # Check if we have at least one opponent's disc in this direction
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == opponent:
                to_flip.append((r, c))
                r += dr
                c += dc
                
                # Continue in this direction
                while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                    if board[r][c] is EMPTY:
                        to_flip = []  # Reset if we hit an empty space
                        break
                    if board[r][c] == player_id:
                        flipped.extend(to_flip)  # Add all discs to flip
                        break
                    to_flip.append((r, c))
                    r += dr
                    c += dc
        
        return flipped
    
    def _flip_discs(self, board: List[List[Optional[int]]], row: int, col: int, player_id: int) -> None:
        """Flip all outflanked discs after placing a disc at (row, col)."""
        for r, c in self._get_flipped_positions(board, row, col, player_id):
            board[r][c] = player_id
    
    def _count_discs(self, board: List[List[Optional[int]]]) -> Tuple[int, int]:
        """Count the number of dark and light discs on the board."""
        dark_count = sum(1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r][c] == DARK)
        light_count = sum(1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r][c] == LIGHT)
        return dark_count, light_count
    
    def _board_to_string(self, board: List[List[Optional[int]]]) -> str:
        """Convert the board to a string representation."""
        result = "  a b c d e f g h\n"
        for row in range(BOARD_SIZE):
            result += f"{row + 1} "
            for col in range(BOARD_SIZE):
                if board[row][col] is EMPTY:
                    result += "- "
                elif board[row][col] == DARK:
                    result += "X "  # Dark disc
                else:
                    result += "O "  # Light disc
            result += "\n"
        return result
