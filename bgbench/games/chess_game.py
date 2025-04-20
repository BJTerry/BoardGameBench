from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from bgbench.game import Game
from bgbench.match.view import MatchView, PromptStyle
import chess
import chess.pgn


@dataclass
class ChessState:
    """Represents the current state of a chess game."""

    board: chess.Board
    move_history: List[str]  # List of moves in PGN format

    def to_dict(self) -> dict:
        return {
            "fen": self.board.fen(),
            "pgn": " ".join(self.move_history),
            "is_check": self.board.is_check(),
            "is_checkmate": self.board.is_checkmate(),
            "is_stalemate": self.board.is_stalemate(),
            "is_insufficient_material": self.board.is_insufficient_material(),
            "turn": "white" if self.board.turn else "black",
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChessState":
        """Create a ChessState from a dictionary.
        
        Args:
            data: Dictionary containing state data
            
        Returns:
            ChessState object
        """
        # Extract move history
        pgn_str = data.get("pgn", "")
        move_history = pgn_str.split() if pgn_str else []
        
        # Create a new board from the starting position
        board = chess.Board()
        
        # Replay the move history to get to the correct position
        for move_san in move_history:
            try:
                move = board.parse_san(move_san)
                board.push(move)
            except ValueError:
                # If we can't parse a move, fall back to using the FEN
                if "fen" in data:
                    board = chess.Board(data["fen"])
                break
        
        # Return a new ChessState
        return cls(board=board, move_history=move_history)


@dataclass
class ChessMove:
    """Represents a move in chess."""

    move_san: str  # Move in Standard Algebraic Notation (PGN format)

    def to_dict(self) -> dict:
        return {"move": self.move_san}


class ChessGame(Game[ChessState, ChessMove]):
    """Implementation of chess following FIDE rules."""

    def __init__(self):
        self.initial_board = chess.Board()

    def get_initial_state(self) -> ChessState:
        """Return the initial state of the game."""
        return ChessState(board=chess.Board(), move_history=[])

    def parse_move(self, move_str: str) -> Optional[ChessMove]:
        """Parse a move string in PGN format."""
        move_str = move_str.strip()
        if not move_str:
            return None

        # Normalize castling notation
        if move_str.lower() in ["o-o", "0-0"]:
            move_str = "O-O"  # Kingside castling
        elif move_str.lower() in ["o-o-o", "0-0-0"]:
            move_str = "O-O-O"  # Queenside castling

        return ChessMove(move_san=move_str)

    def validate_move(
        self, state: ChessState, player_id: int, move: ChessMove
    ) -> Tuple[bool, str]:
        """Validate if a move is legal in the current state."""
        # Check if it's the player's turn
        if self.get_current_player(state) != player_id:
            return False, "Not your turn"

        try:
            # Try to parse the move in the current position
            chess_move = state.board.parse_san(move.move_san)
            # Check if the move is legal
            if chess_move not in state.board.legal_moves:
                return False, "Illegal move"
            return True, ""
        except ValueError as e:
            return False, str(e)

    def apply_move(
        self, state: ChessState, player_id: int, move: ChessMove
    ) -> ChessState:
        """Apply move to state and return new state."""
        # Create a new state with copied board
        new_state = ChessState(
            board=state.board.copy(), move_history=state.move_history.copy()
        )

        # Apply the move
        new_state.board.push_san(move.move_san)

        # Add move to history
        new_state.move_history.append(move.move_san)

        return new_state

    def get_current_player(self, state: ChessState) -> int:
        """Return the ID of the player whose turn it is."""
        return 0 if state.board.turn else 1

    def get_player_view(
        self,
        state: ChessState,
        player_id: int,
        history: Optional[List[Dict[str, Any]]] = None,
        prompt_style: PromptStyle = PromptStyle.HEADER,
    ) -> MatchView:
        """Return what this player can see of the current state."""
        visible_state = {
            "position": state.board.fen(),
            "move_history": " ".join(state.move_history),
            "you_are": "white" if player_id == 0 else "black",
            "in_check": state.board.is_check(),
            "legal_moves": [state.board.san(move) for move in state.board.legal_moves],
        }

        is_terminal = (
            state.board.is_checkmate()
            or state.board.is_stalemate()
            or state.board.is_insufficient_material()
            or state.board.is_fifty_moves()
            or state.board.is_repetition()
        )

        winner = None
        if state.board.is_checkmate():
            # The player who just moved won
            winner = 1 - self.get_current_player(state)

        return MatchView(
            rules_explanation=self.get_rules_explanation(),
            visible_state=visible_state,
            valid_moves=[state.board.san(move) for move in state.board.legal_moves],
            is_terminal=is_terminal,
            winner=winner,
            history=history if history else [],
            move_format_instructions=self.get_move_format_instructions(),
            prompt_style=prompt_style,
        )

    def get_move_format_instructions(self) -> str:
        return (
            "Enter your move in standard chess notation (PGN format).\n"
            "Examples:\n"
            "- e4 (pawn to e4)\n"
            "- Nf3 (knight to f3)\n"
            "- O-O (kingside castling)\n"
            "- O-O-O (queenside castling)\n"
            "- exd5 (pawn on e-file captures on d5)\n"
            "- Bxf7+ (bishop captures on f7 with check)\n"
            "- e8=Q (pawn promotion to queen)\n"
            "- Qxf7# (queen captures on f7 with checkmate)"
        )

    def get_next_state(self, state: ChessState, move: ChessMove) -> ChessState:
        """Return the next state after applying the move."""
        return self.apply_move(state, self.get_current_player(state), move)

    def is_terminal(self, state: ChessState) -> bool:
        return (
            state.board.is_checkmate()
            or state.board.is_stalemate()
            or state.board.is_insufficient_material()
            or state.board.is_fifty_moves()
            or state.board.is_repetition()
        )

    def get_winner(self, state: ChessState) -> Optional[int]:
        if not state.board.is_checkmate():
            return None  # Draw or game not over
        # The player who just moved won
        return 1 - self.get_current_player(state)

    def get_rules_explanation(self) -> str:
        return (
            "Chess is a two-player strategy game played on an 8x8 board.\n"
            "Each player starts with 16 pieces: 1 king, 1 queen, 2 rooks, 2 knights,\n"
            "2 bishops, and 8 pawns.\n\n"
            "Key rules:\n"
            "1. White moves first\n"
            "2. Players alternate turns, moving one piece per turn\n"
            "3. A piece cannot move through other pieces (except knights)\n"
            "4. Pieces capture by moving to an opponent's piece's square\n"
            "5. The goal is to checkmate the opponent's king\n\n"
            "Special moves:\n"
            "- Castling: King moves 2 squares toward rook, rook moves to other side\n"
            "- En passant: Pawn captures a pawn that just moved 2 squares\n"
            "- Pawn promotion: Pawn reaching opposite end becomes another piece\n\n"
            "Game ends when:\n"
            "- Checkmate: King is attacked and cannot escape\n"
            "- Stalemate: No legal moves but king not in check\n"
            "- Draw: By agreement, repetition, or insufficient material"
        )
        
    def serialize_state(self, state: ChessState) -> Dict[str, Any]:
        """Serialize the game state into a JSON-compatible dictionary.

        This method ensures that all game-specific state is properly serialized
        into a format that can be stored in the database and later deserialized.

        Args:
            state: The ChessState to serialize

        Returns:
            A JSON-compatible dictionary representing the game state
        """
        # Use the to_dict method of ChessState
        return state.to_dict()

    def deserialize_state(self, state_data: Dict[str, Any]) -> ChessState:
        """Deserialize state data into a ChessState object.
        
        Args:
            state_data: Dictionary containing serialized state data, the output of ChessState.to_dict()
            
        Returns:
            Deserialized ChessState object
        """
        # Simply pass the state data to ChessState.from_dict
        return ChessState.from_dict(state_data)
