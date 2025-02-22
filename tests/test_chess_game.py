import pytest
from bgbench.games.chess_game import ChessGame, ChessState, ChessMove
import chess

@pytest.fixture
def game():
    return ChessGame()

@pytest.fixture
def initial_state(game):
    return game.get_initial_state()

def test_initial_state(game, initial_state):
    """Test the initial game state setup."""
    assert isinstance(initial_state.board, chess.Board)
    assert initial_state.board.fen() == chess.STARTING_FEN
    assert len(initial_state.move_history) == 0
    assert game.get_current_player(initial_state) == 0  # White moves first

def test_parse_move():
    """Test parsing various types of chess moves."""
    game = ChessGame()
    
    # Test valid moves
    move = game.parse_move("e4")
    assert move is not None and move.move_san == "e4"
    move = game.parse_move("Nf3")
    assert move is not None and move.move_san == "Nf3"
    move = game.parse_move("O-O")
    assert move is not None and move.move_san == "O-O"
    move = game.parse_move("o-o")
    assert move is not None and move.move_san == "O-O"
    move = game.parse_move("0-0")
    assert move is not None and move.move_san == "O-O"
    move = game.parse_move("O-O-O")
    assert move is not None and move.move_san == "O-O-O"
    move = game.parse_move("e8=Q")
    assert move is not None and move.move_san == "e8=Q"
    move = game.parse_move("exd5")
    assert move is not None and move.move_san == "exd5"
    

def test_validate_move(game, initial_state):
    """Test move validation in various scenarios."""
    # Valid first moves
    assert game.validate_move(initial_state, 0, ChessMove("e4"))[0]
    assert game.validate_move(initial_state, 0, ChessMove("d4"))[0]
    assert game.validate_move(initial_state, 0, ChessMove("Nf3"))[0]
    
    # Invalid moves
    # Wrong player
    assert not game.validate_move(initial_state, 1, ChessMove("e4"))[0]
    # Invalid notation
    assert not game.validate_move(initial_state, 0, ChessMove("e9"))[0]
    # Invalid piece movement
    assert not game.validate_move(initial_state, 0, ChessMove("e5"))[0]  # Can't move e-pawn to e5 directly
    assert not game.validate_move(initial_state, 0, ChessMove("Ke2"))[0]  # Can't move king through pieces


def test_en_passant(game, initial_state):
    """Test en passant capture rules."""
    # Setup position for en passant
    moves = ["e4", "a6", "e5", "d5"]
    state = initial_state
    
    for i, move in enumerate(moves):
        state = game.apply_move(state, i % 2, ChessMove(move))
    
    # Verify en passant is possible
    assert game.validate_move(state, 0, ChessMove("exd6"))[0]

def test_pawn_promotion(game, initial_state):
    """Test pawn promotion rules."""
    # Setup a position where promotion is possible
    board = chess.Board("8/4P3/8/8/8/8/8/8 w - - 0 1")
    state = ChessState(board=board, move_history=[])
    
    # Test different promotion moves
    assert game.validate_move(state, 0, ChessMove("e8=Q"))[0]
    assert game.validate_move(state, 0, ChessMove("e8=R"))[0]
    assert game.validate_move(state, 0, ChessMove("e8=B"))[0]
    assert game.validate_move(state, 0, ChessMove("e8=N"))[0]

def test_check_detection(game, initial_state):
    """Test detection of check positions."""
    # Setup a check position
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 0 4")
    state = ChessState(board=board, move_history=[])
    
    assert state.board.is_check()
    view = game.get_player_view(state, 0)
    assert view.visible_state["in_check"]

def test_checkmate_detection(game, initial_state):
    """Test detection of checkmate."""
    # Fool's mate position
    moves = ["f3", "e5", "g4", "Qh4"]
    state = initial_state
    
    for i, move in enumerate(moves):
        state = game.apply_move(state, i % 2, ChessMove(move))
    
    assert state.board.is_checkmate()
    view = game.get_player_view(state, 0)
    assert view.is_terminal
    assert view.winner == 1  # Black wins

def test_stalemate_detection(game, initial_state):
    """Test detection of stalemate."""
    # Setup a stalemate position
    board = chess.Board("k7/8/1Q6/8/8/8/8/K7 b - - 0 1")
    state = ChessState(board=board, move_history=[])
    
    assert state.board.is_stalemate()
    view = game.get_player_view(state, 1)
    assert view.is_terminal
    assert view.winner is None  # Draw

def test_insufficient_material(game, initial_state):
    """Test detection of insufficient material draws."""
    # King vs King
    board = chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 0 1")
    state = ChessState(board=board, move_history=[])
    
    assert state.board.is_insufficient_material()
    view = game.get_player_view(state, 0)
    assert view.is_terminal
    assert view.winner is None

def test_move_history(game, initial_state):
    """Test move history tracking."""
    moves = ["e4", "e5", "Nf3", "Nc6"]
    state = initial_state
    
    for i, move in enumerate(moves):
        state = game.apply_move(state, i % 2, ChessMove(move))
    
    assert state.move_history == moves
    view = game.get_player_view(state, 0)
    assert view.visible_state["move_history"] == "e4 e5 Nf3 Nc6"

def test_player_view(game, initial_state):
    """Test player view information."""
    view = game.get_player_view(initial_state, 0)
    
    assert view.visible_state["position"] == chess.STARTING_FEN
    assert view.visible_state["you_are"] == "white"
    assert not view.visible_state["in_check"]
    assert isinstance(view.valid_moves, list)
    assert "e4" in view.valid_moves
    assert "d4" in view.valid_moves
    assert "Nf3" in view.valid_moves

def test_to_dict(game, initial_state):
    """Test state serialization."""
    state_dict = initial_state.to_dict()
    
    assert state_dict["fen"] == chess.STARTING_FEN
    assert state_dict["pgn"] == ""
    assert not state_dict["is_check"]
    assert not state_dict["is_checkmate"]
    assert not state_dict["is_stalemate"]
    assert state_dict["turn"] == "white"
