import pytest
from typing import List, Optional
from bgbench.games.reversi_game import ReversiGame, ReversiMove, ReversiState, DARK, LIGHT, EMPTY

@pytest.fixture
def game():
    return ReversiGame()

@pytest.fixture
def initial_state(game):
    return game.get_initial_state()

def test_initial_state(initial_state):
    """Test that the initial state has the correct setup."""
    # Check the four center pieces
    assert initial_state.board[3][3] == LIGHT
    assert initial_state.board[3][4] == DARK
    assert initial_state.board[4][3] == DARK
    assert initial_state.board[4][4] == LIGHT
    
    # Check that the rest of the board is empty
    for row in range(8):
        for col in range(8):
            if not ((row == 3 and col == 3) or 
                    (row == 3 and col == 4) or 
                    (row == 4 and col == 3) or 
                    (row == 4 and col == 4)):
                assert initial_state.board[row][col] is EMPTY
    
    # Check that dark moves first
    assert initial_state.current_player == DARK

def test_parse_move(game):
    """Test parsing moves from strings."""
    # Test valid algebraic notation
    move = game.parse_move("e4")
    assert move.row == 3
    assert move.col == 4
    
    # Test pass move
    move = game.parse_move("pass")
    assert move.row is None
    assert move.col is None
    
    # Test invalid move
    move = game.parse_move("invalid")
    assert move is None

def test_valid_moves(game, initial_state):
    """Test that valid moves are correctly identified."""
    valid_moves = game.get_valid_moves(initial_state, DARK)
    # In the initial position, dark has 4 valid moves
    expected_moves = {(2, 3), (3, 2), (4, 5), (5, 4)}
    assert valid_moves == expected_moves

def test_validate_move(game, initial_state):
    """Test move validation."""
    # Valid move
    is_valid, _ = game.validate_move(initial_state, DARK, ReversiMove(2, 3))
    assert is_valid
    
    # Invalid move - already occupied
    is_valid, _ = game.validate_move(initial_state, DARK, ReversiMove(3, 3))
    assert not is_valid
    
    # Invalid move - doesn't flip any discs
    is_valid, _ = game.validate_move(initial_state, DARK, ReversiMove(0, 0))
    assert not is_valid
    
    # Invalid move - wrong player
    is_valid, _ = game.validate_move(initial_state, LIGHT, ReversiMove(2, 3))
    assert not is_valid

def test_apply_move(game, initial_state):
    """Test applying a move."""
    # Dark places a disc at d3 (row 2, col 3)
    new_state = game.apply_move(initial_state, DARK, ReversiMove(2, 3))
    
    # Check that the disc was placed
    assert new_state.board[2][3] == DARK
    
    # Check that the opponent's disc was flipped
    assert new_state.board[3][3] == DARK  # Was LIGHT
    
    # Check that it's now light's turn
    assert new_state.current_player == LIGHT
    
    # Check that the move was recorded in history
    assert new_state.move_history == [(DARK, 2, 3)]

def test_game_end(game):
    """Test game end conditions."""
    # Create a board with dark winning and one empty space
    # Initialize with Optional[int] type to satisfy type checker
    board: List[List[Optional[int]]] = []
    for _ in range(8):
        row: List[Optional[int]] = []
        for _ in range(8):
            row.append(DARK)
        board.append(row)
    
    # Set up a position where placing at (7,7) would flip an opponent's disc
    board[7][7] = EMPTY  # Empty space for the move
    board[7][6] = LIGHT  # Light piece to be flipped
    board[7][5] = DARK   # Dark piece to outflank
    
    # Create a state where there are valid moves
    state = ReversiState(board=board, current_player=DARK)
    
    # Verify the move is valid
    is_valid, _ = game.validate_move(state, DARK, ReversiMove(7, 7))
    assert is_valid, "The test move should be valid"
    
    # Dark makes the final move
    final_state = game.apply_move(state, DARK, ReversiMove(7, 7))
    
    # Game is now terminal because board is full
    assert game.is_terminal(final_state)
    
    # Dark wins
    assert game.get_winner(final_state) == DARK

def test_automatic_pass(game):
    """Test that a player automatically passes if they have no valid moves."""
    # Create a board state where DARK plays, leaving LIGHT with no valid moves
    board: List[List[Optional[int]]] = [[EMPTY for _ in range(8)] for _ in range(8)]
    board[0][0] = DARK
    board[0][1] = LIGHT
    board[1][0] = LIGHT
    board[1][1] = LIGHT
    # ... set up more pieces such that after DARK plays at (0,2), LIGHT has no moves ...
    # Example: Fill a corner such that LIGHT is blocked
    board[0][7] = DARK
    board[1][7] = DARK
    board[2][7] = DARK
    board[0][6] = LIGHT # Light piece that could be flipped by (0,7)
    
    state = ReversiState(board=board, current_player=DARK)
    
    # Assume DARK makes a move at (0, 2) which flips board[0][1]
    # Ensure this move is valid first (it should be if board[0][3] is DARK or empty)
    board[0][3] = DARK # Make the move valid
    state.board = board # Update state board
    
    # Verify DARK move is valid
    dark_move = ReversiMove(0, 2)
    is_valid, _ = game.validate_move(state, DARK, dark_move)
    assert is_valid, "Dark's move should be valid"

    # Apply DARK's move
    new_state = game.apply_move(state, DARK, dark_move)
    
    # Check board changes after DARK's move
    assert new_state.board[0][2] == DARK
    assert new_state.board[0][1] == DARK # Flipped

    # Verify that LIGHT now has no valid moves in new_state
    light_valid_moves = game.get_valid_moves(new_state, LIGHT)
    assert not light_valid_moves, f"LIGHT should have no valid moves, but found: {light_valid_moves}"
    
    # Because LIGHT has no moves, the turn should automatically pass back to DARK
    assert new_state.current_player == DARK
    
    # The pass count should be 1 (due to LIGHT's automatic pass)
    assert new_state.pass_count == 1
    
    # Check that LIGHT's pass was recorded in history
    assert new_state.move_history[-1] == (LIGHT, -1, -1)

def test_consecutive_passes_end_game(game):
    """Test that the game ends if both players consecutively have no valid moves."""
    # Create a board state where neither player has any valid moves, based on the corner scenario.
    # If one corner is empty, but DARK has both adjacent edges and the diagonal,
    # neither player can move into the corner.
    board: List[List[Optional[int]]] = [[LIGHT for _ in range(8)] for _ in range(8)] # Initialize with LIGHT

    # Set the top-left corner (0,0) to EMPTY
    board[0][0] = EMPTY

    # Set the adjacent edges (row 0 and col 0) and the diagonal to DARK
    for i in range(1, 8):
        board[0][i] = DARK  # Top edge
        board[i][0] = DARK  # Left edge
        if i > 0: # Avoid overwriting (0,0)
             board[i][i] = DARK  # Diagonal

    # Create the state with DARK to move
    state = ReversiState(board=board, current_player=DARK, pass_count=0)

    # Verify DARK has no valid moves
    dark_valid_moves = game.get_valid_moves(state, DARK)
    assert not dark_valid_moves, "DARK should have no valid moves"

    # Verify LIGHT also has no valid moves
    light_valid_moves = game.get_valid_moves(state, LIGHT)
    assert not light_valid_moves, f"LIGHT should have no valid moves, but found: {light_valid_moves}"

    # Since neither player has moves, the game should be terminal *immediately*
    # The is_terminal check itself should handle the case where no moves are possible for the current player.
    # If the current player (DARK) has no moves, is_terminal should check if the *other* player (LIGHT) also has no moves.
    assert game.is_terminal(state), "Game should be terminal when neither player can move"

    # Further checks: If we try to apply a pass move for DARK (since it has no moves)
    pass_move = ReversiMove(None, None)
    is_valid, reason = game.validate_move(state, DARK, pass_move)
    # Passing is only valid if the player *has* no other moves.
    assert is_valid, f"Passing should be valid for DARK as they have no moves. Reason: {reason}"

    # Applying the pass move
    next_state = game.apply_move(state, DARK, pass_move)

    # After DARK passes, it becomes LIGHT's turn.
    # Since LIGHT also has no valid moves, the game should *automatically* pass for LIGHT.
    # The turn should return to DARK, and the pass_count should become 2.
    assert next_state.current_player == DARK, "Turn should return to DARK after LIGHT auto-passes"
    assert next_state.pass_count == 2, "Pass count should be 2 after consecutive passes"

    # The game should now definitely be terminal.
    assert game.is_terminal(next_state), "Game should be terminal after two consecutive passes"

    # Check history reflects both passes
    assert len(next_state.move_history) >= 2, "Move history should contain at least two pass moves"
    assert next_state.move_history[-2] == (DARK, -1, -1), "Dark's explicit pass should be recorded"
    assert next_state.move_history[-1] == (LIGHT, -1, -1), "Light's automatic pass should be recorded"

def test_serialization(game, initial_state):
    """Test state serialization and deserialization."""
    # Serialize the state
    state_dict = game.serialize_state(initial_state)
    
    # Deserialize back to a state
    reconstructed_state = game.deserialize_state(state_dict)
    
    # Check that the board is the same
    for row in range(8):
        for col in range(8):
            assert reconstructed_state.board[row][col] == initial_state.board[row][col]
    
    # Check other properties
    assert reconstructed_state.current_player == initial_state.current_player
    assert reconstructed_state.pass_count == initial_state.pass_count
    assert reconstructed_state.move_history == initial_state.move_history
