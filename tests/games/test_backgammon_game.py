import pytest
from bgbench.games.backgammon_game import BackgammonGame, BackgammonState, BackgammonMove, WHITE, BLACK

@pytest.fixture
def game():
    return BackgammonGame()

@pytest.fixture
def initial_state(game):
    # Create a deterministic initial state for testing
    state = BackgammonState()
    
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
    
    # Set initial dice and player for testing
    state.current_player = WHITE
    state.dice = [4, 2]
    
    return state

def test_initial_state(initial_state):
    """Test that the initial state has the correct setup."""
    # Check WHITE checkers
    assert initial_state.points[24] == 2
    assert initial_state.points[13] == 5
    assert initial_state.points[8] == 3
    assert initial_state.points[6] == 5
    
    # Check BLACK checkers
    assert initial_state.points[1] == -2
    assert initial_state.points[12] == -5
    assert initial_state.points[17] == -3
    assert initial_state.points[19] == -5
    
    # Check that WHITE starts
    assert initial_state.current_player == WHITE
    
    # Check initial dice
    assert len(initial_state.dice) == 2
    assert 1 <= initial_state.dice[0] <= 6
    assert 1 <= initial_state.dice[1] <= 6

def test_parse_move(game):
    """Test parsing moves from strings."""
    # Test regular move
    move = game.parse_move("13-9")
    assert move.from_point == 13
    assert move.to_point == 9
    
    # Test move from bar
    move = game.parse_move("bar-20")
    assert move.from_point == 0
    assert move.to_point == 20
    
    # Test bearing off
    move = game.parse_move("6-off")
    assert move.from_point == 6
    assert move.to_point == 0
    
    # Test invalid move format
    move = game.parse_move("invalid")
    assert move is None

def test_validate_move_regular(game, initial_state):
    """Test validation of regular moves."""
    # WHITE moving from 13 to 9 using die 4
    move = BackgammonMove(from_point=13, to_point=9, die=4)
    is_valid, _ = game.validate_move(initial_state, WHITE, move)
    assert is_valid
    
    # WHITE moving to a blocked point (occupied by 2+ BLACK checkers)
    move = BackgammonMove(from_point=8, to_point=1, die=7)
    is_valid, _ = game.validate_move(initial_state, WHITE, move)
    assert not is_valid
    
    # BLACK's turn but WHITE is trying to move
    initial_state.current_player = BLACK
    move = BackgammonMove(from_point=13, to_point=9, die=4)
    is_valid, _ = game.validate_move(initial_state, WHITE, move)
    assert not is_valid

def test_validate_move_from_bar(game, initial_state):
    """Test validation of moves from the bar."""
    # Put a WHITE checker on the bar
    initial_state.bar[WHITE] = 1
    # Ensure point 21 is open
    initial_state.points[21] = 0
    
    # Set up dice that match our desired entry
    initial_state.dice = [4, 2]  # Die value 4 for WHITE entering at 21 (25-4)
    
    # Valid move from bar for WHITE
    move = BackgammonMove(from_point=0, to_point=21, die=4)
    is_valid, _ = game.validate_move(initial_state, WHITE, move)
    assert is_valid
    
    # Invalid move: trying to move from regular point while having checkers on bar
    move = BackgammonMove(from_point=13, to_point=9, die=4)
    is_valid, _ = game.validate_move(initial_state, WHITE, move)
    assert not is_valid
    
    # Invalid move: trying to enter on a blocked point
    # Block point 21
    initial_state.points[21] = -2  # 2 BLACK checkers
    move = BackgammonMove(from_point=0, to_point=21, die=4)
    is_valid, _ = game.validate_move(initial_state, WHITE, move)
    assert not is_valid

def test_validate_move_bearing_off(game):
    """Test validation of bearing off moves."""
    # Create a state where WHITE can bear off
    state = BackgammonState()
    state.current_player = WHITE
    state.dice = [21, 2]  # Use point number as die value
    
    # Put WHITE checkers only in home board
    state.points[24] = 2
    state.points[23] = 3
    state.points[21] = 5
    state.points[19] = 5
    
    # Valid bearing off move with exact die value
    move = BackgammonMove(from_point=21, to_point=0, die=21)
    is_valid, reason = game.validate_move(state, WHITE, move)
    assert is_valid, reason
    
    # Create a state where WHITE cannot bear off (has checkers outside home)
    state = BackgammonState()
    state.current_player = WHITE
    state.dice = [4, 2]
    
    # Put some WHITE checkers outside home board
    state.points[24] = 2
    state.points[23] = 3
    state.points[13] = 5  # Outside home board
    state.points[19] = 5
    
    # Invalid bearing off move
    move = BackgammonMove(from_point=24, to_point=0, die=4)
    is_valid, _ = game.validate_move(state, WHITE, move)
    assert not is_valid

def test_apply_move_regular(game, initial_state):
    """Test applying a regular move."""
    # WHITE moving from 13 to 9 using die 4
    move = BackgammonMove(from_point=13, to_point=9, die=4)
    new_state = game.apply_move(initial_state, WHITE, move)
    
    # Check that a checker was moved
    assert new_state.points[13] == 4  # One less checker at starting point
    assert new_state.points[9] == 1   # One more checker at ending point
    
    # Check that the die was used
    assert 4 not in new_state.dice
    assert 2 in new_state.dice  # Second die still available
    assert 4 in new_state.used_dice

def test_apply_move_from_bar(game, initial_state):
    """Test applying a move from the bar."""
    # Put a WHITE checker on the bar
    initial_state.bar[WHITE] = 1
    # Ensure point 21 is open
    initial_state.points[21] = 0
    
    # Set up dice that match our desired entry
    initial_state.dice = [4, 2]  # 25-4 = 21 (entry point for WHITE with die 4)
    
    # WHITE moving from bar to 21 using die 4
    move = BackgammonMove(from_point=0, to_point=21, die=4)
    new_state = game.apply_move(initial_state, WHITE, move)
    
    # Check that the checker was moved from bar
    assert new_state.bar[WHITE] == 0
    assert new_state.points[21] == 1
    
    # Check that the die was used
    assert 4 not in new_state.dice
    assert 2 in new_state.dice  # Second die still available
    assert 4 in new_state.used_dice

def test_apply_move_hitting(game):
    """Test applying a move that hits an opponent's blot."""
    # Create a state with a BLACK blot
    state = BackgammonState()
    state.current_player = WHITE
    state.dice = [4, 2]
    
    state.points[13] = 1   # WHITE blot
    state.points[9] = -1   # BLACK blot
    
    # WHITE hitting BLACK
    move = BackgammonMove(from_point=13, to_point=9, die=4)
    new_state = game.apply_move(state, WHITE, move)
    
    # Check that BLACK checker was hit and moved to bar
    assert new_state.points[9] == 1  # Now WHITE
    assert new_state.points[13] == 0 # WHITE moved from here
    assert new_state.bar[BLACK] == 1 # BLACK checker on bar

def test_apply_move_bearing_off(game):
    """Test applying a bearing off move."""
    # Create a state where WHITE can bear off
    state = BackgammonState()
    state.current_player = WHITE
    state.dice = [21, 2]  # Include die value matching the point number
    
    # Put WHITE checkers only in home board
    state.points[21] = 1
    
    # WHITE bearing off
    move = BackgammonMove(from_point=21, to_point=0, die=21)
    new_state = game.apply_move(state, WHITE, move)
    
    # Check that the checker was borne off
    assert new_state.points[21] == 0
    assert new_state.borne_off[WHITE] == 1

def test_game_end_condition(game):
    """Test game end condition."""
    # Create a state where WHITE has borne off 14 checkers and has 1 left
    state = BackgammonState()
    state.current_player = WHITE
    state.dice = [24, 2]  # Die value must match the point to bear off
    
    state.points[24] = 1   # Last WHITE checker
    state.borne_off[WHITE] = 14
    
    # Game should not be over yet
    assert not game.is_terminal(state)
    
    # WHITE bears off the last checker
    move = BackgammonMove(from_point=24, to_point=0, die=24)
    new_state = game.apply_move(state, WHITE, move)
    
    # Game should now be over
    assert game.is_terminal(new_state)
    assert game.get_winner(new_state) == WHITE

def test_can_bear_off(game):
    """Test the can_bear_off method."""
    # Create a state where WHITE can bear off (all checkers in home board)
    state = BackgammonState()
    state.points[24] = 5
    state.points[23] = 5
    state.points[21] = 5
    
    assert state.can_bear_off(WHITE)
    
    # Create a state where WHITE cannot bear off (has checkers outside home)
    state = BackgammonState()
    state.points[24] = 5
    state.points[13] = 5  # Outside home board
    state.points[21] = 5
    
    assert not state.can_bear_off(WHITE)
    
    # Create a state where WHITE cannot bear off (has checkers on bar)
    state = BackgammonState()
    state.points[24] = 5
    state.points[23] = 5
    state.points[21] = 4
    state.bar[WHITE] = 1  # Checker on bar
    
    assert not state.can_bear_off(WHITE)

def test_serialization(game, initial_state):
    """Test state serialization and deserialization."""
    # Serialize the state
    state_dict = game.serialize_state(initial_state)
    
    # Deserialize back to a state
    reconstructed_state = game.deserialize_state(state_dict)
    
    # Check that the board is the same
    for point in range(1, 25):
        assert reconstructed_state.points[point] == initial_state.points[point]
    
    # Check other properties
    assert reconstructed_state.current_player == initial_state.current_player
    assert reconstructed_state.bar == initial_state.bar
    assert reconstructed_state.borne_off == initial_state.borne_off
    assert reconstructed_state.dice == initial_state.dice
    assert reconstructed_state.is_double == initial_state.is_double

def test_get_valid_moves(game, initial_state):
    """Test getting valid moves."""
    # Get valid moves for WHITE
    valid_moves = game._get_valid_moves_str(initial_state, WHITE)
    
    # Check that expected moves are present
    assert "13-9" in valid_moves  # Using die 4
    assert "13-11" in valid_moves  # Using die 2
    
    # Put a WHITE checker on the bar
    initial_state.bar[WHITE] = 1
    
    # Get valid moves again - should only show moves from bar
    valid_moves = game._get_valid_moves_str(initial_state, WHITE)
    assert "bar-21" in valid_moves  # Using die 4
    assert "bar-23" in valid_moves  # Using die 2
    assert "13-9" not in valid_moves  # Regular moves not allowed with checkers on bar