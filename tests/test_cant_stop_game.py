import pytest
from bgbench.games.cant_stop_game import CantStopGame, CantStopState, CantStopMove, ColumnState

@pytest.fixture
def game():
    return CantStopGame()

@pytest.fixture
def initial_state(game):
    return game.get_initial_state()

def test_initial_state(initial_state):
    assert initial_state.current_player == 0
    assert len(initial_state.temp_positions) == 0
    assert len(initial_state.active_columns) == 0
    assert len(initial_state.current_dice) == 4
    assert initial_state.awaiting_selection == True
    
    # Check column setup
    assert len(initial_state.columns) == 11  # columns 2-12
    assert initial_state.columns[2].max_height == 3
    assert initial_state.columns[7].max_height == 13
    assert initial_state.columns[12].max_height == 3

def test_parse_move(game):
    # Test valid moves
    assert game.parse_move("select 0 1") == CantStopMove("select", [0, 1])
    assert game.parse_move("stop") == CantStopMove("stop", [])
    assert game.parse_move("roll") == CantStopMove("roll", [])
    
    # Test invalid moves
    assert game.parse_move("invalid") is None
    assert game.parse_move("select 0") is None
    assert game.parse_move("select 0 0") is None  # Same dice
    assert game.parse_move("select 0 4") is None  # Out of range

def test_validate_move(game, initial_state):
    # Force specific dice for testing
    initial_state.current_dice = [1, 2, 3, 4]
    
    # Test valid moves
    valid_move = CantStopMove("select", [0, 1])  # Sum 3
    assert game.validate_move(initial_state, 0, valid_move)[0] == True
    
    # Test wrong player
    assert game.validate_move(initial_state, 1, valid_move)[0] == False
    
    # Test wrong action type
    wrong_action = CantStopMove("roll", [])
    assert game.validate_move(initial_state, 0, wrong_action)[0] == False

def test_dice_combinations(game):
    dice = [1, 2, 3, 4]
    combinations = game._get_possible_combinations(dice)
    expected = [(3, 7), (4, 6), (5, 5)]
    assert combinations == expected

def test_has_valid_move(game, initial_state):
    # Test with dice that allow valid moves
    initial_state.current_dice = [1, 2, 3, 4]  # Sums: 3,7 or 4,6 or 5,5
    assert game._has_valid_move(initial_state) == True
    
    # Test with impossible dice (all sums too high)
    initial_state.current_dice = [6, 6, 6, 6]
    assert game._has_valid_move(initial_state) == False

def test_apply_move_selection(game, initial_state):
    initial_state.current_dice = [1, 2, 3, 4]
    move = CantStopMove("select", [0, 1])  # Sum 3
    
    new_state = game.apply_move(initial_state, 0, move)
    assert new_state.temp_positions.get(3) == 1  # Advanced in column 3
    assert 3 in new_state.active_columns
    assert new_state.awaiting_selection == False

def test_apply_move_stop(game, initial_state):
    # Setup some temporary progress
    initial_state.temp_positions = {3: 2, 7: 1}
    initial_state.active_columns = {3, 7}
    initial_state.awaiting_selection = False
    
    move = CantStopMove("stop", [])
    new_state = game.apply_move(initial_state, 0, move)
    
    # Check progress was made permanent
    assert new_state.columns[3].player_positions[0] == 2
    assert new_state.columns[7].player_positions[0] == 1
    
    # Check reset for next player
    assert new_state.current_player == 1
    assert len(new_state.temp_positions) == 0
    assert len(new_state.active_columns) == 0
    assert new_state.awaiting_selection == True

def test_apply_move_bust(game, initial_state):
    # Setup state where next roll will bust
    initial_state.temp_positions = {3: 2, 7: 1}
    initial_state.active_columns = {3, 7}
    initial_state.awaiting_selection = False
    initial_state.current_dice = [6, 6, 6, 6]  # Force impossible roll
    
    move = CantStopMove("roll", [])
    new_state = game.apply_move(initial_state, 0, move)
    
    # Check that progress was lost and turn switched
    assert len(new_state.temp_positions) == 0
    assert len(new_state.active_columns) == 0
    assert new_state.current_player != initial_state.current_player

def test_win_condition(game, initial_state):
    # Set up a winning state for player 0
    initial_state.columns[2].is_claimed = True
    initial_state.columns[2].claimed_by = 0
    initial_state.columns[3].is_claimed = True
    initial_state.columns[3].claimed_by = 0
    initial_state.columns[4].is_claimed = True
    initial_state.columns[4].claimed_by = 0
    
    view = game.get_player_view(initial_state, 0)
    assert view.is_terminal == True
    assert view.winner == 0
