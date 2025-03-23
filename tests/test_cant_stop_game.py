import pytest
from bgbench.games.cant_stop_game import CantStopGame, CantStopMove


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
    assert game.parse_move("select 7 8") == CantStopMove("select", [7, 8])
    assert game.parse_move("stop") == CantStopMove("stop", [])
    assert game.parse_move("roll") == CantStopMove("roll", [])

    # Test invalid moves
    assert game.parse_move("invalid") is None
    assert game.parse_move("select 7") is None  # Missing second number
    assert game.parse_move("select 1 13") is None  # Out of range
    assert game.parse_move("select foo bar") is None  # Non-numeric


def test_validate_move(game, initial_state):
    # Force specific dice for testing
    initial_state.current_dice = [2, 3, 4, 5]  # Possible sums: 5,9 or 6,8 or 7,7

    # Test valid moves
    valid_move = CantStopMove("select", [5, 9])  # Valid sum combination
    assert game.validate_move(initial_state, 0, valid_move)[0] == True

    # Test invalid sum combination
    invalid_move = CantStopMove("select", [3, 11])  # Impossible with these dice
    assert game.validate_move(initial_state, 0, invalid_move)[0] == False

    # Test wrong player
    assert game.validate_move(initial_state, 1, valid_move)[0] == False

    # Test wrong action when selection expected
    wrong_action = CantStopMove("roll", [])
    assert game.validate_move(initial_state, 0, wrong_action)[0] == False


def test_dice_combinations(game):
    dice = [2, 3, 4, 5]
    combinations = game._get_possible_combinations(dice)
    expected = [(5, 9), (6, 8), (7, 7)]  # All possible sum combinations
    assert sorted(combinations) == sorted(expected)


def test_has_valid_move(game, initial_state):
    # Test with dice that allow valid moves
    initial_state.current_dice = [2, 3, 4, 5]  # Sums: 5,9 or 6,8 or 7,7
    assert game._has_valid_move(initial_state) == True

    # Test with dice that have no valid moves
    # Mark all possible columns as claimed
    for i in range(2, 13):
        initial_state.columns[i].is_claimed = True
    assert game._has_valid_move(initial_state) == False


def test_apply_move_selection(game, initial_state):
    initial_state.current_dice = [2, 3, 4, 5]  # Sums: 5,9 or 6,8 or 7,7
    move = CantStopMove("select", [5, 9])  # Select columns 5 and 9

    new_state = game.apply_move(initial_state, 0, move)
    assert new_state.temp_positions.get(5) == 1  # Advanced in column 5
    assert new_state.temp_positions.get(9) == 1  # Advanced in column 9
    assert 5 in new_state.active_columns
    assert 9 in new_state.active_columns
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
    # Setup state with some progress
    initial_state.temp_positions = {5: 2, 9: 1}
    initial_state.active_columns = {5, 9}
    initial_state.awaiting_selection = False
    initial_state.current_player = 0

    # Mock _has_valid_move to always return False to simulate a bust
    original_has_valid_move = game._has_valid_move
    game._has_valid_move = lambda state: False

    move = CantStopMove("roll", [])
    new_state = game.apply_move(initial_state, 0, move)

    # Restore original method
    game._has_valid_move = original_has_valid_move

    # Check that progress was lost and turn switched
    assert len(new_state.temp_positions) == 0
    assert len(new_state.active_columns) == 0
    assert new_state.current_player == 1  # Switched to player 1


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


def test_initial_temp_progress_starts_from_player_progress(game, initial_state):
    """Test that when a player first selects a column, temp progress starts from their previous progress."""
    # Set up player progress in columns
    column_num = 7
    player_id = 0
    progress_value = 3

    # Set existing progress for the player
    initial_state.columns[column_num].player_positions[player_id] = progress_value
    initial_state.current_player = player_id

    # Set specific dice that will allow 7 and 7 as valid column sums
    # With dice [3, 4, 2, 5], possible combinations are:
    # (3+4=7, 2+5=7), (3+2=5, 4+5=9), (3+5=8, 2+4=6)
    initial_state.current_dice = [3, 4, 2, 5]

    # Make a move that selects column 7 twice (using 3+4 and 2+5)
    move = CantStopMove("select", [7, 7])

    # Apply the move
    new_state = game.apply_move(initial_state, player_id, move)

    # Output for debugging
    print(f"Dice: {initial_state.current_dice}")
    print(
        f"Possible combinations: {game._get_possible_combinations(initial_state.current_dice)}"
    )
    print(f"Temp positions: {new_state.temp_positions}")
    print(f"Active columns: {new_state.active_columns}")

    # Since we selected 7 twice, temp progress should be previous progress + 2
    assert new_state.temp_positions[column_num] == progress_value + 2

    # Verify active columns are properly updated
    assert column_num in new_state.active_columns

    # Check that the temp_progress_position shown in the view is correct
    view = game.get_player_view(new_state, player_id)
    visible_column = view.visible_state["columns"][column_num]
    assert visible_column["temp_progress_position"] == progress_value + 2
