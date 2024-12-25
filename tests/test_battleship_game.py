import pytest
import pytest
from bgbench.games.battleship_game import BattleshipGame, BattleshipState, Ship, Board

@pytest.fixture
def game():
    return BattleshipGame()

@pytest.fixture
def empty_state():
    return BattleshipState(
        boards=[
            Board(ships=[], hits=set(), misses=set()),
            Board(ships=[], hits=set(), misses=set())
        ],
        current_player=0,
        setup_complete=True
    )

@pytest.fixture
def sample_board():
    # Create a board with one ship
    ship = Ship("Destroyer", 2, {(0, 0), (1, 0)})
    return Board(ships=[ship], hits=set(), misses=set())

def test_validate_move_during_play(game, empty_state):
    # Test valid moves
    assert game.validate_move(empty_state, 0, (0, 0)) == (True, "")
    assert game.validate_move(empty_state, 0, (9, 9)) == (True, "")
    
    # Test invalid moves
    assert game.validate_move(empty_state, 0, (10, 0))[0] == False  # Out of bounds
    assert game.validate_move(empty_state, 0, (-1, 0))[0] == False  # Negative coordinate
    
    # Test repeated move
    empty_state.boards[1].hits.add((0, 0))
    assert game.validate_move(empty_state, 0, (0, 0))[0] == False  # Already hit

def test_apply_move_during_play(game, empty_state, sample_board):
    state = empty_state
    state.boards[0] = sample_board
    
    # Test hit
    new_state = game.apply_move(state, 1, (0, 0))
    assert (0, 0) in new_state.boards[0].hits
    
    # Test miss
    new_state = game.apply_move(state, 1, (5, 5))
    assert (5, 5) in new_state.boards[0].misses

def test_game_over(game, empty_state, sample_board):
    state = empty_state
    state.boards[0] = sample_board
    
    # Sink the only ship
    state = game.apply_move(state, 1, (0, 0))
    state = game.apply_move(state, 1, (1, 0))
    
    assert game._is_game_over(state)
    assert game._get_winner(state) == 1
