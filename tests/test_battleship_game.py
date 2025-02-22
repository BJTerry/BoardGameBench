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

def test_ship_placement_rules(game):
    state = BattleshipState(
        boards=[
            Board(ships=[], hits=set(), misses=set()),
            Board(ships=[], hits=set(), misses=set())
        ],
        current_player=0,
        setup_complete=False
    )
    
    # Place first ship (Carrier)
    valid, msg = game.validate_move(state, 0, (0, 0, True))
    assert valid, "Should be able to place first ship"
    state = game.apply_move(state, 0, (0, 0, True))
    
    # Try to place another Carrier
    carrier_move = (2, 2, True)
    valid, msg = game.validate_move(state, 0, carrier_move)
    assert not valid, "Should not be able to place duplicate Carrier"
    assert "already been placed" in msg
    
    # Place remaining ships correctly
    valid_positions = [(0, 2, True), (0, 4, True), (0, 6, True), (0, 8, True)]
    for pos in valid_positions[:-1]:
        valid, msg = game.validate_move(state, 0, pos)
        assert valid, f"Should be able to place ship at {pos}"
        state = game.apply_move(state, 0, pos)
    
    # Verify all ships were placed with correct sizes
    ships = state.boards[0].ships
    assert len(ships) == len(SHIPS), "Should have placed all ships"
    for ship, (expected_name, expected_size) in zip(ships, SHIPS):
        assert ship.name == expected_name, f"Ship name mismatch: {ship.name} vs {expected_name}"
        assert ship.size == expected_size, f"Ship size mismatch for {ship.name}: {ship.size} vs {expected_size}"
        assert len(ship.positions) == expected_size, f"Ship positions count mismatch for {ship.name}"

def test_game_over(game, empty_state, sample_board):
    state = empty_state
    state.boards[0] = sample_board
    
    # Sink the only ship
    state = game.apply_move(state, 1, (0, 0))
    state = game.apply_move(state, 1, (1, 0))
    
    assert game.is_terminal(state)
    assert game.get_winner(state) == 1
