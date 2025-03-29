import pytest
from bgbench.games.battleship_game import (
    BattleshipGame,
    BattleshipState,
    Ship,
    Board,
    SHIPS,
)


@pytest.fixture
def game():
    return BattleshipGame()


@pytest.fixture
def empty_state():
    return BattleshipState(
        boards=[
            Board(ships=[], hits=set(), misses=set()),
            Board(ships=[], hits=set(), misses=set()),
        ],
        current_player=0,
        setup_complete=True,
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
    assert not game.validate_move(empty_state, 0, (10, 0))[0]  # Out of bounds
    assert not game.validate_move(empty_state, 0, (-1, 0))[0]  # Negative coordinate

    # Test repeated move
    empty_state.boards[1].hits.add((0, 0))
    assert not game.validate_move(empty_state, 0, (0, 0))[0]  # Already hit


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
            Board(ships=[], hits=set(), misses=set()),
        ],
        current_player=0,
        setup_complete=False,
    )

    # Place first ship (Carrier)
    valid, msg = game.validate_move(state, 0, (0, 0, True))
    assert valid, "Should be able to place first ship"
    state = game.apply_move(state, 0, (0, 0, True))

    # Try to place a ship on the same spot
    duplicate_move = (0, 0, True)
    valid, msg = game.validate_move(state, 0, duplicate_move)
    assert not valid, "Should not be able to place overlapping space"
    assert "Ships cannot overlap" in msg

    # Place remaining ships correctly
    valid_positions = [
        (0, 1, True),
        (0, 2, True),
        (0, 4, True),
        (0, 6, True),
        (0, 8, True),
    ]
    for pos in valid_positions[:-1]:
        valid, msg = game.validate_move(state, 0, pos)
        assert valid, f"Should be able to place ship at {pos}"
        state = game.apply_move(state, 0, pos)

    # Verify all ships were placed with correct sizes
    ships = state.boards[0].ships
    assert len(ships) == len(SHIPS), "Should have placed all ships"
    for ship, (expected_name, expected_size) in zip(ships, SHIPS):
        assert ship.name == expected_name, (
            f"Ship name mismatch: {ship.name} vs {expected_name}"
        )
        assert ship.size == expected_size, (
            f"Ship size mismatch for {ship.name}: {ship.size} vs {expected_size}"
        )
        assert len(ship.positions) == expected_size, (
            f"Ship positions count mismatch for {ship.name}"
        )


def test_game_over(game, empty_state, sample_board):
    state = empty_state
    state.boards[0] = sample_board

    # Sink the only ship
    state = game.apply_move(state, 1, (0, 0))
    state = game.apply_move(state, 1, (1, 0))

    assert game.is_terminal(state)
    assert game.get_winner(state) == 1


def test_serialize_deserialize_state(game):
    """Test serialization and deserialization of BattleshipState."""
    # Create a state with some ships and shots
    state = game.get_initial_state()
    
    # Add ships to player 0's board
    ship1 = Ship("Destroyer", 2, {(0, 0), (0, 1)})
    ship2 = Ship("Cruiser", 3, {(5, 5), (5, 6), (5, 7)})
    state.boards[0].ships = [ship1, ship2]
    
    # Add ships to player 1's board
    ship3 = Ship("Submarine", 3, {(2, 2), (3, 2), (4, 2)})
    state.boards[1].ships = [ship3]
    
    # Add some hits and misses
    state.boards[0].hits.add((0, 0))
    ship1.hits.add((0, 0))
    state.boards[0].misses.add((1, 1))
    state.boards[1].misses.add((7, 7))
    
    # Set game state
    state.setup_complete = True
    state.current_player = 1
    
    # Serialize the state
    serialized = game.serialize_state(state)
    
    # Verify serialization contains expected data
    assert serialized["setup_complete"] is True
    assert serialized["current_player"] == 1
    assert len(serialized["boards"]) == 2
    assert len(serialized["boards"][0]["ships"]) == 2
    assert len(serialized["boards"][1]["ships"]) == 1
    assert (0, 0) in [tuple(pos) for pos in serialized["boards"][0]["hits"]]
    
    # Deserialize back to a state object
    deserialized = game.deserialize_state(serialized)
    
    # Verify deserialization preserves the data
    assert deserialized.setup_complete is True
    assert deserialized.current_player == 1
    assert len(deserialized.boards) == 2
    assert len(deserialized.boards[0].ships) == 2
    assert len(deserialized.boards[1].ships) == 1
    assert (0, 0) in deserialized.boards[0].hits
    assert (1, 1) in deserialized.boards[0].misses
    assert (7, 7) in deserialized.boards[1].misses
    
    # Verify ship data was preserved
    assert deserialized.boards[0].ships[0].name == "Destroyer"
    assert deserialized.boards[0].ships[0].size == 2
    assert (0, 0) in deserialized.boards[0].ships[0].positions
    assert (0, 0) in deserialized.boards[0].ships[0].hits
    
    # Verify game logic still works with deserialized state
    assert game.get_current_player(deserialized) == 1
    assert not game.is_terminal(deserialized)
