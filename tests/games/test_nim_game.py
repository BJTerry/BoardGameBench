import pytest
from bgbench.games.nim_game import NimGame, NimState, NimMove


@pytest.fixture
def nim_game():
    return NimGame(starting_count=21, max_take=3)


def test_nim_validate_move(nim_game):
    state = NimState(remaining=10, current_player=0)

    # Test valid moves
    assert nim_game.validate_move(state, 0, NimMove(count=3)) == (True, "")
    assert nim_game.validate_move(state, 0, NimMove(count=1)) == (True, "")

    # Test invalid moves
    assert not nim_game.validate_move(state, 0, NimMove(count=4))[0]  # Too many
    assert not nim_game.validate_move(state, 0, NimMove(count=0))[0]  # Too few
    assert not nim_game.validate_move(state, 0, NimMove(count=11))[
        0
    ]  # More than remaining


def test_nim_apply_move(nim_game):
    initial_state = NimState(remaining=10, current_player=0)

    # Test move application
    new_state = nim_game.apply_move(initial_state, 0, NimMove(count=3))
    assert new_state.remaining == 7

    # Test game ending move
    final_state = NimState(remaining=2, current_player=0)
    end_state = nim_game.apply_move(final_state, 0, NimMove(count=2))
    assert end_state.remaining == 0


def test_nim_serialize_deserialize(nim_game):
    """Test serialization and deserialization of NimState."""
    # Create a state
    state = NimState(remaining=15, current_player=1)
    
    # Serialize it
    serialized = nim_game.serialize_state(state)
    
    # Verify serialization
    assert serialized == {"remaining": 15, "current_player": 1}
    
    # Deserialize it
    deserialized = nim_game.deserialize_state(serialized)
    
    # Verify deserialization
    assert deserialized.remaining == 15
    assert deserialized.current_player == 1
    
    # Test roundtrip with game logic
    assert nim_game.get_current_player(deserialized) == 1
    assert not nim_game.is_terminal(deserialized)
