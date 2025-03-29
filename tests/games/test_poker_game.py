import pytest
from bgbench.games.poker_game import PokerGame


def test_poker_serialization():
    """Test that poker serialization works but deserialization raises NotImplementedError."""
    game = PokerGame()
    state = game.get_initial_state()
    
    # Test serialize_state works
    serialized = game.serialize_state(state)
    assert isinstance(serialized, dict)
    assert "big_blind" in serialized
    assert "player_stacks" in serialized
    assert "community_cards" in serialized
    assert "player_hands" in serialized
    assert "serialization_note" in serialized
    
    # Test deserialize_state raises NotImplementedError
    with pytest.raises(NotImplementedError) as excinfo:
        game.deserialize_state({})
    assert "Poker games cannot currently be deserialized" in str(excinfo.value)
