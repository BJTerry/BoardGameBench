import pytest
from bgbench.games.guess_who_game import GuessWhoGame, GuessWhoState, Character

@pytest.fixture
def game():
    return GuessWhoGame()

@pytest.fixture
def sample_characters():
    return [
        Character("Char1", {
            "hair_color": "blonde",
            "hair_style": "long",
            "eye_color": "blue",
            "facial_hair": "clean",
            "gender": "female",
            "accessories": "glasses"
        }),
        Character("Char2", {
            "hair_color": "brown",
            "hair_style": "short",
            "eye_color": "brown",
            "facial_hair": "beard",
            "gender": "male",
            "accessories": "none"
        })
    ]

@pytest.fixture
def initial_state(sample_characters):
    return GuessWhoState(
        characters=sample_characters,
        target_characters=[sample_characters[0], sample_characters[1]],
        possible_characters=[sample_characters.copy(), sample_characters.copy()],
        current_player=0
    )

def test_initial_state(game):
    """Test that initial state is created correctly."""
    state = game.get_initial_state()
    assert len(state.characters) == 24  # Default number of characters
    assert len(state.target_characters) == 2
    assert len(state.possible_characters) == 2
    assert state.current_player == 0
    assert all(len(chars) == 24 for chars in state.possible_characters)

def test_character_generation(game):
    """Test that generated characters have valid traits."""
    state = game.get_initial_state()
    for char in state.characters:
        assert all(trait in char.traits for trait in GuessWhoGame.TRAITS.keys())
        for trait, value in char.traits.items():
            assert value in GuessWhoGame.TRAITS[trait]

def test_parse_move(game):
    """Test move parsing."""
    # Test valid moves
    assert game.parse_move("hair_color blonde") == ("hair_color", "blonde", False)
    assert game.parse_move("NOT hair_color blonde") == ("hair_color", "blonde", True)
    
    # Test invalid moves
    assert game.parse_move("invalid move") is None
    assert game.parse_move("hair_color invalid") is None
    assert game.parse_move("") is None

def test_validate_move(game, initial_state):
    """Test move validation."""
    # Valid moves
    assert game.validate_move(initial_state, 0, ("hair_color", "blonde", False))[0]
    assert game.validate_move(initial_state, 0, ("eye_color", "blue", True))[0]
    
    # Invalid player
    assert not game.validate_move(initial_state, 1, ("hair_color", "blonde", False))[0]
    
    # Invalid trait/value
    assert not game.validate_move(initial_state, 0, ("invalid", "trait", False))[0]

def test_apply_move(game, initial_state):
    """Test move application and state transitions."""
    # Make a correct guess about opponent's trait
    new_state = game.apply_move(initial_state, 0, ("hair_color", "brown", False))
    assert len(new_state.possible_characters[0]) < len(initial_state.possible_characters[0])
    assert new_state.current_player == 1

    # Make an incorrect guess - should not reduce possibilities
    new_state = game.apply_move(initial_state, 0, ("hair_color", "red", False))
    assert len(new_state.possible_characters[0]) == len(initial_state.possible_characters[0])
    assert new_state.current_player == 1

def test_game_over_conditions(game, initial_state):
    """Test game over detection and winner determination."""
    # Modify state so player 0 has narrowed down to correct character
    initial_state.possible_characters[0] = [initial_state.target_characters[1]]
    
    assert game.is_terminal(initial_state)
    assert game.get_winner(initial_state) == 0

def test_get_player_view(game, initial_state):
    """Test player view generation."""
    view = game.get_player_view(initial_state, 0)
    assert isinstance(view.rules_explanation, str)
    assert isinstance(view.move_format_instructions, str)
    assert not view.is_terminal
    assert view.winner is None

def test_serialization(initial_state):
    """Test state serialization."""
    state_dict = initial_state.to_dict()
    assert "characters" in state_dict
    assert "target_characters" in state_dict
    assert "possible_characters" in state_dict
    assert "current_player" in state_dict
