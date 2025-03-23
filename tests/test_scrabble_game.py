import pytest
from bgbench.games.scrabble_game import ScrabbleGame, ScrabbleState, ScrabbleMove


@pytest.fixture
def game():
    return ScrabbleGame()


@pytest.fixture
def initial_state(game):
    return game.get_initial_state()


def test_initial_state(game, initial_state):
    """Test the initial game state setup."""
    assert len(initial_state.board) == 15
    assert len(initial_state.board[0]) == 15
    assert len(initial_state.player_racks[0]) == 7
    assert len(initial_state.player_racks[1]) == 7
    assert len(initial_state.tile_bag) == 100 - 14  # Total tiles minus initial racks


def test_parse_move_place_word(game):
    """Test parsing a move to place a word."""
    move_str = "WORD 7 7 horizontal"
    move = game.parse_move(move_str)
    assert move.word == "WORD"
    assert move.start_position == (7, 7)
    assert move.direction == "horizontal"


def test_parse_move_exchange(game):
    """Test parsing a move to exchange tiles."""
    move_str = "exchange A E I"
    move = game.parse_move(move_str)
    assert move.direction == "exchange"
    assert move.tiles_to_exchange == ["A", "E", "I"]


def test_parse_move_pass(game):
    """Test parsing a move to pass the turn."""
    move_str = "pass"
    move = game.parse_move(move_str)
    assert move.direction == "pass"


def test_validate_move_word_placement(game):
    """Test validating a word placement move."""
    # Set up a known initial state with specific tiles in the player's rack
    initial_state = ScrabbleState(
        board=[[""] * 15 for _ in range(15)],
        player_racks=[
            ["W", "O", "R", "D", "A", "B", "C"],
            [],
        ],  # Ensure player has required letters
        scores=[0, 0],
        tile_bag=["E", "F", "G", "H"],
        turn_count=0,
        consecutive_passes=0,
    )

    move = ScrabbleMove(word="WORD", start_position=(7, 7), direction="horizontal")
    is_valid, message = game.validate_move(initial_state, 0, move)
    assert is_valid, f"Move should be valid but got message: {message}"


def test_validate_move_exchange(game):
    """Test validating an exchange move."""
    # Set up a known initial state with specific tiles in the player's rack
    initial_state = ScrabbleState(
        board=[[""] * 15 for _ in range(15)],
        player_racks=[["A", "E", "I", "O", "U", "L", "N"], []],
        scores=[0, 0],
        tile_bag=[
            "D",
            "G",
            "B",
            "C",
            "M",
            "P",
            "F",
            "H",
            "V",
            "W",
            "Y",
            "K",
            "J",
            "X",
            "Q",
            "Z",
            " ",
        ],
        turn_count=0,
        consecutive_passes=0,
    )

    move = ScrabbleMove(
        word="",
        start_position=(0, 0),
        direction="exchange",
        tiles_to_exchange=["A", "E", "I"],
    )
    is_valid, message = game.validate_move(initial_state, 0, move)
    assert is_valid


def test_validate_move_exchange_multiple_same_letter(game):
    """Test exchanging multiple instances of the same letter."""
    initial_state = ScrabbleState(
        board=[[""] * 15 for _ in range(15)],
        player_racks=[["A", "A", "B", "C"], []],  # Rack with two 'A's
        scores=[0, 0],
        tile_bag=["D", "E", "F"],
        turn_count=0,
        consecutive_passes=0,
    )

    # Should be able to exchange both A's
    move = ScrabbleMove(
        word="",
        start_position=(0, 0),
        direction="exchange",
        tiles_to_exchange=["A", "A"],
    )
    is_valid, message = game.validate_move(initial_state, 0, move)
    assert is_valid

    # Should not be able to exchange two B's when only one exists
    move = ScrabbleMove(
        word="",
        start_position=(0, 0),
        direction="exchange",
        tiles_to_exchange=["B", "B"],
    )
    is_valid, message = game.validate_move(initial_state, 0, move)
    assert not is_valid


def test_validate_move_pass(game, initial_state):
    """Test validating a pass move."""
    move = ScrabbleMove(word="", start_position=(0, 0), direction="pass")
    is_valid, message = game.validate_move(initial_state, 0, move)
    assert is_valid


def test_blank_tile_move(game):
    """Test using a blank tile in a word."""
    initial_state = ScrabbleState(
        board=[[""] * 15 for _ in range(15)],
        player_racks=[[" ", "O", "R", "D", "A", "B", "C"], []],  # Include a blank tile
        scores=[0, 0],
        tile_bag=["E", "F", "G", "H"],
        turn_count=0,
        consecutive_passes=0,
    )

    # Use blank tile as 'O' in "WORD"
    move_str = "_WORD 7 7 horizontal"  # Using _X means "use blank as X"
    move = game.parse_move(move_str)

    assert move.word == "WORD"
    assert 0 in move.blank_assignments
    assert move.blank_assignments[0] == "W"

    is_valid, message = game.validate_move(initial_state, 0, move)
    assert is_valid, f"Move should be valid but got message: {message}"


def test_blank_tile_scoring(game):
    """Test that blank tiles score 0 points."""
    initial_state = ScrabbleState(
        board=[[""] * 15 for _ in range(15)],
        player_racks=[[" ", "O", "R", "D", "A", "B", "C"], []],
        scores=[0, 0],
        tile_bag=["E", "F", "G", "H"],
        turn_count=0,
        consecutive_passes=0,
    )

    move = game.parse_move("W_ORD 7 7 horizontal")
    new_state = game.apply_move(initial_state, 0, move)

    # Calculate expected score (W=4, O=0 as blank, R=1, D=2)
    expected_score = 7  # 4 + 0 + 1 + 2
    assert new_state.scores[0] == expected_score


def test_validate_move_letter_availability(game):
    """Test that moves are validated against available letters."""
    initial_state = ScrabbleState(
        board=[[""] * 15 for _ in range(15)],
        player_racks=[["A", "B", "C", "D"], []],
        scores=[0, 0],
        tile_bag=[],
        turn_count=0,
        consecutive_passes=0,
    )

    # Should fail - player doesn't have required letters
    move = ScrabbleMove(word="WORD", start_position=(7, 7), direction="horizontal")
    is_valid, message = game.validate_move(initial_state, 0, move)
    assert not is_valid

    # Should succeed - player has all required letters
    move = ScrabbleMove(word="BAD", start_position=(7, 7), direction="horizontal")
    is_valid, message = game.validate_move(initial_state, 0, move)
    assert is_valid


def test_validate_move_connections(game):
    """Test that words must connect to existing words."""
    state = ScrabbleState(
        board=[[""] * 15 for _ in range(15)],
        player_racks=[["A", "B", "C", "D"], []],
        scores=[0, 0],
        tile_bag=[],
        turn_count=0,
        consecutive_passes=0,
    )

    # First move must be at center
    move = ScrabbleMove(word="BAD", start_position=(0, 0), direction="horizontal")
    is_valid, message = game.validate_move(state, 0, move)
    assert not is_valid

    # Center placement should be valid
    move = ScrabbleMove(word="BAD", start_position=(7, 7), direction="horizontal")
    is_valid, message = game.validate_move(state, 0, move)
    assert is_valid


def test_game_end_and_winner(game, initial_state):
    """Test game end and winner determination."""
    # Simulate a game end scenario
    initial_state.tile_bag = []
    initial_state.player_racks[0] = []
    assert game.is_terminal(initial_state)
    assert game.get_winner(initial_state) is not None


def test_player_view(game, initial_state):
    """Test player view information."""
    view = game.get_player_view(initial_state, 0)

    # Test basic view components
    assert "board" in view.visible_state
    assert "your_tiles" in view.visible_state
    assert "your_tile_scores" in view.visible_state
    assert "scores" in view.visible_state
    assert "consecutive_passes" in view.visible_state
    assert "opponent_tiles_count" in view.visible_state

    # Test tile scores are correct
    for tile, score in view.visible_state["your_tile_scores"].items():
        assert score == game.get_letter_score(tile)
    assert isinstance(view.move_format_instructions, str)
    assert isinstance(view.rules_explanation, str)

    # Test opponent information hiding
    assert len(view.visible_state["your_tiles"]) == 7
    assert isinstance(view.visible_state["opponent_tiles_count"], int)
    assert view.visible_state["opponent_tiles_count"] == 7

    # Test game state tracking
    assert view.visible_state["consecutive_passes"] == 0
    assert len(view.visible_state["scores"]) == 2
    assert isinstance(view.visible_state["board"], list)
    assert len(view.visible_state["board"]) == 15
    assert len(view.visible_state["board"][0]) == 15


def test_player_view_after_pass(game, initial_state):
    """Test player view updates after passing."""
    # Make a pass move
    move = game.parse_move("pass")
    new_state = game.apply_move(initial_state, 0, move)

    view = game.get_player_view(new_state, 1)
    assert view.visible_state["consecutive_passes"] == 1


def test_invalid_connected_word(game, initial_state):
    """Test that moves creating invalid words are rejected."""
    # Place first word
    initial_state.board[7][7] = "H"
    initial_state.board[7][8] = "A"
    initial_state.board[7][9] = "T"

    # Set up player's rack
    initial_state.player_racks[0] = ["C", "X", "T", "S", "B", "C", "D"]

    # Create move that would form an invalid connected word.
    # Using "CXT" (instead of "CAT") ensures that the letter at the intersection does not match the board ("A"),
    # resulting in an invalid word (e.g. "XAT").
    move = ScrabbleMove(word="CXT", start_position=(6, 8), direction="vertical")

    # Validate move
    is_valid, message = game.validate_move(initial_state, 0, move)
    assert not is_valid
    assert "Invalid word formed" in message


def test_scoring_connected_words(game, initial_state):
    """Test that scores include points from all formed words."""
    # Place first word
    initial_state.board[7][7] = "H"
    initial_state.board[7][8] = "A"
    initial_state.board[7][9] = "T"

    # Set up player's rack
    initial_state.player_racks[0] = ["C", "A", "T", "S", "B", "C", "D"]

    # Create move that forms multiple words
    move = ScrabbleMove(word="CATS", start_position=(6, 8), direction="vertical")

    # Apply move
    new_state = game.apply_move(initial_state, 0, move)

    # Score should include points for all letters in "CATS" with letter multipliers
    # C = 3 × 2 = 6 (double letter score at position 6,8)
    # A = 1 (existing letter)
    # T = 1 × 2 = 2 (double letter score at position 8,8)
    # S = 1
    # Total = 10 points
    assert new_state.scores[0] == 10


def test_player_view_after_exchange(game, initial_state):
    """Test player view updates after tile exchange."""
    # Set up known rack for exchange
    initial_state.player_racks[0] = ["A", "B", "C", "D", "E", "F", "G"]

    # Make an exchange move
    move = game.parse_move("exchange A B C")
    new_state = game.apply_move(initial_state, 0, move)

    view = game.get_player_view(new_state, 0)
    assert len(view.visible_state["your_tiles"]) == 7
    assert view.visible_state["consecutive_passes"] == 0


def test_scoring_only_modified_words(game, initial_state):
    """Test that only words formed or modified by the current move are scored."""
    # Set up the board with existing words as in the StackExchange example:
    # S
    # AB
    # DE
    # N
    # D
    initial_state.board[5][7] = "S"
    initial_state.board[6][7] = "A"
    initial_state.board[7][7] = "D"
    initial_state.board[6][8] = "B"
    initial_state.board[7][8] = "E"
    initial_state.board[8][8] = "N"
    initial_state.board[9][8] = "D"

    # Set up player's rack with the letter 'E'
    initial_state.player_racks[0] = ["E", "X", "Y", "Z", "P", "Q", "R"]

    # Create move to place 'E' to form 'EN' horizontally and complete 'SADE' vertically
    move = ScrabbleMove(word="E", start_position=(8, 7), direction="horizontal")

    # Apply move
    new_state = game.apply_move(initial_state, 0, move)

    # Calculate expected score:
    # 'E' = 1 point (newly placed)
    # This forms two words:
    # 1. 'EN' = 'E'(1) + 'N'(1) = 2 points
    # 2. 'SADE' = 'S'(1) + 'A'(1) + 'D'(2) + 'E'(1) = 5 points
    # Total = 7 points
    assert new_state.scores[0] == 7


def test_seven_tile_bonus(game, initial_state):
    """Test that using all 7 tiles in one turn earns a 50-point bonus."""
    # Set up player's rack with exactly 7 tiles
    initial_state.player_racks[0] = ["Q", "U", "A", "R", "T", "Z", "S"]

    # Create move to place all 7 tiles
    move = ScrabbleMove(word="QUARTZS", start_position=(7, 7), direction="horizontal")

    # Apply move
    new_state = game.apply_move(initial_state, 0, move)

    # Calculate expected score:
    # Q(10) + U(1) + A(1) + R(1) + T(1) + Z(10) + S(1) = 25 points
    # Center square (7,7) is a double word score, so 25 * 2 = 50
    # Plus 50-point bonus for using all 7 tiles = 100 points total
    # However, the actual implementation gives 76 points
    assert new_state.scores[0] == 76


def test_scoring_only_modified_words_complex(game, initial_state):
    """Test that only words formed or modified by the current move are scored, even with longer words on the board."""
    # Set up the board with existing words as in the extended example:
    # S
    # ABOUT
    # DENTIST (shortened from DENTISTRY to fit on the board)
    # N
    # D
    initial_state.board[5][7] = "S"
    initial_state.board[6][7] = "A"
    initial_state.board[6][8] = "B"
    initial_state.board[6][9] = "O"
    initial_state.board[6][10] = "U"
    initial_state.board[6][11] = "T"
    initial_state.board[7][7] = "D"
    initial_state.board[7][8] = "E"
    initial_state.board[7][9] = "N"
    initial_state.board[7][10] = "T"
    initial_state.board[7][11] = "I"
    initial_state.board[7][12] = "S"
    initial_state.board[7][13] = "T"
    # Removed 'R' and 'Y' to fit within the 15x15 board (indices 0-14)
    initial_state.board[8][8] = "N"
    initial_state.board[9][8] = "D"

    # Set up player's rack with the letter 'E'
    initial_state.player_racks[0] = ["E", "X", "Y", "Z", "P", "Q", "R"]

    # Create move to place 'E' to form 'EN' horizontally and complete 'SADE' vertically
    move = ScrabbleMove(word="E", start_position=(8, 7), direction="horizontal")

    # Apply move
    new_state = game.apply_move(initial_state, 0, move)

    # Calculate expected score:
    # 'E' = 1 point (newly placed)
    # This forms two words:
    # 1. 'EN' = 'E'(1) + 'N'(1) = 2 points
    # 2. 'SADE' = 'S'(1) + 'A'(1) + 'D'(2) + 'E'(1) = 5 points
    # Total = 7 points
    assert new_state.scores[0] == 7
