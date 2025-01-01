import pytest
from bgbench.games.war_game import WarGame, WarState, Card
from bgbench.game_view import GameView

@pytest.fixture
def war_game():
    return WarGame()

@pytest.fixture
def simple_state():
    # Create a simple state with known cards
    player1_hand = [Card(14, 0), Card(10, 0)]  # Ace and 10 of Hearts
    player2_hand = [Card(13, 0), Card(9, 0)]   # King and 9 of Hearts
    return WarState(
        player_hands=[player1_hand, player2_hand],
        board=[],
        current_player=0,
        war_state=False,
        cards_needed=1,
        face_down_count=0
    )

def test_card_comparison():
    """Test card comparison operations"""
    ace = Card(14, 0)
    king = Card(13, 0)
    ace2 = Card(14, 1)
    
    assert ace > king
    assert king < ace
    assert ace == ace2  # Only rank matters for comparison
    assert str(ace) == "A♥"
    assert str(king) == "K♥"

def test_initial_state(war_game):
    """Test initial game state setup"""
    state = war_game.get_initial_state()
    
    assert len(state.player_hands) == 2
    assert len(state.player_hands[0]) == 26
    assert len(state.player_hands[1]) == 26
    assert not state.board
    assert state.current_player == 0
    assert not state.war_state
    assert state.cards_needed == 1
    assert state.face_down_count == 0

def test_player_view(war_game, simple_state):
    """Test player view generation"""
    view = war_game.get_player_view(simple_state, 0)
    
    assert isinstance(view, GameView)
    if isinstance(view.visible_state, dict):
        assert view.visible_state.get("your_cards") == 2
        assert view.visible_state.get("opponent_cards") == 2
        assert view.visible_state.get("board") == []
    else:
        # Handle string case - this test expects a dict so fail if we get a string
        assert False, "Expected visible_state to be a dictionary"
    assert "play" in view.valid_moves

def test_normal_play(war_game, simple_state):
    """Test normal card play without war"""
    # Validate initial move
    is_valid, _ = war_game.validate_move(simple_state, 0, "play")
    assert is_valid
    
    # Apply move for player 1
    new_state = war_game.apply_move(simple_state, 0, "play")
    assert len(new_state.board) == 1
    assert new_state.board[0].rank == 14  # Ace played
    
    # Apply move for player 2
    new_state = war_game.apply_move(new_state, 1, "play")
    assert len(new_state.board) == 0  # Board cleared after resolution
    assert len(new_state.player_hands[0]) == 3  # Player 1 won both cards
    assert len(new_state.player_hands[1]) == 1

def test_war_scenario():
    """Test war scenario when equal cards are played"""
    # Create a state that will trigger war
    player1_hand = [Card(10, 0), Card(2, 0), Card(3, 0), Card(4, 0), Card(14, 0)]
    player2_hand = [Card(10, 1), Card(5, 0), Card(6, 0), Card(7, 0), Card(13, 0)]
    state = WarState(
        player_hands=[player1_hand, player2_hand],
        board=[],
        current_player=0,
        war_state=False,
        cards_needed=1,
        face_down_count=0
    )
    
    game = WarGame()
    
    # First player plays
    state = game.apply_move(state, 0, "play")
    assert len(state.board) == 1
    
    # Second player plays matching card
    state = game.apply_move(state, 1, "play")
    assert state.war_state
    assert state.cards_needed == 4
    
    # Players place war cards
    for _ in range(2):  # Both players play their war cards
        state = game.apply_move(state, 0, "play")
        state = game.apply_move(state, 1, "play")
    
    # Verify war resolution
    assert not state.war_state
    assert len(state.board) == 0  # Cards have been collected
    assert len(state.player_hands[0]) + len(state.player_hands[1]) == 10  # All cards accounted for

def test_insufficient_cards_for_war():
    """Test handling when a player doesn't have enough cards for war"""
    player1_hand = [Card(10, 0)]  # Only one card
    player2_hand = [Card(10, 1), Card(5, 0), Card(6, 0), Card(7, 0)]
    state = WarState(
        player_hands=[player1_hand, player2_hand],
        board=[],
        current_player=0,
        war_state=True,
        cards_needed=4,
        face_down_count=0
    )
    
    game = WarGame()
    is_valid, message = game.validate_move(state, 0, "play")
    assert is_valid  # Should be valid but will trigger forfeit
    assert "forfeit" in message.lower()

def test_game_end_detection(war_game):
    """Test game end detection when one player has all cards"""
    # Create end-game state where player 1 has all cards
    state = WarState(
        player_hands=[[Card(2, 0)], []],
        board=[],
        current_player=0,
        war_state=False,
        cards_needed=1,
        face_down_count=0
    )
    
    view = war_game.get_player_view(state, 0)
    assert view.is_terminal
    assert view.winner == 0  # Player 1 wins
