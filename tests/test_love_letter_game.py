import copy
import pytest
from bgbench.games.love_letter_game import LoveLetterGame, LoveLetterState, Card, LoveLetterMove
from bgbench.game_view import PromptStyle

@pytest.fixture
def game():
    return LoveLetterGame()

@pytest.fixture
def initial_state(game):
    return game.get_initial_state()

def test_initial_state(game, initial_state):
    # Check initial deck setup
    assert len(initial_state.deck) == 10  # 16 - 1 removed - 3 face up - 2 in hands
    assert len(initial_state.face_up_cards) == 3
    assert all(isinstance(card, Card) for card in initial_state.face_up_cards)
    assert len(initial_state.hands) == 2
    assert all(isinstance(card, Card) or card is None for card in initial_state.hands)
    assert initial_state.current_player == 0
    assert len(initial_state.protected_players) == 0
    assert initial_state.scores == [0, 0]

def test_parse_move():
    game = LoveLetterGame()
    
    # Test valid moves
    assert game.parse_move("1 1 2") == LoveLetterMove(Card.GUARD, 1, Card.PRIEST)
    assert game.parse_move("4") == LoveLetterMove(Card.HANDMAID)
    assert game.parse_move("5 1") == LoveLetterMove(Card.PRINCE, 1)
    
    # Test invalid moves
    assert game.parse_move("invalid") is None
    assert game.parse_move("9 1") is None
    assert game.parse_move("1 1 9") is None

def test_validate_move(game, initial_state):
    # Set up a known state for testing
    state = initial_state
    state.hands[0] = Card.GUARD
    state.drawn_card = Card.PRIEST
    
    # Test valid moves
    assert game.validate_move(state, 0, LoveLetterMove(Card.GUARD, 1, Card.PRIEST))[0]
    assert game.validate_move(state, 0, LoveLetterMove(Card.PRIEST, 1))[0]
    
    # Test invalid moves
    # Wrong player's turn
    assert not game.validate_move(state, 1, LoveLetterMove(Card.GUARD, 0, Card.PRIEST))[0]
    # Card not in hand
    assert not game.validate_move(state, 0, LoveLetterMove(Card.PRINCESS, 1))[0]
    # Invalid target
    # Test self-targeting rules
    assert not game.validate_move(state, 0, LoveLetterMove(Card.GUARD, 0, Card.PRIEST))[0]  # Guard can't self-target
    assert game.validate_move(state, 0, LoveLetterMove(Card.PRIEST, 0))[0]  # Priest can self-target
    assert game.validate_move(state, 0, LoveLetterMove(Card.PRINCE, 0))[0]  # Prince can self-target

def test_countess_rule(game, initial_state):
    # Test Countess must be played when holding King/Prince
    state = initial_state
    state.hands[0] = Card.COUNTESS
    state.drawn_card = Card.KING
    
    # Should not be allowed to play King when holding Countess
    assert not game.validate_move(state, 0, LoveLetterMove(Card.KING, 1))[0]
    # Should be allowed to play Countess
    assert game.validate_move(state, 0, LoveLetterMove(Card.COUNTESS))[0]

def test_priest_reveals(game, initial_state):
    # Test that Priest reveals are recorded and visible
    state = initial_state
    state.hands[0] = Card.PRIEST
    state.hands[1] = Card.KING
    state.drawn_card = Card.GUARD
    
    # Player 0 uses Priest to look at Player 1's hand
    new_state = game.apply_move(state, 0, LoveLetterMove(Card.PRIEST, 1))
    
    # Check that the reveal was recorded
    assert len(new_state.priest_views) == 1
    viewer, target, card = new_state.priest_views[0]
    assert viewer == 0
    assert target == 1
    assert card == Card.KING
    
    # Check that the reveal appears in both players' views
    view0 = game.get_player_view(new_state, 0)
    view1 = game.get_player_view(new_state, 1)
    assert any("Player 0 saw Player 1's KING" in reveal for reveal in view0.visible_state["priest_reveals"])
    assert any("Player 0 saw Player 1's KING" in reveal for reveal in view1.visible_state["priest_reveals"])

def test_handmaid_protection(game, initial_state):
    state = initial_state
    state.hands[0] = Card.GUARD
    state.drawn_card = Card.PRIEST
    state.protected_players.add(1)
    
    # Should not be able to target protected player
    assert not game.validate_move(state, 0, LoveLetterMove(Card.GUARD, 1, Card.PRIEST))[0]
    assert not game.validate_move(state, 0, LoveLetterMove(Card.PRIEST, 1))[0]

def test_guard_effect(game, initial_state):
    state = initial_state
    state.hands[0] = Card.GUARD
    state.hands[1] = Card.PRIEST
    state.drawn_card = Card.BARON
    state.scores = [0, 0]  # Reset scores to ensure we can track the win

    # Correct guess
    new_state = game.apply_move(state, 0, LoveLetterMove(Card.GUARD, 1, Card.PRIEST))
    assert new_state.scores[0] == 1  # Player 0 should win the round
    
    # Incorrect guess
    state.hands[1] = Card.PRIEST
    new_state = game.apply_move(state, 0, LoveLetterMove(Card.GUARD, 1, Card.BARON))
    assert new_state.hands[1] is not None  # Player should still be in

def test_baron_effect(game, initial_state):
    # Test Baron winning
    state = initial_state
    state.hands[0] = Card.BARON  # Baron in hand
    state.hands[1] = Card.PRIEST  # Target has Priest
    state.drawn_card = Card.KING  # Drew King
    state.deck = [Card.GUARD]  # Ensure we don't start new round
        
    # Baron wins against Priest
    initial_score = state.scores[0]
    new_state = game.apply_move(state, 0, LoveLetterMove(Card.BARON, 1))
    assert new_state.scores[0] == initial_score + 1  # Winner should get a point
    
    # Baron loses
    state.hands[0] = Card.BARON
    state.hands[1] = Card.PRINCESS
    initial_score = state.scores[1]  # Track target's score
    new_state = game.apply_move(state, 0, LoveLetterMove(Card.BARON, 1))
    assert new_state.scores[1] == initial_score + 1  # Target should get a point

def test_player_view(game, initial_state):
    state = initial_state
    state.hands[0] = Card.GUARD
    state.drawn_card = Card.PRIEST
    state.discards[1].append(Card.BARON)
    
    view = game.get_player_view(state, 0, [], PromptStyle.HEADER)
    
    # Check visible information
    assert "GUARD(1)" in view.visible_state["your_hand"]
    assert "PRIEST(2)" in view.visible_state["drawn_card"]
    assert "BARON(3)" in view.visible_state["opponent_discards"]
    assert "deck_size" not in view.visible_state  # Verify deck_size is not included
    assert isinstance(view.valid_moves, list)
    assert len(view.valid_moves) > 0

def test_card_count_verification(game, initial_state):
    # Test that verification passes for initial state
    game._verify_card_counts(initial_state)
    
    # Test that verification fails when cards are missing
    state = copy.deepcopy(initial_state)
    state.deck.pop()  # Remove a card without putting it anywhere else
    with pytest.raises(ValueError, match="Card count mismatch"):
        game._verify_card_counts(state)
    
    # Test that verification fails when cards are duplicated
    state = copy.deepcopy(initial_state)
    state.deck.append(state.deck[0])  # Duplicate a card
    with pytest.raises(ValueError, match="Card count mismatch"):
        game._verify_card_counts(state)

def test_game_end_conditions(game, initial_state):
    # Test Princess discard
    state = initial_state
    state.hands[0] = Card.PRINCESS
    state.hands[1] = Card.GUARD  # Ensure player 1 has a valid card
    state.drawn_card = Card.GUARD
    initial_score = state.scores[1]

    new_state = game.apply_move(state, 0, LoveLetterMove(Card.PRINCESS))
        
    assert new_state.scores[1] == initial_score + 1  # Other player should win the round
