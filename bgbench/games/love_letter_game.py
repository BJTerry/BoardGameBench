from dataclasses import dataclass
from enum import IntEnum
from typing import List, Dict, Any, Optional, Tuple, Set, Union, cast
from bgbench.game import Game
from bgbench.game_view import GameView, PromptStyle
import random
import copy

class Card(IntEnum):
    GUARD = 1
    PRIEST = 2
    BARON = 3
    HANDMAID = 4
    PRINCE = 5
    KING = 6
    COUNTESS = 7
    PRINCESS = 8

@dataclass
class LoveLetterState:
    """Represents the current state of a Love Letter game."""
    deck: List[Card]  # Remaining cards in deck
    hands: List[Optional[Card]]  # Current card each player holds (None if out)
    discards: List[List[Card]]  # List of discarded cards per player
    current_player: int
    protected_players: Set[int]  # Players protected by Handmaid
    removed_card: Optional[Card]  # Card removed at start of round
    face_up_cards: List[Card]  # Cards revealed at start (2-player game)
    scores: List[int]  # Tokens of affection per player
    drawn_card: Optional[Card] = None  # Card drawn at start of turn
    
    def to_dict(self) -> dict:
        return {
            "deck": [c.value for c in self.deck],
            "hands": [c.value if c else None for c in self.hands],
            "discards": [[c.value for c in player_discards] for player_discards in self.discards],
            "current_player": self.current_player,
            "protected_players": list(self.protected_players),
            "removed_card": self.removed_card.value if self.removed_card else None,
            "face_up_cards": [c.value for c in self.face_up_cards],
            "scores": self.scores,
            "drawn_card": self.drawn_card.value if self.drawn_card else None
        }

@dataclass
class LoveLetterMove:
    """Represents a move in Love Letter."""
    card: Card  # Card being played
    target_player: Optional[int] = None  # Target player for card effects
    named_card: Optional[Card] = None  # For Guard's guess

    def to_dict(self) -> dict:
        return {
            "card": self.card.value,
            "target_player": self.target_player,
            "named_card": self.named_card.value if self.named_card else None
        }

class LoveLetterGame(Game[LoveLetterState, LoveLetterMove]):
    """Implementation of Love Letter card game."""
    
    def __init__(self):
        # Create deck with correct card quantities (16 total)
        self.deck = (
            [Card.GUARD] * 5 +      # 5 Guards
            [Card.PRIEST] * 2 +     # 2 Priests
            [Card.BARON] * 2 +      # 2 Barons
            [Card.HANDMAID] * 2 +   # 2 Handmaids
            [Card.PRINCE] * 2 +     # 2 Princes
            [Card.KING] +           # 1 King
            [Card.COUNTESS] +       # 1 Countess
            [Card.PRINCESS]         # 1 Princess
        )
        self.target_score = 7  # First to 7 tokens wins in 2-player game

    def validate_move(self, state: LoveLetterState, player_id: int, move: LoveLetterMove) -> Tuple[bool, str]:
        """Validate if a move is legal in the current state."""
        # Check if it's the player's turn
        if state.current_player != player_id:
            return False, "Not your turn"

        # Check if player has the card they want to play
        available_cards = [state.hands[player_id], state.drawn_card]
        if move.card not in available_cards:
            return False, "You don't have that card"

        # Check Countess rule - must play Countess if holding King or Prince
        if (Card.COUNTESS in available_cards and 
            (Card.KING in available_cards or Card.PRINCE in available_cards) and
            move.card != Card.COUNTESS):
            return False, "Must play Countess when holding King or Prince"

        # Validate target player if required
        if move.target_player is not None:
            # Check target is valid player number
            if move.target_player not in [0, 1]:
                return False, "Invalid target player"
            # Can't target self with most cards
            if move.target_player == player_id and move.card != Card.PRINCE:
                return False, "Cannot target yourself"
            # Can't target protected player
            if move.target_player in state.protected_players:
                return False, "Target player is protected by Handmaid"
            # Can't target player who is out
            if state.hands[move.target_player] is None:
                return False, "Target player is out of the round"

        # Card-specific validation
        if move.card == Card.GUARD:
            if move.target_player is None or move.named_card is None:
                return False, "Guard requires target player and named card"
            if move.named_card == Card.GUARD:
                return False, "Guard cannot name Guard"
        elif move.card in [Card.PRIEST, Card.BARON, Card.KING, Card.PRINCE]:
            if move.target_player is None:
                return False, "This card requires a target player"

        return True, ""
        
    def parse_move(self, move_str: str) -> Optional[LoveLetterMove]:
        """Parse a move string into a LoveLetterMove object."""
        try:
            parts = move_str.strip().upper().split()
            if not parts:
                return None
                
            # Parse card to play
            try:
                card = Card(int(parts[0]))
            except (ValueError, IndexError):
                return None
                
            # Parse target player if provided
            target_player = None
            if len(parts) > 1 and parts[1].isdigit():
                target_player = int(parts[1])
                
            # Parse named card for Guard
            named_card = None
            if len(parts) > 2 and card == Card.GUARD:
                try:
                    named_card = Card(int(parts[2]))
                except ValueError:
                    return None
                    
            return LoveLetterMove(card, target_player, named_card)
        except (ValueError, IndexError):
            return None

    def apply_move(self, state: LoveLetterState, player_id: int, move: LoveLetterMove) -> LoveLetterState:
        """Apply move to state and return new state."""
        # Make a deep copy to avoid modifying the original state
        state = copy.deepcopy(state)

        # Clear Handmaid protection from previous round
        state.protected_players.clear()

        # Get the card being played and discard it
        played_card = move.card

        # Determine which card the player is keeping
        if state.hands[player_id] == played_card:
            # Played the original hand card; keep the drawn card
            state.hands[player_id] = state.drawn_card
        elif state.drawn_card == played_card:
            # Played the drawn card; keep the original hand card
            # Hand remains unchanged
            pass
        else:
            # Played card is not in hand or drawn card
            raise ValueError("Played card not found in hand or drawn card")

        # Add played card to player's discard pile
        state.discards[player_id].append(cast(Card, played_card))

        # Apply card effects
        if played_card == Card.GUARD:
            if move.target_player is not None and move.named_card is not None:
                target_card = state.hands[move.target_player]
                if target_card == move.named_card:
                    # Correct guess - target is out
                    state.discards[move.target_player].append(cast(Card, target_card))
                    state.hands[move.target_player] = None
                    # Clear drawn card since player was eliminated
                    state.drawn_card = None
                    # Check if round is over
                    active_players = sum(1 for hand in state.hands if hand is not None)
                    if active_players <= 1:
                        winner = next((i for i, hand in enumerate(state.hands) if hand is not None), None)
                        if winner is not None:
                            state.scores[winner] += 1
                            # Start new round while preserving scores
                            new_state = self.get_initial_state()
                            new_state.scores = state.scores.copy()
                            return new_state

        elif played_card == Card.PRIEST:
            # Effect handled in player view - no state change needed
            pass

        elif played_card == Card.BARON:
            if move.target_player is not None:
                player_card = state.hands[player_id]
                target_card = state.hands[move.target_player]
                if player_card is not None and target_card is not None:
                    if player_card.value > target_card.value:
                        # Player wins, target is out
                        state.discards[move.target_player].append(cast(Card, target_card))
                        state.hands[move.target_player] = None
                        # Keep the higher card in player's hand
                        state.hands[player_id] = player_card
                        # Clear drawn card since we're keeping original hand
                        state.drawn_card = None
                        
                        # Check active players and update score
                        active_players = sum(1 for hand in state.hands if hand is not None)
                        if active_players <= 1:
                            state.scores[player_id] += 1
                            # Start new round while preserving scores
                            new_state = self.get_initial_state()
                            new_state.scores = state.scores.copy()
                            return new_state
                    elif target_card.value > player_card.value:
                        # Target wins, player is out
                        state.discards[player_id].append(cast(Card, player_card))
                        state.hands[player_id] = None
                        # Check active players and update score
                        active_players = sum(1 for hand in state.hands if hand is not None)
                        if active_players <= 1:
                            winner = next((i for i, hand in enumerate(state.hands) if hand is not None), None)
                            if winner is not None:
                                state.scores[winner] += 1
                                # Start new round while preserving scores
                                new_state = self.get_initial_state()
                                new_state.scores = state.scores.copy()
                                return new_state
                    # Clear drawn card in both cases
                    state.drawn_card = None

        elif played_card == Card.HANDMAID:
            state.protected_players.add(player_id)

        elif played_card == Card.PRINCE:
            # Target discards their hand and draws a new card
            if move.target_player is not None and state.hands[move.target_player]:
                if state.hands[move.target_player] is not None:
                    discarded = state.hands[move.target_player]
                    state.discards[cast(int, move.target_player)].append(cast(Card, discarded))
                if state.deck:
                    state.hands[cast(int, move.target_player)] = state.deck.pop()
                else:
                    state.hands[cast(int, move.target_player)] = state.removed_card

        elif played_card == Card.KING:
            # Swap hands between player and target
            if move.target_player is not None:
                state.hands[player_id], state.hands[move.target_player] = \
                    state.hands[move.target_player], state.hands[player_id]

        elif played_card == Card.PRINCESS:
            # Playing Princess means you're out immediately
            state.discards[player_id].append(played_card)
            state.hands[player_id] = None
            # Clear drawn card since player is eliminated
            state.drawn_card = None
            
            # Count active players
            active_players = sum(1 for hand in state.hands if hand is not None)
            
            # Find winner
            winner = next((i for i, hand in enumerate(state.hands) if hand is not None), None)
            
            if winner is not None:
                state.scores[winner] += 1
                
                # Start new round while preserving scores
                new_state = self.get_initial_state()
                new_state.scores = state.scores.copy()
                return new_state

        # Clear the drawn card
        state.drawn_card = None

        # Check if round is over
        active_players = sum(1 for hand in state.hands if hand is not None)
        if active_players <= 1:
            # Award token to winner
            winner = next((i for i, hand in enumerate(state.hands) if hand is not None), None)
            if winner is not None:
                state.scores[winner] += 1
            # Set up new round if game not over
            if max(state.scores) < self.target_score:
                state = self.get_initial_state()
                state.scores = state.scores  # Preserve scores
        else:
            # Move to next player who's still in
            next_player = (player_id + 1) % len(state.hands)
            while state.hands[next_player] is None:
                next_player = (next_player + 1) % len(state.hands)
            state.current_player = next_player

            # Draw card for next player's turn if they are still in the game
            if state.hands[state.current_player] is not None:
                if state.deck:
                    state.drawn_card = state.deck.pop()
                    # Update deck size after drawing
                    state.deck_size = len(state.deck)
                else:
                    # If deck is empty, highest card wins the round
                    active_players = [i for i, hand in enumerate(state.hands) if hand is not None]
                    if len(active_players) > 1:
                        # Find player with highest card
                        highest_value = max(state.hands[p].value for p in active_players if state.hands[p] is not None)
                        winners = [p for p in active_players if state.hands[p] is not None and state.hands[p].value == highest_value]
                        for winner in winners:
                            state.scores[winner] += 1
                        # Start new round while preserving scores
                        new_state = self.get_initial_state()
                        new_state.scores = state.scores.copy()
                        return new_state
                    state.drawn_card = None
            else:
                # Skip drawing if the next player is eliminated
                state.drawn_card = None

        return state

    def get_player_view(self, state: LoveLetterState, player_id: int,
                       history: Optional[List[Dict[str, Any]]] = None,
                       prompt_style: PromptStyle = PromptStyle.HEADER) -> GameView:
        """Return what this player can see of the current state."""
        # Format visible information
        visible_state = {
            "your_hand": self._format_visible_cards(cast(List[Union[Card, None]], [state.hands[player_id]] if state.hands[player_id] is not None else [])),
            "drawn_card": self._format_visible_cards(cast(List[Union[Card, None]], [state.drawn_card] if state.drawn_card else [])),
            "your_discards": self._format_visible_cards(cast(List[Union[Card, None]], state.discards[player_id])),
            "opponent_discards": self._format_visible_cards(state.discards[1-player_id]),
            "face_up_cards": self._format_visible_cards(state.face_up_cards),
            "protected_players": list(state.protected_players),
            "deck_size": len(state.deck),
            "scores": state.scores
        }

        # Format move history
        move_history = []
        if history:
            for turn in history:
                move = turn.get("move")
                if move and isinstance(move, LoveLetterMove):
                    player = turn["player"]
                    card_name = move.card.name
                    target_desc = ""
                    if move.target_player is not None:
                        target_desc = f" targeting Player {move.target_player}"
                        if move.card == Card.GUARD and move.named_card:
                            target_desc += f", guessing {move.named_card.name}"
                    move_history.append(f"Turn {turn['turn']}: Player {player} played {card_name}{target_desc}")

        visible_state["move_history"] = move_history

        rules_explanation = (
            "Love Letter is a game of risk and deduction.\n"
            "On your turn, draw a card and play one of your two cards.\n"
            "Each card has a special effect when played.\n"
            "Card values and effects:\n"
            "1 Guard: Guess another player's card (2-8)\n"
            "2 Priest: Look at another player's hand\n"
            "3 Baron: Compare hands with another player\n"
            "4 Handmaid: Protection until your next turn\n"
            "5 Prince: Force player to discard their hand\n"
            "6 King: Trade hands with another player\n"
            "7 Countess: Must discard if with King/Prince\n"
            "8 Princess: Lose if discarded\n"
            f"First to {self.target_score} tokens wins."
        )

        return GameView(
            rules_explanation=rules_explanation,
            visible_state=visible_state,
            valid_moves=self._get_valid_moves(state, player_id),
            is_terminal=max(state.scores) >= self.target_score,
            winner=next((i for i, score in enumerate(state.scores) if score >= self.target_score), None),
            history=history if history else [],
            move_format_instructions=self.get_move_format_instructions(),
            prompt_style=prompt_style
        )

    def _get_valid_moves(self, state: LoveLetterState, player_id: int) -> List[str]:
        """Return list of valid moves in string format."""
        if state.current_player != player_id:
            return []
            
        valid_moves = []
        available_cards = [state.hands[player_id], state.drawn_card]
        
        for card in available_cards:
            if card is None:
                continue
                
            # Must play Countess if holding King/Prince
            if (Card.COUNTESS in available_cards and 
                (Card.KING in available_cards or Card.PRINCE in available_cards)):
                return [str(Card.COUNTESS.value)]
                
            # Add basic play without target
            valid_moves.append(str(card.value))
            
            # Add targeted moves if applicable
            if card in [Card.GUARD, Card.PRIEST, Card.BARON, Card.PRINCE, Card.KING]:
                for target in range(2):
                    if (target != player_id and 
                        state.hands[target] is not None and 
                        target not in state.protected_players):
                        if card == Card.GUARD:
                            # Add all possible guard guesses
                            for guess in range(2, 9):  # Can't guess Guard
                                valid_moves.append(f"{card.value} {target} {guess}")
                        else:
                            valid_moves.append(f"{card.value} {target}")
                            
        return valid_moves

    def get_move_format_instructions(self) -> str:
        return (
            "Format: CARD [TARGET_PLAYER] [NAMED_CARD]\n"
            "- CARD: number 1-8 representing the card to play\n"
            "- TARGET_PLAYER: (optional) player number to target (0 or 1)\n"
            "- NAMED_CARD: (only for Guard) number 2-8 representing the guessed card\n"
            "Card numbers:\n"
            "1=Guard, 2=Priest, 3=Baron, 4=Handmaid, 5=Prince,\n"
            "6=King, 7=Countess, 8=Princess\n"
            "Examples:\n"
            "- '1 1 2' plays Guard, targets player 1, guesses Priest\n"
            "- '4' plays Handmaid (no target needed)\n"
            "- '5 1' plays Prince targeting player 1"
        )

    def _format_visible_cards(self, cards: Union[List[Card], List[Optional[Card]]]) -> str:
        """Format a list of cards for display."""
        if not cards:
            return "None"
        return ", ".join(f"{card.name}({card.value})" for card in cards if card is not None)

    def get_initial_state(self) -> LoveLetterState:
        """Set up initial game state."""
        # Shuffle deck
        deck = self.deck.copy()
        random.shuffle(deck)
        
        # Remove one card face-down
        removed_card = deck.pop()
        
        # In 2-player game, remove 3 cards face-up
        face_up_cards = [deck.pop() for _ in range(3)]
        
        # Deal one card to each player
        hands: List[Optional[Card]] = [deck.pop(), deck.pop()]
        
        return LoveLetterState(
            deck=deck,
            hands=hands,
            discards=[[], []],  # Empty discard piles for each player
            current_player=0,
            protected_players=set(),
            removed_card=removed_card,
            face_up_cards=face_up_cards,
            scores=[0, 0],
            drawn_card=None
        )

    def get_current_player(self, state: LoveLetterState) -> int:
        """Return the ID of the player whose turn it is."""
        return state.current_player

    def get_next_state(self, state: LoveLetterState, move: LoveLetterMove) -> LoveLetterState:
        """Return the next state after applying the move."""
        # This is just a wrapper around apply_move since we already have that implementation
        return self.apply_move(state, self.get_current_player(state), move)
