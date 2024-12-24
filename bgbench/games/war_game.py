from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from bgbench.game import Game
from bgbench.game_view import GameView
import random

@dataclass
class Card:
    rank: int  # 2-14 (2-Ace)
    suit: int  # 0-3 (Hearts, Diamonds, Clubs, Spades)

    def __str__(self):
        ranks = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
        suits = ['♥', '♦', '♣', '♠']
        rank_str = ranks.get(self.rank, str(self.rank))
        return f"{rank_str}{suits[self.suit]}"

@dataclass
class WarState:
    player_hands: List[List[Card]]
    board: List[Card]
    current_player: int
    war_state: bool = False
    cards_needed: int = 1  # Increases to 4 during war (3 face down + 1 face up)

class WarGame(Game):
    def __init__(self):
        # Create deck of 52 cards
        self.deck = [
            Card(rank, suit)
            for rank in range(2, 15)  # 2 through Ace
            for suit in range(4)
        ]
        random.shuffle(self.deck)
        self.starting_hands = [self.deck[:26], self.deck[26:]]

    def get_rules_explanation(self) -> str:
        return (
            "We are playing War with a standard 52-card deck. Each player has half the deck. "
            "On each turn, both players reveal their top card. The player with the higher card "
            "wins both cards and puts them at the bottom of their deck. "
            "If the cards match in rank, we enter 'War': "
            "Each player places three cards face down and one card face up. "
            "The player with the higher face-up card wins all cards played. "
            "If there's another tie, the War process repeats. "
            "The game ends when one player has all the cards."
        )

    def get_move_format_instructions(self) -> str:
        return "On your turn, simply reveal the top card of your deck."

    def get_initial_state(self) -> WarState:
        return WarState(
            player_hands=[self.starting_hands[0][:], self.starting_hands[1][:]],
            board=[],
            current_player=0
        )

    def get_player_view(self, state: WarState, player_id: int, history: List[Dict[str, Any]] = None) -> GameView:
        visible_state = {
            "your_hand": state.player_hands[player_id],
            "opponent_cards": len(state.player_hands[1 - player_id]),
            "board": state.board
        }
        return GameView(
            visible_state=visible_state,
            valid_moves=[state.player_hands[player_id][0]] if state.player_hands[player_id] else [],
            is_terminal=not any(state.player_hands),
            winner=0 if not state.player_hands[1] else 1 if not state.player_hands[0] else None,
            history=history if history else []
        )

    def parse_move(self, move_str: str) -> Optional[Card]:
        try:
            # Since we only ever play the top card, we just need to validate
            # that the string represents a valid card format
            if not move_str or len(move_str) < 2:
                return None
            return None  # Just return None as we'll get the actual card from the hand
        except ValueError:
            return None

    def validate_move(self, state: WarState, player_id: int, move: Card) -> Tuple[bool, str]:
        if state.current_player != player_id:
            return False, "It's not your turn."
        if not state.player_hands[player_id]:
            return False, "You have no cards left."
            
        cards_required = state.cards_needed
        if len(state.player_hands[player_id]) < cards_required:
            return False, f"You need {cards_required} cards for this play but only have {len(state.player_hands[player_id])}."
            
        # In War, we always just play the top card(s)
        return True, ""

    def apply_move(self, state: WarState, player_id: int, move: Card) -> WarState:
        # Remove played card from hand
        state.board.append(move)
        state.player_hands[player_id].pop(0)
        
        if len(state.board) == state.cards_needed * 2:  # Both players played required cards
            if not state.war_state:  # Normal play
                card1, card2 = state.board[-2:]  # Last two cards
                if card1.rank > card2.rank:
                    state.player_hands[0].extend(state.board)
                    state.board.clear()
                elif card2.rank > card1.rank:
                    state.player_hands[1].extend(state.board)
                    state.board.clear()
                else:  # War!
                    state.war_state = True
                    state.cards_needed = 4  # 3 face down + 1 face up
            else:  # Resolving war
                card1, card2 = state.board[-2:]  # Compare last cards
                if card1.rank != card2.rank:
                    winner = 0 if card1.rank > card2.rank else 1
                    state.player_hands[winner].extend(state.board)
                    state.board.clear()
                    state.war_state = False
                    state.cards_needed = 1
                # If another tie, war_state stays True and we continue
        
        return state

    def get_current_player(self, state: WarState) -> int:
        return state.current_player

    def get_next_state(self, state: WarState, move: int) -> WarState:
        new_state = self.apply_move(state, state.current_player, move)
        new_state.current_player = 1 - state.current_player
        return new_state
