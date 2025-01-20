from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from bgbench.game import Game
from bgbench.game_view import GameView, PromptStyle
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

    def to_dict(self) -> dict:
        """Convert card to JSON-serializable dictionary."""
        return {
            "rank": self.rank,
            "suit": self.suit
        }

    def __eq__(self, other):
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank == other.rank

    def __lt__(self, other):
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank < other.rank

@dataclass
class WarState:
    player_hands: List[List[Card]]
    board: List[Card]
    current_player: int
    war_state: bool = False
    cards_needed: int = 1  # Increases to 4 during war (3 face down + 1 face up)
    face_down_count: int = 0  # Track face-down cards during war

    def to_dict(self) -> dict:
        """Convert state to JSON-serializable dictionary."""
        return {
            "player_hands": [[card.to_dict() for card in hand] for hand in self.player_hands],
            "board": [card.to_dict() for card in self.board],
            "current_player": self.current_player,
            "war_state": self.war_state,
            "cards_needed": self.cards_needed,
            "face_down_count": self.face_down_count
        }

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
        return (
            "On your turn, your top card will be played automatically. "
            "During war, three cards will be placed face down and one face up. "
            "Just acknowledge your move with 'play' to continue."
        )

    def get_initial_state(self) -> WarState:
        return WarState(
            player_hands=[self.starting_hands[0][:], self.starting_hands[1][:]],
            board=[],
            current_player=0,
            war_state=False,
            cards_needed=1,
            face_down_count=0
        )

    def get_player_view(self, state: WarState, player_id: int, 
                       history: Optional[List[Dict[str, Any]]] = None,
                       prompt_style: PromptStyle = PromptStyle.HEADER) -> GameView:
        visible_state = {
            "your_cards": len(state.player_hands[player_id]),
            "opponent_cards": len(state.player_hands[1 - player_id]),
            "board": state.board,
            "war_state": state.war_state,
            "cards_needed": state.cards_needed,
            "face_down_count": state.face_down_count
        }
        
        # During normal play or when it's time for face-up card in war
        can_play = (
            len(state.player_hands[player_id]) >= state.cards_needed and
            (not state.war_state or state.face_down_count >= 3)
        )
        
        # Game is terminal if one player has all cards
        is_terminal = len(state.player_hands[0]) == 0 or len(state.player_hands[1]) == 0
        winner = None
        if is_terminal:
            winner = 1 if len(state.player_hands[0]) == 0 else 0
        
        return GameView(
            visible_state=visible_state,
            valid_moves=["play"] if can_play else [],
            is_terminal=is_terminal,
            winner=winner,
            history=history if history else [],
            move_format_instructions=self.get_move_format_instructions(),
            rules_explanation=self.get_rules_explanation(),
        )

    def parse_move(self, move_str: str) -> Optional[str]:
        # Any non-empty string is valid as we just need acknowledgment
        return move_str.strip() if move_str.strip() else None

    def validate_move(self, state: WarState, player_id: int, move: str) -> Tuple[bool, str]:
        if state.current_player != player_id:
            return False, "It's not your turn."
        if not state.player_hands[player_id]:
            return False, "You have no cards left."
            
        cards_required = state.cards_needed
        if len(state.player_hands[player_id]) < cards_required:
            if state.war_state:
                # Not enough cards for war, must forfeit
                return True, "Not enough cards for war - must forfeit"
            return False, f"You need {cards_required} cards for this play but only have {len(state.player_hands[player_id])}."
            
        return True, ""

    def apply_move(self, state: WarState, player_id: int, move: str) -> WarState:
        if not state.player_hands[player_id]:
            return state

        # Handle war state
        if state.war_state:
            # Play face down cards first
            while state.face_down_count < 3 and len(state.player_hands[player_id]) > 1:
                face_down = state.player_hands[player_id].pop(0)
                state.board.append(face_down)
                state.face_down_count += 1
            
            # Then play face up card if we have enough cards
            if len(state.player_hands[player_id]) > 0:
                face_up = state.player_hands[player_id].pop(0)
                state.board.append(face_up)
        else:
            # Normal play - just one card
            top_card = state.player_hands[player_id].pop(0)
            state.board.append(top_card)
        
        # Check if both players have played
        if len(state.board) >= 2 and len(state.board) % 2 == 0:
            card1, card2 = state.board[-2:]  # Last two cards
            
            if not state.war_state:
                if card1.rank > card2.rank:
                    state.player_hands[0].extend(state.board)
                    state.board.clear()
                elif card2.rank > card1.rank:
                    state.player_hands[1].extend(state.board)
                    state.board.clear()
                else:  # War!
                    state.war_state = True
                    state.cards_needed = 4  # 3 face down + 1 face up
                    state.face_down_count = 0
            else:  # Resolving war
                # Compare the last two face-up cards
                if card1.rank != card2.rank:
                    winner = 0 if card1.rank > card2.rank else 1
                    state.player_hands[winner].extend(state.board)
                    state.board.clear()
                    state.war_state = False
                    state.cards_needed = 1
                    state.face_down_count = 0
                else:  # Another war!
                    state.war_state = True
                    state.cards_needed = 4
                    state.face_down_count = 0  # Reset for next war round
        
        # Shuffle collected cards when adding to hand
        if not state.board:
            for hand in state.player_hands:
                if hand:
                    random.shuffle(hand)
                    
        return state

    def get_current_player(self, state: WarState) -> int:
        return state.current_player

    def get_next_state(self, state: WarState, move: str) -> WarState:
        new_state = self.apply_move(state, state.current_player, str(move))
        new_state.current_player = 1 - state.current_player
        return new_state

    def is_terminal(self, state: WarState) -> bool:
        return len(state.player_hands[0]) == 0 or len(state.player_hands[1]) == 0

    def get_winner(self, state: WarState) -> Optional[int]:
        if not self.is_terminal(state):
            return None
        return 1 if len(state.player_hands[0]) == 0 else 0
