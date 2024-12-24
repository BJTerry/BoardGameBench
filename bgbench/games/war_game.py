from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from bgbench.game import Game
from bgbench.game_view import GameView
import random

@dataclass
class WarState:
    player_hands: List[List[int]]
    board: List[int]
    current_player: int

class WarGame(Game):
    def __init__(self):
        self.deck = list(range(1, 53))  # Simplified deck representation
        random.shuffle(self.deck)
        self.starting_hands = [self.deck[:26], self.deck[26:]]

    def get_rules_explanation(self) -> str:
        return (
            "We are playing War. Each player has a deck of cards. "
            "On each turn, both players reveal the top card of their deck. "
            "The player with the higher card wins the round and takes both cards. "
            "If the cards are equal, it's a 'war': each player places three cards face down "
            "and then reveals a fourth card. The player with the higher fourth card wins all the cards. "
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

    def parse_move(self, move_str: str) -> Optional[int]:
        try:
            return int(move_str)
        except ValueError:
            return None

    def validate_move(self, state: WarState, player_id: int, move: int) -> Tuple[bool, str]:
        if state.current_player != player_id:
            return False, "It's not your turn."
        if not state.player_hands[player_id]:
            return False, "You have no cards left."
        if move != state.player_hands[player_id][0]:
            return False, "You must play the top card of your deck."
        return True, ""

    def apply_move(self, state: WarState, player_id: int, move: int) -> WarState:
        state.board.append(move)
        state.player_hands[player_id].pop(0)
        if len(state.board) == 2:
            # Resolve the round
            if state.board[0] > state.board[1]:
                state.player_hands[0].extend(state.board)
            elif state.board[1] > state.board[0]:
                state.player_hands[1].extend(state.board)
            state.board.clear()
        return state

    def get_current_player(self, state: WarState) -> int:
        return state.current_player

    def get_next_state(self, state: WarState, move: int) -> WarState:
        new_state = self.apply_move(state, state.current_player, move)
        new_state.current_player = 1 - state.current_player
        return new_state
