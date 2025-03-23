from pokerkit import Card, NoLimitTexasHoldem as PKNoLimitTexasHoldem, Automation, State
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from bgbench.game import Game
from bgbench.game_view import GameView, PromptStyle
from copy import deepcopy


@dataclass
class PokerState:
    internal_state: State
    # The small_bland player alternates between each hand, and has to be mapped to the player indices of internal_state.
    # If this is 0, player 0 is actor_index 0 in internal state. If it's 1, player 0 is actor_index 1 in internal_state.
    big_blind: int
    last_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        internal_actor = self.internal_state.actor_index
        if internal_actor is None:
            current_player = None
        elif self.big_blind == 0:
            current_player = internal_actor
        else:
            current_player = 1 - internal_actor
        return {
            "current_player": current_player,
            "actor_index": internal_actor,
            "player_stacks": self.internal_state.stacks,
            "pot": sum(pot.amount for pot in self.internal_state.pots),
            "community_cards": [str(c) for c in self.internal_state.board_cards],
            "player_hands": [
                [str(c) for c in cards] for cards in self.internal_state.hole_cards
            ],
            "min_current_bet": self.internal_state.min_completion_betting_or_raising_to_amount
            or 0,
            "last_action": self.last_action,
        }

    def player_to_pk_player(self, player_idx: int) -> int:
        if self.big_blind == 0:
            return player_idx
        else:
            return 1 - player_idx

    def pk_player_to_player(self, player_idx: int) -> int:
        return self.player_to_pk_player(player_idx)

    def winner(self) -> Optional[int]:
        # A winner of the poker tournament is someone who can't post their blind, either because they have 0 left, or
        # they don't have enough to post their blind. Since blind is posted automatically, this requires that the
        # actor_index is None and the small_blind has less than small_blind amount, or big_blind has less than big_blind
        # amount.
        print(
            f"Evaluating winner, statuses: {self.internal_state.statuses}, stacks: {self.internal_state.stacks}, actor_index: {self.internal_state.actor_index}, blind_statuses: {self.internal_state.blind_or_straddle_posting_statuses}"
        )
        if self.internal_state.actor_index is None:
            if (not self.internal_state.statuses[0]) and self.internal_state.stacks[
                0
            ] < 10:
                # Small blind has lost
                return 1 - self.big_blind
            elif (not self.internal_state.statuses[1]) and self.internal_state.stacks[
                1
            ] < 20:
                return self.big_blind
        return None

    def current_player(self) -> int:
        if self.internal_state.actor_index is None:
            # Game is over
            return 0
        return self.pk_player_to_player(self.internal_state.actor_index)


def format_cards(cards: List[Card]):
    return " ".join(str(card) for card in cards)


class PokerGame(Game[PokerState, str]):
    """Poker game implementation using PokerKit as the underlying engine."""

    def __init__(self):
        pass

    def get_initial_state(self) -> PokerState:
        return PokerState(
            PKNoLimitTexasHoldem.create_state(
                automations=(
                    Automation.ANTE_POSTING,
                    Automation.BET_COLLECTION,
                    Automation.BLIND_OR_STRADDLE_POSTING,
                    Automation.CARD_BURNING,
                    Automation.HOLE_DEALING,
                    Automation.BOARD_DEALING,
                    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
                    Automation.HAND_KILLING,
                    Automation.CHIPS_PUSHING,
                    Automation.CHIPS_PULLING,
                ),  # type: ignore
                ante_trimming_status=False,
                raw_starting_stacks=(100, 100),  # Starting stacks of 100 each
                player_count=2,
                raw_antes=0,  # No antes
                raw_blinds_or_straddles=(10, 20),  # Small blind 10, big blind 20
                min_bet=20,  # Min bet is big blind size
            ),
            0,
        )

    def parse_move(self, move_str: str) -> Optional[str]:
        parts = move_str.lower().strip().split()
        if not parts:
            return None

        action = parts[0]
        if action in ["fold", "check", "call"]:
            return action
        elif action in ["bet", "raise"] and len(parts) > 1:
            try:
                amount = int(parts[1])
                return f"{action} {amount}"
            except ValueError:
                return None
        return None

    def validate_move(
        self, state: PokerState, player_id: int, move: str
    ) -> Tuple[bool, str]:
        internal_state = state.internal_state
        if state.current_player() != player_id:
            return False, "Not your turn"

        parts = move.split()
        action = parts[0]

        if action == "fold":
            if not internal_state.can_fold():
                return False, "Cannot fold right now"
            return True, ""
        elif action in ("check", "call"):
            if internal_state.can_check_or_call():
                return True, ""
            else:
                return False, "Cannot check or call right now"
        elif action in ["bet", "raise"]:
            try:
                amount = int(parts[1])
                if internal_state.can_complete_bet_or_raise_to(amount):
                    return True, ""
                else:
                    if internal_state.can_complete_bet_or_raise_to():
                        return (
                            False,
                            "Cannot complete bet or raise to any amount right now",
                        )
                    else:
                        return False, f"Cannot complete bet or raise to {amount}"
            except (IndexError, ValueError):
                return False, "Invalid bet/raise format"
        return False, "Invalid action"

    def apply_move(self, state: PokerState, player_id: int, move: str) -> PokerState:
        raise NotImplementedError()

    def get_current_player(self, state) -> int:
        return state.current_player()

    def _get_valid_moves(self, state: PokerState, player_id: int) -> List[str]:
        """Get list of valid moves for the current player."""
        valid_moves = []
        internal_state = state.internal_state

        if internal_state.can_fold():
            valid_moves.append("fold")

        if internal_state.can_check_or_call():
            valid_moves.extend(["check", "call"])

        min_amount = internal_state.min_completion_betting_or_raising_to_amount
        if min_amount is not None:
            valid_moves.extend([f"bet {min_amount}", f"raise {min_amount}"])
        return valid_moves

    def get_player_view(
        self,
        state: PokerState,
        player_id: int,
        history: Optional[List[Dict[str, Any]]] = None,
        prompt_style: PromptStyle = PromptStyle.HEADER,
    ) -> GameView:
        # Create a view that only shows the player's own cards
        board_cards = (state.internal_state.board_cards or [[]])[0]
        pk_player = state.player_to_pk_player(player_id)
        if pk_player == 0:
            blind_size = "big"
        else:
            blind_size = "small"
        print(f"actor index {state.internal_state.actor_index}")
        print(f"pk_player {pk_player}")
        visible_state = {
            "your_hand": format_cards(state.internal_state.hole_cards[pk_player]),
            "community_cards": format_cards(board_cards),
            # 2-player hold-em can't have side pots
            "pot": sum(p.amount for p in state.internal_state.pots),
            "your_bet": state.internal_state.bets[pk_player],
            "opponent_bet": state.internal_state.bets[1 - pk_player],
            "your_stack": state.internal_state.stacks[pk_player],
            "opponent_stack": state.internal_state.stacks[1 - pk_player],
            "blind": f"you are the {blind_size} blind",
        }

        return GameView(
            visible_state=visible_state,
            valid_moves=self._get_valid_moves(state, player_id),
            is_terminal=state.winner() is not None,
            winner=state.winner(),
            rules_explanation=self.get_rules_explanation(),
            move_format_instructions=self.get_move_format_instructions(),
            prompt_style=prompt_style,
        )

    def get_move_format_instructions(self) -> str:
        return (
            "Valid moves are:\n"
            "- 'fold' to fold your hand\n"
            "- 'check' to check if no bets to call\n"
            "- 'call' to match the current bet\n"
            "- 'bet X' to bet X chips (X must be >= min bet)\n"
            "- 'raise X' to raise total bet to X chips (X must be > current bet)\n"
        )

    def get_rules_explanation(self) -> str:
        return (
            "No-Limit Texas Hold'em Poker Rules:\n\n"
            "OBJECTIVE:\n"
            "- Win chips from your opponent by making the best hand or getting them to fold\n\n"
            "GAMEPLAY:\n"
            "1. Each player starts with 1000 chips\n"
            "2. Small blind (10) and big blind (20) are posted automatically\n"
            "3. Each player gets 2 private cards\n"
            "4. Betting rounds:\n"
            "   - Pre-flop: Bet after seeing your 2 cards\n"
            "   - Flop: 3 community cards dealt, betting round\n"
            "   - Turn: 4th community card dealt, betting round\n"
            "   - River: 5th community card dealt, final betting round\n\n"
            "5. Betting Options:\n"
            "   - Check: Pass action if no bet to call\n"
            "   - Call: Match the current bet\n"
            "   - Bet/Raise: Increase the betting amount\n"
            "   - Fold: Give up your hand and lose the pot\n\n"
            "6. Winning:\n"
            "   - Win all chips from your opponent\n"
            "   - Best 5-card hand using any combination of your 2 cards and 5 community cards\n"
            "   - Hand rankings from highest: Royal Flush, Straight Flush, Four of a Kind,\n"
            "     Full House, Flush, Straight, Three of a Kind, Two Pair, Pair, High Card"
        )

    def is_terminal(self, state: PokerState) -> bool:
        return state.winner() is not None

    def get_winner(self, state: PokerState) -> Optional[int]:
        return state.winner()

    def get_next_state(self, state: PokerState, move: str) -> PokerState:
        """Apply the move and return the new state."""
        new_state = deepcopy(state)
        new_state.last_action = move

        parts = move.split()
        action = parts[0]
        if action == "fold":
            new_state.internal_state.fold()
        elif action in ("check", "call"):
            new_state.internal_state.check_or_call()
        elif action in ("bet", "raise"):
            amount = int(parts[1])
            new_state.internal_state.complete_bet_or_raise_to(amount)
        else:
            RuntimeError(f"Invalid move {move}")

        is_hand_finished = not new_state.internal_state.status
        if is_hand_finished and new_state.winner() is None:
            # No final winner yet, we need a new game state while alternating the players.
            return PokerState(
                PKNoLimitTexasHoldem.create_state(
                    automations=(
                        Automation.ANTE_POSTING,
                        Automation.BET_COLLECTION,
                        Automation.BLIND_OR_STRADDLE_POSTING,
                        Automation.CARD_BURNING,
                        Automation.HOLE_DEALING,
                        Automation.BOARD_DEALING,
                        Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
                        Automation.HAND_KILLING,
                        Automation.CHIPS_PUSHING,
                        Automation.CHIPS_PULLING,
                    ),  # type: ignore
                    ante_trimming_status=False,
                    raw_starting_stacks=(
                        # Stacks alternate between games
                        new_state.internal_state.stacks[1],
                        new_state.internal_state.stacks[0],
                    ),
                    player_count=2,
                    raw_antes=0,  # No antes
                    raw_blinds_or_straddles=(
                        10,
                        20,
                    ),
                    min_bet=20,
                ),
                1 - new_state.big_blind,
                new_state.last_action,
            )
        return new_state
