from typing import List, Any, Protocol
from dataclasses import dataclass


@dataclass
class Pot:
    amount: int


class NoLimitTexasHoldem(Protocol):
    player_indices: List[int]
    stacks: List[int]
    pots: List[Pot]
    board_cards: List[Any]
    hole_cards: List[List[Any]]
    min_completion_betting_or_raising_to_amount: int
    status: bool

    def can_fold(self) -> bool: ...
    def fold(self) -> None: ...
    def check_or_call(self) -> None: ...
    def complete_bet_or_raise_to(self, amount: int) -> None: ...
