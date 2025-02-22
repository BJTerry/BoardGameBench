from dataclasses import dataclass
from typing import Union

@dataclass
class ChainOfThoughtMove:
    """Represents a move with explicit reasoning steps."""
    reasoning: str  # The model's reasoning process
    selected_move: str     # The actual move in the required format

def extract_move(result: Union[str, ChainOfThoughtMove]) -> str:
    """Extract the actual move from either a direct string or ChainOfThoughtMove."""
    if isinstance(result, str):
        return result
    if isinstance(result, ChainOfThoughtMove):
        return result.selected_move
    raise ValueError(f"Cannot extract move from result type: {type(result)}")
