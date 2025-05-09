from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Optional, Dict, TypeVar, Generic
from bgbench.match.view import MatchView, PromptStyle

StateType = TypeVar("StateType")
MoveType = TypeVar("MoveType")


class Game(ABC, Generic[StateType, MoveType]):
    """Abstract base class for all games.

    This class defines the interface that all games must implement.
    It uses generics to provide type safety for state and move types.
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_initial_state(self) -> StateType:
        """Return the initial state of the game.

        Returns:
            A fresh game state object.
        """
        pass

    @abstractmethod
    def get_player_view(
        self,
        state: StateType,
        player_id: int,
        history: Optional[List[Dict[str, Any]]] = None,
        prompt_style: PromptStyle = PromptStyle.HEADER,
    ) -> MatchView:
        """Return what this player can see of the current state.

        Args:
            state: Current game state
            player_id: ID of the player viewing the state
            history: Optional list of previous moves and their results

        Returns:
            A MatchView object containing all information visible to this player.
        """
        pass

    @abstractmethod
    def parse_move(self, move_str: str) -> Optional[MoveType]:
        """Parse move from LLM response string.

        Args:
            move_str: The raw string from the LLM

        Returns:
            Parsed move object if valid, None if parsing failed
        """
        pass

    @abstractmethod
    def validate_move(
        self, state: StateType, player_id: int, move: MoveType
    ) -> Tuple[bool, str]:
        """Validate if a move is legal in the current state.

        Args:
            state: Current game state
            player_id: ID of player making the move
            move: The move to validate

        Returns:
            Tuple of (is_valid, explanation_string)
        """
        pass

    @abstractmethod
    def get_current_player(self, state: StateType) -> int:
        """Return the ID of the player whose turn it is.

        Args:
            state: Current game state

        Returns:
            Player ID (0 or 1)
        """
        pass

    @abstractmethod
    def apply_move(self, state: StateType, player_id: int, move: MoveType) -> StateType:
        """Apply move to state and return new state.

        This method should:
        1. Validate the move is legal
        2. Apply the move to create a new state
        3. Update the current player

        Args:
            state: Current game state
            player_id: ID of player making the move
            move: The move to apply

        Returns:
            New game state after applying the move

        Raises:
            ValueError: If the move is invalid
        """
        pass

    @abstractmethod
    def is_terminal(self, state: StateType) -> bool:
        """Return True if the game has ended."""
        pass

    @abstractmethod
    def get_winner(self, state: StateType) -> Optional[int]:
        """Return the ID of the winner if the game has ended, otherwise None."""
        pass

    @abstractmethod
    def get_next_state(self, state: StateType, move: MoveType) -> StateType:
        """Return the next state after applying the move.

        Args:
            state: Current game state
            move: The move to apply

        Returns:
            New game state
        """
        pass
        
    @abstractmethod
    def serialize_state(self, state: StateType) -> Dict[str, Any]:
        """Serialize the game state into a JSON-compatible dictionary.

        This method must ensure that all game-specific state is properly serialized
        into a format that can be stored in the database and later deserialized.

        Args:
            state: The game state to serialize

        Returns:
            A JSON-compatible dictionary representing the game state
        """
        pass
        
    @abstractmethod
    def deserialize_state(self, state_data: Dict[str, Any]) -> StateType:
        """Deserialize a dictionary into a game state object.

        This method must be able to reconstruct the game state from the dictionary
        produced by serialize_state.

        Args:
            state_data: The dictionary to deserialize

        Returns:
            A reconstructed game state object
        """
        pass
