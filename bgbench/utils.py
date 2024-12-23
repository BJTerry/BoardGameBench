from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Protocol

@dataclass
class GameView:
    """What a player can see of the game state."""
    visible_state: Dict[str, Any]
    valid_moves: List[Any]
    is_terminal: bool
    winner: Optional[int] = None

class Game(ABC):
    @abstractmethod
    def get_rules_explanation(self) -> str:
        """Return a clear explanation of the game rules."""
        pass

    @abstractmethod
    def get_move_format_instructions(self) -> str:
        """Explain how moves should be formatted in responses."""
        pass

    @abstractmethod
    def get_initial_state(self) -> Any:
        """Return the initial state of the game."""
        pass

    @abstractmethod
    def get_player_view(self, state: Any, player_id: int) -> GameView:
        """Return what this player can see of the current state."""
        pass

    @abstractmethod
    def parse_move(self, move_str: str) -> Optional[Any]:
        """Parse move from LLM response string."""
        pass

    @abstractmethod
    def validate_move(self, state: Any, player_id: int, move: Any) -> Tuple[bool, str]:
        """Returns (is_valid, explanation)."""
        pass

    @abstractmethod
    def apply_move(self, state: Any, player_id: int, move: Any) -> Any:
        """Apply move to state and return new state."""
        pass
class LLMInterface(Protocol):
    async def complete(self, messages: List[Dict[str, str]]) -> str:
        ...

class LLMPlayer:
    def __init__(self, name: str, llm: LLMInterface):
        self.name = name
        self.llm = llm
        self.conversation_history = []

    async def make_move(self, game_view: GameView) -> Any:
        # Prepare the message for the LLM
        messages = [
            {"role": "system", "content": game_view.visible_state},
            {"role": "user", "content": "What is your move?"}
        ]
        # Get the move from the LLM
        move = await self.llm.complete(messages)
        self.conversation_history.append({"role": "assistant", "content": move})
        return move

class GameRunner:
    def __init__(self, game: Game, player1: LLMPlayer, player2: LLMPlayer):
        self.game = game
        self.players = [player1, player2]

    async def play_game(self) -> Tuple[int, List[Dict[str, Any]]]:
        state = self.game.get_initial_state()
        history = []
        current_player = 0

        while not state.is_terminal:
            player = self.players[current_player]
            game_view = self.game.get_player_view(state, current_player)
            move = await player.make_move(game_view)
            valid, explanation = self.game.validate_move(state, current_player, move)
            if valid:
                state = self.game.apply_move(state, current_player, move)
                history.append({"player": current_player, "move": move, "state_before": game_view.visible_state})
                current_player = 1 - current_player
            else:
                print(f"Invalid move by {player.name}: {explanation}")

        return state.winner, history

