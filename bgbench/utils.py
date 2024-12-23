from typing import List, Dict, Any, Protocol, Tuple
from .game import Game, GameView
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
            {"role": "system", "content": str(game_view.visible_state)},
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

        while True:
            game_view = self.game.get_player_view(state, current_player)
            if game_view.is_terminal:
                break
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

