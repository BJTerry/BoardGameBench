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
        system_message = (
            f"Game state: {str(game_view.visible_state)}\n"
            "Respond with only a number representing how many objects to take."
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": "What is your move? Respond with only a number."}
        ]
        # Get the move from the LLM
        response = await self.llm.complete(messages)
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Try to extract a number from the response
        try:
            # Remove any non-numeric characters and convert to int
            move = int(''.join(c for c in response if c.isdigit()))
            return move
        except ValueError:
            return 0  # Return invalid move if parsing fails

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

        final_view = self.game.get_player_view(state, current_player)
        return final_view.winner, history

