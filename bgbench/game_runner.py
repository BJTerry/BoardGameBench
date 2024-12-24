import logging
from typing import List, Dict, Any, Tuple
from bgbench.game import Game
from bgbench.llm_player import LLMPlayer

logger = logging.getLogger("bgbench")

class GameRunner:
    def __init__(self, game: Game, player1: LLMPlayer, player2: LLMPlayer):
        self.game = game
        self.players = [player1, player2]

    async def play_game(self) -> Tuple[LLMPlayer, List[Dict[str, Any]]]:
        state = self.game.get_initial_state()
        history = []
        
        while True:
            current_player = self.game.get_current_player(state)
            game_view = self.game.get_player_view(state, current_player)
            
            if game_view.is_terminal:
                break
                
            player = self.players[current_player]
            move = await player.make_move(game_view)
            valid, explanation = self.game.validate_move(state, current_player, move)
            
            if not valid:
                logger.warning(f"Invalid move by {player.name}: {explanation}")
                continue
                
            state = self.game.get_next_state(state, move)
            history.append({
                "player": current_player, 
                "move": move, 
                "state_before": game_view.visible_state
            })

        final_view = self.game.get_player_view(state, current_player)
        winner_idx = 0 if final_view.winner is None else final_view.winner
        return self.players[winner_idx], history
