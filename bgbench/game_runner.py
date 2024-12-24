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
            game_view = self.game.get_player_view(state, current_player, history)
            
            if game_view.is_terminal:
                break
                
            player = self.players[current_player]
            move_str = await player.make_move(game_view)
            move = self.game.parse_move(move_str)
            if move is None:
                logger.warning(f"Invalid move format by {player.name}: {move_str}")
                retry_count = 1
                MAX_RETRIES = 5
                
                while retry_count < MAX_RETRIES:
                    logger.warning(f"Retry attempt {retry_count} of {MAX_RETRIES}")
                    move_str = await player.make_move(game_view, "Invalid move format. Please follow the format instructions exactly.")
                    move = self.game.parse_move(move_str)
                    if move is not None:
                        break
                    retry_count += 1
                    logger.warning(f"Invalid move format by {player.name}: {move_str}")
                
                if move is None:
                    logger.warning(f"{player.name} exceeded {MAX_RETRIES} invalid move format attempts and concedes the game")
                    return self.players[1 - current_player], history

            valid, explanation = self.game.validate_move(state, current_player, move)
            
            if not valid:
                logger.warning(f"Invalid move by {player.name}: {explanation}")
                # Try again with the invalid move feedback, max 5 attempts
                retry_count = 1
                MAX_RETRIES = 5
                
                while retry_count < MAX_RETRIES:
                    logger.warning(f"Retry attempt {retry_count} of {MAX_RETRIES}")
                    move_str = await player.make_move(game_view, explanation)
                    move = self.game.parse_move(move_str)
                    if move is None:
                        logger.warning(f"Invalid move format by {player.name}: {move_str}")
                        retry_count += 1
                        continue
                        
                    valid, explanation = self.game.validate_move(state, current_player, move)
                    if valid:
                        break
                    retry_count += 1
                    logger.warning(f"Invalid move by {player.name}: {explanation}")
                
                if not valid:
                    logger.warning(f"{player.name} exceeded {MAX_RETRIES} invalid move attempts and concedes the game")
                    # Return the other player as winner
                    return self.players[1 - current_player], history

            # Print formatted turn information
            turn_number = len(history) + 1
            logger.info(f"\nTurn {turn_number}")
            logger.info(f"Current Player: {player.name}")
            logger.info("Game State:")
            logger.info(f"{game_view.visible_state}")
            logger.info(f"Move: {move}\n")
                
            state = self.game.get_next_state(state, move)
            history.append({
                "player": current_player, 
                "move": move, 
                "state_before": game_view.visible_state,
                "turn": turn_number
            })

        final_view = self.game.get_player_view(state, current_player)
        winner_idx = 0 if final_view.winner is None else final_view.winner
        return self.players[winner_idx], history
