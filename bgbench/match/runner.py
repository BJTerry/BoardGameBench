import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy.orm import Session
from bgbench.game import Game
from bgbench.match.view import MatchView
from bgbench.llm.player import LLMPlayer
# Match state related imports handled by MatchStateData and MatchStateManager
from bgbench.match.state_manager import MatchStateManager
from bgbench.match.match_state import MatchStateData

logger = logging.getLogger("bgbench")


class MatchRunner:
    def __init__(
        self,
        game: Game,
        player1: LLMPlayer,
        player2: LLMPlayer,
        db_session: Session,
        game_id: int,
        player1_id: int,
        player2_id: int,
        experiment_name: Optional[str] = None,
        match_state_manager: Optional[MatchStateManager] = None,
        initial_state: Optional[Any] = None,
    ):
        self.match_state_manager = match_state_manager
        self.initial_state = initial_state
        self.game = game # The abstract ruleset
        self.players = [player1, player2]
        self.session = db_session
        self.match_id = game_id # Renamed game_id to match_id for clarity
        self.turn_count = 0
        self.start_time = None
        self.experiment_name = experiment_name or self.game.__class__.__name__

        # Set database session, game_id, and player_id for players
        logger.debug(
            f"Setting match_id={self.match_id} for players {player1.name} and {player2.name}"
        )
        player1.db_session = db_session
        player1.game_id = self.match_id # LLMPlayer expects game_id for interaction logging
        player1.player_id = player1_id

        player2.db_session = db_session
        player2.game_id = self.match_id # LLMPlayer expects game_id for interaction logging
        player2.player_id = player2_id

        logger.debug(f"Player {player1.name} has player_id={player1.player_id}")
        logger.debug(f"Player {player2.name} has player_id={player2.player_id}")

    async def play_game(
        self,
    ) -> Tuple[Optional[LLMPlayer], List[Dict[str, Any]], Optional[str]]:
        """Play a game between two LLM players.

        Returns:
            Tuple of:
            - Winner (None for draw)
            - Game history
            - Concession reason (if game was conceded)
        """
        self.start_time = datetime.now()
        state = self.initial_state if self.initial_state is not None else self.game.get_initial_state()
        history = []

        # Record initial match state
        if self.match_state_manager:
            # Serialize the game state
            game_state = self.game.serialize_state(state)
                
            # Create a proper MatchStateData object
            initial_state_data = MatchStateData(
                turn=0,  # Initial state is turn 0
                current_player_id=self.game.get_current_player(state),
                timestamp=self.start_time,
                game_state=game_state,
            )
                
            self.match_state_manager.save_state(self.session, self.match_id, initial_state_data)

        while True:
            if self.game.is_terminal(state):
                break

            current_player: int = self.game.get_current_player(state)
            game_view: MatchView = self.game.get_player_view(
                state,
                current_player,
                history,
                prompt_style=self.players[current_player].prompt_style,
            )

            player = self.players[current_player]
            self.turn_count += 1

            player = self.players[current_player]
            self.turn_count += 1

            # Log detailed match state before the move
            if self.match_state_manager:
                # Serialize the game state
                game_state = self.game.serialize_state(state)
                
                # Create a proper MatchStateData object
                turn_state_data = MatchStateData(
                    turn=self.turn_count,
                    current_player_id=current_player,
                    timestamp=datetime.now(),
                    game_state=game_state,
                )

                # Save state and get the ID
                current_match_state_id: Optional[int] = self.match_state_manager.save_state(
                    self.session, self.match_id, turn_state_data
                )
            else:
                current_match_state_id = None # No state saved, pass None

            invalid_moves: List[Dict[str, str]] = []
            retry_count = 0
            MAX_RETRIES = 5
            while True:
                # Determine current player name for logging context
                current_player_name = player.name

                if retry_count == MAX_RETRIES:
                    logger.warning(
                        f"[Match:{self.match_id}] [Player:{current_player_name}] exceeded {MAX_RETRIES} "
                        f"invalid move format attempts and concedes the game"
                    )
                    concession_reason = (
                        f"Exceeded {MAX_RETRIES} invalid move format attempts"
                    )
                    winner = self.players[1 - current_player]
                    return winner, history, concession_reason

                # Pass the captured match_state_id and invalid_moves correctly
                move_str = await player.make_move(
                    game_view=game_view,
                    match_state_id=current_match_state_id,
                    invalid_moves=invalid_moves
                )
                move = self.game.parse_move(move_str)
                if move is None:
                    logger.warning(
                        f"[Match:{self.match_id}] [Player:{current_player_name}] "
                        f"Invalid move format: {move_str}"
                    )
                    invalid_moves.append(
                        {
                            "move": move_str,
                            "explanation": "Invalid move format. Please follow the format instructions exactly.",
                        }
                    )
                    retry_count += 1
                    continue
                valid, explanation = self.game.validate_move(
                    state, current_player, move
                )

                if not valid:
                    logger.warning(
                        f"[Match:{self.match_id}] [Player:{current_player_name}] "
                        f"Invalid move: {explanation} (Move attempted: '{move_str}')"
                    )
                    invalid_moves.append(
                        {
                            "move": move_str,
                            "explanation": explanation,
                        }
                    )
                    retry_count += 1
                    continue

                # Print formatted turn information with match/player context
                turn_number = len(history) + 1
                log_prefix = f"[Match:{self.match_id}] [Player:{current_player_name}]"
                logger.debug(f"\n{log_prefix} Turn {turn_number}")
                # logger.debug(f"{log_prefix} Full Game State:\n{state}") # Maybe too verbose for debug
                logger.debug(f"{log_prefix} Visible State:\n{game_view.visible_state}")
                logger.debug(f"{log_prefix} Move: {move}\n")

                state = self.game.get_next_state(state, move)
                history.append(
                    {
                        "player": current_player,
                        "move": move,
                        "state_before": game_view.visible_state,
                        "turn": turn_number,
                    }
                )
                break

        winner_idx = self.game.get_winner(state)
        winner = None if winner_idx is None else self.players[winner_idx]

        return winner, history, None  # No concession reason for normal game completion
