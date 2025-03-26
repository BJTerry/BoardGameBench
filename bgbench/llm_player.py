import logging
from typing import Any, Optional, List, Dict
from bgbench.llm_integration import ResponseStyle, create_llm, complete_prompt
from dataclasses import dataclass, field
import time
from bgbench.game_view import GameView, PromptStyle
from bgbench.models import LLMInteraction

logger = logging.getLogger("bgbench")


@dataclass
class LLMPlayer:
    name: str
    model_config: Dict[str, Any]
    prompt_style: PromptStyle = PromptStyle.HEADER
    response_style: ResponseStyle = ResponseStyle.DIRECT
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    db_session: Optional[Any] = None
    game_id: Optional[int] = None
    player_id: Optional[int] = None
    _llm: Optional[Dict[str, Any]] = field(default=None)

    def __post_init__(self):
        if self._llm is None:
            self._llm = create_llm(**self.model_config)

    async def make_move(
        self, game_view: GameView, invalid_moves: Optional[List[Dict[str, str]]] = None
    ) -> Any:
        start_time = time.time()
        if self.db_session is None or self.game_id is None:
            raise ValueError(
                "LLMPlayer must have db_session and game_id set before making moves"
            )

        # Add debug logging
        logger.debug(f"Player {self.name} making move for game_id={self.game_id}")

        if self._llm is None:
            raise ValueError("LLM configuration is not initialized")

        # Get the prompt blocks with cache settings from the game view
        prompt_messages = game_view.format_prompt()

        # Create instruction message - this won't be cached as it's different for each move
        move_instruction = "What is your move? Respond with ONLY your move in the exact format specified."

        # Add information about invalid moves if needed
        if invalid_moves:
            invalid_moves_text = "Previous Invalid Moves:\n"
            for i, invalid_move in enumerate(invalid_moves, 1):
                move_text = invalid_move["move"]
                if len(move_text) > 100:
                    move_text = move_text[:100]
                    invalid_moves_text += (
                        f"Attempt {i} (truncated to 100 characters): {move_text}\n"
                        f"Reason: {invalid_move['explanation']}\n\n"
                    )
                else:
                    invalid_moves_text += (
                        f"Attempt {i}: {move_text}\n"
                        f"Reason: {invalid_move['explanation']}\n\n"
                    )
            move_instruction = (
                f"{invalid_moves_text}"
                f"Please carefully review the invalid moves above and try again. {move_instruction}"
            )

        # Add the move instruction to the prompt messages using correct structure
        prompt_messages.append(
            {"role": "user", "content": [{"type": "text", "text": move_instruction}]}
        )

        try:
            # Get response from LLM using the structured prompt messages with caching
            # Now complete_prompt returns the full list of messages used (including system message)
            response, token_info, full_messages = await complete_prompt(
                self._llm, prompt_messages
            )
            end_time = time.time()

            # Add debug logging for token info and cost
            logger.debug(f"LLM response received with token info: {token_info}")

            # Extract move and add to conversation history
            move = response.strip()
            self.conversation_history.append({"role": "assistant", "content": move})

            # Log the interaction if we have a database session
            if self.db_session and self.game_id:
                # Check that player_id is set
                if not hasattr(self, "player_id") or self.player_id is None:
                    raise ValueError(
                        f"Player {self.name} does not have player_id set. This must be set before making moves."
                    )

                # Add debug logging
                logger.debug(
                    f"Logging interaction for player_id={self.player_id}, game_id={self.game_id}"
                )
                cost = 0
                if token_info is not None and isinstance(token_info, dict):
                    cost = token_info.get("cost", 0) or 0
                logger.debug(f"Cost from token_info: ${float(cost):.6f}")

                # Use the full message list from complete_prompt for accurate logging
                llm_interaction = LLMInteraction(
                    game_id=self.game_id,
                    player_id=self.player_id,
                    prompt=full_messages,  # Use full messages including system message
                    response=move,
                    cost=float(cost),
                )
                llm_interaction.log_interaction(
                    self.db_session,
                    full_messages,  # Log the complete message list
                    move,
                    start_time,
                    end_time,
                    token_info["prompt_tokens"],
                    token_info["completion_tokens"],
                    token_info["total_tokens"],
                    token_info["cost"],
                )

            return move

        except Exception as e:
            logger.error(f"Error making move: {str(e)}")
            raise
