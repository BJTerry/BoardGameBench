import logging
from typing import Any, Optional, List, Dict, Tuple
from dataclasses import dataclass, field
import time
from pydantic_ai import Agent
from bgbench.game_view import GameView
from bgbench.models import LLMInteraction, Player

logger = logging.getLogger("bgbench")

@dataclass
class LLMPlayer:
    name: str
    llm: Agent
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    db_session: Optional[Any] = None
    game_id: Optional[int] = None

    async def make_move(self, game_view: GameView, invalid_move_explanation: Optional[str] = None) -> Any:
        """Generate a move using the LLM agent.
        
        Args:
            game_view: The current game state and rules
            invalid_move_explanation: Optional explanation if previous move was invalid
            
        Returns:
            The move string from the LLM
            
        Raises:
            ValueError: If db_session or game_id is not set
        """
        start_time = time.time()
        if self.db_session is None or self.game_id is None:
            raise ValueError("LLMPlayer must have db_session and game_id set before making moves")
        
        # Construct the prompt including system instructions
        system_instructions = ("You are playing a game. Your goal is to win by making valid moves according to the rules. "
                             "Always respond with ONLY your move in the exact format specified - no explanation or other text.")
        
        prompt = (
            f"{system_instructions}\n\n"
            f"Rules: {game_view.rules_explanation}\n\n"
            f"Current Game State:\n{str(game_view.visible_state)}\n\n"
            f"Move Format: {game_view.move_format_instructions}\n\n"
            "What is your move? Respond with ONLY your move in the exact format specified."
        )
        if invalid_move_explanation:
            # Add error context for retry
            last_move = self.conversation_history[-1]["content"] if self.conversation_history else "unknown"
            prompt += (
                f"Your last move was: {last_move}\n"
                f"Invalid move: {invalid_move_explanation}\n"
                "Try again. Respond with ONLY your move in the exact format specified."
            )

        # Get the move from the LLM and track performance metrics
        result = await self.llm.run(prompt)
        end_time = time.time()
        response = result.data.strip()
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Extract token usage if available
        token_usage = getattr(result, 'usage', None)
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        
        if token_usage and not callable(token_usage):
            prompt_tokens = getattr(token_usage, 'prompt_tokens', None)
            completion_tokens = getattr(token_usage, 'completion_tokens', None)
            total_tokens = getattr(token_usage, 'total_tokens', None)
        
        # Log the interaction if we have a database session
        if self.db_session and self.game_id:
            # Get the actual system prompt text
            system_prompt_text = ("You are playing a game. Your goal is to win by making valid moves according to the rules. "
                                "Always respond with ONLY your move in the exact format specified - no explanation or other text.")
            
            prompt_dict = {
                "system_prompt": system_prompt_text,
                "user_prompt": prompt,
            }
            # Get the player record from the database
            player = self.db_session.query(Player).filter_by(name=self.name).first()
            if player is None:
                raise ValueError(f"Player {self.name} not found in database")
                
            llm_interaction = LLMInteraction(
                game_id=self.game_id,
                player_id=player.id,
                prompt=prompt_dict,
                response=response
            )
            llm_interaction.log_interaction(
                self.db_session,
                prompt_dict,
                response,
                start_time,
                end_time,
                prompt_tokens,
                completion_tokens,
                total_tokens
            )
        
        return response
