import logging
from typing import Any, Optional, List, Dict
from dataclasses import dataclass, field
from pydantic_ai import Agent, RunContext
from bgbench.game_view import GameView

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
        """
        
        # Construct the prompt including system instructions
        system_instructions = ("You are playing a game. Your goal is to win by making valid moves according to the rules. "
                             "Always respond with ONLY your move in the exact format specified - no explanation or other text.")
        
        if not invalid_move_explanation:
            prompt = (
                f"{system_instructions}\n\n"
                f"Rules: {game_view.rules_explanation}\n\n"
                f"Move Format: {game_view.move_format_instructions}\n\n"
                f"Current Game State:\n{str(game_view.visible_state)}\n\n"
                "What is your move? Respond with ONLY your move in the exact format specified."
            )
        else:
            # Add error context for retry
            last_move = self.conversation_history[-1]["content"] if self.conversation_history else "unknown"
            prompt = (
                f"{system_instructions}\n\n"
                f"Your last move was: {last_move}\n"
                f"Invalid move: {invalid_move_explanation}\n"
                f"Current Game State:\n{str(game_view.visible_state)}\n\n"
                "Try again. Respond with ONLY your move in the exact format specified."
            )

        # Get the move from the LLM
        result = await self.llm.run(prompt)
        response = result.data.strip()
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Log the interaction if we have a database session
        if self.db_session and self.game_id:
            from bgbench.models import LLMInteraction
            # Get the actual system prompt text
            system_prompt_text = ("You are playing a game. Your goal is to win by making valid moves according to the rules. "
                                "Always respond with ONLY your move in the exact format specified - no explanation or other text.")
            
            prompt_dict = {
                "system_prompt": system_prompt_text,
                "user_prompt": prompt,
            }
            llm_interaction = LLMInteraction(
                game_id=self.game_id,
                prompt=prompt_dict,
                response=response
            )
            llm_interaction.log_interaction(self.db_session, prompt_dict, response)
        
        return response
