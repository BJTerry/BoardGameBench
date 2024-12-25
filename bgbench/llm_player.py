import logging
from typing import List, Any, Protocol, Optional
from openai.types.chat import ChatCompletionMessageParam
from bgbench.game_view import GameView

logger = logging.getLogger("bgbench")

class LLMInterface(Protocol):
    async def complete(self, messages: List[ChatCompletionMessageParam]) -> str:
        ...

class LLMPlayer:
    def __init__(self, name: str, llm: LLMInterface, db_session=None, game_id: Optional[int] = None):
        self.name = name
        self.llm = llm
        self.conversation_history = []
        self.session = db_session
        self.game_id = game_id

    async def make_move(self, game_view: GameView, invalid_move_explanation: Optional[str] = None) -> Any:
        # Construct the conversation sequence
        messages: List[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": (
                    "You are playing a game. Your goal is to win by making valid moves according to the rules. "
                    "Always respond with ONLY your move in the exact format specified - no explanation or other text."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Rules: {game_view.rules_explanation}\n\n"
                    f"Move Format: {game_view.move_format_instructions}\n\n"
                    f"Current Game State:\n{str(game_view.visible_state)}\n\n"
                    "What is your move? Respond with ONLY your move in the exact format specified."
                )
            }
        ]
        
        if invalid_move_explanation:
            # Add the error context if this is a retry
            if self.conversation_history:
                messages.append({"role": "assistant", "content": self.conversation_history[-1]["content"]})
            messages.append({
                "role": "user", 
                "content": (
                    f"Invalid move: {invalid_move_explanation}\n"
                    f"Current Game State:\n{str(game_view.visible_state)}\n\n"
                    "Try again. Respond with ONLY your move in the exact format specified."
                )
            })

        # Get the move from the LLM
        response = await self.llm.complete(messages)
        response = response.strip()
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Log the interaction if we have a database session
        if self.session and self.game_id:
            from bgbench.models import LLMInteraction
            llm_interaction = LLMInteraction(
                game_id=self.game_id,
                prompt={
                    "messages": messages,
                },
                response=response
            )
            llm_interaction.log_interaction(self.session, llm_interaction.prompt, response)
        
        return response
