import logging
from typing import List, Dict, Any, Protocol, Optional
from openai.types.chat import ChatCompletionMessageParam
from bgbench.game_view import GameView

logger = logging.getLogger("bgbench")

class LLMInterface(Protocol):
    async def complete(self, messages: List[ChatCompletionMessageParam]) -> str:
        ...

class LLMPlayer:
    def __init__(self, name: str, llm: LLMInterface):
        self.name = name
        self.llm = llm
        self.conversation_history = []

    async def make_move(self, game_view: GameView, invalid_move_explanation: Optional[str] = None) -> Any:
        # Prepare the message for the LLM
        system_message = (
            "You are playing a game. Here are the rules:\n"
            f"{game_view.rules_explanation}\n\n"
            "Current game state:\n"
            f"{str(game_view.visible_state)}\n\n"
            "Make your move according to these instructions:\n"
            f"{game_view.move_format_instructions}"
        )
        
        messages: List[ChatCompletionMessageParam] = [{"role": "system", "content": system_message}]
        
        if invalid_move_explanation:
            # Include the previous failed move and explanation
            if self.conversation_history:
                messages.append({"role": "assistant", "content": self.conversation_history[-1]["content"]})
            messages.append({"role": "user", "content": f"That move was invalid: {invalid_move_explanation}. Please try again following the move format instructions exactly."})
        else:
            messages.append({"role": "user", "content": "What is your move? Respond with only your move following the format instructions exactly."})

        # Get the move from the LLM
        response = await self.llm.complete(messages)
        response = response.strip()
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
