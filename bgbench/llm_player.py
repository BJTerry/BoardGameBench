import logging
from typing import List, Dict, Any, Protocol
from bgbench.game_view import GameView

logger = logging.getLogger("bgbench")

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
