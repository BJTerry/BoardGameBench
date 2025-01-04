import json
import logging
from typing import Any, Optional, List, Dict, Tuple, Union
from bgbench.moves import ChainOfThoughtMove, extract_move
from bgbench.llm_integration import ResponseStyle, create_llm
from dataclasses import dataclass, field
import time
from pydantic_ai import Agent, capture_run_messages
from bgbench.game_view import GameView, PromptStyle
from bgbench.llm_integration import NON_SYSTEM_MODELS, SYSTEM_PROMPT
from bgbench.models import LLMInteraction, Player

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
    _llm: Optional[Agent[None, Union[str, ChainOfThoughtMove]]] = field(default=None)

    def __post_init__(self):
        if self._llm is None:
            self._llm = create_llm(**self.model_config)

    @property 
    def llm(self) -> Agent[None, Union[str, ChainOfThoughtMove]]:
        # _llm is guaranteed to be set in __post_init__
        return self._llm  # type: ignore

    async def make_move(self, game_view: GameView, invalid_moves: Optional[List[Dict[str, str]]] = None) -> Any:
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
        # Format the game view according to the configured style
        formatted_game_view = game_view.format_prompt()
        
        if self.llm.model in NON_SYSTEM_MODELS:
            prompt = SYSTEM_PROMPT + "\n"
        else:
            prompt = ""
        
        prompt += (
            f"{formatted_game_view}\n\n"
            "What is your move? Respond with ONLY your move in the exact format specified."
        )
        if invalid_moves:
            prompt += "\n\nPrevious Invalid Moves:\n"
            for i, invalid_move in enumerate(invalid_moves, 1):
                prompt += (
                    f"Attempt {i}: {invalid_move['move']}\n"
                    f"Reason: {invalid_move['explanation']}\n\n"
                )
            prompt += "Please carefully review the invalid moves above and try again. Respond with ONLY your move in the exact format specified."

        # Get the move from the LLM and track performance metrics
        with capture_run_messages() as messages:
            try:
                result = await self.llm.run(prompt)
                end_time = time.time()
                
                # Extract move based on response style
                raw_response = result.data
                move = extract_move(raw_response).strip()
                self.conversation_history.append({"role": "assistant", "content": move})
                
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
                    # Get the player record from the database
                    player = self.db_session.query(Player).filter_by(name=self.name).first()
                    if player is None:
                        raise ValueError(f"Player {self.name} not found in database")
                    json_messages = json.loads(result.new_messages_json().decode('utf-8'))
                    llm_interaction = LLMInteraction(
                        game_id=self.game_id,
                        player_id=player.id,
                        prompt=json_messages,
                        response=move,
                    )
                    llm_interaction.log_interaction(
                        self.db_session,
                        json_messages,
                        move,
                        start_time,
                        end_time,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens
                    )
                
                return move
            except Exception:
                print(messages)
                raise
