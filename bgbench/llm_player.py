import logging
from typing import Any, Optional, List, Dict
from bgbench.llm_integration import ResponseStyle, create_llm, complete_prompt
from dataclasses import dataclass, field
import time
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
    _llm: Optional[Dict[str, Any]] = field(default=None)

    def __post_init__(self):
        if self._llm is None:
            self._llm = create_llm(**self.model_config)

    async def make_move(self, game_view: GameView, invalid_moves: Optional[List[Dict[str, str]]] = None) -> Any:
        start_time = time.time()
        if self.db_session is None or self.game_id is None:
            raise ValueError("LLMPlayer must have db_session and game_id set before making moves")
        
        if self._llm is None:
            raise ValueError("LLM configuration is not initialized")
        
        # Construct the prompt including system instructions
        # Format the game view according to the configured style
        formatted_game_view = game_view.format_prompt()
        
        if self.model_config['model'] in NON_SYSTEM_MODELS:
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

        try:
            # Get response from LLM
            response, token_info = await complete_prompt(self._llm, prompt)
            end_time = time.time()
            
            # Extract move and add to conversation history
            move = response.strip()
            self.conversation_history.append({"role": "assistant", "content": move})
            
            # Log the interaction if we have a database session
            if self.db_session and self.game_id:
                # Get the player record from the database
                player = self.db_session.query(Player).filter_by(name=self.name).first()
                if player is None:
                    raise ValueError(f"Player {self.name} not found in database")
                
                messages = [{"role": "user", "content": prompt}]
                llm_interaction = LLMInteraction(
                    game_id=self.game_id,
                    player_id=player.id,
                    prompt=messages,
                    response=move,
                )
                llm_interaction.log_interaction(
                    self.db_session,
                    messages,
                    move,
                    start_time,
                    end_time,
                    token_info["prompt_tokens"],
                    token_info["completion_tokens"],
                    token_info["total_tokens"],
                    token_info["cost"]
                )
            
            return move
            
        except Exception as e:
            logger.error(f"Error making move: {str(e)}")
            raise
