from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union

from enum import Enum
import json

class PromptStyle(Enum):
    XML = "xml"
    HEADER = "header"
    JSON = "json"

class PromptRenderer:
    @staticmethod
    def _render_state(style: PromptStyle, state: Union[str, Dict[str, Any]]) -> str:
        """Render the state according to the specified style."""
        if isinstance(state, str):
            return state
            
        if style == PromptStyle.JSON:
            return str(state)
        elif style == PromptStyle.XML:
            xml_lines = []
            for key, value in state.items():
                if isinstance(value, (list, dict)):
                    value = str(value)
                xml_lines.append(f"<{key}>{value}</{key}>")
            return "\n".join(xml_lines)
        elif style == PromptStyle.HEADER:
            header_lines = []
            for key, value in state.items():
                if isinstance(value, (list, dict)):
                    value = str(value)
                header_lines.append(f"{key.upper()}\n{value}\n")
            return "\n".join(header_lines)
        raise ValueError(f"Unknown prompt style: {style}")

    @staticmethod
    def render(style: PromptStyle, rules: str, state: Union[str, Dict[str, Any]], move_format: str) -> str:
        """Render the full prompt according to the specified style."""
        rendered_state = PromptRenderer._render_state(style, state)
        
        if style == PromptStyle.XML:
            return (f"<rules>\n{rules}\n</rules>\n"
                   f"<state>\n{rendered_state}\n</state>\n"
                   f"<move_format>\n{move_format}\n</move_format>")
        elif style == PromptStyle.HEADER:
            return (f"RULES:\n{rules}\n\n"
                   f"STATE:\n{rendered_state}\n\n"
                   f"MOVE FORMAT:\n{move_format}")
        elif style == PromptStyle.JSON:
            return json.dumps({
                "rules": rules,
                "state": rendered_state,
                "move_format": move_format
            })
        raise ValueError(f"Unknown prompt style: {style}")

@dataclass
class GameView:
    """What a player can see of the game state.
    
    Attributes:
        visible_state: Dictionary of game state visible to this player
        valid_moves: List of legal moves available to this player
        is_terminal: Whether the game has ended
        winner: Player ID of winner if game is over, else None
        history: List of previous moves and their results
        move_format_instructions: How moves should be formatted
        rules_explanation: Explanation of game rules
        error_message: Last error message if any
        prompt_style: PromptStyle to use for formatting
    """
    visible_state: Union[str, Dict[str, Any]]
    valid_moves: List[Any]
    is_terminal: bool
    winner: Optional[int] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    move_format_instructions: Optional[str] = None
    rules_explanation: Optional[str] = None
    error_message: Optional[str] = None
    prompt_style: PromptStyle = PromptStyle.HEADER

    def format_prompt(self) -> str:
        """Format the game view according to the configured prompt style."""
        return PromptRenderer.render(
            self.prompt_style,
            self.rules_explanation or "",
            self.visible_state,
            self.move_format_instructions or ""
        )
