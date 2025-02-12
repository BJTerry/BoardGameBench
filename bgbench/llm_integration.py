import os
import logging
from typing import Optional, List, Dict, Any, Union, Protocol
from enum import Enum
from .moves import ChainOfThoughtMove
import litellm
from litellm.types.utils import ModelResponse, Choices

litellm._turn_on_debug()
logger = logging.getLogger(__name__)

class LLMCompletionProvider(Protocol):
    """Protocol for objects that can provide completions"""
    def completion(self, model: str, messages: List[Dict[str, str]], **kwargs) -> ModelResponse: ...

NON_SYSTEM_MODELS = ["openai/o1-mini", "openai/o1-preview"]
SYSTEM_PROMPT = (
    "You are playing a game. Your goal is to win by making valid moves according to the rules. "
    "Always respond with ONLY your move in the exact format specified - no explanation or other text."
)

class ResponseStyle(Enum):
    DIRECT = "direct"  # Direct response with just the move
    CHAIN_OF_THOUGHT = "chain_of_thought"  # Structured response with reasoning

def create_llm(
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    response_style: ResponseStyle = ResponseStyle.DIRECT,
    **kwargs
) -> Dict[str, Any]:
    """Factory function to create LLM client configuration."""
    
    # Set up system prompt for models that support it
    messages: List[Dict[str, str]] = []
    if model not in NON_SYSTEM_MODELS:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

    return {
        "model": model,  # Pass model name exactly as provided
        "temperature": temperature if model not in NON_SYSTEM_MODELS else None,
        "max_tokens": max_tokens,
        "messages": messages,
        **kwargs
    }

async def complete_prompt(llm_config: Union[Dict[str, Any], LLMCompletionProvider], prompt: str) -> Tuple[str, Dict[str, int]]:
    """Helper function to complete a prompt using litellm."""
    try:
        if isinstance(llm_config, dict):
            # Original dictionary config case
            messages = llm_config["messages"].copy()
            messages.append({"role": "user", "content": prompt})
            
            response = await litellm.acompletion(
                model=llm_config["model"],
                messages=messages,
                temperature=llm_config.get("temperature"),
                max_tokens=llm_config.get("max_tokens"),
            )
        else:
            # If it's a TestLLM or similar object with completion method
            messages = [{"role": "user", "content": prompt}]
            response = llm_config.completion(
                model="test",
                messages=messages
            )
        
        # Handle case where response might be None or missing content
        if not isinstance(response, ModelResponse):
            raise ValueError("Received invalid response type")
            
        choice = response.choices[0]
        if not isinstance(choice, Choices):
            raise ValueError("Invalid response from LLM")
            
        content = choice.message.content
        if content is None:
            raise ValueError("No content in LLM response")
            
        # Extract token counts
        token_info = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
            "completion_tokens": response.usage.completion_tokens if response.usage else None,
            "total_tokens": response.usage.total_tokens if response.usage else None
        }
            
        return content, token_info
        
    except Exception as e:
        logger.error(f"Error completing prompt: {str(e)}")
        raise
