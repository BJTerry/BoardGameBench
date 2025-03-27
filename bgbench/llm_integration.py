import logging
import asyncio
import random
from typing import Optional, List, Dict, Any, Union, Protocol, Tuple, TypedDict
from enum import Enum
import litellm
from litellm.utils import register_model
from litellm.cost_calculator import completion_cost
from litellm.types.utils import ModelResponse, Choices
from litellm.exceptions import RateLimitError

register_model(
    {
        "openrouter/qwen/qwq-32b": {
            "max_tokens": 131000,
            "input_cost_per_token": 0.0000009,
            "output_cost_per_token": 0.0000009,
            "litellm_provider": "openrouter",
            "mode": "chat",
        },
        "openrouter/anthropic/claude-3.5-haiku": {
            "max_tokens": 200000,
            "input_cost_per_token": 0.000001,
            "output_cost_per_token": 0.000005,
            "litellm_provider": "openrouter",
            "mode": "chat",
            "supports_function_calling": True,
            "supports_tool_choice": True,
        },
        "openrouter/openai/o3-mini": {
            "max_tokens": 100000,
            "max_input_tokens": 200000,
            "max_output_tokens": 100000,
            "input_cost_per_token": 0.0000011,
            "output_cost_per_token": 0.0000044,
            "cache_read_input_token_cost": 0.00000055,
            "litellm_provider": "openrouter",
            "mode": "chat",
            "supports_function_calling": True,
            "supports_parallel_function_calling": False,
            "supports_vision": False,
            "supports_prompt_caching": True,
            "supports_response_schema": True,
            "supports_tool_choice": True,
        },
        "gemini/gemini-2.0-flash-thinking-exp-01-21": {
            "max_tokens": 8192,
            "max_input_tokens": 1048576,
            "max_output_tokens": 8192,
            "max_images_per_prompt": 3000,
            "max_videos_per_prompt": 10,
            "max_video_length": 1,
            "max_audio_length_hours": 8.4,
            "max_audio_per_prompt": 1,
            "max_pdf_size_mb": 30,
            "input_cost_per_image": 0,
            "input_cost_per_video_per_second": 0,
            "input_cost_per_audio_per_second": 0,
            "input_cost_per_token": 0,
            "input_cost_per_character": 0,
            "input_cost_per_token_above_128k_tokens": 0,
            "input_cost_per_character_above_128k_tokens": 0,
            "input_cost_per_image_above_128k_tokens": 0,
            "input_cost_per_video_per_second_above_128k_tokens": 0,
            "input_cost_per_audio_per_second_above_128k_tokens": 0,
            "output_cost_per_token": 0,
            "output_cost_per_character": 0,
            "output_cost_per_token_above_128k_tokens": 0,
            "output_cost_per_character_above_128k_tokens": 0,
            "litellm_provider": "gemini",
            "mode": "chat",
            "supports_system_messages": True,
            "supports_function_calling": True,
            "supports_vision": True,
            "supports_response_schema": True,
            "supports_audio_output": True,
            "tpm": 4000000,
            "rpm": 10,
            "source": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#gemini-2.0-flash",
            "supports_tool_choice": True,
        },
    }
)
logger = logging.getLogger(__name__)


class UsageInfo(TypedDict):
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    cost: Optional[float]


class LLMCompletionProvider(Protocol):
    """Protocol for objects that can provide completions"""

    def completion(
        self, model: str, messages: List[Dict[str, str]], **kwargs
    ) -> ModelResponse: ...


NON_SYSTEM_MODELS = ["openai/o1-mini", "openai/o1-preview"]
# Models that don't support cache_control parameters (strip these out before calling)
CACHE_ENABLED_MODELS = [
    "openrouter/anthropic/claude-3.7-sonnet",
    "openrouter/anthropic/claude-3.5-haiku",
]

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
    max_tokens: int = 10000,
    response_style: ResponseStyle = ResponseStyle.DIRECT,
    **kwargs,
) -> Dict[str, Any]:
    """Factory function to create LLM client configuration."""

    # Set up system prompt with appropriate role based on model
    messages: List[Dict[str, Any]] = []

    # For all models, add the system prompt but with different roles
    # Always add cache_control to the system prompt
    if model not in NON_SYSTEM_MODELS:
        # Models that support system role
        messages.append(
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        )
    else:
        # Models that don't support system role, use user role instead
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        )

    return {
        "model": model,  # Pass model name exactly as provided
        "temperature": temperature if model not in NON_SYSTEM_MODELS else None,
        "max_tokens": max_tokens,
        "messages": messages,
        **kwargs,
    }


def _prepare_messages(
    llm_config: Union[Dict[str, Any], LLMCompletionProvider],
    prompt_messages: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Prepare messages for LLM completion.
    
    Args:
        llm_config: Configuration for the LLM
        prompt_messages: List of message blocks with appropriate caching settings
        
    Returns:
        Tuple of (prepared messages, kwargs for completion)
    """
    if isinstance(llm_config, dict):
        # Get base messages from config (like system message)
        messages = llm_config["messages"].copy()
        
        # Add all prompt messages
        messages.extend(prompt_messages)
        
        # For models that don't support cache_control, strip it out
        if llm_config["model"] not in CACHE_ENABLED_MODELS:
            # Remove cache_control from each message block
            for msg in messages:
                if "content" in msg and isinstance(msg["content"], list):
                    for content_block in msg["content"]:
                        if "cache_control" in content_block:
                            del content_block["cache_control"]
        
        # Prepare kwargs for optional parameters (only non-None values)
        kwargs = {}
        
        if llm_config.get("temperature") is not None:
            kwargs["temperature"] = llm_config.get("temperature")
            
        if llm_config.get("max_tokens") is not None:
            kwargs["max_tokens"] = llm_config.get("max_tokens")
            
        if llm_config.get("provider") is not None:
            kwargs["provider"] = llm_config.get("provider")
            
        return messages, kwargs
    else:
        # If it's a TestLLM or similar object with completion method
        # For testing, we flatten the structured prompt to a single message
        # Extract the text content from the nested structure
        combined_content = "\n\n".join(
            [
                block["text"]
                for msg in prompt_messages
                for block in msg["content"]
                if "type" in block and block["type"] == "text"
            ]
        )
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": combined_content}],
            }
        ]
        return messages, {}


async def _execute_completion(
    llm_config: Union[Dict[str, Any], LLMCompletionProvider],
    messages: List[Dict[str, Any]],
    kwargs: Dict[str, Any]
) -> Any:
    """
    Execute the completion request with the prepared messages.
    
    Args:
        llm_config: Configuration for the LLM
        messages: Prepared messages
        kwargs: Additional parameters for completion
        
    Returns:
        The model response
    """
    if isinstance(llm_config, dict):
        # Always pass model and messages as required parameters
        return await litellm.acompletion(
            model=llm_config["model"], messages=messages, **kwargs
        )
    else:
        # If it's a TestLLM or similar object with completion method
        return llm_config.completion(model="test", messages=messages)


def _process_response(response: ModelResponse) -> Tuple[str, UsageInfo]:
    """
    Process the response from the LLM.
    
    Args:
        response: The model response
        
    Returns:
        Tuple of (content, usage info)
    """
    # Handle case where response might be None or missing content
    if not isinstance(response, ModelResponse):
        raise ValueError("Received invalid response type")
        
    choice = response.choices[0]
    if not isinstance(choice, Choices):
        raise ValueError("Invalid response from LLM")
        
    content = choice.message.content
    if content is None:
        raise ValueError("No content in LLM response")
        
    # Extract token counts with proper typing
    cost = None
    try:
        cost = completion_cost(response)
    except Exception as e:
        print(f"Couldn't calculate cost {e}")
    token_info: UsageInfo = {
        "prompt_tokens": getattr(response, "usage", {}).get("prompt_tokens"),
        "completion_tokens": getattr(response, "usage", {}).get(
            "completion_tokens"
        ),
        "total_tokens": getattr(response, "usage", {}).get("total_tokens"),
        "cost": cost,
    }
    
    return content, token_info


async def _handle_rate_limit(
    e: RateLimitError,
    model_info: str,
    retry_attempt: int,
    total_wait_time: float,
    base_delay: float,
    max_timeout: float
) -> Tuple[float, int, float]:
    """
    Handle rate limit exception with exponential backoff.
    
    Args:
        e: The rate limit error
        model_info: Model name for logging
        retry_attempt: Current retry attempt number
        total_wait_time: Total time waited so far
        base_delay: Base delay for backoff
        max_timeout: Maximum timeout before giving up
        
    Returns:
        Tuple of (delay applied, incremented retry attempt, new total wait time)
    """
    # Calculate delay with exponential backoff and jitter
    # 2^retry_attempt * base_delay * (0.5 to 1.5 jitter)
    jitter = 0.5 + random.random()
    delay = min(2 ** retry_attempt * base_delay * jitter, max_timeout - total_wait_time)
    
    # If we've exceeded the max timeout, log and re-raise
    if total_wait_time + delay >= max_timeout:
        logger.error(f"Rate limit exceeded max retry time for {model_info}: {str(e)}")
        raise
    
    # Log warning about rate limit and retry
    logger.warning(f"Rate limit for {model_info}, retrying in {delay:.2f}s (attempt {retry_attempt + 1})")
    
    # Wait and update counters
    await asyncio.sleep(delay)
    return delay, retry_attempt + 1, total_wait_time + delay


async def complete_prompt(
    llm_config: Union[Dict[str, Any], LLMCompletionProvider],
    prompt_messages: List[Dict[str, Any]],
) -> Tuple[str, UsageInfo, List[Dict[str, Any]]]:
    """
    Helper function to complete a prompt using litellm with caching support.

    Args:
        llm_config: Configuration for the LLM
        prompt_messages: List of message blocks with appropriate caching settings

    Returns:
        Tuple of (response content, usage info, full message list used)
    """
    # Max timeout in seconds (5 minutes)
    MAX_RETRY_TIMEOUT = 300
    # Initial retry delay in seconds
    base_delay = 1
    # Track total wait time
    total_wait_time = 0
    
    # Get model info for logging
    model_info = llm_config["model"] if isinstance(llm_config, dict) else "unknown model"
    
    # Attempt with exponential backoff for rate limits
    retry_attempt = 0
    
    while True:
        try:
            # Step 1: Prepare messages and configuration
            messages, kwargs = _prepare_messages(llm_config, prompt_messages)
            
            # Step 2: Execute the completion
            response = await _execute_completion(llm_config, messages, kwargs)
            
            # Step 3: Process the response
            content, token_info = _process_response(response)
            
            # Return the content, token info, and the full message list that was used
            return content, token_info, messages

        except RateLimitError as e:
            # Handle rate limiting with exponential backoff
            delay, retry_attempt, total_wait_time = await _handle_rate_limit(
                e, model_info, retry_attempt, total_wait_time, 
                base_delay, MAX_RETRY_TIMEOUT
            )
            
        except Exception as e:
            # For any other exceptions, log and raise immediately
            logger.error(f"Error completing prompt {model_info}: {str(e)}")
            raise
