import logging
import asyncio
import random
from typing import Optional, List, Dict, Any, Union, Protocol, Tuple, TypedDict
from enum import Enum
import litellm
from litellm.utils import register_model
from litellm.cost_calculator import completion_cost
from litellm.types.utils import ModelResponse, Choices
from litellm.exceptions import RateLimitError, BadRequestError

# Custom exceptions for response processing errors
class LLMResponseError(Exception):
    """Base class for LLM response errors that should be retried"""
    pass


class InvalidResponseTypeError(LLMResponseError):
    """Raised when response is not a ModelResponse"""
    pass


class EmptyChoicesError(LLMResponseError):
    """Raised when response has no choices"""
    pass


class InvalidChoiceError(LLMResponseError):
    """Raised when choice is not a Choices object"""
    pass


class NoContentError(LLMResponseError):
    """Raised when response content is None"""
    pass

register_model(
    {
        "openrouter/qwen/qwq-32b": {
            "max_tokens": 1310072,
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
        "openrouter/google/gemini-2.5-pro-preview-03-25": {
            "max_tokens": 65536,
            "max_input_tokens": 1048576,
            "max_output_tokens": 65536,
            "max_images_per_prompt": 3000,
            "max_videos_per_prompt": 10,
            "max_video_length": 1,
            "max_audio_length_hours": 8.4,
            "max_audio_per_prompt": 1,
            "max_pdf_size_mb": 30,
            "input_cost_per_audio_token": 0.0000007,
            "input_cost_per_token": 0.00000125,
            "input_cost_per_token_above_200k_tokens": 0.0000025,
            "output_cost_per_token": 0.00001,
            "output_cost_per_token_above_200k_tokens": 0.000015,
            "litellm_provider": "openrouter",
            "mode": "chat",
            "rpm": 10000,
            "tpm": 10000000,
            "supports_system_messages": True,
            "supports_function_calling": True,
            "supports_vision": True,
            "supports_response_schema": True,
            "supports_audio_output": False,
            "supports_tool_choice": True,
            "supported_modalities": ["text", "image", "audio", "video"],
            "supported_output_modalities": ["text"],
            "source": "https://ai.google.dev/gemini-api/docs/pricing#gemini-2.5-pro-preview",
        },
        "openrouter/x-ai/grok-3-mini-beta": {
            "max_tokens": 131072,
            "max_input_tokens": 131072,
            "max_output_tokens": 131072,
            "input_cost_per_token": 0.0000003,
            "output_cost_per_token": 0.0000005,
            "litellm_provider": "openrouter",
            "mode": "chat",
            "supports_function_calling": True,
            "supports_tool_choice": True,
            "supports_reasoning": True,
            "supports_response_schema": False,
            "source": "https://x.ai/api#pricing",
        },
        "openrouter/openai/o4-mini": {
            "max_tokens": 100000,
            "max_input_tokens": 200000,
            "max_output_tokens": 100000,
            "input_cost_per_token": 1.1e-6,
            "output_cost_per_token": 4.4e-6,
            "cache_read_input_token_cost": 2.75e-7,
            "litellm_provider": "openrouter",
            "mode": "chat",
            "supports_function_calling": True,
            "supports_parallel_function_calling": False,
            "supports_vision": True,
            "supports_prompt_caching": True,
            "supports_response_schema": True,
            "supports_reasoning": True,
            "supports_tool_choice": True,
        },
        "openrouter/openai/o4-mini-high": {
            "max_tokens": 100000,
            "max_input_tokens": 200000,
            "max_output_tokens": 100000,
            "input_cost_per_token": 1.1e-6,
            "output_cost_per_token": 4.4e-6,
            "cache_read_input_token_cost": 2.75e-7,
            "litellm_provider": "openrouter",
            "mode": "chat",
            "supports_function_calling": True,
            "supports_parallel_function_calling": False,
            "supports_vision": True,
            "supports_prompt_caching": True,
            "supports_response_schema": True,
            "supports_reasoning": True,
            "supports_tool_choice": True,
        },
        "openrouter/openai/o3": {
            "max_tokens": 100000,
            "max_input_tokens": 200000,
            "max_output_tokens": 100000,
            "input_cost_per_token": 1e-5,
            "output_cost_per_token": 4e-5,
            "cache_read_input_token_cost": 2.5e-6,
            "litellm_provider": "openrouter",
            "mode": "chat",
            "supports_function_calling": True,
            "supports_parallel_function_calling": False,
            "supports_vision": True,
            "supports_prompt_caching": True,
            "supports_response_schema": True,
            "supports_reasoning": True,
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


NON_SYSTEM_MODELS = [
    "openai/o1-mini",
    "openai/o1-preview",
    "openrouter/openai/o1-mini",
    "openrouter/openai/o1-preview",
    ]

CACHE_ENABLED_MODELS = [
    "openrouter/anthropic/claude-3.7-sonnet:thinking",
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
        raise InvalidResponseTypeError("Received invalid response type")
    
    # Extract token counts with proper typing first
    cost = None
    try:
        cost = completion_cost(response)
    except Exception as e:
        logger.warning(f"Couldn't calculate cost: {e}")
    token_info: UsageInfo = {
        "prompt_tokens": getattr(response, "usage", {}).get("prompt_tokens"),
        "completion_tokens": getattr(response, "usage", {}).get(
            "completion_tokens"
        ),
        "total_tokens": getattr(response, "usage", {}).get("total_tokens"),
        "cost": cost,
    }
    
    # Check for max tokens before other errors
    max_tokens = getattr(response, "model_params", {}).get("max_tokens")
    if max_tokens and token_info["completion_tokens"] == max_tokens:
        model_name = getattr(response, "model", "unknown model")
        logger.warning(f"Response maxed out completion tokens: used {token_info['completion_tokens']} of {max_tokens} for {model_name} - consider increasing limit")
    
    # Handle empty choices list    
    if not response.choices:
        raise EmptyChoicesError("Empty choices in LLM response")
        
    choice = response.choices[0]
    if not isinstance(choice, Choices):
        raise InvalidChoiceError("Invalid response from LLM")
        
    content = choice.message.content
    if content is None:
        raise NoContentError("No content in LLM response")
    
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


async def _handle_retry(
    error: Exception,
    model_info: str,
    retry_attempt: int,
    total_wait_time: float,
    base_delay: float,
    max_timeout: float,
    max_retries: int,
    error_type: str,
) -> Tuple[float, int, float]:
    """
    Handle retry logic for various error types with exponential backoff.
    
    Args:
        error: The exception that occurred
        model_info: Model name for logging
        retry_attempt: Current retry attempt number
        total_wait_time: Total time waited so far
        base_delay: Base delay for backoff
        max_timeout: Maximum timeout before giving up (in seconds)
        max_retries: Maximum number of retries allowed
        error_type: Type of error for logging
        
    Returns:
        Tuple of (delay applied, incremented retry attempt, new total wait time)
    """
    # Check if we've exceeded max retries
    if retry_attempt >= max_retries:
        logger.error(f"Max retries ({max_retries}) exceeded for {model_info}: {str(error)}")
        raise
        
    # Check if we've exceeded max timeout
    if total_wait_time >= max_timeout:
        logger.error(f"Max timeout ({max_timeout}s) exceeded for {model_info}: {str(error)}")
        raise
        
    # Calculate delay with exponential backoff and jitter
    jitter = 0.5 + random.random()
    # Make sure we don't exceed max_timeout
    delay = min(2 ** retry_attempt * base_delay * jitter, max_timeout - total_wait_time)
    
    # Log warning about error and retry
    logger.warning(f"{error_type.capitalize()} error for {model_info}: {str(error)}, retrying in {delay:.2f}s (attempt {retry_attempt + 1})")
    
    # Wait and update counters
    await asyncio.sleep(delay)
    return delay, retry_attempt + 1, total_wait_time + delay


async def complete_prompt(
    llm_config: Union[Dict[str, Any], LLMCompletionProvider],
    prompt_messages: List[Dict[str, Any]],
) -> Tuple[str, UsageInfo, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Helper function to complete a prompt using litellm with caching support.

    Args:
        llm_config: Configuration for the LLM
        prompt_messages: List of message blocks with appropriate caching settings

    Returns:
        Tuple of (response content, usage info, full message list used, error info)
        Error info is None for successful completions
    """
    # Max timeout in seconds (5 minutes)
    MAX_RETRY_TIMEOUT = 300
    # Initial retry delay in seconds
    base_delay = 1
    # Track total wait time
    total_wait_time = 0
    # Maximum retries for response errors
    MAX_RESPONSE_RETRIES = 3
    
    # Get model info for logging
    model_info = llm_config["model"] if isinstance(llm_config, dict) else "unknown model"
    
    # Attempt with exponential backoff for rate limits and response errors
    rate_limit_retry_attempt = 0
    response_retry_attempt = 0
    
    # We'll track attempt counts for retries
    
    while True:
        try:
            # Step 1: Prepare messages and configuration
            messages, kwargs = _prepare_messages(llm_config, prompt_messages)
            
            # Step 2: Execute the completion
            response = await _execute_completion(llm_config, messages, kwargs)
            
            # Step 3: Process the response
            content, token_info = _process_response(response)
            
            # Check for token limit warnings and capture them
            error_info = None
            max_tokens = getattr(response, "model_params", {}).get("max_tokens")
            if max_tokens and token_info["completion_tokens"] == max_tokens:
                model_name = getattr(response, "model", "unknown model")
                warning_msg = f"Response maxed out completion tokens: used {token_info['completion_tokens']} of {max_tokens} for {model_name}"
                logger.warning(warning_msg)
                
                # Create error info but don't treat as failure
                error_info = {
                    "error_occurred": True,
                    "error_type": "TOKEN_LIMIT_WARNING",
                    "error_message": warning_msg,
                    "error_details": {
                        "completion_tokens": token_info["completion_tokens"],
                        "max_tokens": max_tokens,
                        "model": model_name
                    },
                    "retry_count": rate_limit_retry_attempt + response_retry_attempt
                }
            
            # Return the content, token info, the full message list, and error info
            return content, token_info, messages, error_info

        except RateLimitError as e:
            # Handle rate limiting with exponential backoff
            delay, rate_limit_retry_attempt, total_wait_time = await _handle_retry(
                e, model_info, rate_limit_retry_attempt, total_wait_time, 
                base_delay, MAX_RETRY_TIMEOUT, 999999, "rate limit"  # Using large int instead of infinity
            )
            # Log rate limit error details for monitoring
            logger.debug(
                f"Rate limit error details: type=RATE_LIMIT, message={str(e)}, "
                f"retry_attempt={rate_limit_retry_attempt}"
            )
            
        except LLMResponseError as e:
            # Handle response processing errors with limited retries
            delay, response_retry_attempt, total_wait_time = await _handle_retry(
                e, model_info, response_retry_attempt, total_wait_time,
                base_delay, MAX_RETRY_TIMEOUT, MAX_RESPONSE_RETRIES, "response"
            )
            # Log error details for monitoring, don't store in variable since we're continuing
            logger.debug(
                f"Response error details: type=RESPONSE_ERROR, message={str(e)}, "
                f"error_class={e.__class__.__name__}, retry_attempt={response_retry_attempt}"
            )

        except BadRequestError as e:
            logger.error(f"Error completing prompt {model_info}: {str(e)} {e.litellm_debug_info}", exc_info=True)
            logger.error(f"UNHANDLED_ERROR: {e.__class__.__name__}, retry count: {rate_limit_retry_attempt + response_retry_attempt}")
            raise
            
        except Exception as e:
            # For any other exceptions, log with full traceback and raise immediately
            logger.error(f"Error completing prompt {model_info}: {str(e)}", exc_info=True)
            # Create error record for tracking but we're raising immediately so not using the variable
            # Just log all error information before raising
            logger.error(f"UNHANDLED_ERROR: {e.__class__.__name__}, retry count: {rate_limit_retry_attempt + response_retry_attempt}")
            raise
