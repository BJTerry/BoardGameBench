import os
from typing import Optional, List, Dict, Any, cast
import logging
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings
from typing import TypedDict
from enum import Enum
from typing import Union, Type
from .moves import ChainOfThoughtMove

NON_SYSTEM_MODELS = ["openai/o1-mini", "openai/o1-preview"]
SYSTEM_PROMPT = (
    "You are playing a game. Your goal is to win by making valid moves according to the rules. "
    "Always respond with ONLY your move in the exact format specified - no explanation or other text."
)


class ResponseStyle(Enum):
    DIRECT = "direct"  # Direct response with just the move
    CHAIN_OF_THOUGHT = "chain_of_thought"  # Structured response with reasoning

class OurModelSettings(TypedDict, total=True):
    temperature: float
    max_tokens: int
    top_p: float
    timeout: float
    response_style: ResponseStyle

def convert_to_agent_settings(settings: OurModelSettings, model: str) -> ModelSettings:
    """Convert our settings to Agent's ModelSettings"""
    return ModelSettings(
        top_p=settings["top_p"],
        timeout=settings["timeout"],
        **({'temperature': settings['temperature']} if model not in NON_SYSTEM_MODELS else {}),
        **({'max_completion_tokens': settings['max_tokens']} if model in NON_SYSTEM_MODELS else {}),
        **({'max_tokens': settings['max_tokens']} if model not in NON_SYSTEM_MODELS else {}),
    )

logger = logging.getLogger(__name__)

def create_llm(
    model: str, 
    temperature: float = 0.0, 
    max_tokens: int = 1000,
    response_style: ResponseStyle = ResponseStyle.DIRECT,
    **kwargs
) -> Agent[None, Union[str, ChainOfThoughtMove]]:
    """Factory function to create appropriate Agent instance based on model name."""
    settings = OurModelSettings(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0,  # Default value
        timeout=60.0,  # Default timeout in seconds
        response_style=response_style
    )
    model_settings = convert_to_agent_settings(settings, model)

    result_type: Union[type[str], type[ChainOfThoughtMove]] = str if response_style == ResponseStyle.DIRECT.value else ChainOfThoughtMove
    
    if model.startswith('openrouter'):
        # For Claude models, we use OpenRouter to access Anthropic
        openrouter_key = os.getenv('OPENROUTER_API_KEY')
        if not openrouter_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
            
        model_name = model[len('openrouter/'):]
        agent = Agent(
            OpenAIModel(
                model_name,
                base_url='https://openrouter.ai/api/v1',
                api_key=openrouter_key,
            ),
            model_settings=model_settings,
            result_type=result_type
        )
        logger.info(f"Initialized OpenRouter Agent for model {model} via OpenRouter")
    elif model.startswith("openai"):
        # For other models (GPT, etc), we use OpenAI provider
        openai_key = os.getenv("OPENAI_API_KEY", "")
        agent = Agent(
            OpenAIModel(
                model[len('openai/'):],
                api_key=openai_key,
            ),
            model_settings=model_settings,
            result_type=result_type
        )
        logger.info(f"Initialized OpenAI Agent with model {model}")
    else:
        raise ValueError(f"Invalid model provider for {model}")

    if model not in NON_SYSTEM_MODELS:
        @agent.system_prompt
        def system_prompt():
            return SYSTEM_PROMPT

    return cast(Agent[None, Union[str, ChainOfThoughtMove]], agent)

def create_test_llm(test_responses: List[str]) -> Agent:
    """Create an agent with overridden responses for testing."""
    from pydantic_ai.models.test import TestModel
    
    agent = Agent(
        TestModel(test_responses),  # Added test_responses parameter
        model_settings=convert_to_agent_settings(OurModelSettings(
            temperature=0.0,
            max_tokens=1000,
            top_p=1.0,
            timeout=60.0,
            response_style=ResponseStyle.DIRECT,
        ), "test_model")
    )
    return agent

async def complete_prompt(agent: Agent, prompt: str, system_prompt: Optional[str] = None) -> str:
    """Helper function to complete a prompt using an agent."""
    try:
        if system_prompt:
            @agent.system_prompt
            def _():
                return system_prompt
        result = await agent.run(prompt)
        return result.data
    except Exception as e:
        logger.error(f"Error completing prompt: {str(e)}")
        raise
