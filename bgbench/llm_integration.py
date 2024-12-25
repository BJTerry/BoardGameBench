import os
from typing import Optional, List
import logging
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings

logger = logging.getLogger(__name__)

def create_llm(model: str, temperature: float = 0.0, max_tokens: int = 1000) -> Agent:
    """Factory function to create appropriate Agent instance based on model name."""
    model_settings = ModelSettings(
        temperature=temperature,
        max_tokens=max_tokens
    )
    
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
            model_settings=model_settings
        )
        logger.info(f"Initialized OpenRouter Agent for Anthropic model {model} via OpenRouter")
    elif model.startswith("openai"):
        # For other models (GPT, etc), we use OpenAI provider
        openai_key = os.getenv("OPENAI_API_KEY", "")
        agent = Agent(
            OpenAIModel(
                model[len('openai/'):],
                api_key=openai_key,
            ),
            model_settings=model_settings,
        )
        logger.info(f"Initialized OpenAI Agent with model {model}")
    else:
        raise ValueError(f"Invalid model provider for {model}")
    
    return agent

def create_test_llm(test_responses: List[str]) -> Agent:
    """Create an agent with overridden responses for testing."""
    from pydantic_ai.models.test import TestModel
    
    agent = Agent(
        TestModel(),
        model_settings=ModelSettings(
            temperature=0.0,
            max_tokens=1000
        )
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
