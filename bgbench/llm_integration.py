from typing import Protocol, List, Dict
import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

class LLMInterface(Protocol):
    """Protocol for LLM API implementations."""
    async def complete(self, messages: List[Dict[str, str]]) -> str:
        ...

class AnthropicLLM(LLMInterface):
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1000):
        # Configure via OpenRouter for Anthropic models
        openrouter_model = OpenAIModel(
            f"anthropic/{model}",
            base_url='https://openrouter.ai/api/v1',
            api_key=os.getenv('OPENROUTER_API_KEY'),
        )
        self.agent = Agent(
            openrouter_model,
            model_settings={
                'temperature': temperature,
                'max_tokens': max_tokens
            }
        )

    async def complete(self, messages: List[Dict[str, str]]) -> str:
        result = await self.agent.run(messages=messages)
        return result.data

class OpenAILLM(LLMInterface):
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1000):
        # Configure direct OpenAI access
        openai_model = OpenAIModel(
            model,
            api_key=os.getenv('OPENAI_API_KEY'),
        )
        self.agent = Agent(
            openai_model,
            model_settings={
                'temperature': temperature,
                'max_tokens': max_tokens
            }
        )
        
    async def complete(self, messages: List[Dict[str, str]]) -> str:
        result = await self.agent.run(messages=messages)
        return result.data

def create_llm(model: str, temperature: float = 0.0, max_tokens: int = 1000) -> LLMInterface:
    """Factory function to create appropriate LLM instance based on model name."""
    if model.startswith('claude'):
        return AnthropicLLM(model, temperature, max_tokens)
    else:
        return OpenAILLM(model, temperature, max_tokens)
