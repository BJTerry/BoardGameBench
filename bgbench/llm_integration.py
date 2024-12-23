from typing import Protocol, List, Dict
import os
from openai import AsyncOpenAI

class LLMInterface(Protocol):
    """Protocol for LLM API implementations."""
    async def complete(self, messages: List[Dict[str, str]]) -> str:
        ...

class AnthropicLLM(LLMInterface):
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1000):
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        self.model = f"anthropic/{model}"
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def complete(self, messages: List[Dict[str, str]]) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

class OpenAILLM(LLMInterface):
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1000):
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    async def complete(self, messages: List[Dict[str, str]]) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

def create_llm(model: str, temperature: float = 0.0, max_tokens: int = 1000) -> LLMInterface:
    """Factory function to create appropriate LLM instance based on model name."""
    if model.startswith('claude'):
        return AnthropicLLM(model, temperature, max_tokens)
    else:
        return OpenAILLM(model, temperature, max_tokens)
