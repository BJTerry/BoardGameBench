from typing import Protocol, List, Dict
from llm import LLMClient, LLMConfig as LLMClientConfig
from enum import Enum

class LLMProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    # Add other providers as needed

class LLMConfig:
    """Configuration for LLM API calls."""
    def __init__(
        self,
        provider: LLMProvider,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1000
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

class LLMInterface(Protocol):
    """Protocol for LLM API implementations."""
    async def complete(self, messages: List[Dict[str, str]]) -> str:
        ...

class AnthropicLLM(LLMInterface):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = LLMClient(LLMClientConfig(
            provider="anthropic",
            api_key=config.api_key,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        ))
        
    async def complete(self, messages: List[Dict[str, str]]) -> str:
        response = await self.client.complete(messages)
        return response["content"][0]["text"]

class OpenAILLM(LLMInterface):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = LLMClient(LLMClientConfig(
            provider="openai",
            api_key=config.api_key,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        ))
        
    async def complete(self, messages: List[Dict[str, str]]) -> str:
        response = await self.client.complete(messages)
        return response["choices"][0]["message"]["content"]
