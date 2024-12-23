from typing import Protocol, List, Dict
import aiohttp
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
        self.api_url = "https://api.anthropic.com/v1/messages"
        
    async def complete(self, messages: List[Dict[str, str]]) -> str:
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        
        data = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=data) as response:
                if response.status != 200:
                    raise Exception(f"API call failed: {await response.text()}")
                result = await response.json()
                return result["content"][0]["text"]

class OpenAILLM(LLMInterface):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
    async def complete(self, messages: List[Dict[str, str]]) -> str:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=headers, json=data) as response:
                if response.status != 200:
                    raise Exception(f"API call failed: {await response.text()}")
                result = await response.json()
                return result["choices"][0]["message"]["content"]
