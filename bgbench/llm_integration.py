from typing import Protocol, List, Dict
import aisuite as ai
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
        # Initialize the aisuite client
        self.client = ai.Client()
        
    async def complete(self, messages: List[Dict[str, str]]) -> str:
        # Convert messages to Anthropic format
        formatted_messages = []
        for msg in messages:
            if msg["role"] == "system":
                formatted_messages.append({"role": "assistant", "content": str(msg["content"])})
            else:
                formatted_messages.append({"role": msg["role"], "content": str(msg["content"])})
                
        response = await self.client.chat.completions.create(
            model=self.config.model,  # Remove anthropic: prefix
            messages=formatted_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        return response.choices[0].message.content

class OpenAILLM(LLMInterface):
    def __init__(self, config: LLMConfig):
        self.config = config
        # Initialize the aisuite client
        self.client = ai.Client()
        
    async def complete(self, messages: List[Dict[str, str]]) -> str:
        response = await self.client.chat.completions.create(
            model=f"openai:{self.config.model}",
            messages=messages,
            temperature=self.config.temperature
        )
        return response.choices[0].message.content
