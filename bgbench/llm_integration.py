from typing import Protocol, List, Dict
import os
import logging
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class LLMInterface(Protocol):
    """Protocol for LLM API implementations."""
    async def complete(self, messages: List[Dict[str, str]]) -> str:
        ...

class AnthropicLLM(LLMInterface):
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1000):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        logger.info(f"Initialized AnthropicLLM with model {model}")
        self.model = f"anthropic/{model}"
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def complete(self, messages: List[Dict[str, str]]) -> str:
        logger.debug(f"Sending request to {self.model}")
        for msg in messages:
            logger.debug(f"Message ({msg['role']}): {msg['content'][:100]}...")
            
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            content = response.choices[0].message.content
            logger.debug(f"Response: {content[:100]}...")
            return content
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {str(e)}")
            raise

class OpenAILLM(LLMInterface):
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1000):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        self.client = AsyncOpenAI(
            api_key=api_key
        )
        logger.info(f"Initialized OpenAILLM with model {model}")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    async def complete(self, messages: List[Dict[str, str]]) -> str:
        logger.debug(f"Sending request to {self.model}")
        for msg in messages:
            logger.debug(f"Message ({msg['role']}): {msg['content'][:100]}...")
            
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            content = response.choices[0].message.content
            logger.debug(f"Response: {content[:100]}...")
            return content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise

def create_llm(model: str, temperature: float = 0.0, max_tokens: int = 1000) -> LLMInterface:
    """Factory function to create appropriate LLM instance based on model name."""
    if model.startswith('claude'):
        return AnthropicLLM(model, temperature, max_tokens)
    else:
        return OpenAILLM(model, temperature, max_tokens)
