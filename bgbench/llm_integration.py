from typing import Protocol, List, Union, Iterable
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessage
import os
import logging
from openai import AsyncOpenAI

class LLMInterface(Protocol):
    async def complete(self, messages: List[ChatCompletionMessageParam]) -> str:
        ...

logger = logging.getLogger(__name__)

def extract_content(message: Union[ChatCompletionMessageParam, ChatCompletionMessage]) -> str:
    """Extracts the content from a message, handling different content types."""
    if isinstance(message, ChatCompletionMessage):
        content = message.content
    else:
        content = message.get("content", "")
    if isinstance(content, str):
        return content
    elif isinstance(content, Iterable):
        return "".join([c["text"] if "text" in c else "" for c in content])
    return ""

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

    async def complete(self, messages: List[ChatCompletionMessageParam]) -> str:
        logger.info(f"Sending request to {self.model}")
        logger.info(f"Number of messages in conversation: {len(messages)}")
        for msg in messages:
            content = extract_content(msg)
            logger.debug(f"Message ({msg['role']}): {content[:10000]}...")
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            if response.choices:
                content = extract_content(response.choices[0].message)
                logger.info(f"Received response ({len(content)} chars): {content[:5000]}...")
                return content
            raise ValueError("No content in response")
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
            
    async def complete(self, messages: List[ChatCompletionMessageParam]) -> str:
        logger.debug(f"Sending request to {self.model}")
        for msg in messages:
            content = extract_content(msg)
            logger.debug(f"Message ({msg['role']}): {content[:10000]}...")
            
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            if response.choices:
                content = extract_content(response.choices[0].message)
                logger.debug(f"Response: {content[:10000]}...")
                return content
            raise ValueError("No content in response")
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise

def create_llm(model: str, temperature: float = 0.0, max_tokens: int = 1000) -> LLMInterface:
    """Factory function to create appropriate LLM instance based on model name."""
    if model.startswith('claude'):
        return AnthropicLLM(model, temperature, max_tokens)
    else:
        return OpenAILLM(model, temperature, max_tokens)
