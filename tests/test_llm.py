from litellm.types.utils import ModelResponse, Choices, Message

class TestLLM:
    def __init__(self, response: str = "default test response"):
        self.response = response
        self.last_prompt = None
        
    def completion(self, model: str, messages: list, **kwargs) -> ModelResponse:
        """Simple test double that returns predetermined responses"""
        if messages and messages[-1]["content"]:
            # Extract text from the content array - assuming first text block
            if isinstance(messages[-1]["content"], list) and messages[-1]["content"][0].get("type") == "text":
                self.last_prompt = messages[-1]["content"][0]["text"]
            else:
                self.last_prompt = messages[-1]["content"]
        else:
            self.last_prompt = None
        
        # Create a proper ModelResponse object
        return ModelResponse(
            id="test_id",
            choices=[
                Choices(
                    message=Message(
                        role="assistant",
                        content=self.response
                    ),
                    finish_reason="stop",
                    index=0
                )
            ],
            created=1234567890,
            model=model,
            object="chat.completion",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20
            }
        )

    def set_response(self, response: str):
        self.response = response
