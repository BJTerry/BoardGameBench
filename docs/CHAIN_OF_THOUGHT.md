# Chain of Thought Support

Some language models require explicit chain-of-thought prompting to produce reliable results, while others have this capability built-in. BoardGameBench supports both types of models through different result schemas.

## Model Types

### Direct Response Models
Models like Claude and GPT-4 have strong built-in reasoning capabilities. For these models, we use a simple string result type:

```python
agent = Agent(
    model="openai:gpt-4",
    result_type=str
)
```

The model will return moves directly in the required format.

### Chain of Thought Models
Models that benefit from explicit reasoning steps use a structured result type:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ChainOfThoughtMove:
    thinking: str  # The model's reasoning process
    move: str     # The actual move in the required format
    confidence: Optional[float] = None  # Optional confidence score

agent = Agent(
    model="openai:mistral-7b",
    result_type=ChainOfThoughtMove,
    system_prompt="Always explain your reasoning before making a move."
)
```

## Move Extraction

To handle both types of results consistently, use the move extractor:

```python
from bgbench.moves import extract_move

# For direct response models
result1 = agent1.run_sync("Make your move")
move1 = extract_move(result1.data)  # Returns the string directly

# For chain of thought models
result2 = agent2.run_sync("Make your move")
move2 = extract_move(result2.data)  # Returns result2.data.move
```

## Implementation Details

### Move Extractor
```python
from typing import Union
from dataclasses import is_dataclass

def extract_move(result: Union[str, ChainOfThoughtMove]) -> str:
    """Extract the actual move from either a direct string or ChainOfThoughtMove."""
    if isinstance(result, str):
        return result
    if is_dataclass(result) and hasattr(result, "move"):
        return result.move
    raise ValueError(f"Cannot extract move from result type: {type(result)}")
```

### Validation
The framework validates moves in both cases:
- For direct responses: validates the move string format directly
- For chain of thought: validates the move field format after extraction

### Logging and Analysis
Chain of thought responses provide additional benefits:
- Reasoning steps can be analyzed for model understanding
- Confidence scores can be used for move quality assessment
- Thinking process can be used for debugging and improvement

## Usage Example

```python
from bgbench.moves import ChainOfThoughtMove, extract_move
from bgbench import Agent

# Direct response model
direct_agent = Agent(
    model="openai:gpt-4",
    result_type=str
)

# Chain of thought model
cot_agent = Agent(
    model="openai:mistral-7b",
    result_type=ChainOfThoughtMove,
    system_prompt=(
        "Think through your move step by step:\n"
        "1. Analyze the current game state\n"
        "2. Consider possible moves\n"
        "3. Evaluate the consequences\n"
        "4. Choose and explain your move\n"
        "Provide your thinking and final move in the required format."
    )
)

# Both can be used with the same game interface
game = ChessGame()
state = game.get_initial_state()

# Direct response
result1 = direct_agent.run_sync("Make your move", state)
move1 = extract_move(result1.data)
game.validate_move(state, 0, move1)

# Chain of thought
result2 = cot_agent.run_sync("Make your move", state)
move2 = extract_move(result2.data)
game.validate_move(state, 0, move2)
```

## Best Practices

1. Use direct response for models with proven reasoning capabilities
2. Use chain of thought for:
   - Models that benefit from explicit reasoning
   - Debugging and analysis purposes
   - When you need confidence scores
   - Educational or research applications where reasoning visibility is important

3. Always use the `extract_move()` function rather than accessing moves directly
4. Include clear reasoning prompts in system_prompt for chain of thought models
5. Consider logging both thinking and moves for analysis
