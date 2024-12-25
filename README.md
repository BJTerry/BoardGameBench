# LLM Game Evaluation Framework

A framework for evaluating and comparing Large Language Models (LLMs) through their performance in playing games against each other. The system maintains an Elo rating for each model based on their performance across different games.

## Overview

This framework allows you to:
- Run games between different LLM models
- Maintain Elo ratings across multiple game types
- Add new games and LLM providers easily
- Support both perfect and hidden information games

## Features

- Modular game implementation system
- Support for multiple LLM providers (Anthropic, OpenAI)
- Elo rating system with uncertainty handling
- Async support for efficient game execution
- Conversation management for LLM context
- Built-in move validation and error handling

## Getting Started

1. Install dependencies:
```bash
pip install aiohttp
```

2. Configure your LLM providers:
```python
claude_config = LLMConfig(
    provider=LLMProvider.ANTHROPIC,
    api_key="your_anthropic_key",
    model="claude-3-opus-20240229"
)
```

3. Run a sample game:
```python
async def main():
    game = NimGame(12, 3)  # 12 objects, max take 3
    claude = LLMPlayer("Claude", AnthropicLLM(claude_config))
    gpt4 = LLMPlayer("GPT-4", OpenAILLM(gpt4_config))
    
    runner = GameRunner(game, claude, gpt4)
    winner, history = await runner.play_game()
```

## Implementing New Games

To add a new game, extend the `Game` class:

```python
class YourGame(Game):
    def get_rules_explanation(self) -> str:
        return "Explain your game rules here..."
    
    def get_move_format_instructions(self) -> str:
        return "Explain how moves should be formatted..."
    
    # Implement other required methods...
```

## Adding New LLM Providers

To add a new LLM provider, implement the `LLMInterface`:

```python
class YourLLM(LLMInterface):
    async def complete(self, messages: List[Dict[str, str]]) -> str:
        # Implement API calls to your LLM provider
```

## Testing

Unit tests are an essential part of ensuring the reliability and correctness of the framework. They help verify that each component behaves as expected.

### Running Tests

The project uses pytest for testing. To run the tests:

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_war_game.py

# Run tests with verbose output
poetry run pytest -v
```

### Test Infrastructure

The testing infrastructure includes:

- **pytest-asyncio**: For testing async code
- **pytest-mock**: For mocking dependencies
- **pydantic-ai TestModel**: For simulating LLM responses
- **Fixtures**: Common test fixtures in `conftest.py`

### Test Coverage

Tests cover:
- Game logic (move validation, state transitions)
- LLM interactions (move generation, error handling)
- Database operations (logging, state tracking)
- Edge cases and error conditions

### Writing Tests

New tests should:
- Use appropriate fixtures from `conftest.py`
- Mock external dependencies
- Test both success and failure cases
- Follow existing patterns for similar components

### Maintaining Tests

- **Add Tests for New Features**: Create corresponding unit tests for new functionality
- **Update Existing Tests**: Modify tests when changing existing features
- **Test Coverage**: Maintain high coverage of critical paths
- **Continuous Integration**: Tests run automatically on each commit

## Project Structure

- `game.py` - Base game interface and implementations
- `llm.py` - LLM provider interfaces and implementations
- `elo.py` - Elo rating system
- `runner.py` - Game execution logic

## Contributing

See DESIGN.md for architectural details and contribution guidelines.

## License

MIT License
