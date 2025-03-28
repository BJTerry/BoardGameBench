# LLM Game Evaluation Framework

Visit [BoardGameBench.com](https://www.boardgamebench.com) to see the latest results.

A framework for evaluating and comparing Large Language Models (LLMs) through their performance in playing games against each other. The system maintains an Elo rating for each model based on their performance across different games.

## Overview

This framework allows you to:
- Run games between different LLM models
- Maintain Elo ratings across multiple game types
- Add new games and LLM providers easily
- Support both perfect and hidden information games

## Features

- Modular game implementation system
- Support for multiple LLM providers (Anthropic, OpenAI, OpenRouter) through litellm
- Multiple scheduling strategies: Full Ranking, Top Identification, Sigma Minimization (see `scheduler.py`)
- Bayesian Elo rating with uncertainty handling (see `rating.py`)
- Async support for efficient game execution
- Conversation management for LLM context
- Built-in move validation and error handling
- Experiment export in standardized formats (see `export.py`)

## Getting Started

1. Install dependencies:
```bash
poetry install
```

2. Configure your LLM providers:
- Copy .env.example to .env and set your keys

3. Database Setup:
   - **SQLite** (default): No additional setup required
   - **PostgreSQL** (recommended for production):
     - Install PostgreSQL (see docs/POSTGRES.md for details)
     - Set up database and user:
       ```sql
       CREATE DATABASE bgbench;
       CREATE USER bgbench_user WITH PASSWORD 'bgbench_password';
       GRANT ALL PRIVILEGES ON DATABASE bgbench TO bgbench_user;
       ```
     - Configure .env with PostgreSQL connection details
     - Initialize the database:
       ```bash
       poetry run python -m bgbench.init_db
       ```
     - Migrate existing data (optional):
       ```bash
       poetry run python -m bgbench.migrate_db
       ```

4. Run a sample game:
```bash
poetry run python -m bgbench.main --game nim
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

### Running Tests

The project uses pytest for testing. To run the tests:

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_war_game.py
```

## Contributing

See DESIGN.md for architectural details and contribution guidelines. Note that this project is primarily coded by LLMs
with imperfect human supervision. 

If you are interested in using this in your own research, please let me know.

## License

Copyright (c) 2025 BJ Terry

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.