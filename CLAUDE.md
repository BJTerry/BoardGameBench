# BoardGameBench Development Guide

## Commands
- **Install dependencies**: `poetry install`
- **Run tests**: `poetry run pytest`
- **Run single test**: `poetry run pytest tests/test_file.py::test_function`
- **Type checking**: `poetry run pyright`
- **Linting**: `poetry run ruff check .`
- **Format code**: `poetry run ruff format .`
- **Run application**: `poetry run python -m bgbench.main`

## Code Style Guidelines
- **Simplicity over cleverness**: Write readable code, avoid clever one-liners
- **Types**: Use type hints for all function parameters and return values
- **Naming**: Use descriptive, meaningful names for variables, functions, and classes
- **Functions**: Keep functions short (<20 lines), single responsibility, 3 or fewer parameters
- **OOP**: Follow Single Responsibility Principle, favor composition over inheritance
- **Dependencies**: Make dependencies explicit through parameters, not hidden in methods
- **Error Handling**: Validate input data, return meaningful error messages
- **Comments**: Add comments to explain *why* code works a certain way, not *what* it does. Use detailed comments for complex/tricky code. Code should be self-documenting for behavior.
- **Formatting**: Consistent indentation, spacing between logical sections
- **Testing**: Write tests first (TDD) when possible, test public interfaces
- **Logging**: Never remove debug logging until confirming code is fixed

## Project Structure
- Group related code together
- Keep a flat structure where possible
- Group by feature rather than type