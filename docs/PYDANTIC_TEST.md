# Project Plan: Improving Testability with Pydantic-AI

## Overview

The goal of this project is to enhance the test coverage and overall testability of the existing system by migrating from direct usage of OpenAI's library to using `pydantic-ai`. By following the best practices outlined in `pydantic-ai`'s documentation, we aim to enable more effective and efficient unit testing.

## Objectives

- **Migrate** the system to use `pydantic-ai` as the underlying framework for LLM interactions.
- **Refactor** the codebase to align with `pydantic-ai`'s architecture and best practices.
- **Implement** unit tests using `pytest` and `pydantic-ai`'s `TestModel` and `FunctionModel`.
- **Ensure** no real API calls are made during testing by setting `ALLOW_MODEL_REQUESTS=False`.
- **Achieve** comprehensive test coverage of the existing system components.

## Project Phases

### Phase 1: Preparation and Familiarization [✓ COMPLETE]

**1.1** - **Understand `pydantic-ai`** [COMPLETE]

- Read the `pydantic-ai` documentation thoroughly, focusing on:
  - Agents
  - Models
  - Tools
  - Testing and Evals (Unit Tests section)
- Identify how `pydantic-ai` can replace the current LLM integration.

**1.2** - **Assess the Current System** [COMPLETE]

- Review the existing codebase to:
  - Identify all areas where OpenAI's library is directly used.
  - Understand the dependencies and flow between components like `llm_integration.py` and `llm_player.py`.
  - List the modules and classes that will be affected by the migration.

### Phase 2: Migration to Pydantic-AI [✓ COMPLETE]

**2.1** - **Install `pydantic-ai`** [COMPLETE]

- Add `pydantic-ai` to the project dependencies in `pyproject.toml`:

  ```toml
  [tool.poetry.dependencies]
  pydantic-ai = "^0.0.15"
  ```

- Run `poetry install` to install the new dependency.

**2.2** - **Replace Direct LLM Calls with Pydantic-AI Agents**

- **2.2.1** - **Refactor `llm_integration.py`** [COMPLETE]

  - Replace `AnthropicLLM` and `OpenAILLM` classes with `pydantic-ai` agents.
  - Remove the `LLMInterface` protocol, as `pydantic-ai` provides the necessary abstraction.
  - Implement agents using `pydantic_ai.Agent` for each model (e.g., `AnthropicAgent`, `OpenAIAgent`).

- **2.2.2** - **Adjust LLM Creation** [COMPLETE]

  - Modify the `create_llm` function to return instances of `pydantic-ai` agents.
  - Ensure that model initialization aligns with `pydantic-ai`'s requirements.

**2.3** - **Update `llm_player.py` to Use Pydantic-AI Agents**

- **2.3.1** - **Modify the `LLMPlayer` Class** [COMPLETE]

  - Replace usage of `LLMInterface` with `pydantic_ai.Agent`.
  - Update the `make_move` method to utilize `Agent.run()` for generating moves.
  - Ensure that the conversation history is managed according to `pydantic-ai`'s messaging system.

- **2.3.2** - **Adjust Message Formatting** [COMPLETE]

  - Use `pydantic-ai`'s message parts (`SystemPromptPart`, `UserPromptPart`, etc.) for constructing prompts.
  - Ensure that the prompts and responses are formatted correctly for the agents.

**2.4** - **Remove Deprecated Code** [COMPLETE]

- Eliminate any classes, methods, or functions that are no longer necessary after the migration.
- Clean up imports and resolve any dependency issues.

### Phase 3: Refactoring for Testability [✓ COMPLETE]

**3.1** - **Implement Best Practices from `pydantic-ai`**

- **3.1.1** - **Use `Agent.override` for Testing**

  - Modify agents to allow overriding models during testing.
  - Utilize `TestModel` and `FunctionModel` for simulating LLM responses in tests.

- **3.1.2** - **Pass Dependencies via `RunContext`**

  - Refactor methods to accept dependencies through `RunContext`, facilitating better isolation during tests.

**3.2** - **Set `ALLOW_MODEL_REQUESTS=False` Globally**

- In the testing environment, set `pydantic_ai.models.ALLOW_MODEL_REQUESTS = False` to prevent real API calls.
- Ensure this setting is applied before tests are run.

### Phase 4: Writing Unit Tests [IN PROGRESS]

**4.1** - **Set Up `pytest`**

- **4.1.1** - **Install and Configure `pytest`**

  - Add `pytest` to the development dependencies in `pyproject.toml`:

    ```toml
    [tool.poetry.group.dev.dependencies]
    pytest = "^7.0.0"
    ```

  - Create a `tests/` directory for test modules. [✓ COMPLETE]
  - Configure any necessary `pytest` settings.

**4.2** - **Write Unit Tests Using `TestModel` and `FunctionModel`** [✓ COMPLETE]

- **4.2.1** - **Test Game Logic** [✓ COMPLETE]

  - Write tests for each game (e.g., `NimGame`, `BattleshipGame`, `WarGame`) to verify:
    - Move validation (`validate_move`) [✓ COMPLETE]
    - State transitions (`apply_move`) [✓ COMPLETE]
    - Correct handling of game state and views [✓ COMPLETE]
    - Game end conditions and winner detection [✓ COMPLETE]

- **4.2.2** - **Test `LLMPlayer` Interactions** [✓ COMPLETE]

  - Use `TestModel` to simulate LLM responses in tests for `LLMPlayer`. [✓ COMPLETE]
  - Verify that `LLMPlayer` correctly processes game views and generates moves. [✓ COMPLETE]
  - Test database logging functionality [✓ COMPLETE]
  - Verify system prompt consistency [✓ COMPLETE]

- **4.2.3** - **Advanced Testing with `FunctionModel`** [✓ COMPLETE]

  - For more control, use `FunctionModel` to provide specific responses from the LLM in tests. [✓ COMPLETE]
  - Simulate error cases and retries to test edge conditions. [✓ COMPLETE]
  - Test invalid move handling and retry logic [✓ COMPLETE]

**4.3** - **Enhance Assertions with `dirty-equals` and `inline-snapshot`**

- Install `dirty-equals` and `inline-snapshot` for more expressive assertions:

  ```toml
  [tool.poetry.group.dev.dependencies]
  dirty-equals = "^0.6.0"
  inline-snapshot = "^1.2.0"
  ```

- Use these libraries in tests to compare complex data structures and capture snapshots.

**4.4** - **Ensure Comprehensive Test Coverage**

- Aim for high coverage of all critical paths in the system.
- Include tests for:
  - Error handling and exceptions
  - Boundary conditions and invalid inputs
  - Database interactions (using mocks or test databases)

### Phase 5: Updating Documentation

**5.1** - **Revise `README.md`**

- Update instructions for setting up the development environment.
- Include guidelines on how to run tests.
- Document any changes to how LLMs are integrated.

**5.2** - **Update `DESIGN.md`**

- Reflect architectural changes due to the migration to `pydantic-ai`.
- Describe how agents and models are now structured.
- Explain testing strategies and the benefits of the new approach.
