# LLM Game Evaluation Framework Design

See CONVENTIONS.md for coding style and practices to follow when implementing components.

## Architecture Overview

The framework enables systematic evaluation of LLM game-playing capabilities through a modular, extensible architecture.

### Core Components

1. Game Engine
   - Abstract Game interface defining core game mechanics
   - Game-specific state classes with type hints
   - Strict move validation
   - Player views through MatchView with configurable prompt styles

2. LLM Integration
   - Unified interface for OpenRouter and OpenAI models
   - Advanced conversation management with history tracking
   - Robust move parsing with chain-of-thought support
   - Configurable system prompts and response styles
   - Comprehensive error handling and retry logic

3. Game Implementations
   - NimGame, BattleshipGame, WarGame with consistent interfaces
   - Type-safe state management
   - Clear validation rules
   - Detailed game history tracking
   - See [GAME_CONTRIBUTIONS.md](docs/GAME_CONTRIBUTIONS.md) for detailed implementation guidelines.
   See GAME_CONTRIBUTIONS.md for detailed implementation guidelines.

4. Arena System
   - Automated match scheduling
   - Elo rating calculations
   - Confidence-based match termination
   - Comprehensive experiment tracking
   - Resumable matches and experiments

## Key Design Decisions

### 1. Game State & Views

Each game provides:
- Strongly typed state management
- Configurable view formatting (JSON/XML/Text)
- Detailed move validation
- Comprehensive game history
- Clear win condition tracking

Required Game Interface Methods:
- `get_rules_explanation(self) -> str`: Returns a string explaining the game rules.
- `get_move_format_instructions(self) -> str`: Returns instructions for formatting moves (e.g., "WORD 7 7 horizontal" for Scrabble).

### 2. LLM Integration

Advanced integration features:
- Multiple response styles (Direct/Chain-of-thought)
- Configurable prompt formatting
- Model-specific optimizations
- Detailed interaction logging
- Performance metrics tracking

### 3. Concurrency and Scheduling

- Concurrency: The `Arena` class leverages `asyncio` for parallel game execution, using `_active_tasks` for task management, `_lock` for state synchronization, and `ongoing_matches` to track progress (see `arena.py`).
- Scheduling Strategies: Configurable via `scheduler.py`:
  - `SigmaMinimizationScheduler`: Reduces uncertainty for uncertain players.
  - `TopIdentificationScheduler`: Identifies the best model.
  - `FullRankingScheduler`: Optimizes overall rankings.

### 4. Bayesian Rating System

- Bayesian Elo System: Uses PyMC for MCMC sampling, modeling wins/draws with a multi-outcome categorical approach. Default skill priors are set, with typical chain lengths for sampling (see `rating.py`).

### 5. Resumable Matches

- **State Persistence**: Match states are saved as immutable snapshots in the database, preserving the full history of each match.
- **MatchStateData**: A structured dataclass encapsulates match state data with proper typing and serialization support.
- **Game Interface Extensions**: Games implement `serialize_state` and `deserialize_state` methods to convert between game-specific state objects and database-friendly dictionaries.
- **MatchStateManager**: Handles saving and retrieving match state snapshots from the database.
- **Resumption Process**: The Arena identifies incomplete matches, loads their latest states, and prioritizes them for scheduling.
- **MatchRunner Integration**: MatchRunner accepts an initial state and state manager, saving state snapshots throughout the match.

### 6. Database Integration

- Experiment Tracking: The `Arena` loop integrates with the database schema (`models.py`):
  - Game start creates a `GameMatch`.
  - Moves are logged as `LLMInteraction`.
  - Results update `Player` ratings in `rating.py`.
  - Match states are saved as `MatchState` records for resumption.
- Comprehensive experiment tracking:
  - Full game history
  - Player statistics
  - LLM interaction details
  - Performance metrics
  - Rating progression
  - Match state snapshots for resumption

## Testing Framework

- **Framework:** The project uses `pytest` for unit testing.
- **Approach:** Write tests to cover core game logic, edge cases, and full game scenarios.
- **Guidelines:** Refer to `GAME_CONTRIBUTIONS.md` for detailed instructions on writing tests for new games.

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


### Best Practices

1. **Code Organization**
   - Clear separation of concerns
   - Type safety throughout
   - Comprehensive error handling
   - Detailed logging

2. **Testing Strategy**
   - Isolated component testing
   - Comprehensive mocking
   - Edge case coverage
   - Performance validation

3. **Documentation**
   - Clear API documentation
   - Usage examples
   - Configuration guides
   - Troubleshooting information

## Current Status

1. Core Features
   - Robust game engine with implementations for Chess, Azul, Scrabble, and more (see `bgbench/games/`).
   - Advanced LLM integration
   - Advanced scheduling strategies in the Arena system:
     - `SigmaMinimizationScheduler`: Targets players with high uncertainty.
     - `TopIdentificationScheduler`: Identifies the top-performing model.
     - `FullRankingScheduler`: Clarifies pairwise rankings.
   - Bayesian Elo rating system with uncertainty handling (see `rating.py`).
   - Resumable matches and experiments with state persistence (see `match_state_manager.py`).
   - Comprehensive testing

2. Database Features
   - Full experiment logging
   - Detailed player statistics
   - Complete game history
   - LLM interaction tracking
   - Performance metrics

3. Testing Coverage
   - Component isolation
   - Integration verification
   - Error handling
   - Performance validation

## Next Steps

1. Enhancements
   - Additional game implementations
   - Extended metrics tracking
   - Advanced analysis tools
   - Performance optimizations

2. Documentation
   - API reference updates
   - Configuration guides
   - Best practices documentation
   - Troubleshooting guides

3. Future Features
   - Tournament support
   - Advanced analytics
   - Real-time monitoring
   - Performance benchmarking
