# Resumable Games and Experiments Design & Implementation Plan

## 1. Goal

Allow experiments run via the `Arena` to be stopped (e.g., Ctrl+C, budget exceeded) and resumed later. When resumed, any games that were in progress should continue from the exact state they were left in.

## 2. Design Overview

### 2.1. State Persistence: Historical Snapshots

-   **Immutable History:** Game states will be saved historically. Each significant state change (e.g., before a player's move) will result in a *new* `GameState` record in the database, rather than overwriting a single record. This preserves the full game history for debugging and analysis.
-   **Database Schema:**
    -   The `GameState` table will store snapshots of the game's state (`state_data`).
    -   Each `GameState` record will be linked to its `GameMatch` via `game_id`.
    -   A `timestamp` field will be added to `GameState` to allow ordering and retrieval of the latest state.
    -   The direct relationship from `GameMatch` to `GameState` (`uselist=False`) will be removed to allow multiple state records per game.

### 2.2. State Management Abstraction

-   **`GameStateManager`:** A new class, `GameStateManager`, will be introduced to encapsulate database interactions related to game states.
    -   It will provide methods like `save_state(session, game_id, state_data)` and `get_latest_state_data(session, game_id)`.
    -   This decouples `GameRunner` and `Arena` from direct `GameState` model manipulation, improving separation of concerns and testability.

### 2.3. State Serialization & Deserialization

-   **Serialization:** The existing `bgbench.serialization.serialize_value` function will be used by `GameStateManager` to convert state objects into JSON-compatible dictionaries before saving. Game-specific `StateType` dataclasses must ensure their `to_dict` methods (if used) or structure are compatible.
-   **Deserialization:** Each game's `StateType` dataclass (e.g., `AzulState`, `ChessState`) will require a `classmethod from_dict(cls, data: Dict[str, Any]) -> StateType` to reconstruct the state object from the dictionary loaded from the database.

### 2.4. Resumption Process in `Arena`

-   **Identification:** When resuming (`Arena._resume_experiment`), the system will query the database for `GameMatch` records associated with the experiment where `complete` is `False`.
-   **Loading State:** For each incomplete game, `Arena` will use `GameStateManager.get_latest_state_data` to retrieve the most recent state snapshot.
-   **Reconstruction:** The retrieved `state_data` dictionary will be passed to the appropriate `StateType.from_dict` method (determined by the game type) to reconstruct the live state object.
-   **Tracking:** `Arena` will maintain an internal list of these "resumable games," storing the player identifiers, the reconstructed state object, and the existing `game_id`.

### 2.5. Scheduling Resumed Games

-   **Prioritization:** The main `Arena.evaluate_all` loop will prioritize scheduling games from the "resumable games" list over starting new games.
-   **Concurrency Limits:** Resumed games are still subject to the configured concurrency limits (`max_parallel_games`, `max_concurrent_games_per_pair`). The `Arena` will check these limits *before* launching a resumed game task.
-   **Fallback:** If the highest-priority resumable game cannot be scheduled due to limits (e.g., that specific player pair already has the maximum concurrent games running), the `Arena` will then attempt to schedule a *new* game using the standard `find_next_available_match` logic.

### 2.6. Game Execution (`GameRunner`)

-   **Initialization:** `GameRunner` will be modified to accept an optional `initial_state` object and an instance of the `GameStateManager`.
-   **Starting:** If `initial_state` is provided (i.e., resuming), the `GameRunner` will start execution from that state instead of calling `game.get_initial_state()`.
-   **Saving State:** Within its game loop (e.g., before each player move), `GameRunner` will call `game_state_manager.save_state`, passing the current state object to be serialized and saved as a *new* historical record.

## 3. Implementation Plan Checklist

This checklist outlines the steps to implement the resumable games feature. We will tackle these sections incrementally.

### 3.0. Terminology Refactor (Codebase-wide)

-   [x] Rename `GameState` model class to `MatchState` in `bgbench/models.py`.
-   [x] Update `MatchState.__tablename__` to `match_states`.
-   [x] Update `MatchState` model to use `match_id` instead of `game_id`.
-   [x] Update all imports and references to `GameState` to use `MatchState` (`bgbench/game_runner.py`, `bgbench/arena.py`).
-   [x] Update variable names like `game_state` to `match_state` or `match_state_record` where appropriate.
-   [x] Ensure consistent use of "match" (instance of play) and "game" (ruleset) in comments and logs.
-   [x] Generate and apply database migration (`b20f910fb6ce`) to:
    -   Rename the `game_states` table to `match_states`.
    -   Rename the `game_id` column to `match_id`.
    -   Add the `timestamp` column to `# Resumable Games and Experiments Design & Implementation Plan

## 1. Goal

Allow experiments run via the `Arena` to be stopped (e.g., Ctrl+C, budget exceeded) and resumed later. When resumed, any games that were in progress
should continue from the exact state they were left in.

## 2. Design Overview

### 2.1. State Persistence: Historical Snapshots

- **Immutable History:** Game states will be saved historically. Each significant state change (e.g., before a player's move) will result in a *new*
`MatchState` record in the database, rather than overwriting a single record. This preserves the full game history for debugging and analysis.
- **Type-Safe State Management:** A `MatchStateData` dataclass will provide a clear contract for state data with proper typing.
- **Database Schema:**
  - The `MatchState` table will store snapshots of the game's state (`state_data`).
  - Each `MatchState` record will be linked to its `GameMatch` via `match_id`.
  - A `timestamp` field will be added to `MatchState` to allow ordering and retrieval of the latest state.
  - The direct relationship from `GameMatch` to `MatchState` (`uselist=False`) will be removed to allow multiple state records per game.

### 2.2. MatchStateData Structure

```python
@dataclass
class MatchStateData:
    """Represents the state of a match that can be saved to and loaded from the database."""

    # Common fields across all game types
    turn: int  # The current turn number (0 for initial state)
    current_player_id: int  # ID of the player whose turn it is
    timestamp: datetime  # When this state was recorded

    # Game-specific state (serialized by the Game implementation)
    game_state: Dict[str, Any]  # Game-specific state that can be deserialized by the Game

    # Optional metadata
    metadata: Optional[Dict[str, Any]] = None  # Additional information (e.g., visible_state, history)
```

### 2.3. Game Interface Extensions

The `Game` abstract base class will be extended with methods to handle state serialization and deserialization:

```python
class Game(ABC, Generic[StateType, MoveType]):
    # Existing methods...

    @abstractmethod
    def serialize_state(self, state: StateType) -> Dict[str, Any]:
        """
        Serialize the game state into a JSON-compatible dictionary.

        This method must ensure that all game-specific state is properly serialized
        into a format that can be stored in the database and later deserialized.

        Args:
            state: The game state to serialize

        Returns:
            A JSON-compatible dictionary representing the game state
        """
        pass

    @abstractmethod
    def deserialize_state(self, state_data: Dict[str, Any]) -> StateType:
        """
        Deserialize a dictionary into a game state object.

        This method must be able to reconstruct the game state from the dictionary
        produced by serialize_state.

        Args:
            state_data: The dictionary to deserialize

        Returns:
            A reconstructed game state object
        """
        pass
```

### 2.4. State Management Abstraction

- **`MatchStateManager`:** A class that encapsulates database interactions related to match states.
  - It will provide methods like `save_state(session, match_id, state_data)` and `get_latest_state(session, match_id)`.
  - This decouples `GameRunner` and `Arena` from direct `MatchState` model manipulation, improving separation of concerns and testability.

```python
class MatchStateManager:
    """Handles saving and retrieving match state snapshots from the database."""

    def save_state(
        self,
        session: Session,
        match_id: int,
        state_data: MatchStateData
    ) -> None:
        """
        Saves the given match state to the database.

        Args:
            session: The SQLAlchemy session
            match_id: The ID of the match
            state_data: The MatchStateData object to save
        """
        # Implementation...

    def get_latest_state(
        self,
        session: Session,
        match_id: int
    ) -> Optional[MatchStateData]:
        """
        Retrieves the most recent state for a given match.

        Args:
            session: The SQLAlchemy session
            match_id: The ID of the match

        Returns:
            The most recent MatchStateData, or None if no state exists
        """
        # Implementation...
```

### 2.5. Resumption Process in `Arena`

- **Identification:** When resuming (`Arena._resume_experiment`), the system will query the database for `GameMatch` records associated with the
experiment where `complete` is `False`.
- **Loading State:** For each incomplete game, `Arena` will use `MatchStateManager.get_latest_state` to retrieve the most recent state snapshot.
- **Reconstruction:** The retrieved `MatchStateData` object will be used to reconstruct the game state using
`game.deserialize_state(state_data.game_state)`.
- **Tracking:** `Arena` will maintain an internal list of these "resumable matches," storing the player identifiers, the reconstructed state object, and
the existing `match_id`.

### 2.6. Scheduling Resumed Games

- **Prioritization:** The main `Arena.evaluate_all` loop will prioritize scheduling games from the "resumable games" list over starting new games.
- **Concurrency Limits:** Resumed games are still subject to the configured concurrency limits (`max_parallel_games`, `max_concurrent_games_per_pair`).
The `Arena` will check these limits *before* launching a resumed game task.
- **Fallback:** If the highest-priority resumable game cannot be scheduled due to limits (e.g., that specific player pair already has the maximum
concurrent games running), the `Arena` will then attempt to schedule a *new* game using the standard `find_next_available_match` logic.

### 2.7. Game Execution (`GameRunner`)

- **Initialization:** `GameRunner` will be modified to accept an optional `initial_state` object and an instance of the `MatchStateManager`.
- **Starting:** If `initial_state` is provided (i.e., resuming), the `GameRunner` will start execution from that state instead of calling
`game.get_initial_state()`.
- **Saving State:** Within its game loop (e.g., before each player move), `GameRunner` will create a `MatchStateData` object and call
`match_state_manager.save_state`, passing the object to be saved as a *new* historical record.

## 3. Implementation Plan Checklist

This checklist outlines the steps to implement the resumable games feature. We will tackle these sections incrementally.

### 3.1. Core Data Structures

- [x] Create a new file `bgbench/match_state.py` to define the `MatchStateData` dataclass.
- [x] Implement the `MatchStateData` dataclass with proper type hints and documentation.
- [x] Add serialization/deserialization methods to convert between `MatchStateData` and database-friendly dictionaries.

### 3.2. Game Interface Extensions

- [x] Update the `Game` abstract base class in `bgbench/game.py` to add the `serialize_state` and `deserialize_state` abstract methods.
- [x] Add documentation for these methods explaining the contract that implementations must follow.
- [x] Update existing game implementations to implement these methods:
  - [x] `NimGame` in `bgbench/games/nim_game.py`
  - [x] `ChessGame` in `bgbench/games/chess_game.py`
  - [x] `AzulGame` in `bgbench/games/azul_game.py`
  - [x] `BattleshipGame` in `bgbench/games/battleship_game.py`
  - [x] `CantStopGame` in `bgbench/games/cant_stop_game.py`
  - [x] `GuessWhoGame` in `bgbench/games/guess_who_game.py`
  - [x] `LoveLetterGame` in `bgbench/games/love_letter_game.py`
  - [x] `ScrabbleGame` in `bgbench/games/scrabble_game.py`
  - [x] `WarGame` in `bgbench/games/war_game.py`
  - [x] `PokerGame` in `bgbench/games/poker_game.py` (serialization only, deserialization raises NotImplementedError)

### 3.3. State Management

- [x] Update `MatchStateManager` in `bgbench/match_state_manager.py` to use the new `MatchStateData` class.
- [x] Implement `save_state(self, session: Session, match_id: int, state_data: MatchStateData) -> None`.
- [x] Implement `get_latest_state(self, session: Session, match_id: int) -> Optional[MatchStateData]`.
- [x] Add unit tests for the updated `MatchStateManager` methods.

### 3.4. Game Runner Modifications

### 3.4. Game Runner Modifications

- [x] Update `GameRunner.__init__` in `bgbench/game_runner.py` to accept `match_state_manager: MatchStateManager` and `initial_state: Optional[StateType]
= None` parameters.
- [x] Modify `play_game` to use `self.initial_state` if provided, otherwise call `self.game.get_initial_state()`.
- [x] Update the game loop to create `MatchStateData` objects and save them using `match_state_manager.save_state`.
- [x] Add unit tests for the updated `GameRunner` functionality.

### 3.5. Arena Resumption Logic

- [x] Add `match_state_manager: MatchStateManager` instance variable to `Arena` in `bgbench/arena.py`.
- [x] Update `_resume_experiment` to:
  - Query for incomplete matches
  - Retrieve the latest state for each match
  - Deserialize the game state
  - Store resumable matches in a list
- [x] Modify `run_single_game` to accept `resumed_state` and `existing_match_id` parameters.
- [x] Update `evaluate_all` to prioritize resumable matches.
- [x] Add unit tests for the updated `Arena` functionality.

### 3.7. Documentation

- [x] Update `DESIGN.md` to reflect the new approach.
- [x] Add examples of implementing `serialize_state` and `deserialize_state` for different game types.
- [x] Document the `MatchStateData` class and its usage.
- [x] Update this file (`RESUMABLE_MATCHES.md`) by checking off completed items.

### 3.8. Refactoring & Modularity (Post-Feature Implementation)

-   [x] Create a dedicated `bgbench.match` module to encapsulate match-related logic:
    -   [x] Create `bgbench/match/runner.py` (refactored from `game_runner.py`):
        -   [x] Rename `GameRunner` to `MatchRunner`
        -   [ ] Focus on orchestrating the execution of a single match
        -   [ ] Handle player interactions and move application
        -   [ ] Track match progress and termination
    -   [x] Create `bgbench/match/state_manager.py` (refactored from `match_state_manager.py`):
        -   [ ] Move `MatchStateManager` class here
        -   [ ] Keep references to models in `models.py` (don't move DB models)
    -   [x] Create `bgbench/match/view.py` (refactored from `game_view.py`):
        -   [x] Rename `GameView` to `MatchView`
        -   [ ] Focus on formatting game state for presentation to players
    -   [ ] Create `bgbench/match/result.py`:
        -   [ ] Implement `MatchResult` class to encapsulate match outcomes
        -   [ ] Provide methods for recording results in the database
    -   [ ] Create `bgbench/match/utils.py`:
        -   [ ] Add helper functions for match-related operations
    -   [ ] Create `bgbench/match/exceptions.py`:
        -   [ ] Define match-specific exception classes
-   [x] Update `Arena` to use the new match module components:
    -   [x] Use `MatchRunner` instead of `GameRunner`
    -   [x] Update resumption logic to work with the new module structure
-   [x] Update imports across the codebase to reference the new module structure
-   [x] Ensure all tests are updated to work with the new module structure
