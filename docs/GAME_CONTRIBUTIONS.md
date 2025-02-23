# Game Contributions Guide

This document explains how to add new game engines to BoardGameBench. New games allow LLMs to play against one another without human intervention. Follow these guidelines to ensure your game integrates smoothly with the platform’s architecture, conforms to security and serialization standards, and meets the expectations for hidden information and proper game state management.

---

## 1. Overview

BoardGameBench uses a modular design where every game engine:
- **Inherits from a common Game interface** defined in `bgbench/game.py`
- **Defines its own state and move data structures** (typically using Python dataclasses)
- **Implements required game methods** for initialization, move parsing/validation, state updating, and generating player-specific views
- **Ensures hidden information is maintained** so LLM players see only what they are allowed to see according to the game’s rules
- **Provides a state that is JSON-serializable** for logging, database storage, and debugging

---

## 2. File Structure and Organization

- **Location:** All game engines should be placed in the `bgbench/games/` directory.
- **Naming Convention:** Name your file `<game_name>_game.py` (e.g., `tic_tac_toe_game.py`).
- **Module Registration:** Update `bgbench/games/__init__.py` to import and register your new game class in the `AVAILABLE_GAMES` dictionary.

---

## 3. Game Interface Requirements

Every game must subclass the abstract class `Game[StateType, MoveType]` defined in `bgbench/game.py` and implement the following methods:

- **`get_initial_state()`**  
  Create and return the initial game state.  
  _Tip:_ Use dataclasses to structure the state and include a `to_dict()` method for JSON serialization.

- **`get_player_view(state, player_id, history, prompt_style)`**  
  Generate a `GameView` object that filters the state to show each player only the information they are allowed to see.  
  _Key Point:_ Hide secret or sensitive information (e.g., opponent’s hand or hidden cards).

- **`parse_move(move_str)`**  
  Convert an LLM response string into a move object.  
  _Tip:_ Ensure robust error handling for malformed inputs.

- **`validate_move(state, player_id, move)`**  
  Check if a move is legal given the current state and the rules of the game. Return a tuple of `(is_valid, explanation)`.  
  _Consideration:_ Verify turn order and any game-specific constraints.

- **`apply_move(state, player_id, move)`**  
  Apply a validated move to the state, returning a new state.  
  _Best Practice:_ Use deep copies of the state to avoid side effects.

- **`get_current_player(state)`**  
  Return the ID of the player whose turn it is.

- **`is_terminal(state)`**  
  Determine if the game has ended.

- **`get_winner(state)`**  
  Return the winning player’s ID if the game is over, otherwise `None`.

- **`get_next_state(state, move)`**  
  Compute and return the next state after a move is applied. This may be a simple wrapper around `apply_move`.

---

## 4. Creating the Game State and Move Data Structures

- **State Objects:**  
  Use Python dataclasses to define game state. Ensure your state includes a `to_dict()` method so that it can be easily serialized for database storage and logging.

- **Move Objects:**  
  Define move classes or structures that capture the essential data for a move. Include a `to_dict()` method if necessary, and a clear `__str__()` or similar representation for debugging.

---

## 5. Implementing Game Methods

When implementing game methods, keep these common patterns in mind:

- **Initialization (`get_initial_state`):**  
  Set up any game boards, card decks, or initial counters. Follow the game’s rule book or design document closely.

- **Parsing and Validation:**  
  - **`parse_move`:** Convert raw strings to structured move data.
  - **`validate_move`:** Ensure moves follow both the game rules and the turn order. Return meaningful error messages for invalid moves.

- **State Transitions:**  
  In `apply_move` and `get_next_state`, update the game state immutably (using deep copies) to avoid unintended side effects.

- **Turn Management:**  
  Correctly update the current player and check for terminal conditions after each move.

---

## 6. Ensuring Proper Player Views and Hidden Information

- **Player-Specific Views:**  
  The `get_player_view` method must carefully filter game state data so that LLMs (acting as players) see only the information they are allowed to know.
  - **Example:** In card games, the opponent’s hand should not be visible.
  - **Tip:** Build a dictionary of “visible_state” with keys like `your_board`, `target_board`, or similar, and avoid including internal state details.

- **Documentation:**  
  Clearly document in your code which parts of the state are public and which are private.

---

## 7. Serialization and Database Integration

- **State Serialization:**  
  All state classes should implement a `to_dict()` method that produces a JSON-serializable representation.  
  _Why:_ This is used for logging match progress, debugging, and persisting game states in the database.

- **Consistency:**  
  Ensure that every piece of the game state (including nested objects) can be converted to a standard dictionary format.

---

## 8. Testing and Validation

- **Unit Testing:**  
  Create tests for each game engine to verify that:
  - Moves are correctly parsed and validated.
  - State transitions follow the rules.
  - Player views do not leak hidden information.
  - Terminal conditions and winner detection are accurate.

- **Simulation:**  
  Run simulated matches (LLM vs. LLM) to check for edge cases and ensure the game behaves as expected under various scenarios.

---

## 9. Common Patterns and Best Practices

- **Use of Dataclasses:**  
  They simplify state and move management and automatically support serialization patterns.

- **Defensive Copying:**  
  Use `copy.deepcopy` when updating state to avoid unintended mutations.

- **Error Handling:**  
  Provide clear and concise error messages during move validation to help with debugging and testing.

- **Consistency with Existing Games:**  
  Refer to the implementations of games like [Nim](./nim_game.py), [Chess](./chess_game.py), and [Battleship](./battleship_game.py) for examples of:
  - State structure
  - Player view filtering
  - Move parsing and application

- **Documentation and Comments:**  
  Write clear comments in your code explaining key decisions, especially around hidden information and state serialization.

---

## 10. Contribution Process

**Implement Your Game:**  
   Follow the guidelines above and ensure your code conforms to the overall structure of BoardGameBench.

**Add Tests:**  
   Include unit tests and sample game simulations to verify correctness.

**Update Documentation:**  
   Add or update any relevant documentation and register your game in the main game list in `bgbench/games/__init__.py`.