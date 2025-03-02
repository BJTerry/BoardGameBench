# Game Contributions Guide

This guide explains how to implement game engines within the architecture described in [DESIGN.md](../DESIGN.md),
following the coding conventions in [CONVENTIONS.md](../CONVENTIONS.md). New games allow LLMs to play against one another without human intervention. Follow these guidelines to ensure your game integrates smoothly with the platform’s architecture, conforms to security and serialization standards, and meets the expectations for hidden information and proper game state management.

---

## 11. Enabling LLM to Perform Moves

To ensure that the LLM can effectively perform moves in a game, it is crucial to provide clear instructions, robust parsing, and comprehensive validation. This involves three key components:

1. **Move Format Instructions:**
   - **Purpose:** Guide the LLM on how to format its moves.
   - **Implementation:** Use the `get_move_format_instructions` method to provide a detailed string that explains how moves should be formatted.
   - **Example:** In Scrabble, the instructions might be: "To make a move, specify the word, starting position (row, column), and direction (horizontal or vertical). Example: 'WORD 7 7 horizontal'."

2. **Move Parsing:**
   - **Purpose:** Convert the LLM's response into a structured move object.
   - **Implementation:** Implement the `parse_move` method to handle the conversion from a string to a move object.
   - **Considerations:** Ensure robust error handling for malformed inputs and provide meaningful error messages.

3. **Move Validation:**
   - **Purpose:** Ensure that the move is legal according to the game's rules.
   - **Implementation:** Implement the `validate_move` method to check the legality of the move.
   - **Considerations:** Verify turn order, game-specific constraints, and any other rules that apply to the move.

### Example from Scrabble

In the Scrabble game implementation:
- **Move Format Instructions:** Clearly instruct the LLM on how to specify a word, its starting position, and direction.
- **Move Parsing:** Convert the LLM's string input into a `ScrabbleMove` object, handling any errors gracefully.
- **Move Validation:** Check that the word fits on the board, is connected to existing words, and is in the dictionary.

By ensuring these components are well-implemented, the LLM can effectively interact with the game and perform all possible moves.

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

- **Move Format Instructions:**  
  Every game must provide clear move format instructions to guide the LLM on how to play the game. This is crucial for ensuring that the LLM can generate valid moves.
  - **Implementation:** Use the `get_move_format_instructions` method to return a string that explains how moves should be formatted. This string should be included in the `GameView`.
  - **Example:** In `chess_game.py`, the move format instructions are provided in standard chess notation (PGN format).

- **GameView Setup:**  
  The `GameView` class in `game_view.py` is used to encapsulate what a player can see of the game state. It includes:
  - `visible_state`: A dictionary or string representing the current state visible to the player.
  - `valid_moves`: A list of legal moves available to the player.
  - `is_terminal`: A boolean indicating if the game has ended.
  - `winner`: The player ID of the winner if the game is over, otherwise `None`.
  - `history`: A list of previous moves and their results.
  - `move_format_instructions`: Instructions on how moves should be formatted.
  - `rules_explanation`: An explanation of the game rules.
  - `prompt_style`: The style in which the prompt should be formatted (e.g., JSON, XML, HEADER).

- **Example from ChessGame:**  
  In `chess_game.py`, the `get_player_view` method constructs a `GameView` by:
  - Providing the current board position in FEN format.
  - Listing legal moves in PGN format.
  - Including move format instructions to guide the LLM.
  - Indicating if the game is in a terminal state and who the winner is, if applicable.

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

To ensure the robustness and correctness of your game implementation, it is crucial to create comprehensive unit tests. Follow these guidelines to write effective tests using `pytest`, the project's testing framework.

### Setting Up Tests

1. **Use Fixtures:**
   - Create fixtures for setting up the game instance and initial state.
   - Example:
     ```python
     import pytest
     from bgbench.games.your_game import YourGame, YourState

     @pytest.fixture
     def game():
         return YourGame()

     @pytest.fixture
     def initial_state(game):
         return game.get_initial_state()
     ```

2. **Test Core Game Logic:**
   - Write tests for move parsing, validation, and application.
   - Example:
     ```python
     def test_parse_move(game):
         move = game.parse_move("your_move_format")
         assert move is not None
     ```

3. **Test Edge Cases:**
   - Include tests for invalid moves, special rules, and game-ending conditions.
   - Example:
     ```python
     def test_invalid_move(game, initial_state):
         move = game.parse_move("invalid_move")
         is_valid, _ = game.validate_move(initial_state, 0, move)
         assert not is_valid
     ```

4. **Simulate Game Scenarios:**
   - Simulate full games to ensure correct behavior under various scenarios.
   - Example:
     ```python
     def test_game_scenario(game, initial_state):
         moves = ["move1", "move2", "move3"]
         state = initial_state
         for move_str in moves:
             move = game.parse_move(move_str)
             state = game.apply_move(state, 0, move)
         assert game.is_terminal(state)
     ```

### Best Practices

- **Consistent Naming:** Use descriptive names for test functions to indicate what they are testing.
- **Comprehensive Coverage:** Ensure tests cover all aspects of the game logic, including edge cases.
- **Use Assertions:** Use assertions to verify expected outcomes and behaviors.

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
