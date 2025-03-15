# Azul Implementation Plan

## Overview

Azul is a tile-placement game where players score points by strategically placing colorful tiles on their wall. The goal is to score the most points by completing specific patterns on your personal player board. Below is a detailed implementation plan, including complete rules, necessary state management, and interaction logic tailored for LLM benchmarking.

## 1. Game Objective

Players earn points by strategically placing tiles to decorate a palace wall. The game ends when at least one player completes a horizontal line of 5 tiles on their wall.

## 2. Game Components
- **Tiles:** 100 tiles, 20 each of 5 different colors.
- **Player Boards:** Individual boards with Pattern Lines, Wall, Score Track, and Floor line.
- **Factory Displays:** Disks used to present tiles.

## 3. Game Setup

- Each player receives a player board and a scoring marker placed at zero.
- Factory displays: 5 (2-player), 7 (3-player), or 9 (4-player), arranged around the table center.
- Fill bag with 100 tiles (20 of each color).
- Randomly place exactly 4 tiles on each factory display.
- The first player marker goes to the player who recently visited Portugal (or randomly chosen).

## 3. Data Structures

### AzulState
- `factory_displays`: Tiles currently on each factory display.
- `center_tiles`: Tiles moved to center area, initially empty.
- `player_boards`: List of PlayerBoard objects (one per player).
- `tile_bag`: List of remaining tiles.
- `lid`: Temporary storage for discarded tiles.
- `current_player`: ID of the current player.
- `starting_player`: ID of the player holding the first player marker.

### PlayerBoard
- `pattern_lines`: 5 rows (1-5 spaces each), tiles placed right-to-left.
- `wall`: 5x5 grid with specific color constraints.
- `floor_line`: Accumulates penalties for excess or unwanted tiles.
- `score`: Player’s current score.

## 3. Gameplay

Each round consists of three phases:

### A. Factory Offer Phase

- Players take turns clockwise, starting with the starting player.
- On your turn, choose tiles either:
  - From a single factory display (all tiles of one color) and move remaining tiles to the center.
  - From the table center (first to choose from center also takes starting marker).
- Add tiles to one pattern line on your board:
  - Fill lines from right to left.
  - Only same-color tiles can occupy a single line.
  - Excess tiles go to the floor line (negative points).
  - You can't place tiles of a color already placed on the corresponding wall line.

Phase ends when all tiles have been taken from factories and center.

### B. Wall-Tiling Phase

Simultaneous for all players:
- Move the rightmost tile from each completed pattern line to the matching wall spot.
- Score points immediately for each placed tile (see Scoring).
- Remove leftover tiles from completed pattern lines to the lid; incomplete lines remain for the next round.
- Deduct points for tiles in the floor line; discard these tiles into the lid.

### C. Preparing Next Round

- Check for game-end condition (a completed horizontal wall line).
- If the game continues, refill factory displays from tile bag (refill bag from lid if needed).
- Next round begins with the player holding the starting player marker.

## 4. Scoring

- Score points immediately for tiles moved to the wall:
  - Isolated tile: 1 point.
  - Tiles adjacent horizontally or vertically form groups:
    - Score 1 point per tile in contiguous horizontal/vertical line (including the placed tile).
- Floor line: Lose points indicated above occupied spaces.

### End-of-Game Scoring Bonuses:
- 2 points for each complete horizontal line.
- 7 points for each complete vertical line.
- 10 points for each color fully completed (5 tiles).

Highest score wins. Break ties by most completed horizontal lines, then share victory.

## 4. Data Classes and State Management

Define using Python dataclasses:
- **PlayerBoard:** pattern_lines (list of lists), wall (5x5 grid), floor_line, score.
- **AzulMove:** Source ('factory' or 'center'), source_id (factory number or 'center'), color, pattern_line_number.

## 4. Required Methods Implementation

- `get_initial_state()`: Initialize the state per setup rules.
- `parse_move(move_str)`: Convert move string to AzulMove.
- `validate_move(state, player_id, move)`: Check legality per rules.
- `apply_move(state, player_id, move)`: Execute moves, updating pattern lines, floor line, and center tiles.
- `get_current_player(state)`: Return current player's ID.
- `is_terminal(state)`: Check if horizontal line completed.
- `get_winner(state)`: Determine winner from scores.
- `get_next_state(state, move)`: Compute state after applying move.

## 5. Player Views

- **Visible**: All factory displays, center tiles, player's own pattern lines, wall, floor line, and own score.
- **Hidden Information**: Opponent boards’ incomplete pattern lines are hidden.

## 6. Move Format Instructions (LLM interaction)

Example instructions for the LLM:
- "Pick tiles by specifying source ('factory' or 'center'), source id (factory number or 'center'), tile color, and pattern line number (1-5). Example: `factory 3 blue 4` or `center center red 2`."

## 6. Serialization & Logging

- Implement `.to_dict()` method for `AzulState` and `AzulMove`.
- Clearly log each player's state, including current score, board status, and tiles selected.

## 7. Testing Plan

- Use `pytest` with fixtures for initial state.
- Test tile picking logic, pattern-line filling rules, floor-line handling.
- Simulate full rounds to verify scoring logic, particularly wall adjacency scoring.
- Check for game-end triggering and final scoring.

## Implementation Checklist:

## Final Implementation Steps (Similar to SCRABBLE.md Format)

1. **Define Game State and Move Structures**
   - [x] Create the `AzulState`, `PlayerBoard`, and `AzulMove` data classes.
   - [x] Implement a `to_dict()` method for serialization.

2. **Initialize Game State**
   - [x] Set up the bag with 100 tiles (5 colors × 20).
   - [x] Shuffle tiles, distribute 4 to each factory display, and store the rest in `tile_bag`.
   - [x] Initialize each player's board, score, and floor line.
   - [x] Mark the starting player.

3. **Implement Move Parsing and Validation**
   - [x] Write `parse_move` to interpret LLM commands (e.g., `factory 2 red 3`).
   - [x] Verify legality in `validate_move`:
      - [x] Check that the chosen color exists in the selected source.
      - [x] Check if the pattern line can accommodate that color without conflicts.
      - [x] Ensure no color conflict on the corresponding wall row.
      - [x] Check for excess tiles that must go to the floor line.
   - [x] Validate that if picking from center for the first time, place first-player marker.

4. **Apply Moves and Update State**
   - [x] Implement `apply_move`:
      - [x] Remove chosen tiles from factory or center.
      - [x] Move leftover tiles (if from a factory) to the center.
      - [x] Place selected tiles into the appropriate pattern line.
      - [x] Any overflow tiles go into the floor line.
      - [x] If center is chosen for the first time, place first-player marker in floor line.

5. **Wall-Tiling Phase**
   - [x] After all tiles are claimed:
      - [x] For each completed pattern line, move the rightmost tile to the wall.
      - [x] Score each tile immediately (count horizontal and vertical adjacency).
      - [x] Discard the rest of the tiles in that pattern line to the lid.
      - [x] Subtract points for each tile in the floor line and move them to the lid.

6. **Round Transition & Next Rounds**
   - [x] Check `is_terminal` if any player has a full horizontal line.
   - [x] If not terminal:
      - [x] Refill each factory display with 4 new tiles from `tile_bag` (or refill `tile_bag` from `lid` if empty).
      - [x] `current_player` becomes whoever holds the first-player marker.

7. **End of Game and Final Scoring**
   - [x] If a horizontal line is completed, the game ends immediately.
   - [x] Apply bonus scoring:
      - [x] +2 points for each completed horizontal line.
      - [x] +7 points for each completed vertical line.
      - [x] +10 points for each color with all 5 tiles on the wall.
   - [x] Use `get_winner` to compare final scores.

8. **Testing and Validation**
   - [x] Create unit tests covering:
      - [x] Initial tile distribution.
      - [x] Picking tiles from factory and center.
      - [x] Placing tiles into pattern lines and floor line.
      - [x] Scoring logic for adjacency and floor line penalties.
      - [x] Refilling factories for subsequent rounds.
      - [x] End-of-game detection and final bonus scoring.
      - [x] Edge cases (insufficient tiles, repeated colors in a row, overfilling pattern lines).

9. **Documentation and Maintenance**
   - [x] Provide a rules summary in `get_rules_explanation`.
      - [x] Outline factory pick rules, center pick rules.
      - [x] Pattern line placement and constraints.
      - [x] Floor line penalty rules.
      - [x] Scoring adjacency rules.
      - [x] End-of-game scoring.
   - [x] Maintain code in `bgbench/games/azul_game.py` with thorough docstrings.
   - [x] Add unit tests to `tests/test_azul_game.py`.

**Status: Implementation Complete**


