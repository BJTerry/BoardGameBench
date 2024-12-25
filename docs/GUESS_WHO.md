**Project Plan to Add Guess Who! to the System**

To add the game "Guess Who!" to your system, we'll need to implement several components. Below is a step-by-step plan outlining the necessary changes:

---

### 1. Create the `GuessWhoGame` Class

- **File**: `bgbench/games/guess_who_game.py`

- **Action**: Implement the `GuessWhoGame` class that inherits from `Game`.

```python
# bgbench/games/guess_who_game.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from bgbench.game import Game
from bgbench.game_view import GameView
import copy
import random

# Define Character and Trait data structures
@dataclass
class Character:
    name: str
    traits: Dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "traits": self.traits
        }

@dataclass
class GuessWhoState:
    characters: List[Character]
    possible_characters: List[List[Character]]  # List per player
    current_player: int

    def to_dict(self) -> dict:
        return {
            "characters": [char.to_dict() for char in self.characters],
            "possible_characters": [
                [char.to_dict() for char in chars] for chars in self.possible_characters
            ],
            "current_player": self.current_player
        }

class GuessWhoGame(Game):
    # Implement required methods here
    pass
```

---

### 2. Define Traits and Generate Characters

- **Action**: Create a method to generate a list of characters with random traits.

```python
# In GuessWhoGame class

TRAITS = {
    "hair_color": ["blonde", "brown", "black", "red", "white"],
    "hair_style": ["bald", "long", "short", "curly"],
    "eye_color": ["blue", "brown", "green"],
    "facial_hair": ["beard", "mustache", "clean-shaven"],
    "gender": ["male", "female"],
    "accessories": ["glasses", "hat", "earrings", "none"]
}

def generate_characters(self, num_characters: int) -> List[Character]:
    characters = []
    for i in range(num_characters):
        traits = {trait: random.choice(options) for trait, options in TRAITS.items()}
        character = Character(name=f"Character_{i+1}", traits=traits)
        characters.append(character)
    return characters
```

---

### 3. Implement `get_initial_state`

- **Action**: Initialize the game state with generated characters and possible characters for each player.

```python
# In GuessWhoGame class

def get_initial_state(self) -> GuessWhoState:
    characters = self.generate_characters(num_characters=24)
    possible_characters = [copy.deepcopy(characters) for _ in range(2)]
    return GuessWhoState(
        characters=characters,
        possible_characters=possible_characters,
        current_player=0
    )
```

---

### 4. Implement `get_rules_explanation`

- **Action**: Provide a rules explanation specific to Guess Who.

```python
# In GuessWhoGame class

def get_rules_explanation(self) -> str:
    return (
        "We are playing Guess Who!. Both players have the same set of characters, each with specific traits. "
        "On your turn, you may guess a characteristic or its negation (e.g., 'black hair' or 'NOT black hair'). "
        "You will receive feedback based on the opponent's chosen character. "
        "The goal is to narrow down the possible characters to one. "
        "First player to correctly identify the opponent's character wins."
    )
```

---

### 5. Implement `get_move_format_instructions`

- **Action**: Instruct players on how to format their moves.

```python
# In GuessWhoGame class

def get_move_format_instructions(self) -> str:
    return (
        "FORMAT: '<trait> <value>' or 'NOT <trait> <value>'\n"
        "- Trait must be one of: hair_color, hair_style, eye_color, facial_hair, gender, accessories\n"
        "- Value must be a valid option for the trait.\n"
        "Examples:\n"
        "- 'hair_color blonde'\n"
        "- 'NOT eye_color blue'"
    )
```

---

### 6. Implement `get_player_view`

- **Action**: Present the player's current view, including possible characters and move history.

```python
# In GuessWhoGame class

def get_player_view(
    self,
    state: GuessWhoState,
    player_id: int,
    history: Optional[List[Dict[str, Any]]] = None
) -> GameView:
    visible_state = {
        "possible_characters": [char.to_dict() for char in state.possible_characters[player_id]]
    }
    return GameView(
        rules_explanation=self.get_rules_explanation(),
        visible_state=visible_state,
        valid_moves=self._get_valid_moves(),
        is_terminal=self._is_game_over(state),
        winner=self._get_winner(state),
        history=history if history else [],
        move_format_instructions=self.get_move_format_instructions()
    )
```

---

### 7. Implement `_get_valid_moves`

- **Action**: Provide a list of valid moves (traits and values).

```python
# In GuessWhoGame class

def _get_valid_moves(self) -> List[str]:
    moves = []
    for trait, options in TRAITS.items():
        for value in options:
            moves.append(f"{trait} {value}")
            moves.append(f"NOT {trait} {value}")
    return moves
```

---

### 8. Implement `parse_move`

- **Action**: Parse the player's input into a structured move.

```python
# In GuessWhoGame class

def parse_move(self, move_str: str) -> Optional[Dict[str, Any]]:
    tokens = move_str.strip().upper().split()
    if not tokens:
        return None
    negation = False
    if tokens[0] == 'NOT':
        negation = True
        tokens = tokens[1:]
    if len(tokens) != 2:
        return None
    trait = tokens[0].lower()
    value = tokens[1].lower()
    if trait not in TRAITS or value not in [v.lower() for v in TRAITS[trait]]:
        return None
    return {
        "trait": trait,
        "value": value,
        "negation": negation
    }
```

---

### 9. Implement `validate_move`

- **Action**: Ensure the move is properly formatted and valid.

```python
# In GuessWhoGame class

def validate_move(
    self,
    state: GuessWhoState,
    player_id: int,
    move: Dict[str, Any]
) -> Tuple[bool, str]:
    if state.current_player != player_id:
        return False, "It's not your turn."
    if move is None:
        return False, "Invalid move format."
    trait = move["trait"]
    value = move["value"]
    if trait not in TRAITS or value not in [v.lower() for v in TRAITS[trait]]:
        return False, f"Invalid trait or value: {trait} {value}"
    return True, ""
```

---

### 10. Implement `apply_move`

- **Action**: Update the game state based on the player's move.

```python
# In GuessWhoGame class

def apply_move(
    self,
    state: GuessWhoState,
    player_id: int,
    move: Dict[str, Any]
) -> GuessWhoState:
    state = copy.deepcopy(state)
    opponent_id = 1 - player_id
    opponent_character = state.characters[opponent_id]
    trait = move["trait"]
    value = move["value"]
    negation = move["negation"]

    # Determine if the opponent's character has the trait
    opponent_has_trait = opponent_character.traits[trait].lower() == value

    # If negation, invert the result
    trait_matches = opponent_has_trait != negation

    # Update possible characters based on the answer
    state.possible_characters[player_id] = [
        char for char in state.possible_characters[player_id]
        if (char.traits[trait].lower() == value) == trait_matches
    ]

    # Advance to next player
    state.current_player = opponent_id

    return state
```

---

### 11. Implement `_is_game_over` and `_get_winner`

- **Action**: Determine if the game has ended and identify the winner.

```python
# In GuessWhoGame class

def _is_game_over(self, state: GuessWhoState) -> bool:
    return any(len(chars) == 1 for chars in state.possible_characters)

def _get_winner(self, state: GuessWhoState) -> Optional[int]:
    for player_id, chars in enumerate(state.possible_characters):
        if len(chars) == 1:
            return player_id
    return None
```

---

### 12. Ensure Serialization Support

- **Action**: Verify all classes implement `to_dict` methods for serialization.

---

### 13. Write Unit Tests

- **File**: `tests/test_guess_who_game.py`

- **Action**: Create unit tests for the `GuessWhoGame` class.

```python
# tests/test_guess_who_game.py

import pytest
from bgbench.games.guess_who_game import GuessWhoGame

def test_initial_state():
    game = GuessWhoGame()
    state = game.get_initial_state()
    assert len(state.characters) == 24
    assert len(state.possible_characters[0]) == 24
    assert state.current_player == 0

# Add more tests for validate_move, apply_move, etc.
```

---

### 14. Update `bgbench/main.py`

- **Action**: Integrate the new game into the game selection logic.

```python
# In bgbench/main.py

from bgbench.games.guess_who_game import GuessWhoGame

# Update game initialization
if args.game == 'guess_who':
    game = GuessWhoGame()
```

---

### 15. Update Documentation

- **Files**: `README.md`, `DESIGN.md`

- **Action**: Add descriptions and usage instructions for the new game.

---

### 16. Test the Game

- **Action**: Run the game using the main script to ensure it functions correctly.

```bash
poetry run python -m bgbench.main --game guess_who --debug
```

---

**Notes:**

- **Trait Normalization**: Ensure trait values are consistently formatted (e.g., lowercase) to avoid mismatches.

- **Random Character Assignment**: Consider assigning a random character to each player at the start.

- **LLM Interaction**: Since players are LLMs, ensure the move instructions are clear to generate valid moves.

- **Error Handling**: Implement robust error handling in `parse_move` and `validate_move`.

- **Concurrency**: Account for asynchronous execution if needed.