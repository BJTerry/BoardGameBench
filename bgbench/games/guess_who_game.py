from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from bgbench.game import Game
from bgbench.game_view import GameView, PromptStyle
import random
import copy

@dataclass
class Character:
    """Represents a character in Guess Who with their traits."""
    name: str
    traits: Dict[str, Any]

    def to_dict(self) -> dict:
        """Convert character to JSON-serializable dictionary."""
        return {
            "name": self.name,
            "traits": self.traits
        }

@dataclass
class GuessWhoState:
    """Represents the current state of a Guess Who game."""
    characters: List[Character]  # All available characters
    target_characters: List[Character]  # One per player - their assigned character
    possible_characters: List[List[Character]]  # List per player of their remaining possibilities
    current_player: int

    def to_dict(self) -> dict:
        """Convert state to JSON-serializable dictionary."""
        return {
            "characters": [char.to_dict() for char in self.characters],
            "target_characters": [char.to_dict() for char in self.target_characters],
            "possible_characters": [
                [char.to_dict() for char in chars]
                for chars in self.possible_characters
            ],
            "current_player": self.current_player
        }

class GuessWhoGame(Game):
    """Implementation of Guess Who game."""

    # Define possible traits and their values
    TRAITS = {
        "hair_color": ["blonde", "brown", "black", "red", "white"],
        "hair_style": ["bald", "long", "short", "curly"],
        "eye_color": ["blue", "brown", "green"],
        "facial_hair": ["beard", "mustache", "clean"],
        "gender": ["male", "female"],
        "accessories": ["glasses", "hat", "earrings", "none"]
    }

    def __init__(self, num_characters: int = 24):
        """Initialize the game with a specified number of characters."""
        self.num_characters = num_characters

    def get_initial_state(self) -> GuessWhoState:
        """Generate initial game state with random characters."""
        # Generate random characters
        characters = self._generate_characters()
        # Randomly select target characters for each player
        target_characters = random.sample(characters, 2)
        # Initially, all characters are possible for both players
        possible_characters = [copy.deepcopy(characters) for _ in range(2)]
        
        return GuessWhoState(
            characters=characters,
            target_characters=target_characters,
            possible_characters=possible_characters,
            current_player=0
        )

    def _generate_characters(self) -> List[Character]:
        """Generate a list of random characters with unique combinations of traits."""
        characters = []
        used_combinations = set()

        while len(characters) < self.num_characters:
            traits = {
                trait: random.choice(values)
                for trait, values in self.TRAITS.items()
            }
            # Create a hashable representation of traits
            trait_tuple = tuple(sorted(traits.items()))
            if trait_tuple not in used_combinations:
                used_combinations.add(trait_tuple)
                characters.append(Character(
                    name=f"Character_{len(characters) + 1}",
                    traits=traits
                ))

        return characters

    def get_rules_explanation(self) -> str:
        """Return a string explaining the rules of Guess Who."""
        return (
            "We are playing Guess Who! Each player has a secret character, and players take turns "
            "asking questions about the opponent's character's traits. Questions must be about specific "
            "traits (e.g., 'hair_color black' or 'NOT hair_color black'). The first player to "
            "narrow down their possibilities to just one character and correctly identify the "
            "opponent's character wins!"
        )

    def get_move_format_instructions(self) -> str:
        """Return instructions for how to format moves."""
        return (
            "'<trait> <value>' or 'NOT <trait> <value>'\n"
            "Valid traits and values:\n" + 
            "\n".join(f"- {trait}: {', '.join(values)}" 
                     for trait, values in self.TRAITS.items()) +
            "\nExamples:\n"
            "- 'hair_color black'\n"
            "- 'NOT eye_color blue'\n"
            "Remember to respond with ONLY your move in the exact format specified."
        )

    def get_player_view(self, state: GuessWhoState, player_id: int,
                       history: Optional[List[Dict[str, Any]]] = None,
                       prompt_style: PromptStyle = PromptStyle.HEADER) -> GameView:
        """Return the game state from a player's perspective."""
        
        # Structure the visible state based on the prompt style
        visible_state = {
            "all_characters": [char.to_dict() for char in state.characters],
            "possible_characters": [char.to_dict() for char in state.possible_characters[player_id]],
            "remaining_count": len(state.possible_characters[player_id]),
            "traits": self.TRAITS  # Include trait definitions for reference
        }

        # Create the GameView with the specified prompt style
        return GameView(
            rules_explanation=self.get_rules_explanation(),
            visible_state=visible_state,
            valid_moves=self._get_valid_moves(),
            is_terminal=self.is_terminal(state),
            winner=self.get_winner(state),
            history=history if history else [],
            move_format_instructions=self.get_move_format_instructions(),
            prompt_style=prompt_style
        )

    def _get_valid_moves(self) -> List[str]:
        """Return all valid moves (questions about traits)."""
        moves = []
        for trait, values in self.TRAITS.items():
            for value in values:
                moves.append(f"{trait} {value}")
                moves.append(f"NOT {trait} {value}")
        return moves

    def parse_move(self, move_str: str) -> Optional[Tuple[str, str, bool]]:
        """Parse a move string into (trait, value, is_negation)."""
        try:
            parts = move_str.strip().split()
            is_negation = False
            
            if parts[0].upper() == "NOT":
                is_negation = True
                parts = parts[1:]
                
            if len(parts) != 2:
                return None
                
            trait, value = parts
            trait = trait.lower()
            value = value.lower()
            
            if trait not in self.TRAITS or value not in self.TRAITS[trait]:
                return None
                
            return (trait, value, is_negation)
        except (IndexError, AttributeError):
            return None

    def validate_move(self, state: GuessWhoState, player_id: int,
                     move: Tuple[str, str, bool]) -> Tuple[bool, str]:
        """Validate if a move is legal."""
        if state.current_player != player_id:
            return False, "It's not your turn."
            
        if move is None:
            return False, "Invalid move format."
            
        trait, value, _ = move
        if trait not in self.TRAITS or value not in self.TRAITS[trait]:
            return False, f"Invalid trait or value: {trait} {value}"
            
        return True, ""

    def apply_move(self, state: GuessWhoState, player_id: int,
                  move: Tuple[str, str, bool]) -> GuessWhoState:
        """Apply a move to the game state."""
        state = copy.deepcopy(state)
        opponent_id = 1 - player_id
        trait, value, is_negation = move
        
        # Get opponent's character's trait value
        opponent_value = state.target_characters[opponent_id].traits[trait]
        
        # Determine if the guess matches
        is_match = (opponent_value == value) != is_negation
        
        # Update possible characters based on the answer
        state.possible_characters[player_id] = [
            char for char in state.possible_characters[player_id]
            if (char.traits[trait] == value) == is_match
        ]
        
        # Switch to other player
        state.current_player = opponent_id
        
        return state

    def get_current_player(self, state: GuessWhoState) -> int:
        """Return the ID of the current player."""
        return state.current_player

    def get_next_state(self, state: GuessWhoState, move: Any) -> GuessWhoState:
        """Apply the move and return the next game state."""
        new_state = self.apply_move(state, state.current_player, move)
        new_state.current_player = 1 - state.current_player
        return new_state

    def is_terminal(self, state: GuessWhoState) -> bool:
        """Check if the game is over."""
        return any(len(chars) == 1 for chars in state.possible_characters)

    def get_winner(self, state: GuessWhoState) -> Optional[int]:
        """Return the winner if game is over, None otherwise."""
        if not self.is_terminal(state):
            return None
        
        # Winner is the player who has narrowed down to one possibility
        for player_id, chars in enumerate(state.possible_characters):
            if len(chars) == 1:
                # Verify they found the correct character
                if chars[0].name == state.target_characters[1-player_id].name:
                    return player_id
        return None
