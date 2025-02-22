from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from bgbench.game import Game
from bgbench.game_view import GameView, PromptStyle
import string
import copy

@dataclass
class Ship:
    name: str
    size: int
    positions: Set[Tuple[int, int]]  # Set of (x,y) coordinates
    hits: Set[Tuple[int, int]] = field(default_factory=set)  # Track hits on this ship
    sunk_reported: bool = False
    
    def __post_init__(self):
        self.hits = set()
    
    def to_dict(self) -> dict:
        """Convert ship to JSON-serializable dictionary."""
        return {
            "name": self.name,
            "size": self.size,
            "positions": list(self.positions),
            "hits": list(self.hits),
            "sunk_reported": self.sunk_reported
        }
    
    @property
    def is_sunk(self) -> bool:
        return len(self.hits) == len(self.positions)

@dataclass
class Board:
    ships: List[Ship]
    hits: Set[Tuple[int, int]]  # All hits on this board
    misses: Set[Tuple[int, int]]  # All misses on this board
    
    def to_dict(self) -> dict:
        """Convert board to JSON-serializable dictionary."""
        return {
            "ships": [ship.to_dict() for ship in self.ships],
            "hits": list(self.hits),
            "misses": list(self.misses)
        }
    
    def is_valid_shot(self, x: int, y: int) -> bool:
        # Can't shoot same spot twice (whether hit or miss)
        return 0 <= x < 10 and 0 <= y < 10 and (x, y) not in self.hits and (x, y) not in self.misses
    
    def receive_shot(self, x: int, y: int) -> Tuple[str, Optional[str]]:
        if not self.is_valid_shot(x, y):
            raise ValueError("Invalid shot location")
            
        for ship in self.ships:
            if (x, y) in ship.positions:
                ship.hits.add((x, y))
                self.hits.add((x, y))
                # Only report sunk if this hit was the final one needed
                if ship.is_sunk and not ship.sunk_reported:
                    ship.sunk_reported = True
                    return "hit", f"sunk {ship.name}"
                return "hit", None
        self.misses.add((x, y))
        return "miss", None

@dataclass
class BattleshipState:
    boards: List[Board]  # One per player
    current_player: int
    setup_complete: bool = False
    
    def to_dict(self) -> dict:
        """Convert state to JSON-serializable dictionary."""
        return {
            "boards": [board.to_dict() for board in self.boards],
            "current_player": self.current_player,
            "setup_complete": self.setup_complete
        }
    
SHIPS = [
    ("Carrier", 5),
    ("Battleship", 4),
    ("Cruiser", 3),
    ("Submarine", 3),
    ("Destroyer", 2)
]

class BattleshipGame(Game):
    def __init__(self):
        self.size = 10  # 10x10 grid
        
    def get_initial_state(self) -> BattleshipState:
        """Return the initial state of the Battleship game."""
        return BattleshipState(
            boards=[
                Board(ships=[], hits=set(), misses=set()),
                Board(ships=[], hits=set(), misses=set())
            ],
            current_player=0,
            setup_complete=False
        )
    
    def get_move_format_instructions_setup(self) -> str:
        return (
            "[letter][number] [direction]\n"
            "- letter must be A-J (column)\n"
            "- number must be 1-10 (row)\n"
            "- direction must be h (horizontal) or v (vertical)\n"
            "- do not add any explanatory text before or after your move, just the move\n"
            "Examples:\n"
            "- 'A1 h' places horizontally starting at A1\n"
            "- 'B2 v' places vertically starting at B2"
        )
    
    def get_move_format_instructions_gameplay(self) -> str:
        return (
            "[letter][number] during gameplay\n"
            "- letter must be A-J (column)\n"
            "- number must be 1-10 (row)\n"
            "- do not add any explanatory text before or after your move, just the move\n"
            "Examples: 'B5' or 'H10'"
        )
    
    def _format_board(self, board: Board, show_ships: bool) -> str:
        grid = [[' ' for _ in range(self.size)] for _ in range(self.size)]
        
        # Mark ships if visible
        if show_ships:
            for ship in board.ships:
                for x, y in ship.positions:
                    grid[y][x] = 'S'
        
        # Mark hits and misses
        for x, y in board.hits:
            grid[y][x] = 'H'
        for x, y in board.misses:
            grid[y][x] = 'M'
            
        # Format grid
        result = "  A B C D E F G H I J\n"
        for i, row in enumerate(grid):
            result += f"{i+1:2d} {' '.join(row)}\n"
        return result
    
    def get_player_view(self, state: BattleshipState, player_id: int, 
                       history: Optional[List[Dict[str, Any]]] = None,
                       prompt_style: PromptStyle = PromptStyle.HEADER) -> GameView:
        opponent_id = 1 - player_id
        
        # Format shot history
        shot_history = []
        if history:
            for turn in history:
                if turn["player"] == player_id:
                    move = turn["move"]
                    if state.setup_complete:
                        if len(move) == 2:  # Attack move
                            x, y = move
                            coord = f"{string.ascii_uppercase[x]}{y+1}"
                            # Check the result of this specific shot
                            if (x, y) in state.boards[opponent_id].hits:
                                # Find the ship that was hit at this coordinate
                                hit_ship = next((ship for ship in state.boards[opponent_id].ships 
                                               if (x, y) in ship.positions), None)
                                if hit_ship and hit_ship.is_sunk and len(hit_ship.hits) == len(hit_ship.positions):
                                    result = f"hit and sunk {hit_ship.name}"
                                else:
                                    result = "hit"
                            else:
                                result = "miss"
                            shot_history.append(f"Turn {turn['turn']}: {coord} - {result}")
                    else:  # Setup move
                        x, y, is_horizontal = move
                        direction = "horizontally" if is_horizontal else "vertically"
                        coord = f"{string.ascii_uppercase[x]}{y+1}"
                        ships_placed = len(state.boards[player_id].ships)
                        if ships_placed < len(SHIPS):
                            ship_name = SHIPS[ships_placed][0]
                            shot_history.append(f"Turn {turn['turn']}: Placed {ship_name} {direction} at {coord}")
        
        visible_state = {
            "your_board": self._format_board(state.boards[player_id], True),
            "target_board": self._format_board(state.boards[opponent_id], False),
            "setup_complete": state.setup_complete,
            "shot_history": shot_history
        }
        
        if not state.setup_complete:
            ships_placed = len(state.boards[player_id].ships)
            if ships_placed < len(SHIPS):
                next_ship = SHIPS[ships_placed]
                visible_state["next_ship_to_place"] = next_ship
                visible_state["remaining_ships"] = SHIPS[ships_placed + 1:]
        
        move_instructions = (
            self.get_move_format_instructions_setup()
            if not state.setup_complete
            else self.get_move_format_instructions_gameplay()
        )
        
        rules_explanation = (
            "We are playing Battleship on a 10x10 grid (A-J × 1-10). "
            "First, place your ships by specifying start coordinate and direction (h/v). "
            "Ships must be placed in this exact order: "
            "Carrier (length 5), Battleship (length 4), Cruiser (length 3), Submarine (length 3), Destroyer (length 2). "
            "During play, call shots using coordinates (e.g., 'B5'). "
            "On your boards:\n"
            "- 'S' marks your ship locations\n"
            "- 'H' marks hits (both on your ships and hits you've made)\n"
            "- 'M' marks misses (both on your board and your missed shots)\n"
            "You will see two boards:\n"
            "1. Your board: Shows your ships and all enemy shots\n"
            "2. Target board: Shows your hits and misses, but not enemy ship locations\n"
            "The shot history shows all moves made and their results.\n"
            "First to sink all opponent's ships wins."
        )

        return GameView(
            rules_explanation=rules_explanation,
            visible_state=visible_state,
            valid_moves=self._get_valid_moves(state, player_id),
            is_terminal=self.is_terminal(state),
            winner=self.get_winner(state),
            history=history if history else [],
            move_format_instructions=move_instructions,
            prompt_style=prompt_style
        )
    
    def _get_valid_moves(self, state: BattleshipState, player_id: int) -> List[str]:
        if not state.setup_complete:
            # During setup, return all possible placements
            return [f"{col}{row} {dir}" 
                   for col in string.ascii_uppercase[:self.size]
                   for row in range(1, self.size + 1)
                   for dir in ['h', 'v']]
        
        # During play, return all unshot coordinates
        return [f"{col}{row}" 
                for col in string.ascii_uppercase[:self.size]
                for row in range(1, self.size + 1)
                if state.boards[1-player_id].is_valid_shot(
                    string.ascii_uppercase.index(col), row-1)]
    
    def parse_move(self, move_str: str) -> Optional[Tuple[Any, ...]]:
        try:
            move_str = str(move_str).strip().upper()
            if not move_str:
                return None
                
            if ' ' in move_str:  # Setup move
                coord, direction = move_str.split()
                if (len(coord) < 2 or coord[0] not in string.ascii_uppercase[:self.size] or
                    not coord[1:].isdigit() or direction not in ['H', 'V']):
                    return None
                x = string.ascii_uppercase.index(coord[0])
                y = int(coord[1:]) - 1
                return (x, y, direction == 'H')
            else:  # Shot move
                if (len(move_str) < 2 or move_str[0] not in string.ascii_uppercase[:self.size] or
                    not move_str[1:].isdigit()):
                    return None
                x = string.ascii_uppercase.index(move_str[0])
                y = int(move_str[1:]) - 1
                return (x, y)
        except (ValueError, AttributeError, TypeError):
            return None
    
    def validate_move(self, state: BattleshipState, player_id: int, move: Tuple[Any, ...]) -> Tuple[bool, str]:
        if state.current_player != player_id:
            return False, "It's not your turn."
            
        if not state.setup_complete:
            if len(move) != 3:
                return False, "Setup move should be coordinate and direction"
            x, y, is_horizontal = move
            # Get next ship to place
            ships_placed = len(state.boards[player_id].ships)
            if ships_placed >= len(SHIPS):
                return False, "All ships already placed"

            # Check if placement is valid
            positions = self._get_ship_positions(x, y, SHIPS[ships_placed][1], is_horizontal)
            if not positions:
                return False, "Ship placement out of bounds"

            # Get the next ship that should be placed
            ships_placed = len(state.boards[player_id].ships)
            if ships_placed >= len(SHIPS):
                return False, "All ships have already been placed"
                
            next_ship_name, next_ship_size = SHIPS[ships_placed]
            if len(positions) != next_ship_size:
                return False, f"Must place {next_ship_name} (size {next_ship_size}) next"
            # Check for overlap
            for ship in state.boards[player_id].ships:
                if positions & ship.positions:
                    return False, "Ships cannot overlap"
            return True, ""
        else:
            if len(move) != 2:
                return False, "Shot move should be coordinate only"
            x, y = move
            if not (0 <= x < self.size and 0 <= y < self.size):
                return False, "Shot location out of bounds"
            if not state.boards[1-player_id].is_valid_shot(x, y):
                return False, "Cannot shoot the same location twice"
            return True, ""
    
    def _get_ship_positions(self, x: int, y: int, size: int, is_horizontal: bool) -> Set[Tuple[int, int]]:
        positions = set()
        for i in range(size):
            new_x = x + (i if is_horizontal else 0)
            new_y = y + (0 if is_horizontal else i)
            if new_x >= self.size or new_y >= self.size:
                return set()
            positions.add((new_x, new_y))
        return positions
    
    def apply_move(self, state: BattleshipState, player_id: int, move: Tuple[Any, ...]) -> BattleshipState:
        state = copy.deepcopy(state)
        if not state.setup_complete:
            x, y, is_horizontal = move
            ships_placed = len(state.boards[player_id].ships)
            ship_name, ship_size = SHIPS[ships_placed]
            positions = self._get_ship_positions(x, y, ship_size, is_horizontal)
            state.boards[player_id].ships.append(Ship(ship_name, ship_size, positions))
            
            # Check if setup is complete
            if all(len(board.ships) == len(SHIPS) for board in state.boards):
                state.setup_complete = True
        else:
            x, y = move
            result, sunk = state.boards[1-player_id].receive_shot(x, y)
            
        return state
    
    def get_current_player(self, state: BattleshipState) -> int:
        return state.current_player
    
    def get_next_state(self, state: BattleshipState, move: Any) -> BattleshipState:
        new_state = self.apply_move(state, state.current_player, move)
        if state.setup_complete:  # Only change turns during actual play
            new_state.current_player = 1 - state.current_player
        elif len(new_state.boards[state.current_player].ships) == len(SHIPS):
            new_state.current_player = 1 - state.current_player
        return new_state
    
    def is_terminal(self, state: BattleshipState) -> bool:
        if not state.setup_complete:
            return False
        return any(all(ship.is_sunk for ship in board.ships) for board in state.boards)
    
    def get_winner(self, state: BattleshipState) -> Optional[int]:
        if not self.is_terminal(state):
            return None
        # Return the player who sunk all of their opponent's ships
        # Player 0 wins if player 1's ships are all sunk, and vice versa
        return 1 if all(ship.is_sunk for ship in state.boards[0].ships) else 0
