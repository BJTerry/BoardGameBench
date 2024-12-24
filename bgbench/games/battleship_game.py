from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set
from bgbench.game import Game
from bgbench.game_view import GameView
import random
import string

@dataclass
class Ship:
    name: str
    size: int
    positions: Set[Tuple[int, int]]  # Set of (x,y) coordinates
    hits: Set[Tuple[int, int]] = None  # Track hits on this ship
    
    def __post_init__(self):
        self.hits = set()
    
    @property
    def is_sunk(self) -> bool:
        return len(self.hits) == len(self.positions)

@dataclass
class Board:
    ships: List[Ship]
    hits: Set[Tuple[int, int]]  # All hits on this board
    misses: Set[Tuple[int, int]]  # All misses on this board
    
    def is_valid_shot(self, x: int, y: int) -> bool:
        return (x, y) not in self.hits and (x, y) not in self.misses
    
    def receive_shot(self, x: int, y: int) -> Tuple[str, Optional[str]]:
        for ship in self.ships:
            if (x, y) in ship.positions:
                ship.hits.add((x, y))
                self.hits.add((x, y))
                if ship.is_sunk:
                    return "hit", f"sunk {ship.name}"
                return "hit", None
        self.misses.add((x, y))
        return "miss", None

@dataclass
class BattleshipState:
    boards: List[Board]  # One per player
    current_player: int
    setup_complete: bool = False
    
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
        
    def get_rules_explanation(self) -> str:
        return (
            "We are playing Battleship on a 10x10 grid (A-J × 1-10). "
            "First, place your ships by specifying start coordinate and direction (h/v). "
            "Ships are: Carrier (5), Battleship (4), Cruiser (3), Submarine (3), Destroyer (2). "
            "During play, call shots using coordinates (e.g., 'B5'). "
            "Responses will be 'hit', 'miss', or 'hit and sunk <ship>'. "
            "First to sink all opponent's ships wins."
        )
    
    def get_move_format_instructions(self) -> str:
        if not self.setup_complete:
            return (
                "Place your ships by specifying coordinate and direction for each, "
                "e.g., 'A1 h' for horizontal at A1 or 'B2 v' for vertical at B2"
            )
        return "Call your shot using grid coordinates (e.g., 'B5')"
    
    def get_initial_state(self) -> BattleshipState:
        return BattleshipState(
            boards=[
                Board(ships=[], hits=set(), misses=set()),
                Board(ships=[], hits=set(), misses=set())
            ],
            current_player=0,
            setup_complete=False
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
    
    def get_player_view(self, state: BattleshipState, player_id: int, history: List[Dict[str, Any]] = None) -> GameView:
        opponent_id = 1 - player_id
        
        visible_state = {
            "your_board": self._format_board(state.boards[player_id], True),
            "target_board": self._format_board(state.boards[opponent_id], False),
            "setup_complete": state.setup_complete
        }
        
        if not state.setup_complete:
            ships_to_place = [ship for ship in SHIPS 
                            if not any(s.name == ship[0] for s in state.boards[player_id].ships)]
            visible_state["ships_to_place"] = ships_to_place
        
        return GameView(
            visible_state=visible_state,
            valid_moves=self._get_valid_moves(state, player_id),
            is_terminal=self._is_game_over(state),
            winner=self._get_winner(state),
            history=history if history else []
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
        move_str = move_str.strip().upper()
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
            ship_name, ship_size = SHIPS[ships_placed]
            # Check if placement is valid
            positions = self._get_ship_positions(x, y, ship_size, is_horizontal)
            if not positions:
                return False, "Ship placement out of bounds"
            # Check for overlap
            for ship in state.boards[player_id].ships:
                if positions & ship.positions:
                    return False, "Ships cannot overlap"
            return True, ""
        else:
            if len(move) != 2:
                return False, "Shot move should be coordinate only"
            x, y = move
            if not state.boards[1-player_id].is_valid_shot(x, y):
                return False, "Invalid or repeated shot location"
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
    
    def _is_game_over(self, state: BattleshipState) -> bool:
        if not state.setup_complete:
            return False
        return any(all(ship.is_sunk for ship in board.ships) for board in state.boards)
    
    def _get_winner(self, state: BattleshipState) -> Optional[int]:
        if not self._is_game_over(state):
            return None
        return next(i for i, board in enumerate(state.boards)
                   if not all(ship.is_sunk for ship in board.ships))
