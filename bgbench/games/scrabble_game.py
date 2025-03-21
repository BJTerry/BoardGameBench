import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from bgbench.game import Game
from bgbench.game_view import GameView, PromptStyle

@dataclass
class ScrabbleState:
    board: List[List[str]] = field(default_factory=lambda: [[''] * 15 for _ in range(15)])
    player_racks: List[List[str]] = field(default_factory=lambda: [[], []])
    scores: List[int] = field(default_factory=lambda: [0, 0])
    tile_bag: List[str] = field(default_factory=list)
    turn_count: int = 0
    consecutive_passes: int = 0  # New field to track consecutive passes

    def to_dict(self) -> Dict:
        return {
            "board": self.board,
            "player_racks": self.player_racks,
            "scores": self.scores,
            "tile_bag": self.tile_bag,
            "consecutive_passes": self.consecutive_passes
        }

@dataclass
class ScrabbleMove:
    word: str
    start_position: Tuple[int, int]
    direction: str  # 'horizontal', 'vertical', 'exchange', or 'pass'
    tiles_to_exchange: Optional[List[str]] = None
    blank_assignments: Dict[int, str] = field(default_factory=dict)  # Maps position in word to letter assignment

    def to_dict(self) -> Dict:
        return {
            "word": self.word,
            "start_position": self.start_position,
            "direction": self.direction,
            "tiles_to_exchange": self.tiles_to_exchange
        }

class ScrabbleGame(Game[ScrabbleState, ScrabbleMove]):
    def __init__(self):
        self.tile_distribution = {
            'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12, 'F': 2, 'G': 3, 'H': 2,
            'I': 9, 'J': 1, 'K': 1, 'L': 4, 'M': 2, 'N': 6, 'O': 8, 'P': 2,
            'Q': 1, 'R': 6, 'S': 4, 'T': 6, 'U': 4, 'V': 2, 'W': 2, 'X': 1,
            'Y': 2, 'Z': 1, ' ': 2  # Blank tiles
        }
        self.word_list = self.load_word_list()

    def load_word_list(self) -> set:
        with open('data/scrabble-dict.txt', 'r') as file:
            return set(word.strip().upper() for word in file)

    def get_initial_state(self) -> ScrabbleState:
        tile_bag = [letter for letter, count in self.tile_distribution.items() for _ in range(count)]
        random.shuffle(tile_bag)

        player_racks = [tile_bag[:7], tile_bag[7:14]]
        tile_bag = tile_bag[14:]

        return ScrabbleState(
            board=[[''] * 15 for _ in range(15)],
            player_racks=player_racks,
            scores=[0, 0],
            tile_bag=tile_bag
        )

    def parse_move(self, move_str: str) -> Optional[ScrabbleMove]:
        parts = move_str.split()
        action = parts[0].lower()

        if action == "exchange":
            tiles_to_exchange = parts[1:]
            return ScrabbleMove(word="", start_position=(0, 0), direction="exchange", tiles_to_exchange=tiles_to_exchange)

        if action == "pass":
            return ScrabbleMove(word="", start_position=(0, 0), direction="pass")

        # Handle word placement with potential blank tiles
        newly_placed_positions = set()
        raw_word = parts[0].upper()
        try:
            start_position = (int(parts[2]), int(parts[1]))
        except ValueError:
            return None
        direction = parts[3].lower()

        if direction not in ['horizontal', 'vertical']:
            return None
        
        move = ScrabbleMove(word="", start_position=(0, 0), direction=direction)
        # Process blanks marked with _
        blank_assignments = {}
        word = ""
        i = 0
        while i < len(raw_word):
            if raw_word[i] == '_':
                i += 1  # Move to the letter after _
                if i < len(raw_word):
                    word += raw_word[i]
                    blank_assignments[len(word) - 1] = raw_word[i]
                i += 1
                if move.direction == 'horizontal':
                    newly_placed_positions.add((start_position[1], start_position[0] + len(word) - 1))
                else:
                    newly_placed_positions.add((start_position[1] + len(word) - 1, start_position[0]))
            else:
                if move.direction == 'horizontal':
                    newly_placed_positions.add((start_position[1], start_position[0] + len(word) - 1))
                else:
                    newly_placed_positions.add((start_position[1] + len(word) - 1, start_position[0]))
                word += raw_word[i]
                i += 1

        return ScrabbleMove(word, start_position, direction, blank_assignments=blank_assignments)


    def has_required_letters(self, state: ScrabbleState, player_id: int, word: str, position: Tuple[int, int], direction: str, blank_assignments: Dict[int, str]) -> bool:
        # Count letters needed
        # BLANK TILE CONVENTION:
        # blank_assignments maps position -> letter, where the letter is what the blank represents
        # Example: {1: 'O'} means position 1 uses a blank tile as 'O'
        needed_letters = {}
        for i, letter in enumerate(word):
            if i in blank_assignments:  # This position uses a blank tile
                needed_letters[' '] = needed_letters.get(' ', 0) + 1
            else:
                needed_letters[letter] = needed_letters.get(letter, 0) + 1
        
        # Subtract letters that are already on the board at connection points
        x, y = position
        for i, letter in enumerate(word):
            if direction == 'horizontal':
                board_x, board_y = x, y + i
            else:
                board_x, board_y = x + i, y
                
            if state.board[board_x][board_y] == letter:
                if i not in blank_assignments:  # Only subtract if not using a blank
                    needed_letters[letter] = needed_letters.get(letter, 0) - 1
                
        # Count letters in player's rack
        rack_letters = {}
        for letter in state.player_racks[player_id]:
            rack_letters[letter] = rack_letters.get(letter, 0) + 1
            
        # Check if player has all needed letters
        for letter, count in needed_letters.items():
            if count > rack_letters.get(letter, 0):
                return False
        return True

    def _find_formed_word_position(
        self, board: List[List[str]], formed_word: str
    ) -> Tuple[int, int, str]:
         """
         Find 'formed_word' in board, return (start_x, start_y, direction).
         Called after the move has been placed on temp_board.
         """
         for x in range(15):
             for y in range(15):
                 # Check horizontal
                 if y <= 15 - len(formed_word):
                    if ''.join(board[x][y + i] for i in range(len(formed_word))) == formed_word:
                        # Ensure bounding for horizontal
                        if y > 0 and board[x][y - 1] != '':
                            continue
                        if (y + len(formed_word)) < 15 and board[x][y + len(formed_word)] != '':
                            continue
                        return x, y, 'horizontal'

                 # Check vertical
                 if x <= 15 - len(formed_word):
                    if ''.join(board[x + i][y] for i in range(len(formed_word))) == formed_word:
                        # Ensure bounding for vertical
                        if x > 0 and board[x - 1][y] != '':
                            continue
                        if (x + len(formed_word)) < 15 and board[x + len(formed_word)][y] != '':
                            continue
                        return x, y, 'vertical'

         # Should never happen if the word is truly on the board
         raise ValueError(f"Couldn't locate formed word '{formed_word}' on the board.")

    def is_connected(self, state: ScrabbleState, move: ScrabbleMove) -> bool:
        # First move must be placed at center (7,7)
        if all(all(cell == '' for cell in row) for row in state.board):
            return move.start_position == (7, 7)
            
        x, y = move.start_position
        word = move.word
        direction = move.direction
        
        # Check if word connects to existing letters
        has_connection = False
        for i, letter in enumerate(word):
            if direction == 'horizontal':
                curr_x, curr_y = x, y + i
            else:
                curr_x, curr_y = x + i, y
                
            # Check the current position
            if state.board[curr_x][curr_y] != '':
                has_connection = True
                
            # Check adjacent positions
            adjacents = [
                (curr_x-1, curr_y), (curr_x+1, curr_y),
                (curr_x, curr_y-1), (curr_x, curr_y+1)
            ]
            
            for adj_x, adj_y in adjacents:
                if (0 <= adj_x < 15 and 0 <= adj_y < 15 and 
                    state.board[adj_x][adj_y] != '' and 
                    (adj_x, adj_y) != (x, y)):
                    has_connection = True
                    
        return has_connection

    def validate_move(self, state: ScrabbleState, player_id: int, move: ScrabbleMove) -> Tuple[bool, str]:
        old_words_info = self._get_all_words_on_board(state.board)
        old_words = set(info[3] for info in old_words_info)

        if move.direction == "exchange":
            if move.tiles_to_exchange is None:
                return False, "No tiles specified for exchange"

            # Count the tiles in player's rack
            rack_counts = {}
            for tile in state.player_racks[player_id]:
                rack_counts[tile] = rack_counts.get(tile, 0) + 1

            # Count the tiles to exchange
            exchange_counts = {}
            for tile in move.tiles_to_exchange:
                exchange_counts[tile] = exchange_counts.get(tile, 0) + 1

            # Verify player has enough of each tile
            for tile, count in exchange_counts.items():
                if count > rack_counts.get(tile, 0):
                    return False, f"Not enough '{tile}' tiles in rack"

            return True, "Exchange is valid"

        if move.direction == "pass":
            return True, "Pass is valid"

        # Validate word placement
        x, y = move.start_position
        if move.direction == 'horizontal':
            if y + len(move.word) > 15:
                return False, "Word does not fit horizontally on the board"
        else:
            if x + len(move.word) > 15:
                return False, "Word does not fit vertically on the board"

        # Check if player has required letters
        if not self.has_required_letters(state, player_id, move.word, move.start_position, move.direction, move.blank_assignments):
            return False, "Player does not have required letters"

        # Check if the word is connected to existing words
        if not self.is_connected(state, move):
            return False, "Word must connect to existing words"

        # Save all words on the board before applying the move
        old_words_info = self._get_all_words_on_board(state.board)
        old_words = set(info[3] for info in old_words_info)

        # Temporarily place the new tiles on a copy of the board to see new words
        temp_state = ScrabbleState(
            board=[row[:] for row in state.board],
            player_racks=state.player_racks,
            scores=state.scores,
            tile_bag=state.tile_bag,
            turn_count=state.turn_count
        )

        # Place the word on the temporary board
        x, y = move.start_position
        for i, letter in enumerate(move.word):
            if move.direction == 'horizontal':
                temp_state.board[x][y + i] = letter
            else:
                temp_state.board[x + i][y] = letter

        # Get all words on the board after the move
        new_words_info = self._get_all_words_on_board(temp_state.board)
        new_words = set(info[3] for info in new_words_info)

        # Only words in new_words - old_words are newly formed or modified
        newly_formed_words = new_words - old_words

        # Verify each newly formed word against the dictionary
        for word in newly_formed_words:
            if word not in self.word_list:
                return False, f"Invalid word formed: {word}"

        return True, "Move is valid"

    def _is_connected_to_existing(self, state: ScrabbleState, move: ScrabbleMove) -> bool:
        """Check if the word being placed connects to or modifies existing words on the board."""
        x, y = move.start_position
        
        # Check if any letter in the word is placed on an existing letter
        for i in range(len(move.word)):
            if move.direction == 'horizontal':
                curr_x, curr_y = x, y + i
            else:
                curr_x, curr_y = x + i, y
                
            # If this position already has a letter, the word modifies an existing word
            if state.board[curr_x][curr_y] != '':
                return True
                
            # Check adjacent positions (excluding the word being placed)
            adjacents = [
                (curr_x-1, curr_y), (curr_x+1, curr_y),
                (curr_x, curr_y-1), (curr_x, curr_y+1)
            ]
            
            for adj_x, adj_y in adjacents:
                if (0 <= adj_x < 15 and 0 <= adj_y < 15 and 
                    state.board[adj_x][adj_y] != ''):
                    # Don't count adjacent positions that are part of the word being placed
                    is_part_of_word = False
                    for j in range(len(move.word)):
                        if move.direction == 'horizontal' and adj_x == x and adj_y == y + j:
                            is_part_of_word = True
                            break
                        elif move.direction == 'vertical' and adj_x == x + j and adj_y == y:
                            is_part_of_word = True
                            break
                    
                    if not is_part_of_word:
                        return True
                        
        return False

    def get_letter_score(self, letter: str) -> int:
        if letter.islower():
            return 0

        letter_scores = {
            'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4,
            'I': 1, 'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3,
            'Q': 10, 'R': 1, 'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8,
            'Y': 4, 'Z': 10, ' ': 0  # Blank tiles
        }
        return letter_scores.get(letter, 0)

    def _get_word_start(self, board: List[List[str]], start: Tuple[int, int], direction: str) -> Tuple[int, int]:
        """Return the coordinates of the first letter in the contiguous word that includes `start`."""
        x, y = start
        if direction == 'horizontal':
            while y > 0 and board[x][y - 1] != '':
                y -= 1
        else:  # vertical
            while x > 0 and board[x - 1][y] != '':
                x -= 1
        return x, y

    def apply_move(self, state: ScrabbleState, player_id: int, move: ScrabbleMove) -> ScrabbleState:
        if move.direction == "exchange":
            self.exchange_tiles(state, player_id, move.tiles_to_exchange)
            state.consecutive_passes = 0
        elif move.direction == "pass":
            self.pass_turn(state)
        else:
            old_words_info = self._get_all_words_on_board(state.board)
            old_words = set(info[3] for info in old_words_info)
            x, y = move.start_position
            total_score = 0
            state.consecutive_passes = 0
            
            # Create a temporary board for scoring
            temp_board = [row[:] for row in state.board]
            
            # Place the word on the temporary board
            placed_coords = set()
            for i, letter in enumerate(move.word):
                curr_x = x if move.direction == 'horizontal' else x + i
                curr_y = y + i if move.direction == 'horizontal' else y
                
                # Only add to placed_coords if the square was empty before
                if state.board[curr_x][curr_y] == '':
                    placed_coords.add((curr_x, curr_y))
                
                # Always update the temp board
                if i in move.blank_assignments:
                    board_letter = letter.lower()  # Mark blank as lowercase
                else:
                    board_letter = letter
                temp_board[curr_x][curr_y] = board_letter
                
            # Track how many tiles were placed in this move
            tiles_placed = len(placed_coords)

            # Gather all runs from the temp board
            newly_detected = self._get_all_words_on_board(temp_board)
            formed_words = []
            for (r, c, dirn, w_str) in newly_detected:
                # Build coordinate set for this contiguous run
                coords = []
                if dirn == 'horizontal':
                    coords = [(r, c + i) for i in range(len(w_str))]
                else:
                    coords = [(r + i, c) for i in range(len(w_str))]

                # Only score if the run includes at least one newly placed tile and is not an old word
                if (set(coords) & placed_coords) and (w_str not in old_words):
                    formed_words.append(w_str)

            for word in formed_words:
                word_score = 0
                word_multiplier = 1

                # Find the actual starting square & direction of this specific word
                start_x, start_y, direction = self._find_formed_word_position(temp_board, word)

                curr_x, curr_y = start_x, start_y
                while 0 <= curr_x < 15 and 0 <= curr_y < 15 and temp_board[curr_x][curr_y] != '':
                    letter = temp_board[curr_x][curr_y]
                    # Check if this tile is newly placed
                    is_new_tile = state.board[curr_x][curr_y] == ''

                    letter_score = self.get_letter_score(letter)
                    if is_new_tile:
                        letter_multiplier = self.get_tile_multiplier(curr_x, curr_y)
                        letter_score *= letter_multiplier
                        if self.is_double_word(curr_x, curr_y):
                            word_multiplier *= 2
                        elif self.is_triple_word(curr_x, curr_y):
                            word_multiplier *= 3

                    word_score += letter_score

                    if direction == 'horizontal':
                        curr_y += 1
                    else:
                        curr_x += 1

                word_score *= word_multiplier
                total_score += word_score
            
            # Add 50-point bonus if all 7 tiles were used
            if tiles_placed == 7:
                total_score += 50

            # Update the actual board
            for i, letter in enumerate(move.word):
                if i in move.blank_assignments:
                    board_letter = letter.lower()
                else:
                    board_letter = letter
                if move.direction == 'horizontal':
                    state.board[x][y + i] = board_letter
                else:
                    state.board[x + i][y] = board_letter

            # Update player's score
            state.scores[player_id] += total_score

            # Update player's rack
            self.update_player_rack(state, player_id, move.word, move.blank_assignments)

        # Increment turn count
        state.turn_count += 1
        return state

    def get_current_player(self, state: ScrabbleState) -> int:
        return state.turn_count % 2

    def is_terminal(self, state: ScrabbleState) -> bool:
        # Check if the game has ended
        if not state.tile_bag and any(len(rack) == 0 for rack in state.player_racks):
            return True
        # End the game if both players pass consecutively
        if state.consecutive_passes >= 2:
            return True
        return False

    def get_winner(self, state: ScrabbleState) -> Optional[int]:
        if not self.is_terminal(state):
            return None

        # Calculate final scores, deducting remaining tiles
        final_scores = state.scores[:]
        for player_id, rack in enumerate(state.player_racks):
            final_scores[player_id] -= sum(self.get_letter_score(tile) for tile in rack)

        # Determine the winner
        if final_scores[0] > final_scores[1]:
            return 0
        elif final_scores[1] > final_scores[0]:
            return 1
        return None  # Tie

    def exchange_tiles(self, state: ScrabbleState, player_id: int, tiles_to_exchange: Optional[List[str]]) -> None:
        if tiles_to_exchange is None:
            return

        # Ensure the player can exchange tiles correctly
        for tile in tiles_to_exchange:
            if tile in state.player_racks[player_id]:
                state.player_racks[player_id].remove(tile)
        
        while len(state.player_racks[player_id]) < 7 and state.tile_bag:
            state.player_racks[player_id].append(state.tile_bag.pop())

        state.tile_bag.extend(tiles_to_exchange)
        random.shuffle(state.tile_bag)

    def pass_turn(self, state: ScrabbleState) -> None:
        # Update consecutive passes counter
        state.consecutive_passes += 1

    def get_tile_multiplier(self, x: int, y: int) -> int:
        # Double letter scores
        if (x, y) in {(3, 0), (11, 0), (6, 2), (8, 2), (0, 3), (7, 3), (14, 3),
                      (2, 6), (6, 6), (8, 6), (12, 6), (3, 7), (11, 7),
                      (2, 8), (6, 8), (8, 8), (12, 8), (0, 11), (7, 11),
                      (14, 11), (6, 12), (8, 12), (3, 14), (11, 14)}:
            return 2
        # Triple letter scores    
        if (x, y) in {(5, 1), (9, 1), (1, 5), (5, 5), (9, 5), (13, 5),
                      (1, 9), (5, 9), (9, 9), (13, 9), (5, 13), (9, 13)}:
            return 3
        return 1

    def is_double_word(self, x: int, y: int) -> bool:
        return (x, y) in {(1, 1), (2, 2), (3, 3), (4, 4), (13, 1), (12, 2), (11, 3), (10, 4),
                          (1, 13), (2, 12), (3, 11), (4, 10), (13, 13), (12, 12), (11, 11), (10, 10)}

    def is_triple_word(self, x: int, y: int) -> bool:
        return (x, y) in {(0, 0), (0, 7), (0, 14), (7, 0), (7, 14), (14, 0), (14, 7), (14, 14)}

    def get_next_state(self, state: ScrabbleState, move: ScrabbleMove) -> ScrabbleState:
        # Apply the move to the current state and return the new state
        return self.apply_move(state, self.get_current_player(state), move)

    def _find_word_position(self, board: List[List[str]], word: str) -> Tuple[int, int]:
        """Find the starting position of a word on the board."""
        for x in range(15):
            for y in range(15):
                # Check horizontal
                if y <= 15 - len(word):
                    if all(board[x][y + i] == word[i] for i in range(len(word))):
                        return x, y
                # Check vertical
                if x <= 15 - len(word):
                    if all(board[x + i][y] == word[i] for i in range(len(word))):
                        return x, y
        return 0, 0  # Should never happen if word exists on board

    def _find_word_direction(self, board: List[List[str]], word: str, x: int, y: int) -> str:
        """Determine if a word at position (x,y) is horizontal or vertical."""
        if y <= 15 - len(word) and all(board[x][y + i] == word[i] for i in range(len(word))):
            return 'horizontal'
        return 'vertical'

    def update_player_rack(self, state: ScrabbleState, player_id: int, word: str, blank_assignments: Dict[int, str]) -> None:
        # Remove used letters from player's rack
        for i, letter in enumerate(word):
            if i in blank_assignments:
                # Remove a blank tile
                state.player_racks[player_id].remove(' ')
            elif letter in state.player_racks[player_id]:
                state.player_racks[player_id].remove(letter)
            
        # Draw new tiles to fill rack back to 7
        while len(state.player_racks[player_id]) < 7 and state.tile_bag:
            state.player_racks[player_id].append(state.tile_bag.pop())

    def get_rules_explanation(self) -> str:
        return (
            "Scrabble Rules:\n"
            "1. Game Play:\n"
            "   - Players take turns forming words on the board using their letter tiles\n"
            "   - First word must be placed through the center square (7,7)\n"
            "   - All subsequent words must connect to existing words\n"
            "   - Words can be placed horizontally (left to right) or vertically (top to bottom)\n\n"
            "2. Scoring:\n"
            "   - You score ONLY for words formed or modified in your turn\n"
            "   - For each word you form or modify, count ALL letters in that word\n"
            "   - Existing words on the board that aren't modified don't score again\n"
            "   - Letter multipliers (DLS, TLS) apply only to newly placed tiles\n"
            "   - Word multipliers (DWS, TWS) apply to the entire word score\n"
            "   - If you form multiple words in one turn, you score for each word\n"
            "   - Using all 7 tiles in one turn earns a 50-point bonus (bingo)\n\n"
            "3. Game End:\n"
            "   - Game ends when:\n"
            "     * All tiles have been drawn and one player uses all their tiles\n"
            "     * Both players pass twice in succession\n"
            "   - Final scores are calculated by:\n"
            "     * Subtracting the value of remaining tiles from each player's score\n"
            "     * Player with highest final score wins"
        )

    def get_player_view(self, state: ScrabbleState, player_id: int, history: Optional[List[Dict[str, Any]]] = None, prompt_style: PromptStyle = PromptStyle.HEADER) -> GameView:
        tile_scores = {tile: self.get_letter_score(tile) for tile in state.player_racks[player_id]}
        
        visible_state = {
            "board": state.board,
            "your_tiles": state.player_racks[player_id],
            "your_tile_scores": tile_scores,
            "scores": state.scores,
            "consecutive_passes": state.consecutive_passes,
            "opponent_tiles_count": len(state.player_racks[1 - player_id])
        }
        
        move_format_instructions = (
            "To make a move, you can:\n"
            "1. Place a word: specify the word, starting position (row, column), and direction (horizontal or vertical).\n"
            "   Note: Board positions are 0-indexed, with (0,0) at the top-left corner and (14,14) at the bottom-right.\n"
            "   The center square is at position (7,7).\n"
            "   For blank tiles, use _X where X is the letter you want the blank to represent. Example:\n"
            "   - Regular word: 'WORD 7 7 horizontal'\n"
            "   - With blank tile: 'W_ORD 7 7 horizontal' (blank used as O)\n"
            "2. Exchange tiles: specify 'exchange' followed by the tiles you want to exchange. Example: 'exchange A E I'\n"
            "3. Pass your turn: simply specify 'pass'"
        )

        return GameView(
            visible_state=visible_state,
            valid_moves=[],
            is_terminal=self.is_terminal(state),
            winner=self.get_winner(state),
            history=history or [],
            move_format_instructions=move_format_instructions,
            rules_explanation=self.get_rules_explanation(),
            prompt_style=prompt_style
        )
    def _get_all_words_on_board(self, board: List[List[str]]) -> List[Tuple[int, int, str, str]]:
        words = []
        size = len(board)

        # Horizontal words
        for row in range(size):
            col = 0
            while col < size:
                if board[row][col] != '':
                    start_col = col
                    word_letters = board[row][col]
                    col += 1
                    while col < size and board[row][col] != '':
                        word_letters += board[row][col]
                        col += 1
                    run_length = len(word_letters)
                    if run_length >= 2:
                        words.append((row, start_col, 'horizontal', word_letters))
                else:
                    col += 1

        # Vertical words
        for col in range(size):
            row = 0
            while row < size:
                if board[row][col] != '':
                    start_row = row
                    word_letters = board[row][col]
                    row += 1
                    while row < size and board[row][col] != '':
                        word_letters += board[row][col]
                        row += 1
                    run_length = len(word_letters)
                    if run_length >= 2:
                        words.append((start_row, col, 'vertical', word_letters))
                else:
                    row += 1
        return words
