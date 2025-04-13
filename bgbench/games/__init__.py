from bgbench.games.poker_game import PokerGame
from .nim_game import NimGame
from .battleship_game import BattleshipGame
from .war_game import WarGame
from .guess_who_game import GuessWhoGame
from .love_letter_game import LoveLetterGame
from .chess_game import ChessGame
from .cant_stop_game import CantStopGame
from .scrabble_game import ScrabbleGame
from .azul_game import AzulGame
from .reversi_game import ReversiGame

AVAILABLE_GAMES = {
    "nim": NimGame,
    "battleship": BattleshipGame,
    "war": WarGame,
    "guess_who": GuessWhoGame,
    "love_letter": LoveLetterGame,
    "chess": ChessGame,
    "cant_stop": CantStopGame,
    "poker": PokerGame,
    "scrabble": ScrabbleGame,
    "azul": AzulGame,
    "reversi": ReversiGame,
}

__all__ = [
    "AVAILABLE_GAMES",
]
