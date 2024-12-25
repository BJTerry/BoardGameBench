from .nim_game import NimGame
from .battleship_game import BattleshipGame
from .war_game import WarGame
from .guess_who_game import GuessWhoGame

AVAILABLE_GAMES = {
    'nim': NimGame,
    'battleship': BattleshipGame,
    'war': WarGame,
    'guess_who': GuessWhoGame
}

__all__ = ['NimGame', 'BattleshipGame', 'WarGame', 'GuessWhoGame', 'AVAILABLE_GAMES']
