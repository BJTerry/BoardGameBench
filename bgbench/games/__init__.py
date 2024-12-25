from .nim_game import NimGame
from .battleship_game import BattleshipGame
from .war_game import WarGame

AVAILABLE_GAMES = {
    'nim': NimGame,
    'battleship': BattleshipGame,
    'war': WarGame
}

__all__ = ['NimGame', 'BattleshipGame', 'WarGame', 'AVAILABLE_GAMES']
