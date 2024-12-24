import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from bgbench.game import Game
from bgbench.llm_player import LLMPlayer
from bgbench.game_runner import GameRunner
from bgbench.rating import PlayerRating, EloSystem

logger = logging.getLogger("bgbench")

@dataclass
class ArenaPlayer:
    llm_player: LLMPlayer
    rating: PlayerRating

class Arena:
    def __init__(self, game: Game, confidence_threshold: float = 0.70):
        self.game = game
        self.players: List[ArenaPlayer] = []
        self.elo_system = EloSystem()
        self.confidence_threshold = confidence_threshold
        
    def add_player(self, player: LLMPlayer, initial_rating: float = 1500):
        rating = PlayerRating(name=player.name, rating=initial_rating, games_played=0)
        self.players.append(ArenaPlayer(player, rating))

    def calculate_match_uncertainty(self, player_a: ArenaPlayer, player_b: ArenaPlayer) -> float:
        prob = self.elo_system.probability_stronger(player_a.rating, player_b.rating)
        # Uncertainty is highest when prob is close to 0.5
        return 1.0 - abs(prob - 0.5) * 2

    def find_best_match(self) -> Optional[Tuple[ArenaPlayer, ArenaPlayer]]:
        if len(self.players) < 2:
            return None
            
        best_uncertainty = -1
        best_pair = None
        
        for i, player_a in enumerate(self.players):
            for player_b in self.players[i+1:]:
                prob = self.elo_system.probability_stronger(
                    player_a.rating, player_b.rating)
                if prob < self.confidence_threshold:
                    uncertainty = self.calculate_match_uncertainty(player_a, player_b)
                    if uncertainty > best_uncertainty:
                        best_uncertainty = uncertainty
                        best_pair = (player_a, player_b)
        
        return best_pair

    def log_standings(self):
        """Log current ratings for all players"""
        logger.info("\nCurrent Standings:")
        sorted_players = sorted(self.players, key=lambda p: p.rating.rating, reverse=True)
        for player in sorted_players:
            logger.info(f"{player.llm_player.name}: {player.rating.rating:.0f} "
                       f"({player.rating.games_played} games)")

    async def evaluate_all(self) -> Dict[str, float]:
        while True:
            match = self.find_best_match()
            if not match:
                break  # All pairs have reached confidence threshold
                
            player_a, player_b = match
            runner = GameRunner(self.game, player_a.llm_player, player_b.llm_player)
            winner, _ = await runner.play_game()
            
            # Update ratings
            new_rating_a, new_rating_b = self.elo_system.update_ratings(
                player_a.rating, player_b.rating, 
                winner.name == player_a.llm_player.name
            )
            player_a.rating = new_rating_a
            player_b.rating = new_rating_b
            
            # Log current standings
            self.log_standings()
            
        return {p.llm_player.name: p.rating.rating for p in self.players}
