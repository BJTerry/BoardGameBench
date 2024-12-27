import math
from dataclasses import dataclass
from typing import Tuple

@dataclass
class PlayerRating:
    name: str
    rating: float
    games_played: int
    k_factor: float = 32.0  # K-factor determines how quickly ratings change

class EloSystem:
    def __init__(self, initial_rating: float = 1500, confidence_threshold: float = 0.70):
        self.initial_rating = initial_rating
        self.confidence_threshold = confidence_threshold
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A against player B."""
        return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))
    
    def update_ratings(self, player_a: PlayerRating, player_b: PlayerRating, a_wins: bool) -> Tuple[PlayerRating, PlayerRating]:
        """Update ratings based on game result."""
        expected_a = self.expected_score(player_a.rating, player_b.rating)
        actual_a = 1.0 if a_wins else 0.0
        
        # Calculate rating changes
        change = player_a.k_factor * (actual_a - expected_a)
        
        # Create new rating objects
        new_a = PlayerRating(
            name=player_a.name,
            rating=player_a.rating + change,
            games_played=player_a.games_played + 1,
            k_factor=player_a.k_factor
        )
        
        new_b = PlayerRating(
            name=player_b.name,
            rating=player_b.rating - change,
            games_played=player_b.games_played + 1,
            k_factor=player_b.k_factor
        )
        
        return new_a, new_b
    
    def probability_stronger(self, player_a: PlayerRating, player_b: PlayerRating) -> float:
        """
        Calculate the probability that player A is actually stronger than player B,
        taking into account the uncertainty in the ratings.
        """
        rating_diff = player_a.rating - player_b.rating
        # Uncertainty decreases with more games played
        total_games = player_a.games_played + player_b.games_played
        if total_games == 0:
            # Maximum uncertainty when no games played
            return 0.5
            
        uncertainty = math.sqrt(2 * (400 ** 2) / total_games)
        
        if uncertainty < 0.0001:  # Avoid division by zero
            return 1.0 if rating_diff > 0 else 0.0
        
        # Using normal distribution to calculate probability
        return 0.5 * (1 + math.erf(rating_diff / (uncertainty * math.sqrt(2))))

    def calculate_match_uncertainty(self, player_a: PlayerRating, player_b: PlayerRating) -> float:
        """Calculate uncertainty for a match between two players.
        Returns 1.0 for equal ratings or few games, decreasing as ratings diverge and games increase."""
        prob = self.probability_stronger(player_a, player_b)
        
        # Consider number of games played
        min_games = min(player_a.games_played, player_b.games_played)
        games_factor = min(1.0, min_games / 5)  # Requires at least 5 games for full effect
        
        # Scale uncertainty based on probability difference from 0.5
        rating_uncertainty = max(0.0, 1.0 - abs(0.5 - prob) * 4)
        
        # Combine factors - high uncertainty if either few games or similar ratings
        return max(rating_uncertainty, 1.0 - games_factor)

    def is_match_needed(self, player_a: PlayerRating, player_b: PlayerRating) -> bool:
        """Determine if we need more games between these players."""
        prob = self.probability_stronger(player_a, player_b)
        return prob < self.confidence_threshold
