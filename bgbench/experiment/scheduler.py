"""
Module implementing different match scheduling strategies for BGBench arena.

This module provides a common interface for scheduling matches between players,
with different implementations optimizing for different goals:
1. Top Identification Strategy: focuses on identifying the single best model with high confidence
2. Full Ranking Uncertainty Strategy: focuses on reducing overall pairwise uncertainty among all models
3. Sigma Minimization Strategy: focuses on reducing uncertainty (sigma) for players with highest uncertainty
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, cast, Any, TYPE_CHECKING
import numpy as np
import logging
from bgbench.experiment.rating import EloSystem, GameResult

# Handle circular imports
if TYPE_CHECKING:
    from bgbench.experiment.arena import ArenaPlayer

logger = logging.getLogger(__name__)


@dataclass
class MatchFilterSpec:
    """Filter specification for match scheduling."""

    # Experiment design filters
    selected_player_names: Optional[List[str]] = None
    ignored_player_names: Optional[List[str]] = None

    # Match dynamics filters
    max_games_per_pairing: int = 10

    # Statistical filters
    confidence_threshold: float = 0.70

    # Concurrency filters
    max_concurrent_games_per_pair: int = 1


class MatchScheduler:
    """Base class for match scheduling strategies."""

    def find_matches(
        self,
        players: List["ArenaPlayer"],
        match_history: List[GameResult],
        elo_system: EloSystem,
        ongoing_matches: Optional[Dict[Tuple[int, int], int]] = None,
        filter_spec: Optional[MatchFilterSpec] = None,
        limit: int = 5,
    ) -> List[Tuple["ArenaPlayer", "ArenaPlayer"]]:
        """
        Return a list of (playerA, playerB) tuples for potential matches,
        ordered by relevance according to the scheduling strategy.

        Args:
            players: List of all players in the arena
            match_history: List of all previous game results
            elo_system: The EloSystem used for computing probabilities
            ongoing_matches: Set of (player_id1, player_id2) tuples for matches currently in progress
            filter_spec: Filter specification for match constraints
            limit: Maximum number of matches to return

        Returns:
            A list of (playerA, playerB) tuples for potential matches, ordered by relevance
        """
        # Use default filter spec if none provided
        if filter_spec is None:
            filter_spec = MatchFilterSpec()

        raise NotImplementedError()

    def _get_candidate_pairs(
        self,
        players: List["ArenaPlayer"],
        ongoing_matches: Dict[Tuple[int, int], int],
        filter_spec: MatchFilterSpec,
        elo_system: EloSystem,
        match_history: Optional[List[GameResult]] = None,
    ) -> List[Tuple["ArenaPlayer", "ArenaPlayer", float]]:
        """
        Generate all valid candidate pairs of players that can play a match.

        Args:
            players: List of all players in the arena
            ongoing_matches: Set of (player_id1, player_id2) tuples for matches currently in progress
            filter_spec: Filter specification with match constraints
            elo_system: The EloSystem used for computing probabilities
            match_history: Optional list of game results to consider

        Returns:
            List of (playerA, playerB, relevance_score) tuples for valid candidate pairs,
            where relevance_score is a strategy-specific score for how important this match is
        """
        candidates = []

        # Track number of games between each pair
        games_per_pair: Dict[Tuple[int, int], int] = {}

        # Count existing games between each pair of players
        for player_a in players:
            for player_b in players:
                if player_a.player_model.id == player_b.player_model.id:
                    continue  # Skip same player

                # Create a canonical ordering of player IDs for the pair
                player_ids_tuple = self._get_canonical_pair(
                    player_a.player_model.id, player_b.player_model.id
                )
                games_per_pair[player_ids_tuple] = 0

        # Count games from match history passed to this function
        if match_history:
            # Get mapping of player name to ID
            name_to_id = {
                player.llm_player.name: player.player_model.id for player in players
            }

            for game in match_history:
                # Convert player names to IDs if possible
                player0_id = name_to_id.get(game.player_0)
                player1_id = name_to_id.get(game.player_1)

                # Skip if we can't find both players (shouldn't happen normally)
                if player0_id is None or player1_id is None:
                    continue

                player_ids_tuple = self._get_canonical_pair(player0_id, player1_id)
                games_per_pair[player_ids_tuple] = (
                    games_per_pair.get(player_ids_tuple, 0) + 1
                )

        # Check each pair of players
        for player_a in players:
            for player_b in players:
                if player_a.player_model.id == player_b.player_model.id:
                    continue  # Skip same player

                # Skip if we have selected player filter and neither player is selected
                if filter_spec.selected_player_names:
                    if (
                        player_a.llm_player.name
                        not in filter_spec.selected_player_names
                        and player_b.llm_player.name
                        not in filter_spec.selected_player_names
                    ):
                        continue
                        
                # Skip if either player is in the ignored players list
                if filter_spec.ignored_player_names:
                    if (
                        player_a.llm_player.name
                        in filter_spec.ignored_player_names
                        or player_b.llm_player.name
                        in filter_spec.ignored_player_names
                    ):
                        continue

                # Create a canonical ordering of player IDs for the pair
                player_ids_tuple = self._get_canonical_pair(
                    player_a.player_model.id, player_b.player_model.id
                )

                # Skip if the number of ongoing games for this pair meets or exceeds the limit
                if ongoing_matches.get(player_ids_tuple, 0) >= filter_spec.max_concurrent_games_per_pair:
                    logger.debug(f"Skipping pair {player_ids_tuple}: ongoing games ({ongoing_matches.get(player_ids_tuple, 0)}) >= limit ({filter_spec.max_concurrent_games_per_pair})")
                    continue

                # Skip if they've already played too many games
                if (
                    games_per_pair.get(player_ids_tuple, 0)
                    >= filter_spec.max_games_per_pairing
                ):
                    continue

                # Skip if we don't need a match based on confidence
                if not elo_system.is_match_needed(
                    player_a.llm_player.name,
                    player_b.llm_player.name,
                    filter_spec.confidence_threshold,
                ):
                    continue

                # Calculate relevance score (will be strategy-specific in subclasses)
                relevance_score = self._calculate_pair_relevance(
                    player_a, player_b, elo_system
                )
                candidates.append((player_a, player_b, relevance_score))

        return candidates

    def _get_canonical_pair(self, id1: int, id2: int) -> Tuple[int, int]:
        """Create a canonical ordering of player IDs to use as dictionary keys."""
        return (min(id1, id2), max(id1, id2))

    def _calculate_pair_relevance(
        self, player_a: "ArenaPlayer", player_b: "ArenaPlayer", elo_system: EloSystem
    ) -> float:
        """
        Calculate a relevance score for this pair of players.

        This is a strategy-specific score that determines how important this match is
        according to the scheduling strategy.

        Args:
            player_a: First player
            player_b: Second player
            elo_system: The EloSystem used for computing probabilities

        Returns:
            A relevance score (higher means more important match)
        """
        raise NotImplementedError()


class FullRankingScheduler(MatchScheduler):
    """
    A scheduler that focuses on reducing overall pairwise uncertainty among all models.

    This scheduler prioritizes matches that have the highest uncertainty (closest to 50/50 win probability),
    which is the current default behavior in Arena.
    """

    def find_matches(
        self,
        players: List["ArenaPlayer"],
        match_history: List[GameResult],
        elo_system: EloSystem,
        ongoing_matches: Optional[Dict[Tuple[int, int], int]] = None,
        filter_spec: Optional[MatchFilterSpec] = None,
        limit: int = 5,
    ) -> List[Tuple["ArenaPlayer", "ArenaPlayer"]]:
        """
        Find matches that will most reduce overall pairwise uncertainty.

        Args:
            players: List of all players in the arena
            match_history: List of all previous game results
            elo_system: The EloSystem used for computing probabilities
            ongoing_matches: Set of (player_id1, player_id2) tuples for matches currently in progress
            filter_spec: Filter specification with match constraints
            limit: Maximum number of matches to return

        Returns:
            A list of (playerA, playerB) tuples for potential matches, ordered by relevance
        """
        if ongoing_matches is None:
            ongoing_matches = {}

        if filter_spec is None:
            filter_spec = MatchFilterSpec()

        # Sort players by rating for default tie-breaking
        sorted_players = sorted(players, key=lambda p: p.rating.rating, reverse=True)

        # For brand new experiments with no history, use adjacent players in ratings
        if not match_history:
            logger.debug("No game history yet, starting with adjacent players")
            result = []
            
            # Filter out ignored players for adjacency calculations
            filtered_players = sorted_players
            if filter_spec.ignored_player_names:
                filtered_players = [p for p in sorted_players if p.llm_player.name not in filter_spec.ignored_player_names]
                logger.debug(f"Filtered out {len(sorted_players) - len(filtered_players)} ignored players from adjacency consideration")
            
            # Process adjacent pairs in the filtered ranking
            for i in range(len(filtered_players) - 1):
                if len(result) >= limit:
                    break

                player_a = filtered_players[i]
                player_b = filtered_players[i + 1]

                # Apply selected player filter
                if filter_spec.selected_player_names:
                    if (
                        player_a.llm_player.name
                        not in filter_spec.selected_player_names
                        and player_b.llm_player.name
                        not in filter_spec.selected_player_names
                    ):
                        continue

                pair_tuple = self._get_canonical_pair(
                    player_a.player_model.id, player_b.player_model.id
                )
                # Check against the dict using .get() for the concurrency limit
                if ongoing_matches.get(pair_tuple, 0) < filter_spec.max_concurrent_games_per_pair:
                    result.append((player_a, player_b))
            return result

        # Otherwise, get candidates and sort by relevance score (uncertainty)
        candidates = self._get_candidate_pairs(
            players, ongoing_matches, filter_spec, elo_system, match_history
        )

        if not candidates:
            return []

        # Sort by relevance score (higher is more important)
        candidates.sort(key=lambda x: x[2], reverse=True)

        # Return top matches up to the limit
        return [(pair[0], pair[1]) for pair in candidates[:limit]]

    def _calculate_pair_relevance(
        self, player_a: "ArenaPlayer", player_b: "ArenaPlayer", elo_system: EloSystem
    ) -> float:
        """
        Calculate the relevance score for this pair as the uncertainty of their relative strength.

        The uncertainty is highest (0.5) when the probability is 50/50, and lowest (0.0)
        when the probability is 0 or 1.

        Args:
            player_a: First player
            player_b: Second player
            elo_system: The EloSystem used for computing probabilities

        Returns:
            Uncertainty score between 0.0 and 0.5, where 0.5 is maximum uncertainty
        """
        # Calculate the probability that player_a is stronger than player_b
        prob = elo_system.probability_stronger(
            player_a.llm_player.name, player_b.llm_player.name
        )

        # Uncertainty is highest (0.5) when prob is 0.5, and lowest (0.0) when prob is 0 or 1
        uncertainty = 0.5 - abs(prob - 0.5)

        return uncertainty


class TopIdentificationScheduler(MatchScheduler):
    """
    A scheduler that focuses on identifying the single best model with high confidence.

    This scheduler prioritizes matches that will most reduce uncertainty about which model is #1.
    It does this by simulating the expected information gain for each potential match.
    """

    def __init__(self, samples_per_outcome: int = 100):
        """
        Initialize the TopIdentificationScheduler.

        Args:
            samples_per_outcome: Number of posterior samples to use when estimating information gain
        """
        self.samples_per_outcome = samples_per_outcome

    def find_matches(
        self,
        players: List["ArenaPlayer"],
        match_history: List[GameResult],
        elo_system: EloSystem,
        ongoing_matches: Optional[Dict[Tuple[int, int], int]] = None,
        filter_spec: Optional[MatchFilterSpec] = None,
        limit: int = 5,
    ) -> List[Tuple["ArenaPlayer", "ArenaPlayer"]]:
        """
        Find matches that will most reduce uncertainty about which model is best.

        Args:
            players: List of all players in the arena
            match_history: List of all previous game results
            elo_system: The EloSystem used for computing probabilities
            ongoing_matches: Set of (player_id1, player_id2) tuples for matches currently in progress
            filter_spec: Filter specification with match constraints
            limit: Maximum number of matches to return

        Returns:
            A list of (playerA, playerB) tuples for potential matches, ordered by relevance
        """
        if ongoing_matches is None:
            ongoing_matches = {}

        if filter_spec is None:
            filter_spec = MatchFilterSpec()

        # If no games have been played yet, default to the same strategy as FullRankingScheduler
        if not match_history:
            # Delegate to FullRankingScheduler for initial matches
            return FullRankingScheduler().find_matches(
                players,
                match_history,
                elo_system,
                ongoing_matches,
                filter_spec,
                limit,
            )

        # Get candidate pairs
        candidates = self._get_candidate_pairs(
            players, ongoing_matches, filter_spec, elo_system, match_history
        )

        if not candidates:
            return []

        # Sort by relevance score (higher is more important)
        candidates.sort(key=lambda x: x[2], reverse=True)

        # Return top matches up to the limit
        return [(pair[0], pair[1]) for pair in candidates[:limit]]

    def _calculate_pair_relevance(
        self, player_a: "ArenaPlayer", player_b: "ArenaPlayer", elo_system: EloSystem
    ) -> float:
        """
        Calculate the expected information gain about top model identity from this match.

        This calculates how much a match between these players would reduce uncertainty
        about which model is the best, by simulating different outcomes and their impact
        on our belief about the top model.

        Args:
            player_a: First player
            player_b: Second player
            elo_system: The EloSystem used for computing probabilities

        Returns:
            Expected information gain (higher means more important match for top ID)
        """
        # Get current posterior skill samples from trace
        trace_data = cast(Any, elo_system._trace)
        if trace_data is None:
            # If no trace yet, default to uncertainty
            return 0.5 - abs(0.5 - 0.5)

        # Get all player names from the EloSystem
        if not hasattr(elo_system, "_all_players") or elo_system._all_players is None:
            # If no players are registered yet, use the default uncertainty
            return 0.5 - abs(0.5 - 0.5)

        player_names = elo_system._all_players
        name_to_idx = {name: i for i, name in enumerate(player_names)}

        # Get skill samples and reshape
        skill_samples = trace_data.posterior["skill"].values.astype(
            float
        )  # Convert to float to avoid int errors
        flat_samples = skill_samples.reshape(-1, skill_samples.shape[-1])

        # Get current probability of each player being the best
        current_top_probs = self._calculate_top_model_probs(flat_samples)
        current_uncertainty = self._calculate_top_uncertainty(current_top_probs)

        # Get indices for our players
        idx_a = name_to_idx.get(player_a.llm_player.name)
        idx_b = name_to_idx.get(player_b.llm_player.name)

        if idx_a is None or idx_b is None:
            # One of the players doesn't have ratings yet
            return 0.5  # High priority for unrated players

        # Probability of each outcome
        p_a_wins = elo_system.probability_stronger(
            player_a.llm_player.name, player_b.llm_player.name
        )
        p_b_wins = 1.0 - p_a_wins  # Ignoring draws for simplicity

        # Sample subset of posterior for efficiency
        num_samples = min(self.samples_per_outcome, len(flat_samples))
        sample_indices = np.random.choice(len(flat_samples), num_samples, replace=False)
        samples = flat_samples[sample_indices]

        # Simulate a win for player A
        # This is a simplified Bayesian update - we nudge skills to reflect the new outcome
        a_win_samples = samples.copy()
        update_amount = 10.0  # A simplified update amount
        a_win_samples[:, idx_a] += update_amount / 2
        a_win_samples[:, idx_b] -= update_amount / 2

        a_win_top_probs = self._calculate_top_model_probs(a_win_samples)
        a_win_uncertainty = self._calculate_top_uncertainty(a_win_top_probs)

        # Simulate a win for player B
        b_win_samples = samples.copy()
        b_win_samples[:, idx_a] -= update_amount / 2
        b_win_samples[:, idx_b] += update_amount / 2

        b_win_top_probs = self._calculate_top_model_probs(b_win_samples)
        b_win_uncertainty = self._calculate_top_uncertainty(b_win_top_probs)

        # Expected uncertainty after match
        expected_uncertainty = (
            p_a_wins * a_win_uncertainty + p_b_wins * b_win_uncertainty
        )

        # Information gain is the reduction in uncertainty
        info_gain = current_uncertainty - expected_uncertainty

        return info_gain

    def _calculate_top_model_probs(self, skill_samples: np.ndarray) -> np.ndarray:
        """
        Calculate probability of each model being the best based on skill samples.

        Args:
            skill_samples: Array of skill samples from the posterior [num_samples, num_players]

        Returns:
            Array of probabilities for each player being the best
        """
        # For each sample, identify which player has the highest skill
        top_indices = np.argmax(skill_samples, axis=1)

        # Count how often each player is the best
        unique, counts = np.unique(top_indices, return_counts=True)

        # Initialize probabilities (all zeros)
        top_probs = np.zeros(skill_samples.shape[1])

        # Fill in probabilities based on counts
        for idx, count in zip(unique, counts):
            top_probs[idx] = count / len(skill_samples)

        return top_probs

    def _calculate_top_uncertainty(self, top_probs: np.ndarray) -> float:
        """
        Calculate the uncertainty in our belief about which model is best.

        This uses Shannon entropy as the uncertainty measure.

        Args:
            top_probs: Array of probabilities for each player being the best

        Returns:
            Entropy of the top model probability distribution
        """
        # Filter out zero probabilities to avoid log(0)
        non_zero_probs = top_probs[top_probs > 0]

        # Shannon entropy: -sum(p_i * log(p_i))
        return -np.sum(non_zero_probs * np.log(non_zero_probs))


class SigmaMinimizationScheduler(MatchScheduler):
    """
    A scheduler that focuses on reducing uncertainty (sigma) for players with highest uncertainty.

    This scheduler prioritizes matches that will most likely reduce the sigma value for
    players with high uncertainty in their skill rating. It targets the players with the
    highest sigma values and schedules matches that will provide the most information about
    their true skill level.
    """

    def find_matches(
        self,
        players: List["ArenaPlayer"],
        match_history: List[GameResult],
        elo_system: EloSystem,
        ongoing_matches: Optional[Dict[Tuple[int, int], int]] = None,
        filter_spec: Optional[MatchFilterSpec] = None,
        limit: int = 5,
    ) -> List[Tuple["ArenaPlayer", "ArenaPlayer"]]:
        """
        Find matches that will most reduce uncertainty (sigma) for high-uncertainty players.

        Args:
            players: List of all players in the arena
            match_history: List of all previous game results
            elo_system: The EloSystem used for computing probabilities
            ongoing_matches: Set of (player_id1, player_id2) tuples for matches currently in progress
            filter_spec: Filter specification with match constraints
            limit: Maximum number of matches to return

        Returns:
            A list of (playerA, playerB) tuples for potential matches, ordered by relevance
        """
        if ongoing_matches is None:
            ongoing_matches = {}

        if filter_spec is None:
            filter_spec = MatchFilterSpec()

        # Sort players by rating for default tie-breaking
        sorted_players = sorted(players, key=lambda p: p.rating.rating, reverse=True)

        # For brand new experiments with no history, use adjacent players in ratings
        if not match_history:
            logger.debug("No game history yet, starting with adjacent players")
            result = []
            
            # Filter out ignored players for adjacency calculations
            filtered_players = sorted_players
            if filter_spec.ignored_player_names:
                filtered_players = [p for p in sorted_players if p.llm_player.name not in filter_spec.ignored_player_names]
                logger.debug(f"Filtered out {len(sorted_players) - len(filtered_players)} ignored players from adjacency consideration")
            
            # Process adjacent pairs in the filtered ranking
            for i in range(len(filtered_players) - 1):
                if len(result) >= limit:
                    break

                player_a = filtered_players[i]
                player_b = filtered_players[i + 1]

                # Apply selected player filter
                if filter_spec.selected_player_names:
                    if (
                        player_a.llm_player.name
                        not in filter_spec.selected_player_names
                        and player_b.llm_player.name
                        not in filter_spec.selected_player_names
                    ):
                        continue

                pair_tuple = self._get_canonical_pair(
                    player_a.player_model.id, player_b.player_model.id
                )
                # Check against the dict using .get() for the concurrency limit
                if ongoing_matches.get(pair_tuple, 0) < filter_spec.max_concurrent_games_per_pair:
                    result.append((player_a, player_b))
            return result

        # Otherwise, get candidates and sort by relevance score
        candidates = self._get_candidate_pairs(
            players, ongoing_matches, filter_spec, elo_system, match_history
        )

        if not candidates:
            return []

        # Sort by relevance score (higher is more important)
        candidates.sort(key=lambda x: x[2], reverse=True)

        # Return top matches up to the limit
        return [(pair[0], pair[1]) for pair in candidates[:limit]]

    def _calculate_pair_relevance(
        self, player_a: "ArenaPlayer", player_b: "ArenaPlayer", elo_system: EloSystem
    ) -> float:
        """
        Calculate the relevance score based on potential sigma reduction.

        The score is based on:
        1. The maximum sigma value between the two players (prioritizing high uncertainty players)
        2. The closeness of their skill ratings (matches between similarly skilled players
           provide more information)

        Args:
            player_a: First player
            player_b: Second player
            elo_system: The EloSystem used for computing probabilities

        Returns:
            Relevance score for sigma reduction potential
        """
        # Get sigma values for both players
        sigma_a = player_a.rating.sigma
        sigma_b = player_b.rating.sigma

        # Prioritize matches involving players with high sigma
        max_sigma = max(sigma_a, sigma_b)

        # Calculate probability that player_a is stronger than player_b
        # Matches closer to 0.5 provide more information (50/50 matches are most informative)
        prob = elo_system.probability_stronger(
            player_a.llm_player.name, player_b.llm_player.name
        )
        uncertainty = 0.5 - abs(prob - 0.5)

        # Combine both factors:
        # - Higher max_sigma means we want to reduce uncertainty for a high-uncertainty player
        # - Higher uncertainty (closer to 0.5 probability) means the match will be more informative
        # This formula prioritizes matches that involve high-uncertainty players in informative games
        return max_sigma * uncertainty
