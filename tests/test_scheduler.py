from typing import List, cast
import pytest
from unittest.mock import Mock, MagicMock
from bgbench.arena import ArenaPlayer
from bgbench.scheduler import (
    MatchScheduler,
    FullRankingScheduler,
    TopIdentificationScheduler,
    MatchFilterSpec,  # Import MatchFilterSpec
)
from bgbench.rating import EloSystem, GameResult


class TestSchedulers:
    def test_match_scheduler_interface(self):
        """Test that the base MatchScheduler class requires implementation of methods"""
        scheduler = MatchScheduler()

        # Should raise NotImplementedError since we haven't overridden methods
        with pytest.raises(NotImplementedError):
            scheduler.find_matches([], [], EloSystem())

        with pytest.raises(NotImplementedError):
            scheduler._calculate_pair_relevance(Mock(), Mock(), Mock())

    def test_full_ranking_scheduler(self):
        """Test that FullRankingScheduler prioritizes matches with highest uncertainty"""
        scheduler = FullRankingScheduler()

        # Create mock player objects with necessary attributes
        def create_mock_player(name, player_id, rating=1500):
            player = MagicMock()
            player.llm_player = MagicMock()
            player.llm_player.name = name
            player.player_model = MagicMock()
            player.player_model.id = player_id
            player.rating = MagicMock()
            player.rating.rating = rating
            return player

        # Create some mock players
        player1 = create_mock_player("player1", 1)
        player2 = create_mock_player("player2", 2)
        player3 = create_mock_player("player3", 3)
        player4 = create_mock_player("player4", 4)

        # Create a mock EloSystem
        elo_system = Mock()

        # Mock the is_match_needed method
        elo_system.is_match_needed.return_value = True

        # Set up probabilities
        def mock_probability(a, b):
            if a == "player2" and b == "player3" or a == "player3" and b == "player2":
                return 0.5  # Highest uncertainty
            if a == "player1" and b == "player2" or a == "player2" and b == "player1":
                return 0.6  # Medium uncertainty
            return 0.8  # Low uncertainty

        elo_system.probability_stronger.side_effect = mock_probability

        # Set games attribute for EloSystem
        elo_system.games = []  # No history

        # Test the relevance calculation
        rel_2_3 = scheduler._calculate_pair_relevance(player2, player3, elo_system)
        rel_1_2 = scheduler._calculate_pair_relevance(player1, player2, elo_system)
        rel_3_4 = scheduler._calculate_pair_relevance(player3, player4, elo_system)

        # player2 vs player3 should have highest relevance (uncertainty 0.5)
        assert rel_2_3 > rel_1_2
        assert rel_2_3 > rel_3_4

        # For a simple test with no history, we'll just check that we get adjacent pairs
        players = [player1, player2, player3, player4]
        match_history = []

        # Mock that we're starting a new experiment with no ongoing matches
        # Cast the list of mocks to the expected type for pyright
        typed_players = cast(List[ArenaPlayer], players)
        pairs = scheduler.find_matches(
            typed_players, match_history, elo_system, {}, limit=1
        )

        # For new experiments with no history, it should schedule adjacent players based on initial sort
        assert len(pairs) > 0
        pair = pairs[0]
        assert (pair[0].player_model.id == 1 and pair[1].player_model.id == 2) or (
            pair[0].player_model.id == 2 and pair[1].player_model.id == 1
        )

    # Helper function to create mock players for reuse
    def _create_mock_player(self, name, player_id, rating=1500):
        player = MagicMock()
        player.llm_player = MagicMock()
        player.llm_player.name = name
        player.player_model = MagicMock()
        player.player_model.id = player_id
        player.rating = MagicMock()
        player.rating.rating = rating
        return player

    def test_scheduler_max_concurrent_games_default(self):
        """Test that the default max_concurrent_games_per_pair=1 is respected."""
        scheduler = FullRankingScheduler()
        player1 = self._create_mock_player("p1", 1)
        player2 = self._create_mock_player("p2", 2)
        player3 = self._create_mock_player("p3", 3)
        players = [player1, player2, player3]
        typed_players = cast(List[ArenaPlayer], players)

        elo_system = Mock()
        elo_system.is_match_needed.return_value = True
        # Mock probability_stronger to ensure relevance calculation works
        elo_system.probability_stronger.return_value = 0.6 # Arbitrary non-0.5 value

        # Simulate one game ongoing between p1 and p2
        ongoing_matches = {(1, 2): 1}
        filter_spec = MatchFilterSpec(max_concurrent_games_per_pair=1) # Default

        # Use a dummy match history
        match_history = [GameResult(player_0="p1", player_1="p3", winner="p1")]

        pairs = scheduler.find_matches(
            typed_players, match_history, elo_system, ongoing_matches, filter_spec, limit=10
        )

        # Expected: p1 vs p2 should NOT be scheduled again. p1 vs p3 or p2 vs p3 should be.
        found_p1_p2 = False
        for pA, pB in pairs:
            ids = {pA.player_model.id, pB.player_model.id}
            if ids == {1, 2}:
                found_p1_p2 = True
                break
        assert not found_p1_p2, "Pair (1, 2) scheduled despite ongoing game and limit 1"
        assert len(pairs) > 0 # Should schedule other pairs

    def test_scheduler_max_concurrent_games_increased(self):
        """Test that max_concurrent_games_per_pair > 1 allows concurrent games."""
        scheduler = FullRankingScheduler()
        player1 = self._create_mock_player("p1", 1)
        player2 = self._create_mock_player("p2", 2)
        player3 = self._create_mock_player("p3", 3)
        players = [player1, player2, player3]
        typed_players = cast(List[ArenaPlayer], players)

        elo_system = Mock()
        elo_system.is_match_needed.return_value = True
        elo_system.probability_stronger.return_value = 0.6

        # Use a dummy match history
        match_history = [GameResult(player_0="p1", player_1="p3", winner="p1")]

        # --- Test allowing 2 concurrent games ---
        filter_spec_2 = MatchFilterSpec(max_concurrent_games_per_pair=2)

        # Case 1: One game ongoing between p1 and p2, limit is 2
        ongoing_matches_1 = {(1, 2): 1}
        pairs_limit2_ongoing1 = scheduler.find_matches(
            typed_players, match_history, elo_system, ongoing_matches_1, filter_spec_2, limit=10
        )

        # Expected: p1 vs p2 SHOULD be scheduled again.
        found_p1_p2_limit2_ongoing1 = False
        for pA, pB in pairs_limit2_ongoing1:
            ids = {pA.player_model.id, pB.player_model.id}
            if ids == {1, 2}:
                found_p1_p2_limit2_ongoing1 = True
                break
        assert found_p1_p2_limit2_ongoing1, "Pair (1, 2) not scheduled with limit 2 and 1 ongoing game"

        # Case 2: Two games ongoing between p1 and p2, limit is 2
        ongoing_matches_2 = {(1, 2): 2}
        pairs_limit2_ongoing2 = scheduler.find_matches(
            typed_players, match_history, elo_system, ongoing_matches_2, filter_spec_2, limit=10
        )

        # Expected: p1 vs p2 should NOT be scheduled again.
        found_p1_p2_limit2_ongoing2 = False
        for pA, pB in pairs_limit2_ongoing2:
            ids = {pA.player_model.id, pB.player_model.id}
            if ids == {1, 2}:
                found_p1_p2_limit2_ongoing2 = True
                break
        assert not found_p1_p2_limit2_ongoing2, "Pair (1, 2) scheduled despite 2 ongoing games and limit 2"
        # Ensure other pairs can still be scheduled if available
        assert len(pairs_limit2_ongoing2) > 0 or len(players) <= 2

    def test_top_identification_scheduler_basic(self):
        """Simple test for TopIdentificationScheduler - just verifies it falls back to FullRankingScheduler when no history."""
        # For simplicity, we'll just test the fallback behavior for now
        scheduler = TopIdentificationScheduler()

        # Create some mock players
        def create_mock_player(name, player_id, rating=1500):
            player = MagicMock()
            player.llm_player = MagicMock()
            player.llm_player.name = name
            player.player_model = MagicMock()
            player.player_model.id = player_id
            player.rating = MagicMock()
            player.rating.rating = rating
            return player

        player1 = create_mock_player("player1", 1)
        player2 = create_mock_player("player2", 2)
        player3 = create_mock_player("player3", 3)
        player4 = create_mock_player("player4", 4)

        players = [player1, player2, player3, player4]

        # Create a mock EloSystem with no history
        elo_system = Mock()
        elo_system.games = []  # Empty history
        elo_system.is_match_needed.return_value = True

        # This should fall back to FullRankingScheduler behavior when no history
        # Cast the list of mocks to the expected type for pyright
        typed_players = cast(List[ArenaPlayer], players)
        pairs = scheduler.find_matches(typed_players, [], elo_system, {}, limit=1)

        # Should return adjacent players (player1 vs player2) based on initial sort
        assert len(pairs) > 0
        pair = pairs[0]
        assert (pair[0].player_model.id == 1 and pair[1].player_model.id == 2) or (
            pair[0].player_model.id == 2 and pair[1].player_model.id == 1
        )
