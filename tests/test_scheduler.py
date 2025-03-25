import pytest
from unittest.mock import Mock, MagicMock
from bgbench.scheduler import (
    MatchScheduler,
    FullRankingScheduler,
    TopIdentificationScheduler,
)
from bgbench.bayes_rating import EloSystem


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

        # Mock that we're starting a new experiment
        pairs = scheduler.find_matches(players, match_history, elo_system, set(), limit=1)  # type: ignore

        # For new experiments, we should get player1, player2 as the first pair (adjacent in list)
        assert len(pairs) > 0
        pair = pairs[0]
        assert (pair[0].player_model.id == 1 and pair[1].player_model.id == 2) or (
            pair[0].player_model.id == 2 and pair[1].player_model.id == 1
        )

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

        # This should fall back to FullRankingScheduler behavior
        pairs = scheduler.find_matches(players, [], elo_system, set(), limit=1)  # type: ignore

        # Should return adjacent players (player1 vs player2) since there's no history
        assert len(pairs) > 0
        pair = pairs[0]
        assert (pair[0].player_model.id == 1 and pair[1].player_model.id == 2) or (
            pair[0].player_model.id == 2 and pair[1].player_model.id == 1
        )
