import pytest
import numpy as np
from collections import defaultdict
from bgbench.bayes_rating import EloSystem, PlayerRating, GameResult


class TestBayesRating:
    def test_basic_initialization(self):
        """Test that EloSystem can be initialized with default parameters."""
        elo = EloSystem()
        assert elo.confidence_threshold == 0.70
        assert elo.draws_parameter_prior == 10.0

        elo = EloSystem(confidence_threshold=0.80, draws_parameter_prior=5.0)
        assert elo.confidence_threshold == 0.80
        assert elo.draws_parameter_prior == 5.0

    def test_game_result_creation(self):
        """Test that GameResult objects can be created properly."""
        # Test win
        result = GameResult(player_0="A", player_1="B", winner="A")
        assert result.player_0 == "A"
        assert result.player_1 == "B"
        assert result.winner == "A"

        # Test draw
        result = GameResult(player_0="A", player_1="B", winner=None)
        assert result.player_0 == "A"
        assert result.player_1 == "B"
        assert result.winner is None

    def test_update_ratings_no_history(self):
        """Test updating ratings with empty history."""
        elo = EloSystem()
        history = []
        player_names = ["A", "B"]

        ratings = elo.update_ratings(history, player_names)

        assert len(ratings) == 2
        assert ratings["A"].name == "A"
        assert ratings["A"].rating == 1500.0
        assert ratings["A"].sigma == 400.0
        assert ratings["A"].games_played == 0

        assert ratings["B"].name == "B"
        assert ratings["B"].rating == 1500.0
        assert ratings["B"].sigma == 400.0
        assert ratings["B"].games_played == 0

    def test_update_ratings_single_game(self):
        """Test updating ratings after a single game."""
        elo = EloSystem()
        history = [GameResult(player_0="A", player_1="B", winner="A")]
        player_names = ["A", "B"]

        ratings = elo.update_ratings(history, player_names)

        assert len(ratings) == 2
        assert ratings["A"].name == "A"
        assert ratings["A"].games_played == 1
        assert ratings["B"].name == "B"
        assert ratings["B"].games_played == 1

        # Winner should have higher rating
        assert ratings["A"].rating > ratings["B"].rating

    def test_update_ratings_with_draws(self):
        """Test updating ratings with games that include draws."""
        elo = EloSystem()
        history = [
            GameResult(player_0="A", player_1="B", winner="A"),
            GameResult(player_0="A", player_1="B", winner=None),  # Draw
            GameResult(player_0="B", player_1="A", winner="B"),
        ]
        player_names = ["A", "B"]

        ratings = elo.update_ratings(history, player_names)

        assert len(ratings) == 2
        assert ratings["A"].games_played == 3
        assert ratings["B"].games_played == 3

        # With equal wins and a draw, ratings should be close
        assert abs(ratings["A"].rating - ratings["B"].rating) < 50

    def test_update_ratings_multiple_players(self):
        """Test updating ratings with multiple players."""
        elo = EloSystem()
        history = [
            GameResult(player_0="A", player_1="B", winner="A"),
            GameResult(player_0="B", player_1="C", winner="B"),
            GameResult(player_0="C", player_1="A", winner="A"),
        ]
        player_names = ["A", "B", "C"]

        ratings = elo.update_ratings(history, player_names)

        assert len(ratings) == 3
        for p in player_names:
            assert p in ratings
            assert ratings[p].games_played == 2

        # A won all games, should have highest rating
        assert ratings["A"].rating > ratings["B"].rating
        assert ratings["A"].rating > ratings["C"].rating

    def test_probability_stronger(self):
        """Test calculating probability that one player is stronger than another."""
        elo = EloSystem()
        history = [
            GameResult(player_0="A", player_1="B", winner="A"),
            GameResult(player_0="A", player_1="B", winner="A"),
        ]
        player_names = ["A", "B"]

        # First run update_ratings to generate the trace
        elo.update_ratings(history, player_names)

        # Now we can calculate probability
        prob = elo.probability_stronger("A", "B")

        # A won both games, so probability should be high
        assert prob > 0.5
        assert 0 <= prob <= 1.0

    def test_direct_comparison(self):
        """Test comparing players directly using probability_stronger."""
        elo = EloSystem(confidence_threshold=0.8)

        # Set up a simple history where A wins most games
        history = [
            GameResult(player_0="A", player_1="B", winner="A"),
            GameResult(player_0="A", player_1="B", winner="A"),
            GameResult(player_0="B", player_1="A", winner="B"),  # B wins once
            GameResult(player_0="A", player_1="B", winner="A"),
        ]
        player_names = ["A", "B"]

        # Update ratings to create the trace
        ratings = elo.update_ratings(history, player_names)

        # Get probability using posterior samples
        prob_a_stronger = elo.probability_stronger("A", "B")

        # A won 3/4 games, so probability should be well above 0.5
        assert prob_a_stronger > 0.7

        # Reverse comparison should be 1 - prob
        prob_b_stronger = elo.probability_stronger("B", "A")
        assert abs(prob_b_stronger - (1 - prob_a_stronger)) < 0.01

    def test_get_credible_intervals(self):
        """Test retrieving credible intervals for player skills."""
        elo = EloSystem()
        history = [
            GameResult(player_0="A", player_1="B", winner="A"),
            GameResult(player_0="A", player_1="B", winner="A"),
        ]
        player_names = ["A", "B"]

        # First run update_ratings to generate the trace
        elo.update_ratings(history, player_names)

        # Now get credible intervals
        intervals = elo.get_credible_intervals()

        assert "A" in intervals
        assert "B" in intervals

        a_lower, a_upper = intervals["A"]
        b_lower, b_upper = intervals["B"]

        # Intervals should make sense
        assert a_lower < a_upper
        assert b_lower < b_upper

        # A won all games, so their interval should generally be higher
        assert a_lower > b_lower or a_upper > b_upper

    def test_convergence_with_many_games(self):
        """Test that ratings converge with many games between players of different skill."""
        elo = EloSystem()

        # Create history where A wins 80% of games against B
        history = []
        np.random.seed(42)  # For reproducibility

        for _ in range(50):
            # 80% chance A wins, 20% chance B wins
            winner = "A" if np.random.random() < 0.8 else "B"
            history.append(GameResult(player_0="A", player_1="B", winner=winner))

        player_names = ["A", "B"]
        ratings = elo.update_ratings(history, player_names)

        # A should have significantly higher rating
        assert ratings["A"].rating > ratings["B"].rating

        # With many games, uncertainty should be reduced from initial 400
        assert ratings["A"].sigma < 300
        assert ratings["B"].sigma < 300

        # Probability A is stronger should be high
        prob = elo.probability_stronger("A", "B")
        assert prob > 0.95  # Very confident A is stronger

    def test_round_robin_tournament(self):
        """Test a round-robin tournament with multiple players."""
        elo = EloSystem()

        # Create players with different true skills
        players = ["A", "B", "C", "D"]
        true_skills = {"A": 1800, "B": 1700, "C": 1600, "D": 1500}

        # Simulate matches with some randomness but biased by skill difference
        history = []
        np.random.seed(42)  # For reproducibility

        # Each player plays against every other player twice
        for i, p1 in enumerate(players):
            for j, p2 in enumerate(players):
                if i != j:
                    for _ in range(2):  # Two matches each
                        # Calculate win probability based on skill difference
                        skill_diff = true_skills[p1] - true_skills[p2]
                        p1_win_prob = 1 / (1 + 10 ** (-skill_diff / 400))

                        # Determine winner
                        winner = p1 if np.random.random() < p1_win_prob else p2
                        history.append(
                            GameResult(player_0=p1, player_1=p2, winner=winner)
                        )

        # Update ratings
        ratings = elo.update_ratings(history, players)

        # Check that ratings roughly correspond to true skills
        rating_order = sorted(players, key=lambda p: ratings[p].rating, reverse=True)
        true_skill_order = sorted(players, key=lambda p: true_skills[p], reverse=True)

        # Order should match or be close
        assert (
            rating_order[0] == true_skill_order[0]
        )  # Strongest player identified correctly
        assert (
            rating_order[-1] == true_skill_order[-1]
        )  # Weakest player identified correctly

    def test_is_match_needed(self):
        """Test if matches are needed based on confidence threshold."""
        elo = EloSystem(confidence_threshold=0.8)

        # Create a history with clear results for comparison
        history_one_sided = [
            GameResult(player_0="A", player_1="B", winner="A"),
            GameResult(player_0="A", player_1="B", winner="A"),
            GameResult(player_0="A", player_1="B", winner="A"),
            GameResult(player_0="A", player_1="B", winner="A"),
            GameResult(player_0="A", player_1="B", winner="A"),
        ]

        history_balanced = [
            GameResult(player_0="C", player_1="D", winner="C"),
            GameResult(player_0="C", player_1="D", winner="D"),
            GameResult(player_0="C", player_1="D", winner="D"),
            GameResult(player_0="C", player_1="D", winner="C"),
        ]

        # Test with one-sided results
        elo.update_ratings(history_one_sided, ["A", "B"])
        prob_a_b = elo.probability_stronger("A", "B")
        assert prob_a_b > 0.8  # A is clearly stronger
        # Now test is_match_needed directly
        assert not elo.is_match_needed("A", "B")  # No more matches needed

        # Test with balanced results
        elo.update_ratings(history_balanced, ["C", "D"])
        prob_c_d = elo.probability_stronger("C", "D")
        assert 0.3 < prob_c_d < 0.7  # Close to 50/50
        # Now test is_match_needed directly
        assert elo.is_match_needed("C", "D")  # More matches needed


# To avoid running the tests all the time when imported
if __name__ == "__main__":
    pytest.main(["-xvs", "test_bayes_rating.py"])
