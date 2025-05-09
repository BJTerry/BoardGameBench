import pytest
import datetime
from unittest.mock import patch, MagicMock
from bgbench.experiment.rating import GameResult

from bgbench.data.models import Experiment, Player, GameMatch
from bgbench.experiment.export import (
    is_game_complete,
    is_game_draw,
    count_complete_games,
    count_draws,
    build_match_history,
    get_player_costs,
    format_for_export,
    export_experiment,
    calculate_skill_comparison_data,
    format_skill_comparison_for_export,
)


@pytest.fixture
def setup_experiment(db_session):
    """Setup a test experiment with players and games"""
    experiment = Experiment().create_experiment(db_session, "Test Export")

    # Create players
    player1 = Player(
        name="Player 1",
        model_config={"model": "test-model"},
        experiment_id=experiment.id,
    )
    player2 = Player(
        name="Player 2",
        model_config={"model": "test-model"},
        experiment_id=experiment.id,
    )
    db_session.add_all([player1, player2])
    db_session.flush()

    # Create various types of games
    games = [
        # Completed game with a winner
        GameMatch(
            experiment_id=experiment.id,
            player1_id=player1.id,
            player2_id=player2.id,
            winner_id=player1.id,
            complete=True,
            conceded=False,
        ),
        # Completed game that ended in a draw
        GameMatch(
            experiment_id=experiment.id,
            player1_id=player1.id,
            player2_id=player2.id,
            winner_id=None,
            complete=True,
            conceded=False,
        ),
        # Conceded game
        GameMatch(
            experiment_id=experiment.id,
            player1_id=player1.id,
            player2_id=player2.id,
            winner_id=player2.id,
            complete=True,
            conceded=True,
            concession_reason="Invalid moves exceeded",
        ),
        # Incomplete game (error or timeout)
        GameMatch(
            experiment_id=experiment.id,
            player1_id=player1.id,
            player2_id=player2.id,
            winner_id=None,
            complete=False,
            conceded=False,
        ),
    ]
    db_session.add_all(games)
    db_session.commit()

    return {
        "experiment": experiment,
        "player1": player1,
        "player2": player2,
        "games": games,
    }


class TestGameCompletionUtils:
    def test_is_game_complete(self, setup_experiment):
        """Test is_game_complete utility function"""
        games = setup_experiment["games"]

        # First 3 games should be complete (win, draw, concession)
        for i in range(3):
            assert is_game_complete(games[i]) is True

        # Last game should be incomplete
        assert is_game_complete(games[3]) is False

    def test_is_game_draw(self, setup_experiment):
        """Test is_game_draw utility function"""
        games = setup_experiment["games"]

        # Only the second game is a draw (complete but no winner)
        assert is_game_draw(games[0]) is False  # Complete with winner
        assert is_game_draw(games[1]) is True  # Complete with no winner (draw)
        assert is_game_draw(games[2]) is False  # Complete but conceded (not a draw)
        assert is_game_draw(games[3]) is False  # Incomplete

    def test_count_complete_games(self, setup_experiment):
        """Test count_complete_games utility function"""
        games = setup_experiment["games"]

        # First 3 games are complete
        assert count_complete_games(games) == 3

        # Subset of games
        assert count_complete_games(games[2:]) == 1

    def test_count_draws(self, setup_experiment):
        """Test count_draws utility function"""
        games = setup_experiment["games"]

        # Only the second game is a draw
        assert count_draws(games) == 1

        # Subset with no draws
        assert count_draws(games[0:1]) == 0

        # Subset containing the draw
        assert count_draws(games[1:2]) == 1


class TestMatchHistoryBuilding:
    def test_build_match_history(self, setup_experiment, db_session):
        """Test build_match_history utility function"""
        experiment = setup_experiment["experiment"]
        games = setup_experiment["games"]

        # Get players from the database
        players = experiment.get_players(db_session)

        # Build match history
        match_history = build_match_history(games, players)

        # Should include 3 complete games (win, draw, concession)
        assert len(match_history) == 3

        # Check wins and draws are correctly represented
        assert match_history[0].winner is not None  # First game has a winner
        assert match_history[1].winner is None  # Second game is a draw
        assert (
            match_history[2].winner is not None
        )  # Third game has a winner (concession)

        # Check player names are correct
        assert match_history[0].player_0 == "Player 1"
        assert match_history[0].player_1 == "Player 2"
class TestPlayerCosts:
    # Patch target updated to reflect the new location of export
    @patch("bgbench.experiment.export.func.sum")
    def test_get_player_costs(self, mock_sum, setup_experiment, db_session):
        """Test get_player_costs utility function"""
        experiment = setup_experiment["experiment"]
        player1 = setup_experiment["player1"]
        player2 = setup_experiment["player2"]

        # Mock the query result
        mock_scalar = MagicMock()
        mock_scalar.scalar.side_effect = [
            0.5,
            0.3,
        ]  # Mock costs for player1 and player2
        mock_join = MagicMock()
        mock_join.join.return_value = mock_join
        mock_join.filter.return_value = mock_scalar

        mock_query = MagicMock()
        mock_query.join.return_value = mock_join

        mock_db_session = MagicMock()
        mock_db_session.query.return_value = mock_query

        # Get player costs
        costs = get_player_costs(mock_db_session, experiment.id, [player1, player2])

        # Verify costs were returned correctly
        assert len(costs) == 2
        assert costs["Player 1"] == 0.5
        assert costs["Player 2"] == 0.3


class TestSkillComparisonFunctions:
    def test_calculate_skill_comparison_data(self):
        """Test the skill comparison calculation function"""
        # Create a simple match history
        match_history = [
            GameResult(player_0="Player1", player_1="Player2", winner="Player1"),
            GameResult(player_0="Player1", player_1="Player2", winner="Player1"),
            GameResult(player_0="Player1", player_1="Player2", winner="Player2"),
            GameResult(player_0="Player1", player_1="Player3", winner=None),  # Draw
        ]
        player_names = ["Player1", "Player2", "Player3"]

        # Calculate skill comparison data
        probabilities, records = calculate_skill_comparison_data(
            match_history, player_names
        )

        # Check probabilities exist
        assert ("Player1", "Player2") in probabilities
        assert ("Player2", "Player1") in probabilities
        assert ("Player1", "Player3") in probabilities
        assert ("Player3", "Player1") in probabilities

        # Check the probabilities sum to 1 for complementary pairs
        assert (
            round(
                probabilities[("Player1", "Player2")]
                + probabilities[("Player2", "Player1")],
                5,
            )
            == 1.0
        )

        # Check records are correct
        assert records[("Player1", "Player2")] == (2, 1, 0)  # 2 wins, 1 loss, 0 draws
        assert records[("Player2", "Player1")] == (1, 2, 0)  # 1 win, 2 losses, 0 draws
        assert records[("Player1", "Player3")] == (0, 0, 1)  # 0 wins, 0 losses, 1 draw
        assert records[("Player3", "Player1")] == (0, 0, 1)  # 0 wins, 0 losses, 1 draw
        
    def test_calculate_skill_comparison_with_ignored_players(self):
        """Test that ignored players are included in skill comparison calculations"""
        # Create a match history with an ignored player
        match_history = [
            GameResult(player_0="Player1", player_1="Player2", winner="Player1"),
            GameResult(player_0="Player1", player_1="Player2", winner="Player1"),
            GameResult(player_0="Player1", player_1="Player3", winner="Player3"),  # Player3 is ignored but still included
            GameResult(player_0="Player2", player_1="Player3", winner="Player2"),
        ]
        # All players, including ignored one
        player_names = ["Player1", "Player2", "Player3"]
        
        # Calculate skill comparison data
        probabilities, records = calculate_skill_comparison_data(
            match_history, player_names
        )
        
        # Check that the ignored player's probabilities are included
        assert ("Player1", "Player3") in probabilities
        assert ("Player3", "Player1") in probabilities
        assert ("Player2", "Player3") in probabilities
        assert ("Player3", "Player2") in probabilities
        
        # Check records are correct for the ignored player
        assert records[("Player1", "Player3")] == (0, 1, 0)  # 0 wins, 1 loss, 0 draws
        assert records[("Player3", "Player1")] == (1, 0, 0)  # 1 win, 0 losses, 0 draws
        assert records[("Player2", "Player3")] == (1, 0, 0)  # 1 win, 0 losses, 0 draws
        assert records[("Player3", "Player2")] == (0, 1, 0)  # 0 wins, 1 losses, 0 draws

    def test_format_skill_comparison_for_export(self):
        """Test formatting skill comparison data for export"""
        # Create test data
        player_names = ["Player1", "Player2", "Player3"]

        # Sample probabilities and records
        probabilities = {
            ("Player1", "Player2"): 0.7,
            ("Player2", "Player1"): 0.3,
            ("Player1", "Player3"): 0.6,
            ("Player3", "Player1"): 0.4,
            ("Player2", "Player3"): 0.55,
            ("Player3", "Player2"): 0.45,
        }

        records = {
            ("Player1", "Player2"): (2, 1, 0),
            ("Player2", "Player1"): (1, 2, 0),
            ("Player1", "Player3"): (1, 0, 1),
            ("Player3", "Player1"): (0, 1, 1),
            ("Player2", "Player3"): (1, 1, 0),
            ("Player3", "Player2"): (1, 1, 0),
        }

        # Format for export
        comparisons = format_skill_comparison_for_export(
            probabilities, records, player_names
        )

        # Verify it's a list
        assert isinstance(comparisons, list)

        # We should have n(n-1) comparisons where n is the number of players
        # 3 players means 6 comparisons (each player compared with 2 others)
        assert len(comparisons) == 6

        # Verify each comparison has the correct structure
        for comparison in comparisons:
            assert "model" in comparison
            assert "opponent" in comparison
            assert "probability" in comparison
            assert "wins" in comparison
            assert "losses" in comparison
            assert "draws" in comparison

        # Find specific comparisons to verify
        p1_p2 = next(
            c
            for c in comparisons
            if c["model"] == "Player1" and c["opponent"] == "Player2"
        )
        p3_p1 = next(
            c
            for c in comparisons
            if c["model"] == "Player3" and c["opponent"] == "Player1"
        )

        # Verify Player1 vs Player2 comparison
        assert p1_p2["probability"] == 0.7
        assert p1_p2["wins"] == 2
        assert p1_p2["losses"] == 1
        assert p1_p2["draws"] == 0

        # Verify Player3 vs Player1 comparison
        assert p3_p1["probability"] == 0.4
        assert p3_p1["wins"] == 0
        assert p3_p1["losses"] == 1
        assert p3_p1["draws"] == 1
class TestExportFunctionality:
    # Patch targets updated to reflect the new location of export
    @patch("bgbench.experiment.export.build_match_history")
    @patch("bgbench.experiment.export.get_player_costs")
    def test_format_for_export(
        self, mock_get_costs, mock_build_history, setup_experiment, db_session
    ):
        """Test format_for_export function"""
        experiment = setup_experiment["experiment"]
        player1 = setup_experiment["player1"]
        player2 = setup_experiment["player2"]

        # Mock match history building
        mock_build_history.return_value = [
            MagicMock(player_0="Player 1", player_1="Player 2", winner="Player 1"),
            MagicMock(player_0="Player 1", player_1="Player 2", winner=None),  # Draw
            MagicMock(player_0="Player 1", player_1="Player 2", winner="Player 2"),
        ]

        # Set up player ratings manually to guarantee Player 1 > Player 2
        player1.rating = 1550  # Higher rating
        player2.rating = 1450  # Lower rating

        # Mock player costs
        mock_get_costs.return_value = {"Player 1": 0.5, "Player 2": 0.3}

        # Use a real EloSystem with the probability_stronger method patched
        with patch(
            "bgbench.experiment.rating.EloSystem.probability_stronger"
        ) as mock_prob_stronger:
            # Always return that Player 1 is stronger than Player 2 with >95% probability
            mock_prob_stronger.return_value = 0.96

            # Format for export
            export_data = format_for_export(db_session, experiment.id, "Test Game")

        # Verify basic structure
        assert export_data["gameName"] == "Test Game"
        assert "metadata" in export_data
        assert "results" in export_data
        assert (
            "skillComparisons" in export_data
        )  # Verify skill comparison data is included

        # Verify skill comparison structure
        skill_comparisons = export_data["skillComparisons"]
        assert isinstance(skill_comparisons, list)
        assert len(skill_comparisons) == 2  # Two players * (2-1) = 2 comparisons

        # Verify comparison structure
        for comparison in skill_comparisons:
            assert "model" in comparison
            assert "opponent" in comparison
            assert "probability" in comparison
            assert "wins" in comparison
            assert "losses" in comparison
            assert "draws" in comparison

        # Verify metadata
        metadata = export_data["metadata"]
        assert metadata["totalGamesPlayed"] == 4
        assert metadata["completedGames"] == 3
        assert metadata["draws"] == 1
        assert metadata["experimentId"] == experiment.id

        # Verify player results
        results = export_data["results"]
        assert len(results) == 2  # Two players

        # Verify each player has the expected fields
        for player_result in results:
            assert "modelName" in player_result
            assert "score" in player_result
            assert "rank" in player_result  # New field
            assert "gamesPlayed" in player_result
            assert "winRate" in player_result
            assert "concessions" in player_result
            assert "costPerGame" in player_result
            assert "confidenceInterval" in player_result

        # Find the results by score to avoid confusion with modelName
        # Because the sort is by score, the first result should be rank 1
        results_by_score = sorted(results, key=lambda x: x["score"], reverse=True)
        assert results_by_score[0]["rank"] == 1
        assert results_by_score[1]["rank"] == 2

    @patch("os.makedirs")
    # Patch targets updated to reflect the new location of export
    @patch("bgbench.experiment.export.format_for_export")
    @patch("json.dump")
    @patch("builtins.open")
    @patch("bgbench.experiment.export.datetime")
    def test_export_experiment(
        self,
        mock_datetime,
        mock_open,
        mock_json_dump,
        mock_format,
        mock_makedirs,
        db_session,
    ):
        """Test export_experiment function"""
        # Mock the datetime to have a predictable filename
        mock_datetime.datetime.now.return_value = datetime.datetime(2025, 3, 15)
        mock_datetime.datetime.utcnow.return_value = datetime.datetime(2025, 3, 15)
        mock_datetime.datetime.strftime = datetime.datetime.strftime

        # Mock the formatted data
        mock_format.return_value = {"test": "data"}

        # Call export_experiment
        result = export_experiment(db_session, 1, "TestGame")

        # Verify directory was created
        mock_makedirs.assert_called_once_with("exports", exist_ok=True)

        # Verify the file was opened with the expected name
        mock_open.assert_called_once()
        file_path = mock_open.call_args[0][0]
        assert file_path.startswith("exports/TestGame_1_")

        # Verify the JSON was written - note the indent param is used in the actual code
        mock_json_dump.assert_called_once()
        assert mock_json_dump.call_args[0][0] == {"test": "data"}
        assert mock_json_dump.call_args[0][1] == mock_open().__enter__()

        # Verify function returned the expected path
        assert result == file_path


class TestGameCompletion:
    def test_arena_handles_draws(self, setup_experiment, db_session):
        """Test that the Arena correctly identifies draws"""
        # This would test the arena's handling of draws, but we'd need to mock too many things
        # Instead, we'll test the utility functions that the Arena uses to handle draws

        games = setup_experiment["games"]
        # Count games with different completion states
        complete_games = [g for g in games if is_game_complete(g)]
        draws = [g for g in games if is_game_draw(g)]
        wins = [
            g
            for g in games
            if is_game_complete(g) and not is_game_draw(g) and not g.conceded
        ]
        concessions = [g for g in games if is_game_complete(g) and g.conceded]
        incomplete = [g for g in games if not is_game_complete(g)]

        assert len(complete_games) == 3
        assert len(draws) == 1
        assert len(wins) == 1
        assert len(concessions) == 1
        assert len(incomplete) == 1

    def test_complete_field_migration(self, db_session):
        """Test the complete field migration logic"""
        # Create an experiment
        experiment = Experiment().create_experiment(db_session, "Test Migration")

        # Create players
        player1 = Player(
            name="Player 1",
            model_config={"model": "test-model"},
            experiment_id=experiment.id,
        )
        player2 = Player(
            name="Player 2",
            model_config={"model": "test-model"},
            experiment_id=experiment.id,
        )
        db_session.add_all([player1, player2])
        db_session.flush()

        # Create a game without explicitly setting the complete field
        # The field should default to False
        game = GameMatch(
            experiment_id=experiment.id,
            player1_id=player1.id,
            player2_id=player2.id,
            winner_id=player1.id,
        )
        db_session.add(game)
        db_session.commit()

        # Verify the complete field defaulted to False
        db_game = db_session.query(GameMatch).filter_by(id=game.id).first()
        assert hasattr(db_game, "complete"), "The complete field should exist"
        assert not bool(db_game.complete), "The complete field should default to False"

        # Run the migration logic (updating games with winners to be complete)
        from sqlalchemy import text

        db_session.execute(
            text("UPDATE games SET complete = 1 WHERE winner_id IS NOT NULL")
        )
        db_session.commit()

        # Verify the complete field was updated
        db_game = db_session.query(GameMatch).filter_by(id=game.id).first()
        assert bool(db_game.complete), (
            "The complete field should be True after migration"
        )
