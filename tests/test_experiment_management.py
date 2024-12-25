import pytest
from bgbench.models import Experiment, Game, Player
from bgbench.arena import Arena
from bgbench.games.nim_game import NimGame

class TestExperimentManagement:
    def test_get_experiment_results(self, db_session, test_llm):
        """Test getting experiment results summary"""
        # Create experiment
        exp = Experiment().create_experiment(db_session, "test-results")
        
        # Add players
        players = [
            Player(name="player-a", rating=1500.0),
            Player(name="player-b", rating=1600.0)
        ]
        for player in players:
            db_session.add(player)
        db_session.commit()
        
        # Add some games
        games = [
            Game(experiment_id=exp.id, player_id=players[0].id),
            Game(experiment_id=exp.id, player_id=players[1].id)
        ]
        for game in games:
            db_session.add(game)
        db_session.commit()
        
        # Create arena and get results
        # Use mock LLM factory for testing
        mock_llm_factory = lambda name: test_llm
        arena = Arena(NimGame(12, 3), db_session, experiment_id=exp.id, llm_factory=mock_llm_factory)
        results = arena.get_experiment_results()
        
        # Verify results structure
        assert results["experiment_id"] == exp.id
        assert results["experiment_name"] == "test-results"
        assert results["total_games"] == 2
        assert len(results["games"]) == 2
        
        # Verify player ratings
        ratings = results["player_ratings"]
        assert "player-a" in ratings
        assert "player-b" in ratings
        assert ratings["player-b"] > ratings["player-a"]

    def test_get_player_game_history(self, db_session):
        """Test getting game history for a specific player"""
        # Create experiment and player
        exp = Experiment().create_experiment(db_session, "test-history")
        player = Player(name="test-player", rating=1500.0)
        db_session.add(player)
        db_session.commit()
        
        # Add games with states
        game = Game(experiment_id=exp.id, player_id=player.id)
        db_session.add(game)
        db_session.commit()
        
        # Create arena
        arena = Arena(NimGame(12, 3), db_session, experiment_id=exp.id)
        
        # Get player history
        history = arena.get_player_game_history("test-player")
        
        # Verify history structure
        assert isinstance(history, list)
        assert len(history) == 1
        assert "game_id" in history[0]
        assert "won" in history[0]
        assert history[0]["won"] is True

    def test_experiment_player_management(self, db_session):
        """Test managing players across experiments"""
        # Create two experiments
        exp1 = Experiment().create_experiment(db_session, "exp-1")
        exp2 = Experiment().create_experiment(db_session, "exp-2")
        
        # Add player to first experiment
        player = Player(name="shared-player", rating=1500.0)
        db_session.add(player)
        db_session.commit()
        
        game1 = Game(experiment_id=exp1.id, player_id=player.id)
        db_session.add(game1)
        db_session.commit()
        
        # Verify player appears in first experiment
        arena1 = Arena(NimGame(12, 3), db_session, experiment_id=exp1.id)
        exp1_players = exp1.get_players(db_session)
        assert len(exp1_players) == 1
        assert exp1_players[0].name == "shared-player"
        
        # Verify player doesn't appear in second experiment
        arena2 = Arena(NimGame(12, 3), db_session, experiment_id=exp2.id)
        exp2_players = exp2.get_players(db_session)
        assert len(exp2_players) == 0
