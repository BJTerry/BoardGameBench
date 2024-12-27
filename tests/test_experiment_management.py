from bgbench.models import Experiment, Game, Player
from bgbench.arena import Arena
from bgbench.games.nim_game import NimGame
from bgbench.llm_player import LLMPlayer

class TestExperimentManagement:
    def test_get_experiment_results(self, db_session, test_llm):
        """Test getting experiment results summary"""
        # Create experiment
        exp = Experiment().create_experiment(db_session, "test-results")
        
        # Add players
        players = [
            Player(name="player-a", rating=1500.0, model_config={"model": "test-model", "temperature": 0.0, "max_tokens": 1000}),
            Player(name="player-b", rating=1600.0, model_config={"model": "test-model", "temperature": 0.0, "max_tokens": 1000})
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
        player = Player(name="test-player", rating=1500.0, model_config={"model": "test-model"})
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

    def test_resume_experiment_with_players(self, db_session, test_llm):
        """Test resuming an experiment with existing players"""
        # Create initial experiment
        exp = Experiment().create_experiment(db_session, "resume-test")
        
        # Add players with specific configurations
        players = [
            Player(
                name="test-player-1",
                rating=1500.0,
                model_config={"model": "test-model", "temperature": 0.0, "max_tokens": 1000}
            ),
            Player(
                name="test-player-2",
                rating=1600.0,
                model_config={"model": "test-model", "temperature": 0.0, "max_tokens": 1000}
            )
        ]
        for player in players:
            db_session.add(player)
            exp.players.append(player)
        db_session.commit()
        
        # Create some games
        games = [
            Game(experiment_id=exp.id, player_id=players[0].id),
            Game(experiment_id=exp.id, player_id=players[1].id)
        ]
        for game in games:
            db_session.add(game)
        db_session.commit()
        
        # Create mock LLM factory
        mock_llm_factory = lambda name: test_llm
        
        # Resume experiment
        arena = Arena(NimGame(12, 3), db_session, experiment_id=exp.id, llm_factory=mock_llm_factory)
        
        # Verify players were restored
        assert len(arena.players) == 2
        player_names = {p.llm_player.name for p in arena.players}
        assert "test-player-1" in player_names
        assert "test-player-2" in player_names
        
        # Verify ratings were restored
        for arena_player in arena.players:
            if arena_player.llm_player.name == "test-player-1":
                assert arena_player.rating.rating == 1500.0
            elif arena_player.llm_player.name == "test-player-2":
                assert arena_player.rating.rating == 1600.0

    def test_prevent_new_players_in_resumed_experiment(self, db_session, test_llm):
        """Test that new players cannot be added to resumed experiments"""
        # Create initial experiment
        exp = Experiment().create_experiment(db_session, "no-new-players-test")
        
        # Add initial player
        initial_player = Player(
            name="initial-player",
            rating=1500.0,
            model_config={"model": "test-model", "temperature": 0.0, "max_tokens": 1000}
        )
        db_session.add(initial_player)
        exp.players.append(initial_player)
        db_session.commit()
        
        # Resume experiment
        mock_llm_factory = lambda name: test_llm
        arena = Arena(NimGame(12, 3), db_session, experiment_id=exp.id, llm_factory=mock_llm_factory)
        
        # Try to add new player
        new_player = LLMPlayer("new-player", test_llm)
        arena.add_player(new_player)
        
        # Verify new player was not added
        assert len(arena.players) == 1
        assert arena.players[0].llm_player.name == "initial-player"

    def test_new_experiment_player_addition(self, db_session, test_llm):
        """Test adding players to a new experiment"""
        # Create new arena with fresh experiment
        arena = Arena(NimGame(12, 3), db_session, experiment_name="new-players-test")
        
        # Add players
        players = [
            LLMPlayer("new-player-1", test_llm),
            LLMPlayer("new-player-2", test_llm)
        ]
        
        for player in players:
            arena.add_player(player)
        
        # Verify players were added
        assert len(arena.players) == 2
        player_names = {p.llm_player.name for p in arena.players}
        assert "new-player-1" in player_names
        assert "new-player-2" in player_names
        
        # Verify experiment relationship
        exp_players = arena.experiment.get_players(db_session)
        assert len(exp_players) == 2
        exp_player_names = {p.name for p in exp_players}
        assert "new-player-1" in exp_player_names
        assert "new-player-2" in exp_player_names

    def test_experiment_player_management(self, db_session):
        """Test managing players across experiments"""
        # Create two experiments
        exp1 = Experiment().create_experiment(db_session, "exp-1")
        exp2 = Experiment().create_experiment(db_session, "exp-2")
        
        # Add player to first experiment
        player = Player(name="shared-player", rating=1500.0, model_config={"model": "test-model"})
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
