from bgbench.models import Experiment, GameMatch, Player
from bgbench.arena import Arena
from bgbench.games.nim_game import NimGame

class TestExperimentManagement:
    def test_resume_experiment_with_players(self, db_session, test_llm):
        """Test resuming an experiment with existing players"""
        # Create initial experiment
        exp = Experiment().create_experiment(db_session, "resume-test")
        
        # Add players with specific configurations
        players = [
            Player(
                name="test-player-1",
                rating=1500.0,
                model_config={"model": "test-model", "temperature": 0.0, "max_tokens": 1000},
                experiment_id=exp.id
            ),
            Player(
                name="test-player-2",
                rating=1600.0,
                model_config={"model": "test-model", "temperature": 0.0, "max_tokens": 1000},
                experiment_id=exp.id
            )
        ]
        for player in players:
            db_session.add(player)
            exp.players.append(player)
        db_session.commit()
        
        # Create some games
        games = [
            GameMatch(experiment_id=exp.id, player1_id=players[0].id, player2_id=players[1].id),
            GameMatch(experiment_id=exp.id, player1_id=players[1].id, player2_id=players[0].id)
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
                assert arena_player.rating.rating == 1500.0

    def test_prevent_new_players_in_resumed_experiment(self, db_session, test_llm):
        """Test that new players cannot be added to resumed experiments"""
        # Create initial experiment
        exp = Experiment().create_experiment(db_session, "no-new-players-test")
        
        # Add initial player
        initial_player = Player(
            name="initial-player",
            rating=1500.0,
            model_config={"model": "test-model", "temperature": 0.0, "max_tokens": 1000},
            experiment_id=exp.id
        )
        db_session.add(initial_player)
        exp.players.append(initial_player)
        db_session.commit()
        
        # Resume experiment
        mock_llm_factory = lambda name: test_llm
        arena = Arena(NimGame(12, 3), db_session, experiment_id=exp.id, llm_factory=mock_llm_factory)
        
        # Try to create new arena with additional player config
        new_player_config = {
            "name": "new-player",
            "model_config": {"model": "test-model", "temperature": 0.0, "max_tokens": 1000}
        }
        
        # This should not add the new player since we're using an existing experiment
        arena = Arena(NimGame(12, 3), db_session, 
                     player_configs=[new_player_config],
                     experiment_id=exp.id,
                     llm_factory=mock_llm_factory)
        
        # Verify only original player exists
        assert len(arena.players) == 1
        assert arena.players[0].llm_player.name == "initial-player"

    def test_new_experiment_player_addition(self, db_session, test_llm):
        """Test adding players to a new experiment"""
        # Create new arena with fresh experiment and player configs
        player_configs = [
            {
                "name": "test-player-1",
                "model_config": {"model": "test-model", "temperature": 0.0, "max_tokens": 1000}
            }
        ]
        # Create mock LLM factory that returns our test LLM
        mock_llm_factory = lambda name: test_llm
        
        arena = Arena(NimGame(12, 3), db_session, 
                     player_configs=player_configs,
                     experiment_name="new-players-test",
                     llm_factory=mock_llm_factory)
        
        # Create arena with player configs
        player_configs = [
            {
                "name": "new-player-1",
                "model_config": {"model": "test-model", "temperature": 0.0, "max_tokens": 1000}
            },
            {
                "name": "new-player-2",
                "model_config": {"model": "test-model", "temperature": 0.0, "max_tokens": 1000}
            }
        ]
        
        # Create mock LLM factory that returns our test LLM
        mock_llm_factory = lambda name: test_llm
        
        # Create arena with player configs
        arena = Arena(NimGame(12, 3), db_session, 
                     player_configs=player_configs,
                     experiment_name="new-players-test",
                     llm_factory=mock_llm_factory)
        
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
        player = Player(name="shared-player", rating=1500.0, model_config={"model": "test-model"}, experiment_id=exp1.id)
        db_session.add(player)
        db_session.commit()
        
        # Create second player for games
        player2 = Player(name="player-2", rating=1500.0, model_config={"model": "test-model"}, experiment_id=exp1.id)
        db_session.add(player2)
        db_session.commit()
        
        game1 = GameMatch(experiment_id=exp1.id, player1_id=player.id, player2_id=player2.id)
        db_session.add(game1)
        db_session.commit()
        
        # Verify players appear in first experiment
        Arena(NimGame(12, 3), db_session, experiment_id=exp1.id)
        exp1_players = exp1.get_players(db_session)
        assert len(exp1_players) == 2  # Both players are in exp1
        player_names = {p.name for p in exp1_players}
        assert "shared-player" in player_names
        assert "player-2" in player_names
        
        # Verify player doesn't appear in second experiment
        Arena(NimGame(12, 3), db_session, experiment_id=exp2.id)
        exp2_players = exp2.get_players(db_session)
        assert len(exp2_players) == 0
