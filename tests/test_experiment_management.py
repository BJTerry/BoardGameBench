import asyncio
import pytest
from unittest.mock import patch
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
                model_config={
                    "model": "test-model",
                    "temperature": 0.0,
                    "max_tokens": 1000,
                },
                experiment_id=exp.id,
            ),
            Player(
                name="test-player-2",
                rating=1600.0,
                model_config={
                    "model": "test-model",
                    "temperature": 0.0,
                    "max_tokens": 1000,
                },
                experiment_id=exp.id,
            ),
        ]
        for player in players:
            db_session.add(player)
            exp.players.append(player)
        db_session.commit()

        # Create some games
        games = [
            GameMatch(
                experiment_id=exp.id, player1_id=players[0].id, player2_id=players[1].id
            ),
            GameMatch(
                experiment_id=exp.id, player1_id=players[1].id, player2_id=players[0].id
            ),
        ]
        for game in games:
            db_session.add(game)
        db_session.commit()

        # Create mock LLM factory
        def mock_llm_factory(name):
            return test_llm

        # Resume experiment
        arena = Arena(
            NimGame(12, 3),
            db_session,
            experiment_id=exp.id,
            llm_factory=mock_llm_factory,
        )

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

    def test_allow_new_players_in_resumed_experiment(self, db_session, test_llm):
        """Test that new players can be added to resumed experiments"""
        # Create initial experiment
        exp = Experiment().create_experiment(db_session, "add-new-players-test")

        # Add initial player
        initial_player = Player(
            name="initial-player",
            rating=1500.0,
            model_config={
                "model": "test-model",
                "temperature": 0.0,
                "max_tokens": 1000,
            },
            experiment_id=exp.id,
        )
        db_session.add(initial_player)
        exp.players.append(initial_player)
        db_session.commit()

        # Resume experiment
        def mock_llm_factory(name):
            return test_llm
        arena = Arena(
            NimGame(12, 3),
            db_session,
            experiment_id=exp.id,
            llm_factory=mock_llm_factory,
        )

        # Verify only the initial player exists
        assert len(arena.players) == 1
        assert arena.players[0].llm_player.name == "initial-player"

        # Create new arena with additional player config
        new_player_config = {
            "name": "new-player",
            "model_config": {
                "model": "test-model",
                "temperature": 0.0,
                "max_tokens": 1000,
            },
        }

        # This should now add the new player when resuming an experiment
        arena = Arena(
            NimGame(12, 3),
            db_session,
            player_configs=[new_player_config],
            experiment_id=exp.id,
            llm_factory=mock_llm_factory,
        )

        # Verify both the initial and new player exist
        assert len(arena.players) == 2
        player_names = {p.llm_player.name for p in arena.players}
        assert "initial-player" in player_names
        assert "new-player" in player_names

        # Check that the new player was added to the database
        exp_players = exp.get_players(db_session)
        assert len(exp_players) == 2
        db_player_names = {p.name for p in exp_players}
        assert "initial-player" in db_player_names
        assert "new-player" in db_player_names

    def test_new_experiment_player_addition(self, db_session, test_llm):
        """Test adding players to a new experiment"""
        # Create new arena with fresh experiment and player configs
        player_configs = [
            {
                "name": "test-player-1",
                "model_config": {
                    "model": "test-model",
                    "temperature": 0.0,
                    "max_tokens": 1000,
                },
            }
        ]
        # Create mock LLM factory that returns our test LLM
        def mock_llm_factory(name):
            return test_llm

        arena = Arena(
            NimGame(12, 3),
            db_session,
            player_configs=player_configs,
            experiment_name="new-players-test",
            llm_factory=mock_llm_factory,
        )

        # Create arena with player configs
        player_configs = [
            {
                "name": "new-player-1",
                "model_config": {
                    "model": "test-model",
                    "temperature": 0.0,
                    "max_tokens": 1000,
                },
            },
            {
                "name": "new-player-2",
                "model_config": {
                    "model": "test-model",
                    "temperature": 0.0,
                    "max_tokens": 1000,
                },
            },
        ]

        # Reuse the same LLM factory for the second arena

        # Create arena with player configs
        arena = Arena(
            NimGame(12, 3),
            db_session,
            player_configs=player_configs,
            experiment_name="new-players-test",
            llm_factory=mock_llm_factory,
        )

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
        
    @pytest.mark.asyncio
    async def test_graceful_termination(self, db_session, test_llm):
        """Test that graceful termination works correctly"""
        # Create test player configs
        player_configs = [
            {
                "name": "player-a",
                "model_config": {
                    "model": "test-model",
                    "temperature": 0.0,
                    "max_tokens": 1000,
                },
            },
            {
                "name": "player-b",
                "model_config": {
                    "model": "test-model",
                    "temperature": 0.0,
                    "max_tokens": 1000,
                },
            },
        ]
        
        # Create arena with test players
        def mock_llm_factory(name):
            return test_llm
        arena = Arena(
            NimGame(12, 3),
            db_session,
            player_configs=player_configs,
            experiment_name="termination-test",
            llm_factory=mock_llm_factory,
        )
        
        # Create mock game task that will run for a while
        async def mock_long_game(*args, **kwargs):
            await asyncio.sleep(1)
            return True
        
        # Replace find_next_available_match to always return the same players
        async def mock_find_match(*args, **kwargs):
            return arena.players[0], arena.players[1]
        
        # Patch the methods
        with patch.object(arena, 'run_single_game', side_effect=mock_long_game), \
             patch.object(arena, 'find_next_available_match', side_effect=mock_find_match):
             
            # Start the arena in a task
            arena_task = asyncio.create_task(arena.evaluate_all())
            
            # Wait for games to start
            await asyncio.sleep(0.2)
            
            # Verify games are running
            assert len(arena._active_tasks) > 0
            
            # Simulate first Ctrl+C
            arena.handle_sigint()
            
            # Verify that scheduling has stopped but games continue
            assert arena._stop_scheduling
            assert not arena._force_stop
            
            # Wait a bit and verify no new games were scheduled
            num_tasks = len(arena._active_tasks)
            await asyncio.sleep(0.2)
            assert len(arena._active_tasks) <= num_tasks  # Should not increase
            
            # Simulate second Ctrl+C
            arena.handle_sigint()
            
            # Verify that force stop is now True
            assert arena._force_stop
            
            # Wait for arena to finish
            await arena_task

    def test_experiment_player_management(self, db_session):
        """Test managing players across experiments"""
        # Create two experiments
        exp1 = Experiment().create_experiment(db_session, "exp-1")
        exp2 = Experiment().create_experiment(db_session, "exp-2")

        # Add player to first experiment
        player = Player(
            name="shared-player",
            rating=1500.0,
            model_config={"model": "test-model"},
            experiment_id=exp1.id,
        )
        db_session.add(player)
        db_session.commit()

        # Create second player for games
        player2 = Player(
            name="player-2",
            rating=1500.0,
            model_config={"model": "test-model"},
            experiment_id=exp1.id,
        )
        db_session.add(player2)
        db_session.commit()

        game1 = GameMatch(
            experiment_id=exp1.id, player1_id=player.id, player2_id=player2.id
        )
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
