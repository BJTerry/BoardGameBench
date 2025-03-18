import pytest
import logging
from unittest.mock import MagicMock, patch
import random

async def mock_play_game_no_elo_update(self):
    # Pretend a random player wins, but we skip rating updates.
    winner = random.choice(self.players)
    return winner, [], None  # no rating changes
from bgbench.arena import Arena
from bgbench.games.nim_game import NimGame
from bgbench.models import Experiment, Player as DBPlayer, GameMatch

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.run = MagicMock()
    # Mock the model configuration
    model = MagicMock()
    model.model_name = "test-model"
    llm.model = model
    
    # Mock model settings to return actual values
    model_settings = MagicMock()
    model_settings.get.side_effect = lambda key, default=None: {
        "temperature": 0.0,
        "max_tokens": 1000,
        "top_p": 1.0,
        "timeout": 60.0
    }.get(key, default)
    llm.model_settings = model_settings
    return llm

@pytest.fixture
def mock_llm_factory(mock_llm):
    def factory(name):
        return mock_llm
    return factory

@pytest.fixture
def nim_game():
    return NimGame(12, 3)  # 12 objects, max take 3

class TestArena:
    def test_arena_initialization_new_experiment(self, db_session, nim_game, mock_llm, mock_llm_factory):
        """Test creating a new Arena with a new experiment"""
        player_configs = [{
            "name": "test-player",
            "model_config": {
                "model": "test-model",
                "temperature": 0.0,
                "max_tokens": 1000
            }
        }]

        arena = Arena(
            nim_game, 
            db_session, 
            player_configs=player_configs,
            experiment_name="test-arena",
            llm_factory=mock_llm_factory
        )
        
        assert arena.experiment.name == "test-arena"
        assert len(arena.players) == 1
        assert arena.players[0].llm_player.name == "test-player"

    def test_arena_initialization_resume_experiment(self, db_session, nim_game):
        """Test resuming an Arena from existing experiment"""
        # Create experiment and add some players
        exp = Experiment().create_experiment(db_session, "test-resume")
        player = DBPlayer(
            name="test-player",
            rating=1500.0,
            model_config={"model": "test-model"},
            experiment_id=exp.id
        )
        db_session.add(player)
        db_session.commit()
        
        # Resume experiment
        arena = Arena(nim_game, db_session, experiment_id=exp.id)
        assert arena.experiment.id == exp.id
        assert arena.experiment.name == "test-resume"
        
    def test_resume_experiment_with_draws(self, db_session, nim_game, mock_llm_factory):
        """Test resuming an experiment with completed games including draws"""
        # Create experiment
        exp = Experiment().create_experiment(db_session, "test-resume-draws")
        
        # Create players
        player1 = DBPlayer(name="player1", model_config={"model": "test-model"}, experiment_id=exp.id)
        player2 = DBPlayer(name="player2", model_config={"model": "test-model"}, experiment_id=exp.id)
        db_session.add_all([player1, player2])
        db_session.flush()
        
        # Create games with different completion states
        games = [
            # Complete game with winner
            GameMatch(
                experiment_id=exp.id, 
                player1_id=player1.id, 
                player2_id=player2.id, 
                winner_id=player1.id, 
                complete=True
            ),
            # Complete game that ended in a draw
            GameMatch(
                experiment_id=exp.id, 
                player1_id=player1.id, 
                player2_id=player2.id, 
                winner_id=None, 
                complete=True
            ),
            # Incomplete game (should be deleted on resume)
            GameMatch(
                experiment_id=exp.id, 
                player1_id=player1.id, 
                player2_id=player2.id, 
                winner_id=None, 
                complete=False
            )
        ]
        db_session.add_all(games)
        db_session.commit()
        
        # Resume the experiment
        arena = Arena(nim_game, db_session, experiment_id=exp.id, llm_factory=mock_llm_factory)
        
        # Verify only complete games were kept
        remaining_games = db_session.query(GameMatch).filter_by(experiment_id=exp.id).all()
        assert len(remaining_games) == 2, "Incomplete games should be removed"
        
        # Verify match history includes both complete games (win and draw)
        assert len(arena.match_history) == 2, "Match history should contain both complete games"
        
        # Check if one of the games in match history is a draw
        has_draw = any(match.winner is None for match in arena.match_history)
        assert has_draw, "Match history should include the draw game"

    def test_create_new_experiment_with_players(self, db_session, nim_game, mock_llm, mock_llm_factory):
        """Test creating a new experiment with players"""
        player_configs = [{
            "name": "test-player",
            "model_config": {
                "model": "test-model",
                "temperature": 0.0,
                "max_tokens": 1000
            }
        }]

        arena = Arena(
            nim_game,
            db_session,
            player_configs=player_configs,
            experiment_name="test-new-experiment",
            llm_factory=mock_llm_factory
        )
        
        # Verify players were created correctly
        assert len(arena.players) == 1
        assert arena.players[0].llm_player.name == "test-player"
        
        # Verify player was added to database with correct experiment
        db_player = db_session.query(DBPlayer).filter_by(name="test-player").first()
        assert db_player is not None
        assert db_player.rating == 1500.0
        assert db_player.experiment_id == arena.experiment.id
        assert db_player in arena.experiment.players

    @pytest.mark.asyncio
    async def test_find_best_match(self, db_session, nim_game, mock_llm, mock_llm_factory):
        """Test finding best match between players"""
        player_configs = [
            {
                "name": "player-a",
                "model_config": {"model": "test-model", "temperature": 0.0}
            },
            {
                "name": "player-b", 
                "model_config": {"model": "test-model", "temperature": 0.0}
            },
            {
                "name": "player-c",
                "model_config": {"model": "test-model", "temperature": 0.0}
            }
        ]
        
        arena = Arena(
            nim_game,
            db_session,
            player_configs=player_configs,
            experiment_name="test-match",
            llm_factory=mock_llm_factory
        )
        
        # Update ratings after initialization
        ratings = {"player-a": 1500, "player-b": 1600, "player-c": 1800}
        for player in arena.players:
            db_player = db_session.query(DBPlayer).filter_by(name=player.llm_player.name).first()
            db_player.rating = ratings[player.llm_player.name]
            db_session.commit()
        
        # Find best match
        match = await arena.find_next_available_match()
        assert match is not None
        
        # Should match closest rated players
        player_names = {match[0].llm_player.name, match[1].llm_player.name}
        assert player_names == {"player-a", "player-b"}

    @pytest.mark.asyncio
    async def test_arena_handles_draws(self, db_session, nim_game, mock_llm_factory):
        """Test that the Arena correctly handles games that end in a draw."""
        player_configs = [
            {
                "name": "player-a",
                "model_config": {"model": "test-model", "temperature": 0.0}
            },
            {
                "name": "player-b",
                "model_config": {"model": "test-model", "temperature": 0.0}
            }
        ]
        
        # Create an arena
        arena = Arena(
            nim_game,
            db_session,
            player_configs=player_configs,
            experiment_name="test-draws",
            llm_factory=mock_llm_factory
        )
        
        # Mock the play_game method to return a draw (None for winner)
        async def mock_play_game_draw(self):
            return None, [], None  # No winner (draw), empty history, no concession
            
        # Patch the play_game method to return a draw
        with patch('bgbench.game_runner.GameRunner.play_game', mock_play_game_draw):
            # Run a single game that will end in a draw
            player_a, player_b = arena.players[0], arena.players[1]
            await arena.run_single_game(player_a, player_b)
            
            # Get the game from the database
            game = db_session.query(GameMatch).filter_by(experiment_id=arena.experiment.id).first()
            
            # Verify the game was marked as complete but without a winner
            assert bool(game.complete), "Game should be marked as complete"
            assert game.winner_id is None, "Game should not have a winner (draw)"
            
            # Verify the match history includes the draw
            assert len(arena.match_history) == 1, "Match history should contain the draw"
            assert arena.match_history[0].winner is None, "Match history should record the game as a draw"
            
            # Get experiment results and verify draws are counted
            results = arena.get_experiment_results()
            assert results["draws"] == 1, "Draw should be counted in experiment results"
            
    def test_concessions(self, db_session, nim_game, mock_llm, mock_llm_factory):
        """Test tracking of concessions in games"""
        player_configs = [
            {
                "name": "player-a",
                "model_config": {"model": "test-model", "temperature": 0.0}
            },
            {
                "name": "player-b",
                "model_config": {"model": "test-model", "temperature": 0.0}
            }
        ]
        
            
        arena = Arena(
            nim_game,
            db_session,
            player_configs=player_configs,
            experiment_name="test-concessions",
            llm_factory=mock_llm_factory
        )

        # Get the players from database
        player_a = db_session.query(DBPlayer).filter_by(name="player-a").first()
        player_b = db_session.query(DBPlayer).filter_by(name="player-b").first()

        # Create some test games with concessions
        games_data = [
            # player_a concedes to player_b
            {
                "winner_id": player_b.id,
                "player1_id": player_a.id,
                "player2_id": player_b.id,
                "conceded": True,
                "concession_reason": "test concession 1"
            },
            # player_b concedes to player_a
            {
                "winner_id": player_a.id,
                "player1_id": player_b.id,
                "player2_id": player_a.id,
                "conceded": True,
                "concession_reason": "test concession 2"
            },
            # Normal game, no concession
            {
                "winner_id": player_a.id,
                "player1_id": player_a.id,
                "player2_id": player_b.id,
                "conceded": False,
                "concession_reason": None
            }
        ]

        # Add test games to database
        for game_data in games_data:
            game = GameMatch(
                experiment_id=arena.experiment.id,
                **game_data
            )
            db_session.add(game)
        db_session.commit()

        # Get experiment results and verify concession counts
        results = arena.get_experiment_results()
        
        assert results["player_concessions"]["player-a"] == 1
        assert results["player_concessions"]["player-b"] == 1
        assert results["total_games"] == 3

    def test_log_standings_with_concessions(self, db_session, nim_game, mock_llm, caplog, mock_llm_factory):
        """Test that log_standings correctly shows concession information"""
        # Configure logging to capture output
        caplog.set_level(logging.INFO)
        player_configs = [
            {
                "name": "player-a",
                "model_config": {"model": "test-model", "temperature": 0.0}
            },
            {
                "name": "player-b",
                "model_config": {"model": "test-model", "temperature": 0.0}
            }
        ]
                
        arena = Arena(
            nim_game,
            db_session,
            player_configs=player_configs,
            experiment_name="test-log-concessions",
            llm_factory=mock_llm_factory
        )

        # Get the players from database
        player_a = db_session.query(DBPlayer).filter_by(name="player-a").first()
        player_b = db_session.query(DBPlayer).filter_by(name="player-b").first()

        # Create a game where player_a concedes to player_b
        game = GameMatch(
            experiment_id=arena.experiment.id,
            winner_id=player_b.id,
            player1_id=player_a.id,
            player2_id=player_b.id,
            conceded=True,
            concession_reason="test concession"
        )
        db_session.add(game)
        db_session.commit()

        # Clear the log
        caplog.clear()
        
        # Log standings
        arena.log_standings()

        # Check that concessions are mentioned in the log
        log_text = caplog.text
        assert "player-a: 1500 (0 games, 1 concessions, $0.0000 cost)" in log_text
        assert "player-b: 1500 (0 games, 0 concessions, $0.0000 cost)" in log_text
