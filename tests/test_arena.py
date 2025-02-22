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

    def test_calculate_match_uncertainty(self, db_session, nim_game, mock_llm, mock_llm_factory):
        """Test match uncertainty calculation"""
        player_configs = [
            {
                "name": "player-a",
                "model_config": {
                    "model": "test-model",
                    "temperature": 0.0,
                    "max_tokens": 1000
                }
            },
            {
                "name": "player-b",
                "model_config": {
                    "model": "test-model", 
                    "temperature": 0.0,
                    "max_tokens": 1000
                }
            }
        ]
        
        arena = Arena(
            nim_game,
            db_session,
            player_configs=player_configs,
            experiment_name="test-uncertainty",
            llm_factory=mock_llm_factory
        )
        
        # Update player ratings directly
        arena.players[0].rating.rating = 1500
        arena.players[0].rating.games_played = 10
        arena.players[1].rating.rating = 2000
        arena.players[1].rating.games_played = 10
    
        # With significant rating gap and enough games played, uncertainty should be lower
        uncertainty = arena.elo_system.calculate_match_uncertainty(
            arena.players[0].rating,
            arena.players[1].rating
        )
        assert uncertainty < 1.0

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
    @patch('bgbench.game_runner.GameRunner.play_game', new=mock_play_game_no_elo_update)
    async def test_max_games_between_players(self, db_session, nim_game, mock_llm, mock_llm_factory):
        """Test that the Arena does not schedule more than 10 games between any two players."""
        player_configs = [
            {
                "name": "player-1",
                "model_config": {"model": "test-model", "temperature": 0.0}
            },
            {
                "name": "player-2",
                "model_config": {"model": "test-model", "temperature": 0.0}
            }
        ]

        # Initialize the Arena
        arena = Arena(
            nim_game,
            db_session,
            player_configs=player_configs,
            experiment_name="test-max-games",
            llm_factory=mock_llm_factory,
            confidence_threshold=1.0  # Set to 1.0 so games continue until max limit
        )


        # Run the evaluation loop
        await arena.evaluate_all()

        # Check the number of games between the two players
        games_played = arena._games_played_between(
            arena.players[0].player_model,
            arena.players[1].player_model
        )
        assert games_played == 10, f"Expected 10 games, but found {games_played}"

        # Ensure no more matches are scheduled
        next_match = await arena.find_next_available_match()
        assert next_match is None, "No more matches should be scheduled between the players"
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
