import pytest
from unittest.mock import MagicMock
from bgbench.arena import Arena, ArenaPlayer
from bgbench.games.nim_game import NimGame
from bgbench.llm_player import LLMPlayer
from bgbench.models import Experiment, Player as DBPlayer
from bgbench.rating import PlayerRating

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
def nim_game():
    return NimGame(12, 3)  # 12 objects, max take 3

class TestArena:
    def test_arena_initialization_new_experiment(self, db_session, nim_game):
        """Test creating a new Arena with a new experiment"""
        arena = Arena(nim_game, db_session, experiment_name="test-arena")
        assert arena.experiment.name == "test-arena"
        assert not arena.players

    def test_arena_initialization_resume_experiment(self, db_session, nim_game):
        """Test resuming an Arena from existing experiment"""
        # Create experiment and add some players
        exp = Experiment().create_experiment(db_session, "test-resume")
        player = DBPlayer(name="test-player", rating=1500.0, model_config={"model": "test-model"})
        db_session.add(player)
        db_session.commit()
        
        # Resume experiment
        arena = Arena(nim_game, db_session, experiment_id=exp.id)
        assert arena.experiment.id == exp.id
        assert arena.experiment.name == "test-resume"

    def test_add_player_new_experiment(self, db_session, nim_game, mock_llm):
        """Test adding players to a new experiment"""
        arena = Arena(nim_game, db_session, experiment_name="test-add-player")
        player = LLMPlayer("test-player", mock_llm)
        arena.add_player(player)
        
        assert len(arena.players) == 1
        assert arena.players[0].llm_player.name == "test-player"
        
        # Verify player was added to database
        db_player = db_session.query(DBPlayer).filter_by(name="test-player").first()
        assert db_player is not None
        assert db_player.rating == 1500.0

    def test_calculate_match_uncertainty(self, db_session, nim_game):
        """Test match uncertainty calculation"""
        arena = Arena(nim_game, db_session)
        # Create players with different ratings and some games played
        player_a = ArenaPlayer(
            LLMPlayer("player-a", MagicMock()),
            PlayerRating("player-a", 1500, 10)  # 10 games played
        )
        player_b = ArenaPlayer(
            LLMPlayer("player-b", MagicMock()),
            PlayerRating("player-b", 2000, 10)  # Much higher rating, 10 games played
        )
    
        # With significant rating gap and enough games played, uncertainty should be lower
        uncertainty = arena.calculate_match_uncertainty(player_a, player_b)
        assert uncertainty < 1.0

    def test_find_best_match(self, db_session, nim_game, mock_llm):
        """Test finding best match between players"""
        arena = Arena(nim_game, db_session)
        
        # Add players with different ratings
        players = [
            ("player-a", 1500),
            ("player-b", 1600),
            ("player-c", 1800)
        ]
        
        for name, rating in players:
            player = LLMPlayer(name, mock_llm)
            arena.add_player(player)
            arena.players[-1].rating.rating = rating
        
        # Find best match
        match = arena.find_best_match()
        assert match is not None
        
        # Should match closest rated players
        player_names = {match[0].llm_player.name, match[1].llm_player.name}
        assert player_names == {"player-a", "player-b"}
