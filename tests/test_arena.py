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
        player = DBPlayer(name="test-player", rating=1500.0)
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

    def test_add_player_resumed_experiment(self, db_session, nim_game, mock_llm):
        """Test adding players to a resumed experiment"""
        # Create experiment with existing player
        exp = Experiment().create_experiment(db_session, "test-resume")
        db_player = DBPlayer(name="existing-player", rating=1600.0)
        db_session.add(db_player)
        db_session.commit()
        
        # Resume experiment
        arena = Arena(nim_game, db_session, experiment_id=exp.id)
        
        # Try to add new player
        new_player = LLMPlayer("new-player", mock_llm)
        arena.add_player(new_player)
        
        # Verify new player was not added
        assert len(arena.players) == 0
        
        # Try to add existing player
        existing_player = LLMPlayer("existing-player", mock_llm)
        arena.add_player(existing_player)
        
        # Verify existing player was added with correct rating
        assert len(arena.players) == 1
        assert arena.players[0].llm_player.name == "existing-player"
        assert arena.players[0].rating.rating == 1600.0

    def test_calculate_match_uncertainty(self, db_session, nim_game):
        """Test match uncertainty calculation"""
        arena = Arena(nim_game, db_session)
        player_a = ArenaPlayer(
            LLMPlayer("player-a", MagicMock()),
            PlayerRating("player-a", 1500, 0)
        )
        player_b = ArenaPlayer(
            LLMPlayer("player-b", MagicMock()),
            PlayerRating("player-b", 1500, 0)
        )
        
        # Equal ratings should have maximum uncertainty
        uncertainty = arena.calculate_match_uncertainty(player_a, player_b)
        assert uncertainty == 1.0
        
        # Update player_b rating to create skill gap
        player_b.rating.rating = 1800
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
