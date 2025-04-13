import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from bgbench.arena import Arena
from bgbench.games.nim_game import NimGame, NimState
from bgbench.models import Experiment, Player as DBPlayer, GameMatch
from bgbench.match.match_state import MatchStateData
from datetime import datetime


@pytest.fixture
def nim_game():
    """Create a NimGame instance for testing."""
    return NimGame(12, 3)  # 12 objects, max take 3


@pytest.fixture
def mock_llm_factory():
    """Create a mock LLM factory for testing."""
    def factory(name):
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
            "timeout": 60.0,
        }.get(key, default)
        llm.model_settings = model_settings
        return llm

    return factory


@pytest.fixture
def mock_match_state_manager():
    """Create a mock MatchStateManager for testing."""
    manager = MagicMock()
    
    # Mock get_latest_state to return a test state
    def get_latest_state(session, match_id):
        # Return a simple test state
        return MatchStateData(
            turn=1,
            current_player_id=1,
            timestamp=datetime.now(),
            game_state={"remaining": 9},  # NimState with 9 remaining
            metadata={"test": "data"}
        )
    
    manager.get_latest_state = get_latest_state
    return manager


class TestArenaResumption:
    def test_arena_initializes_resumable_matches_list(self, db_session, nim_game):
        """Test that Arena initializes the resumable matches list."""
        # Create a new arena with player configs
        player_configs = [
            {"name": "test-player", "model_config": {"model": "test-model"}}
        ]
        arena = Arena(nim_game, db_session, experiment_name="test-resumption", player_configs=player_configs)
        
        # Check that the resumable matches list is initialized
        assert hasattr(arena, "_resumable_matches")
        assert isinstance(arena._resumable_matches, list)
        assert len(arena._resumable_matches) == 0
    
    def test_resume_experiment_with_incomplete_matches(
        self, db_session, nim_game, mock_llm_factory, mock_match_state_manager
    ):
        """Test that Arena loads incomplete matches when resuming an experiment."""
        # Create experiment
        exp = Experiment().create_experiment(db_session, "test-resume-incomplete")
        
        # Create players
        player1 = DBPlayer(
            name="player1", model_config={"model": "test-model"}, experiment_id=exp.id
        )
        player2 = DBPlayer(
            name="player2", model_config={"model": "test-model"}, experiment_id=exp.id
        )
        db_session.add_all([player1, player2])
        db_session.flush()
        
        # Create an incomplete match
        incomplete_match = GameMatch(
            experiment_id=exp.id,
            player1_id=player1.id,
            player2_id=player2.id,
            winner_id=None,
            complete=False,
        )
        db_session.add(incomplete_match)
        db_session.commit()
        
        # Patch the match_state_manager to return our mock
        with patch('bgbench.arena.MatchStateManager', return_value=mock_match_state_manager):
            # Patch the game's deserialize_state method to return a NimState
            with patch.object(nim_game, 'deserialize_state', return_value=NimState(remaining=9, current_player=0)):
                # Instead of patching _resume_experiment which causes recursion,
                # we'll patch get_latest_state to return our test state
                # and let the normal _resume_experiment method handle it
                
                # Configure our mock to return a valid state
                mock_match_state_manager.get_latest_state.return_value = MatchStateData(
                    turn=1,
                    current_player_id=1,
                    timestamp=datetime.now(),
                    game_state={"remaining": 9, "current_player": 0},
                    metadata={"test": "data"}
                )
                
                # Directly patch the MatchStateManager class
                with patch('bgbench.arena.MatchStateManager', return_value=mock_match_state_manager):
                    # Resume the experiment
                    arena = Arena(
                        nim_game, 
                        db_session, 
                        experiment_id=exp.id, 
                        llm_factory=mock_llm_factory
                    )
                    
                    # Check that the resumable matches list contains the incomplete match
                    assert len(arena._resumable_matches) == 1
                
                # Check the structure of the resumable match entry
                resumable_match = arena._resumable_matches[0]
                assert len(resumable_match) == 4  # (player_a, player_b, state, match_id)
                
                # Verify the state is a NimState with 9 remaining
                assert isinstance(resumable_match[2], NimState)
                assert resumable_match[2].remaining == 9
                
                # Verify the match ID matches our incomplete match
                assert resumable_match[3] == incomplete_match.id
    
    @pytest.mark.asyncio
    async def test_prioritize_resumable_matches(
        self, db_session, nim_game, mock_llm_factory, mock_match_state_manager
    ):
        """Test that Arena prioritizes resumable matches over new matches."""
        # Create experiment
        exp = Experiment().create_experiment(db_session, "test-prioritize-resumable")
        
        # Create players
        player1 = DBPlayer(
            name="player1", model_config={"model": "test-model"}, experiment_id=exp.id
        )
        player2 = DBPlayer(
            name="player2", model_config={"model": "test-model"}, experiment_id=exp.id
        )
        db_session.add_all([player1, player2])
        db_session.flush()
        
        # Create an incomplete match
        incomplete_match = GameMatch(
            experiment_id=exp.id,
            player1_id=player1.id,
            player2_id=player2.id,
            winner_id=None,
            complete=False,
        )
        db_session.add(incomplete_match)
        db_session.commit()
        
        # Patch the match_state_manager to return our mock
        with patch('bgbench.arena.MatchStateManager', return_value=mock_match_state_manager):
            # Patch the game's deserialize_state method to return a NimState
            with patch.object(nim_game, 'deserialize_state', return_value=NimState(remaining=9, current_player=0)):
                # Configure our mock to return a valid state
                mock_match_state_manager.get_latest_state.return_value = MatchStateData(
                    turn=1,
                    current_player_id=1,
                    timestamp=datetime.now(),
                    game_state={"remaining": 9, "current_player": 0},
                    metadata={"test": "data"}
                )
                
                # Resume the experiment
                arena = Arena(
                    nim_game, 
                    db_session, 
                    experiment_id=exp.id, 
                    llm_factory=mock_llm_factory
                )
                
                # Manually add a resumable match to test prioritization
                p1 = next((p for p in arena.players if p.player_model.id == player1.id), None)
                p2 = next((p for p in arena.players if p.player_model.id == player2.id), None)
                if p1 and p2:
                    state = NimState(remaining=9, current_player=0)
                    arena._resumable_matches.append((p1, p2, state, incomplete_match.id))
                
                # Mock run_single_game to track calls
                mock_run_single_game = AsyncMock()
                arena.run_single_game = mock_run_single_game
                
                # Mock find_next_available_match to return None to ensure it's not used
                mock_find_match = AsyncMock(return_value=None)
                arena.find_next_available_match = mock_find_match
                
                # Run evaluate_all for a short time
                arena_task = asyncio.create_task(arena.evaluate_all())
                
                # Wait a short time for the task to start scheduling
                await asyncio.sleep(0.1)
                
                # Stop the arena
                arena.handle_sigint()
                arena.handle_sigint()  # Force stop
                
                # Wait for the task to complete
                await arena_task
                
                # Verify run_single_game was called with the resumable match
                mock_run_single_game.assert_called_once()
                
                # Get the call arguments
                call_args = mock_run_single_game.call_args[0]
                call_kwargs = mock_run_single_game.call_args[1]
                
                # Verify the state and match_id were passed correctly
                assert 'resumed_state' in call_kwargs
                assert isinstance(call_kwargs['resumed_state'], NimState)
                assert call_kwargs['resumed_state'].remaining == 9
                
                assert 'existing_match_id' in call_kwargs
                assert call_kwargs['existing_match_id'] == incomplete_match.id
                
                # Verify find_next_available_match was not called
                mock_find_match.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_run_single_game_with_resumed_state(
        self, db_session, nim_game, mock_llm_factory
    ):
        """Test that run_single_game correctly uses the resumed state."""
        # Create experiment
        exp = Experiment().create_experiment(db_session, "test-run-resumed")
        
        # Create players
        player1 = DBPlayer(
            name="player1", model_config={"model": "test-model"}, experiment_id=exp.id
        )
        player2 = DBPlayer(
            name="player2", model_config={"model": "test-model"}, experiment_id=exp.id
        )
        db_session.add_all([player1, player2])
        db_session.flush()
        
        # Create an incomplete match
        incomplete_match = GameMatch(
            experiment_id=exp.id,
            player1_id=player1.id,
            player2_id=player2.id,
            winner_id=None,
            complete=False,
        )
        db_session.add(incomplete_match)
        db_session.commit()
        
        # Create arena
        arena = Arena(
            nim_game, 
            db_session, 
            experiment_id=exp.id, 
            llm_factory=mock_llm_factory
        )
        
        # Create arena players
        player_a = arena.players[0]
        player_b = arena.players[1]
        
        # Create a test state
        test_state = NimState(remaining=9, current_player=0)
        
        # Mock MatchRunner to verify it receives the correct state
        with patch('bgbench.arena.MatchRunner') as mock_runner_class:
            # Set up the mock runner
            mock_runner = MagicMock()
            mock_runner.play_game = AsyncMock(return_value=(None, [], None))
            mock_runner_class.return_value = mock_runner
            
            # Run a single game with the resumed state
            await arena.run_single_game(
                player_a, 
                player_b, 
                resumed_state=test_state, 
                existing_match_id=incomplete_match.id
            )
            
            # Verify MatchRunner was created with the correct parameters
            mock_runner_class.assert_called_once()
            _, kwargs = mock_runner_class.call_args
            
            # Check that initial_state was passed correctly
            assert 'initial_state' in kwargs
            assert kwargs['initial_state'] == test_state
            
            # Check that match_state_manager was passed
            assert 'match_state_manager' in kwargs
            assert kwargs['match_state_manager'] == arena.match_state_manager
            
            # Check that game_id matches the existing match ID
            assert 'game_id' in kwargs
            assert kwargs['game_id'] == incomplete_match.id
