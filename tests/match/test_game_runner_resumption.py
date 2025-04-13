import pytest
from unittest.mock import MagicMock, patch
from bgbench.match.runner import MatchRunner
from bgbench.match.state_manager import MatchStateManager
from bgbench.game import Game


class TestMatchRunnerResumption:
    def test_game_runner_uses_initial_state(self):
        """Test that MatchRunner uses initial_state when provided."""
        # Create mocks
        mock_game = MagicMock(spec=Game)
        mock_player1 = MagicMock()
        mock_player2 = MagicMock()
        mock_session = MagicMock()
        
        # Create initial state
        initial_state = {"test": "state"}
        
        # Create MatchRunner with initial state
        runner = MatchRunner(
            game=mock_game,
            player1=mock_player1,
            player2=mock_player2,
            db_session=mock_session,
            game_id=1,
            player1_id=1,
            player2_id=2,
            initial_state=initial_state
        )
        
        # Patch play_game to avoid actually running it
        with patch.object(runner, 'play_game') as mock_play_game:
            # Call play_game (this will be mocked)
            mock_play_game.return_value = (None, [], None)
            # Use asyncio.run to handle the coroutine
            import asyncio
            asyncio.run(runner.play_game())
            
        # Verify get_initial_state was not called
        mock_game.get_initial_state.assert_not_called()
        
        # Verify initial_state is stored correctly
        assert runner.initial_state == initial_state
    
    def test_game_runner_uses_state_manager(self):
        """Test that MatchRunner uses MatchStateManager when provided."""
        # Create mocks
        mock_game = MagicMock(spec=Game)
        mock_player1 = MagicMock()
        mock_player2 = MagicMock()
        mock_session = MagicMock()
        mock_state_manager = MagicMock(spec=MatchStateManager)
        
        # Create MatchRunner with state manager
        runner = MatchRunner(
            game=mock_game,
            player1=mock_player1,
            player2=mock_player2,
            db_session=mock_session,
            game_id=1,
            player1_id=1,
            player2_id=2,
            match_state_manager=mock_state_manager
        )
        
        # Patch play_game to avoid actually running it
        with patch.object(runner, 'play_game') as mock_play_game:
            # Call play_game (this will be mocked)
            mock_play_game.return_value = (None, [], None)
            # Use asyncio.run to handle the coroutine
            import asyncio
            asyncio.run(runner.play_game())
            
        # Verify state manager is stored correctly
        assert runner.match_state_manager == mock_state_manager
    
    def test_game_runner_without_state_manager(self):
        """Test that MatchRunner works correctly without a state manager."""
        # Create mocks
        mock_game = MagicMock(spec=Game)
        mock_player1 = MagicMock()
        mock_player2 = MagicMock()
        mock_session = MagicMock()
        
        # Create MatchRunner without state manager
        runner = MatchRunner(
            game=mock_game,
            player1=mock_player1,
            player2=mock_player2,
            db_session=mock_session,
            game_id=1,
            player1_id=1,
            player2_id=2
        )
        
        # Patch play_game to avoid actually running it
        with patch.object(runner, 'play_game') as mock_play_game:
            # Call play_game (this will be mocked)
            mock_play_game.return_value = (None, [], None)
            # Use asyncio.run to handle the coroutine
            import asyncio
            asyncio.run(runner.play_game())
            
        # Verify match_state_manager is None
        assert runner.match_state_manager is None
    
    def test_play_game_uses_initial_state(self):
        """Test that play_game actually uses the initial state when provided."""
        # Create mocks
        mock_game = MagicMock(spec=Game)
        mock_player1 = MagicMock()
        mock_player2 = MagicMock()
        mock_session = MagicMock()
        
        # Create initial state
        initial_state = {"test": "state"}
        
        # Set up mock game to return a terminal state
        mock_game.is_terminal.return_value = True
        mock_game.get_winner.return_value = 0  # Player 1 wins
        
        # Create MatchRunner with initial state
        runner = MatchRunner(
            game=mock_game,
            player1=mock_player1,
            player2=mock_player2,
            db_session=mock_session,
            game_id=1,
            player1_id=1,
            player2_id=2,
            initial_state=initial_state
        )
        
        # Run play_game (will use the real method this time)
        import asyncio
        winner, history, concession = asyncio.run(runner.play_game())
        
        # Verify get_initial_state was not called
        mock_game.get_initial_state.assert_not_called()
        
        # Verify is_terminal was called with the initial state
        mock_game.is_terminal.assert_called_with(initial_state)
        
        # Verify get_winner was called with the initial state
        mock_game.get_winner.assert_called_with(initial_state)
        
        # Verify the winner is player 1
        assert winner == mock_player1
