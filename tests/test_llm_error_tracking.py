import pytest
import asyncio
from unittest.mock import MagicMock, patch
import json
from sqlalchemy.orm import Session
from bgbench.llm_player import LLMPlayer
from bgbench.match.view import MatchView, PromptStyle
from bgbench.models import LLMInteraction, Player, GameMatch
from bgbench.llm_integration import (
    complete_prompt, 
    RateLimitError,
    LLMResponseError,
    EmptyChoicesError
)
from litellm.types.utils import ModelResponse, Choices, Message


class TestLLMErrorTracking:
    """Tests for LLM error tracking functionality."""

    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        session = MagicMock(spec=Session)
        return session

    @pytest.fixture
    def mock_match_view(self):
        """Create a mock match view."""
        view = MagicMock(spec=MatchView)
        view.format_prompt.return_value = [
            {"role": "user", "content": [{"type": "text", "text": "Test prompt"}]}
        ]
        return view

    @pytest.fixture
    def mock_llm_player(self, mock_db_session):
        """Create a mock LLM player with database session."""
        player = LLMPlayer(
            name="TestPlayer",
            model_config={"model": "test-model"},
            prompt_style=PromptStyle.HEADER,
            db_session=mock_db_session,
            game_id=1,
            player_id=1
        )
        return player

    @pytest.fixture
    def mock_model_response(self):
        """Create a mock model response."""
        response = MagicMock(spec=ModelResponse)
        choice = MagicMock(spec=Choices)
        message = MagicMock(spec=Message)
        message.content = "Test response"
        choice.message = message
        response.choices = [choice]
        response.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        response.model_params = {"max_tokens": 100}
        return response

    @pytest.mark.asyncio
    async def test_token_limit_warning_tracking(self, mock_llm_player, mock_match_view, mock_db_session):
        """Test that token limit warnings are tracked correctly."""
        # Create a response that maxes out tokens
        response = MagicMock(spec=ModelResponse)
        choice = MagicMock(spec=Choices)
        message = MagicMock(spec=Message)
        message.content = "Test response"
        choice.message = message
        response.choices = [choice]
        response.usage = {"prompt_tokens": 10, "completion_tokens": 100, "total_tokens": 110}
        response.model_params = {"max_tokens": 100}
        response.model = "test-model"
        
        # Mock the complete_prompt function to return our response with token limit warning
        with patch('bgbench.llm_player.complete_prompt', autospec=True) as mock_complete:
            # Return response, token info, messages, and error info
            mock_complete.return_value = (
                "Test response",
                {"prompt_tokens": 10, "completion_tokens": 100, "total_tokens": 110, "cost": 0.01},
                [{"role": "user", "content": "Test prompt"}],
                {
                    "error_occurred": True,
                    "error_type": "TOKEN_LIMIT_WARNING",
                    "error_message": "Response maxed out completion tokens: used 100 of 100 for test-model",
                    "error_details": {
                        "completion_tokens": 100,
                        "max_tokens": 100,
                        "model": "test-model"
                    },
                    "retry_count": 0
                }
            )
            
            # Call make_move
            await mock_llm_player.make_move(mock_match_view, None)
            
            # Check that log_interaction was called with error info
            mock_db_session.add.assert_called_once()
            mock_db_session.commit.assert_called_once()
            
            # Get the LLMInteraction object that was created
            llm_interaction = mock_db_session.add.call_args[0][0]
            assert isinstance(llm_interaction, LLMInteraction)
            assert llm_interaction.error_occurred is True
            assert llm_interaction.error_type == "TOKEN_LIMIT_WARNING"
            assert llm_interaction.error_message is not None
            assert "maxed out completion tokens" in llm_interaction.error_message
            assert llm_interaction.error_details is not None
            assert "completion_tokens" in llm_interaction.error_details
            assert "max_tokens" in llm_interaction.error_details
            assert llm_interaction.error_details["completion_tokens"] == 100
            assert llm_interaction.error_details["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_exception_tracking(self, mock_llm_player, mock_match_view, mock_db_session):
        """Test that exceptions during LLM calls are tracked correctly."""
        # Mock the complete_prompt function to raise an exception
        with patch('bgbench.llm_player.complete_prompt', autospec=True) as mock_complete:
            mock_complete.side_effect = Exception("Test exception")
            
            # Call make_move and expect it to raise the exception
            with pytest.raises(Exception, match="Test exception"):
                await mock_llm_player.make_move(mock_match_view, None)
            
            # Check that log_interaction was called with error info
            mock_db_session.add.assert_called_once()
            mock_db_session.commit.assert_called_once()
            
            # Get the LLMInteraction object that was created
            llm_interaction = mock_db_session.add.call_args[0][0]
            assert isinstance(llm_interaction, LLMInteraction)
            assert llm_interaction.error_occurred is True
            assert llm_interaction.error_type == "EXCEPTION"
            assert llm_interaction.error_message == "Test exception"
            assert llm_interaction.error_details is not None
            assert "exception_class" in llm_interaction.error_details
            assert llm_interaction.error_details["exception_class"] == "Exception"
            assert llm_interaction.response == "[ERROR]"

    @pytest.mark.asyncio
    async def test_complete_prompt_error_info(self):
        """Test that complete_prompt returns proper error info for token limit warnings."""
        # Create a mock response with max tokens used
        response = MagicMock(spec=ModelResponse)
        choice = MagicMock(spec=Choices)
        message = MagicMock(spec=Message)
        message.content = "Test response"
        choice.message = message
        response.choices = [choice]
        response.usage = {"prompt_tokens": 10, "completion_tokens": 100, "total_tokens": 110}
        response.model_params = {"max_tokens": 100}
        response.model = "test-model"
        
        # Mock litellm.acompletion to return our response
        with patch('litellm.acompletion', autospec=True) as mock_acompletion:
            mock_acompletion.return_value = response
            
            # Call complete_prompt
            llm_config = {"model": "test-model", "messages": []}
            prompt_messages = [{"role": "user", "content": [{"type": "text", "text": "Test prompt"}]}]
            
            content, token_info, messages, error_info = await complete_prompt(llm_config, prompt_messages)
            
            # Check the error info
            assert error_info is not None
            assert error_info.get("error_occurred") is True
            assert error_info.get("error_type") == "TOKEN_LIMIT_WARNING"
            
            # Check error message with proper type checking
            error_message = error_info.get("error_message")
            assert error_message is not None
            assert isinstance(error_message, str)
            assert "maxed out completion tokens" in error_message
            
            # Check error details with proper type checking
            error_details = error_info.get("error_details")
            assert error_details is not None
            assert isinstance(error_details, dict)
            completion_tokens = error_details.get("completion_tokens")
            assert completion_tokens is not None
            assert completion_tokens == 100
            max_tokens = error_details.get("max_tokens")
            assert max_tokens is not None
            assert max_tokens == 100
            model = error_details.get("model")
            assert model is not None
            assert model == "test-model"

    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(self):
        """Test that rate limit errors are properly handled and would be tracked."""
        # Mock litellm.acompletion to raise RateLimitError first, then succeed
        with patch('litellm.acompletion', autospec=True) as mock_acompletion, \
             patch('asyncio.sleep', return_value=None) as mock_sleep:
            
            # First call raises RateLimitError, second call succeeds
            mock_acompletion.side_effect = [
                RateLimitError("Rate limit exceeded", llm_provider="openai", model="test-model"),
                MagicMock(spec=ModelResponse, 
                          choices=[MagicMock(spec=Choices, message=MagicMock(spec=Message, content="Success"))],
                          usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
            ]
            
            # Call complete_prompt
            llm_config = {"model": "test-model", "messages": []}
            prompt_messages = [{"role": "user", "content": [{"type": "text", "text": "Test prompt"}]}]
            
            # This should succeed after one retry
            content, token_info, messages, error_info = await complete_prompt(llm_config, prompt_messages)
            
            # Verify sleep was called (for backoff)
            mock_sleep.assert_called_once()
            
            # Check the result
            assert content == "Success"
            # No error info for successful completion after retry
            # Use a more explicit check that doesn't trigger type errors
            assert error_info is None or not error_info.get("error_occurred", False)

    @pytest.mark.asyncio
    async def test_llm_response_error_handling(self):
        """Test that LLM response errors are properly handled and would be tracked."""
        # Mock _execute_completion to return a response, but _process_response to raise an error then succeed
        with patch('bgbench.llm_integration._execute_completion', autospec=True) as mock_execute, \
             patch('bgbench.llm_integration._process_response', autospec=True) as mock_process, \
             patch('asyncio.sleep', return_value=None) as mock_sleep:
            
            # _execute_completion always returns a mock response
            mock_execute.return_value = "mock_response"
            
            # _process_response raises EmptyChoicesError first, then succeeds
            mock_process.side_effect = [
                EmptyChoicesError("Empty choices in response"),
                ("Success", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01})
            ]
            
            # Call complete_prompt
            llm_config = {"model": "test-model", "messages": []}
            prompt_messages = [{"role": "user", "content": [{"type": "text", "text": "Test prompt"}]}]
            
            # This should succeed after one retry
            content, token_info, messages, error_info = await complete_prompt(llm_config, prompt_messages)
            
            # Verify sleep was called (for backoff)
            mock_sleep.assert_called_once()
            
            # Check the result
            assert content == "Success"
            # No error info for successful completion after retry
            # Use a more explicit check that doesn't trigger type errors
            assert error_info is None or not error_info.get("error_occurred", False)
