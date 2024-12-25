import pytest
from bgbench.llm_player import LLMPlayer
from bgbench.game_view import GameView
from bgbench.games.nim_game import NimState, NimMove
from bgbench.games.war_game import WarState, Card
from typing import Dict, Any

@pytest.mark.asyncio
async def test_llm_player_make_move_nim(test_llm):
    """Test LLMPlayer with a Nim game scenario"""
    player = LLMPlayer("test_player", test_llm)
    
    # Create a realistic Nim game view
    state = {"remaining": 10}
    game_view = GameView(
        visible_state=state,
        valid_moves=["1", "2", "3"],
        is_terminal=False,
        winner=None,
        history=[],
        move_format_instructions="Enter a number between 1 and 3",
        rules_explanation="Take 1-3 objects. Player who takes the last object wins.",
        error_message=None
    )
    
    # Set expected response
    test_llm.model.custom_result_text = "2"
    
    move = await player.make_move(game_view)
    assert move == "2"
    assert len(player.conversation_history) == 1
    assert "2" in player.conversation_history[0]["content"]

@pytest.mark.asyncio
async def test_llm_player_make_move_war(test_llm):
    """Test LLMPlayer with a War game scenario"""
    player = LLMPlayer("test_player", test_llm)
    
    # Create a realistic War game view
    state = {
        "your_cards": 26,
        "opponent_cards": 26,
        "board": [],
        "war_state": False,
        "cards_needed": 1,
        "face_down_count": 0
    }
    
    game_view = GameView(
        visible_state=state,
        valid_moves=["play"],
        is_terminal=False,
        winner=None,
        history=[],
        move_format_instructions="Type 'play' to play your next card",
        rules_explanation="Standard War card game rules...",
        error_message=None
    )
    
    test_llm.model.custom_result_text = "play"
    
    move = await player.make_move(game_view)
    assert move == "play"

@pytest.mark.asyncio
async def test_llm_player_invalid_move_retry_with_context(test_llm):
    """Test LLMPlayer's handling of invalid moves with context"""
    player = LLMPlayer("test_player", test_llm)
    
    game_view = GameView(
        visible_state={"remaining": 5},
        valid_moves=["1", "2", "3"],
        is_terminal=False,
        winner=None,
        history=[],
        move_format_instructions="Enter a number between 1 and 3",
        rules_explanation="Take 1-3 objects",
        error_message=None
    )
    
    # First move - invalid
    test_llm.model.custom_result_text = "4"
    first_move = await player.make_move(game_view)
    
    # Retry with error context
    test_llm.model.custom_result_text = "2"
    retry_move = await player.make_move(
        game_view,
        invalid_move_explanation="Cannot take 4 objects, maximum is 3"
    )
    
    assert retry_move == "2"
    assert len(player.conversation_history) == 2

@pytest.mark.asyncio
async def test_llm_player_db_logging(test_llm, mocker):
    """Test database logging functionality of LLMPlayer"""
    # Mock database session
    mock_session = mocker.MagicMock()
    mock_interaction = mocker.MagicMock()
    mocker.patch('bgbench.models.LLMInteraction', return_value=mock_interaction)
    
    player = LLMPlayer("test_player", test_llm, db_session=mock_session, game_id=1)
    
    game_view = GameView(
        visible_state={"remaining": 5},
        valid_moves=["1", "2", "3"],
        is_terminal=False,
        winner=None,
        history=[],
        move_format_instructions="Enter a number between 1 and 3",
        rules_explanation="Take 1-3 objects",
        error_message=None
    )
    
    test_llm.model.custom_result_text = "2"
    await player.make_move(game_view)
    
    # Verify logging occurred
    mock_interaction.log_interaction.assert_called_once()
    args = mock_interaction.log_interaction.call_args[0]
    assert args[0] == mock_session
    assert isinstance(args[1], dict)  # prompt dict
    assert args[2] == "2"  # response

@pytest.mark.asyncio
async def test_llm_player_system_prompt_consistency(test_llm, capture_messages):
    """Test that system prompts are consistently included"""
    player = LLMPlayer("test_player", test_llm)
    
    game_view = GameView(
        visible_state={"remaining": 5},
        valid_moves=["1", "2", "3"],
        is_terminal=False,
        winner=None,
        history=[],
        move_format_instructions="Enter a number between 1 and 3",
        rules_explanation="Take 1-3 objects",
        error_message=None
    )
    
    test_llm.model.custom_result_text = "2"
    await player.make_move(game_view)
    
    # Verify system prompt is present and correct
    system_prompts = [
        part.content for msg in capture_messages 
        for part in msg.parts 
        if "You are playing a game" in part.content
    ]
    assert len(system_prompts) == 1
    assert "respond with only your move" in system_prompts[0].lower()
