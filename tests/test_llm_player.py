import pytest
from bgbench.llm_player import LLMPlayer
from bgbench.game_view import GameView
from bgbench.models import LLMInteraction, Experiment, Player, GameMatch

@pytest.mark.asyncio
async def test_llm_player_basic_move(test_llm, db_session):
    """Test basic move generation and response handling"""
    # Set up test environment
    experiment = Experiment().create_experiment(db_session, "Test Experiment")
    player = Player(name="test_player", model_config={"model": "test"}, experiment_id=experiment.id)
    db_session.add(player)
    game = GameMatch(experiment_id=experiment.id, player1_id=1, player2_id=2)
    db_session.add(game)
    db_session.commit()

    llm_player = LLMPlayer("test_player", {"model": "test"}, db_session=db_session, game_id=game.id, _llm=test_llm)
    
    game_view = GameView(
        visible_state={"remaining": 10},
        valid_moves=["1", "2", "3"],
        is_terminal=False,
        winner=None,
        history=[],
        move_format_instructions="Enter a number between 1 and 3",
        rules_explanation="Take 1-3 objects. Player who takes the last object wins.",
        error_message=None
    )
    
    # Test move generation
    test_llm.set_response("2")
    move = await llm_player.make_move(game_view)
    
    # Verify response
    assert move == "2"
    assert len(llm_player.conversation_history) == 1
    assert "2" in llm_player.conversation_history[0]["content"]
    
    # Verify prompt content
    assert "Take 1-3 objects" in test_llm.last_prompt
    assert "Enter a number between 1 and 3" in test_llm.last_prompt

@pytest.mark.asyncio
async def test_llm_player_invalid_move_retry(test_llm, db_session):
    """Test handling of invalid moves with context"""
    experiment = Experiment().create_experiment(db_session, "Test Experiment")
    player = Player(name="test_player", model_config={"model": "test"}, experiment_id=experiment.id)
    db_session.add(player)
    game = GameMatch(experiment_id=experiment.id, player1_id=1, player2_id=2)
    db_session.add(game)
    db_session.commit()

    llm_player = LLMPlayer("test_player", {"model": "test"}, db_session=db_session, game_id=game.id, _llm=test_llm)
    
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
    
    # Test retry with invalid move context
    test_llm.set_response("2")
    move = await llm_player.make_move(
        game_view,
        invalid_moves=[{
            "move": "4",
            "explanation": "Cannot take 4 objects, maximum is 3"
        }]
    )
    
    assert move == "2"
    assert "Cannot take 4 objects" in test_llm.last_prompt

@pytest.mark.asyncio
async def test_llm_player_db_logging(test_llm, mocker):
    """Test database logging of LLM interactions"""
    mock_session = mocker.MagicMock()
    mock_interaction = mocker.MagicMock()
    mocker.patch('bgbench.llm_player.LLMInteraction', return_value=mock_interaction)
    
    llm_player = LLMPlayer("test_player", {"model": "test"}, db_session=mock_session, game_id=1, _llm=test_llm)
    
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
    
    test_llm.set_response("2")
    await llm_player.make_move(game_view)
    
    # Verify logging
    mock_interaction.log_interaction.assert_called_once()
    args = mock_interaction.log_interaction.call_args[0]
    assert args[0] == mock_session
    assert isinstance(args[1], list)
    assert args[2] == "2"
