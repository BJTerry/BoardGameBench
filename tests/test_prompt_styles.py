import pytest
from bgbench.game_view import GameView, PromptStyle, PromptRenderer

@pytest.mark.parametrize("style,expected", [
    (PromptStyle.XML, 
     "<rules>\nTest rules\n</rules>\n<state>\n{'test': 'state'}\n</state>\n<move_format>\nTest format\n</move_format>"),
    (PromptStyle.HEADER,
     "RULES:\nTest rules\n\nSTATE:\n{'test': 'state'}\n\nMOVE FORMAT:\nTest format"),
    (PromptStyle.JSON,
     '{"rules": "Test rules", "state": "{\'test\': \'state\'}", "move_format": "Test format"}')
])
def test_prompt_renderer(style, expected):
    """Test PromptRenderer with different styles"""
    result = PromptRenderer.render(
        style,
        "Test rules",
        "{'test': 'state'}",
        "Test format"
    )
    assert result == expected

def test_prompt_renderer_invalid_style():
    """Test PromptRenderer with invalid style"""
    with pytest.raises(ValueError):
        PromptRenderer.render(
            "invalid",  # type: ignore
            "Test rules",
            "{'test': 'state'}",
            "Test format"
        )

@pytest.mark.parametrize("style", [
    PromptStyle.XML,
    PromptStyle.HEADER,
    PromptStyle.JSON
])
def test_game_view_format_prompt(style):
    """Test GameView's format_prompt with different styles"""
    game_view = GameView(
        visible_state={"test": "state"},
        valid_moves=[],
        is_terminal=False,
        rules_explanation="Test rules",
        move_format_instructions="Test format",
        prompt_style=style
    )
    
    result = game_view.format_prompt()
    assert "Test rules" in result
    if style == PromptStyle.XML:
        assert "<test>state</test>" in result
    elif style == PromptStyle.HEADER:
        assert "TEST\nstate" in result
    elif style == PromptStyle.JSON:
        assert '"test": "state"' in result or "'test': 'state'" in result
    assert "Test format" in result

def test_game_view_default_prompt_style():
    """Test that GameView defaults to HEADER style"""
    game_view = GameView(
        visible_state={},
        valid_moves=[],
        is_terminal=False
    )
    assert game_view.prompt_style == PromptStyle.HEADER
