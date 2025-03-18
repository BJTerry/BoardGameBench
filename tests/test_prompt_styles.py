import pytest
from bgbench.game_view import GameView, PromptStyle, PromptRenderer

@pytest.mark.parametrize("style,expected", [
    (PromptStyle.XML, "<rules>\nTest rules\n</rules>"),
    (PromptStyle.HEADER, "RULES:\nTest rules"),
    (PromptStyle.JSON, '{"rules": "Test rules"}')
])
def test_render_rules(style, expected):
    """Test rules rendering with different styles"""
    result = PromptRenderer.render_rules(style, "Test rules")
    assert result == expected

@pytest.mark.parametrize("style,expected", [
    (PromptStyle.XML, "<game_state>\n{'test': 'state'}\n</game_state>"),
    (PromptStyle.HEADER, "GAME STATE:\n{'test': 'state'}"),
    (PromptStyle.JSON, '{"game_state": "{\'test\': \'state\'}"}')
])
def test_render_game_state(style, expected):
    """Test game state rendering with different styles"""
    result = PromptRenderer.render_game_state(style, "{'test': 'state'}")
    assert result == expected

@pytest.mark.parametrize("style,expected", [
    (PromptStyle.XML, "<move_format>\nTest format\n</move_format>"),
    (PromptStyle.HEADER, "MOVE FORMAT:\nTest format"),
    (PromptStyle.JSON, '{"move_format": "Test format"}')
])
def test_render_move_format(style, expected):
    """Test move format rendering with different styles"""
    result = PromptRenderer.render_move_format(style, "Test format")
    assert result == expected

def test_prompt_renderer_invalid_style():
    """Test PromptRenderer with invalid style"""
    with pytest.raises(ValueError):
        PromptRenderer.render_rules("invalid", "Test rules")  # type: ignore

@pytest.mark.parametrize("style", [
    PromptStyle.XML,
    PromptStyle.HEADER,
    PromptStyle.JSON
])
def test_game_view_format_prompt(style: PromptStyle):
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
    
    # Find the rules message - should have the rendered version
    if style == PromptStyle.XML:
        expected_rules = "<rules>\nTest rules\n</rules>"
    elif style == PromptStyle.HEADER:
        expected_rules = "RULES:\nTest rules"
    elif style == PromptStyle.JSON:
        expected_rules = '{"rules": "Test rules"}'
    
    rules_msg = next((msg for msg in result 
                      if msg["role"] == "user" and 
                      any(content.get("text") == expected_rules for content in msg["content"])), 
                    None)
    assert rules_msg is not None, f"Rules message not found for style {style}"
    
    # Find the state message - it's the third one, after rules and move format
    # Messages are ordered: rules, move_format, state
    if len(result) >= 3:
        state_msg = result[2]  # Get the third message which should be the state
        state_content = state_msg["content"][0]["text"] if state_msg and state_msg["content"] else ""
    else:
        state_content = ""
    
    if style == PromptStyle.XML:
        assert "<game_state>" in state_content
        assert "<test>state</test>" in state_content
    elif style == PromptStyle.HEADER:
        assert "GAME STATE:" in state_content
        assert "TEST\nstate" in state_content
    elif style == PromptStyle.JSON:
        assert '{"game_state":' in state_content
        assert '"test": "state"' in state_content or "'test': 'state'" in state_content
    
    # Find the move format message
    if style == PromptStyle.XML:
        expected_format = "<move_format>\nTest format\n</move_format>"
    elif style == PromptStyle.HEADER:
        expected_format = "MOVE FORMAT:\nTest format"
    elif style == PromptStyle.JSON:
        expected_format = '{"move_format": "Test format"}'
    
    format_msg = next((msg for msg in result 
                      if msg["role"] == "user" and 
                      any(content.get("text") == expected_format for content in msg["content"])),
                     None)
    assert format_msg is not None, f"Move format message not found for style {style}"

def test_game_view_default_prompt_style():
    """Test that GameView defaults to HEADER style"""
    game_view = GameView(
        visible_state={},
        valid_moves=[],
        is_terminal=False
    )
    assert game_view.prompt_style == PromptStyle.HEADER
