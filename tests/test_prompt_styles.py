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
    
    # Find the rules message
    rules_msg = next((msg for msg in result 
                      if msg["role"] == "user" and 
                      any(content.get("text") == "Test rules" for content in msg["content"])), 
                    None)
    assert rules_msg is not None, "Rules message not found"
    
    # Find the state message - it's the third one, after rules and move format
    # Messages are ordered: rules, move_format, state
    if len(result) >= 3:
        state_msg = result[2]  # Get the third message which should be the state
        state_content = state_msg["content"][0]["text"] if state_msg and state_msg["content"] else ""
    else:
        state_content = ""
    
    if style == PromptStyle.XML:
        assert "<test>state</test>" in state_content
    elif style == PromptStyle.HEADER:
        assert "TEST\nstate" in state_content
    elif style == PromptStyle.JSON:
        assert '"test": "state"' in state_content or "'test': 'state'" in state_content
    
    # Find the move format message
    format_msg = next((msg for msg in result 
                      if msg["role"] == "user" and 
                      any(content.get("text") == "Test format" for content in msg["content"])),
                     None)
    assert format_msg is not None, "Move format message not found"

def test_game_view_default_prompt_style():
    """Test that GameView defaults to HEADER style"""
    game_view = GameView(
        visible_state={},
        valid_moves=[],
        is_terminal=False
    )
    assert game_view.prompt_style == PromptStyle.HEADER
