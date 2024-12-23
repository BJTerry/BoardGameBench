from bgbench.nim_game import NimGame
from bgbench.llm_integration import LLMConfig, LLMProvider, AnthropicLLM, OpenAILLM
from bgbench.utils import LLMPlayer, GameRunner

async def main():
    # Initialize LLMs
    claude_config = LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        api_key="your_anthropic_key",
        model="claude-3-opus-20240229",
        temperature=0.0
    )
    claude_llm = AnthropicLLM(claude_config)
    
    gpt4_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        api_key="your_openai_key",
        model="gpt-4-turbo-preview",
        temperature=0.0
    )
    gpt4_llm = OpenAILLM(gpt4_config)
    
    # Create players
    claude_player = LLMPlayer("Claude", claude_llm)
    gpt4_player = LLMPlayer("GPT-4", gpt4_llm)
    
    # Initialize game
    game = NimGame(12, 3)
    
    # Play a single game
    runner = GameRunner(game, claude_player, gpt4_player)
    winner, history = await runner.play_game()
    
    print(f"Winner: {'Claude' if winner == 0 else 'GPT-4'}")
    print("Game history:")
    for move in history:
        print(f"Player {move['player']}: took {move['move']} objects, "
              f"{move['state_before']['remaining']} remaining")
        
    # Access conversation history
    print("\nClaude's thought process:")
    for msg in claude_player.conversation_history:
        if msg["role"] != "system":
            print(f"{msg['role']}: {msg['content']}\n")

def main():
    print("Welcome to bgbench!")
    game = NimGame()
    print(game.get_rules_explanation())

if __name__ == "__main__":
    main()
