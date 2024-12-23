from bgbench.nim_game import NimGame
from bgbench.llm_integration import LLMConfig, LLMProvider, AnthropicLLM, OpenAILLM
from bgbench.utils import LLMPlayer, GameRunner

async def main():
    # Initialize LLMs
    gpt4_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        api_key="your_openai_key",
        model="openai:gpt-4-turbo-preview",
        temperature=0.0
    )
    gpt4_llm = OpenAILLM(gpt4_config)
    
    # Create players
    player_a = LLMPlayer("Claude", gpt4_llm)
    player_b = LLMPlayer("GPT-4", gpt4_llm)
    
    # Initialize game
    game = NimGame(12, 3)
    
    # Play a single game
    runner = GameRunner(game, player_a, player_b)
    winner, history = await runner.play_game()
    
    print(f"Winner: {'Claude' if winner == 0 else 'GPT-4'}")
    print("Game history:")
    for move in history:
        print(f"Player {move['player']}: took {move['move']} objects, "
              f"{move['state_before']['remaining']} remaining")
        
    # Access conversation history
    print("\nClaude's thought process:")
    for msg in player_a.conversation_history:
        if msg["role"] != "system":
            print(f"{msg['role']}: {msg['content']}\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
