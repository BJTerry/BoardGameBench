import os
from dotenv import load_dotenv
from bgbench.nim_game import NimGame
from bgbench.llm_integration import LLMConfig, LLMProvider, AnthropicLLM, OpenAILLM
from bgbench.utils import LLMPlayer, GameRunner

load_dotenv()

async def main():
    # Initialize LLMs
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
        
    gpt4_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        api_key=api_key,
        model="gpt-4o-mini",
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
