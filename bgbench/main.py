import os
import argparse
from dotenv import load_dotenv
from bgbench.logging_config import setup_logging
from bgbench.nim_game import NimGame
from bgbench.llm_integration import create_llm
from bgbench.utils import LLMPlayer, GameRunner

load_dotenv()

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run a game between LLM players')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # Set up logging with debug flag
    setup_logging(debug=args.debug)
    
    # Initialize LLMs
    claude_llm = create_llm("claude-3-haiku", temperature=0.0)
    gpt4_llm = create_llm("gpt-4o", temperature=0.0)
    
    # Create players
    player_a = LLMPlayer("Claude-3-Sonnet", claude_llm)
    player_b = LLMPlayer("GPT-4-Turbo", gpt4_llm)
    
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

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
