import os
import argparse
import logging
from dotenv import load_dotenv
from bgbench.logging_config import setup_logging
from bgbench.games.nim_game import NimGame
from bgbench.llm_integration import create_llm
from bgbench.llm_player import LLMPlayer
from bgbench.arena import Arena

logger = logging.getLogger("bgbench")

load_dotenv()

async def main():
    parser = argparse.ArgumentParser(description='Run a game between LLM players')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    setup_logging(debug=args.debug)
    
    # Create multiple LLM players
    players = [
        LLMPlayer("claude-3-haiku", create_llm("claude-3-haiku", temperature=0.0)),
        LLMPlayer("gpt-4o-mini", create_llm("gpt-4o-mini", temperature=0.0)),
        LLMPlayer("claude-3-opus", create_llm("claude-3-opus", temperature=0.0)),
        LLMPlayer("gpt-4", create_llm("gpt-4", temperature=0.0)),
    ]
    
    game = NimGame(12, 3)
    arena = Arena(game, confidence_threshold=0.70)
    
    for player in players:
        arena.add_player(player)
    
    final_ratings = await arena.evaluate_all()
    
    # Print final standings
    logger.info("\nFinal Ratings:")
    for name, rating in sorted(final_ratings.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"{name}: {rating:.0f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
