import os
import argparse
import logging
from dotenv import load_dotenv
from bgbench.logging_config import setup_logging
import importlib
import pkgutil
from bgbench import games
from bgbench.llm_integration import create_llm
from bgbench.llm_player import LLMPlayer
from bgbench.arena import Arena

logger = logging.getLogger("bgbench")

load_dotenv()

async def main():
    # Dynamically find available games
    game_modules = {
        name: importlib.import_module(f"bgbench.games.{name}")
        for _, name, _ in pkgutil.iter_modules(games.__path__)
    }
    game_names = list(game_modules.keys())

    parser = argparse.ArgumentParser(description='Run a game between LLM players')
    parser.add_argument('--game', choices=game_names, required=True, help='The game to play')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    setup_logging(debug=args.debug)
    
    # Create multiple LLM players
    players = [
        # LLMPlayer("claude-3-haiku", create_llm("claude-3-haiku", temperature=0.0)),
        # LLMPlayer("gpt-4o-mini", create_llm("gpt-4o-mini", temperature=0.0)),
        LLMPlayer("claude-3.5-sonnet", create_llm("claude-3.5-sonnet", temperature=0.0)),
        LLMPlayer("gpt-4o", create_llm("gpt-4o", temperature=0.0)),
    ]
    
    # Instantiate the selected game
    game_module = game_modules[args.game]
    game_class_name = ''.join(word.capitalize() for word in args.game.split('_'))
    game_class = getattr(game_module, game_class_name)
    game = game_class()
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
