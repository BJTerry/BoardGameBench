import os
import argparse
import logging
from dotenv import load_dotenv
from bgbench.logging_config import setup_logging
from bgbench.nim_game import NimGame
from bgbench.llm_integration import create_llm
from bgbench.utils import LLMPlayer, GameRunner
from bgbench.rating import PlayerRating, RatingAnalyzer

logger = logging.getLogger("bgbench")

load_dotenv()

async def main():
    parser = argparse.ArgumentParser(description='Run a game between LLM players')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    setup_logging(debug=args.debug)
    
    # Initialize LLMs and players
    claude_llm = create_llm("claude-3-haiku", temperature=0.0)
    gpt4_llm = create_llm("gpt-4o-mini", temperature=0.0)
    
    player_a = LLMPlayer("claude-3-haiku", claude_llm)
    player_b = LLMPlayer("gpt-4o-mini", gpt4_llm)
    
    # Initialize rating system
    analyzer = RatingAnalyzer(confidence_threshold=0.95)
    rating_a = PlayerRating(name=player_a.name, rating=1500, games_played=0)
    rating_b = PlayerRating(name=player_b.name, rating=1500, games_played=0)
    
    game = NimGame(12, 3)
    game_count = 0
    
    while True:
        game_count += 1
        logger.info(f"\nStarting game {game_count}")
        
        # Play a game
        runner = GameRunner(game, player_a, player_b)
        winner, history = await runner.play_game()
        
        # Update ratings
        rating_a, rating_b = analyzer.elo_system.update_ratings(
            rating_a, rating_b, winner.name == player_a.name
        )
        
        logger.info(f"Game {game_count} winner: {winner.name}")
        logger.info(f"Current ratings: {rating_a.name}: {rating_a.rating:.0f}, "
                   f"{rating_b.name}: {rating_b.rating:.0f}")
        
        # Check if we have reached confidence threshold
        complete, explanation = analyzer.is_evaluation_complete(rating_a, rating_b)
        logger.info(explanation)
        
        if complete:
            break

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
