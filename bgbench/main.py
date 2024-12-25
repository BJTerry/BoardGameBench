import argparse
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL
from bgbench.models import Experiment, Game
from bgbench.logging_config import setup_logging
from bgbench.games import AVAILABLE_GAMES
from bgbench.llm_integration import create_llm
from bgbench.llm_player import LLMPlayer
from bgbench.arena import Arena

logger = logging.getLogger("bgbench")

load_dotenv()

async def main():
    parser = argparse.ArgumentParser(description='Run a game between LLM players')
    parser.add_argument('--game', choices=list(AVAILABLE_GAMES.keys()), required=True, help='The game to play')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    # Experiment management options
    parser.add_argument('--resume', type=int, help='Resume experiment by ID')
    parser.add_argument('--name', help='Name for new experiment')
    parser.add_argument('--export', type=int, help='Export results for experiment ID')
    parser.add_argument('--list', action='store_true', help='List all experiments')
    
    args = parser.parse_args()
    
    setup_logging(debug=args.debug)
    
    # Create multiple LLM players
    players = [
        LLMPlayer("claude-3-haiku", create_llm("claude-3-haiku", temperature=0.0)),
        LLMPlayer("gpt-4o-mini", create_llm("gpt-4o-mini", temperature=0.0)),
        # LLMPlayer("claude-3.5-sonnet", create_llm("claude-3.5-sonnet", temperature=0.0)),
        # LLMPlayer("gpt-4o", create_llm("gpt-4o", temperature=0.0)),
    ]
    
    # Get the game class from our available games
    game_class = AVAILABLE_GAMES[args.game]
    # Set up database session
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    db_session = Session()

    game = game_class()
    
    # Handle experiment management commands
    if args.list:
        experiments = db_session.query(Experiment).all()
        logger.info("\nExisting Experiments:")
        for exp in experiments:
            logger.info(f"ID: {exp.id}, Name: {exp.name}, Description: {exp.description}")
            games = db_session.query(Game).filter_by(experiment_id=exp.id).count()
            logger.info(f"Games played: {games}")
        return

    if args.resume:
        experiment = Experiment.resume_experiment(db_session, args.resume)
        if not experiment:
            logger.error(f"No experiment found with ID {args.resume}")
            return
        logger.info(f"Resuming experiment: {experiment.name}")
        arena = Arena(game, db_session, experiment_id=args.resume)
        for player in players:
            arena.add_player(player)
    elif args.name:
        arena = Arena(game, db_session, experiment_name=args.name)
        for player in players:
            arena.add_player(player)
    else:
        arena = Arena(game, db_session)
        for player in players:
            arena.add_player(player)

    if args.export:
        experiment = Experiment.resume_experiment(db_session, args.export)
        if not experiment:
            logger.error(f"No experiment found with ID {args.export}")
            return
            
        results = arena.get_experiment_results()
        logger.info("\nExperiment Results:")
        logger.info(f"Name: {results['experiment_name']}")
        logger.info(f"Total Games: {results['total_games']}")
        logger.info("\nFinal Ratings:")
        for name, rating in sorted(results['player_ratings'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"{name}: {rating:.0f}")
        
        logger.info("\nGame History:")
        for game in results['games']:
            logger.info(f"Game {game['game_id']}: Winner - {game['winner']}")
        return

    # Run the experiment and print final standings
    final_ratings = await arena.evaluate_all()
    logger.info("\nFinal Ratings:")
    for name, rating in sorted(final_ratings.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"{name}: {rating:.0f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
