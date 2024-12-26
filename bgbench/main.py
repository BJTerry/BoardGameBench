import argparse
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL
from bgbench.models import Experiment, Game, Player
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
    
    # Set up database session
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    db_session = Session()

    # Create players with their full configurations
    player_configs = [
        {
            "name": "llama-3.1-8b",
            "model_config": {
                "model": "openrouter/meta-llama/llama-3.1-8b-instruct:free",
                "temperature": 0.0,
                "max_tokens": 1000
            }
        },
        {
            "name": "claude-3-haiku",
            "model_config": {
                "model": "openrouter/anthropic/claude-3-haiku",
                "temperature": 0.0,
                "max_tokens": 1000
            }
        },
        {
            "name": "gpt-4o-mini",
            "model_config": {
                "model": "openai/gpt-4o-mini",
                "temperature": 0.0,
                "max_tokens": 1000
            }
        }
    ]

    players = []
    for config in player_configs:
        # Create or get existing player
        db_player = (
            db_session.query(Player)
            .filter_by(name=config["name"])
            .first()
        )
        if not db_player:
            db_player = Player.create_player(
                db_session,
                config["name"],
                config["model_config"]
            )
        
        # Create LLM player
        llm = create_llm(**config["model_config"])
        players.append(LLMPlayer(db_player.name, llm))
    
    # Get the game class from our available games
    game_class = AVAILABLE_GAMES[args.game]
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
