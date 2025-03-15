import argparse
import logging
import json
from typing import Any, Dict
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL
from bgbench.models import Experiment, GameMatch
from bgbench.logging_config import setup_logging
from bgbench.games import AVAILABLE_GAMES
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
    parser.add_argument('--players', type=str, help='Path to player configuration JSON file')
    parser.add_argument('--list', action='store_true', help='List all experiments')
    
    # Parallel execution options
    parser.add_argument('--parallel-games', type=int, default=3, help='Number of games to run in parallel')
    parser.add_argument('--cost-budget', type=float, default=2.0, help='Maximum cost budget for the experiment in dollars')
    args = parser.parse_args()

    setup_logging(debug=args.debug)

    with open(args.players, 'r') as f:
        player_configs = json.load(f)

    for entry in player_configs:
        model_conf = entry.get("model_config", {})
        logger.info(f"Player: {entry.get('name')} - Model: {model_conf.get('model')}, Temperature: {model_conf.get('temperature')}, Max Tokens: {model_conf.get('max_tokens')}, Response Style: {model_conf.get('response_style')}, Prompt Style: {entry.get('prompt_style')}")
    
    # Set up database session
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    db_session = Session()


    # Get the game class from our available games
    game_class = AVAILABLE_GAMES[args.game]
    game = game_class()
    
    # Handle experiment management commands
    if args.list:
        experiments = db_session.query(Experiment).all()
        logger.info("\nExisting Experiments:")
        for exp in experiments:
            logger.info(f"ID: {exp.id}, Name: {exp.name}, Description: {exp.description}")
            games = db_session.query(GameMatch).filter_by(experiment_id=exp.id).count()
            logger.info(f"Games played: {games}")
        return

    if args.resume:
        arena = Arena(
            game, 
            db_session, 
            experiment_id=args.resume,
            max_parallel_games=args.parallel_games,
            cost_budget=args.cost_budget
        )
    else:
        arena = Arena(
            game, 
            db_session, 
            player_configs=player_configs,
            experiment_name=args.name,
            max_parallel_games=args.parallel_games,
            cost_budget=args.cost_budget
        )

    if args.export:
        experiment = Experiment.resume_experiment(db_session, args.export)
        if not experiment:
            logger.error(f"No experiment found with ID {args.export}")
            return
            
        print_results(arena.get_experiment_results())
        return

    # Run the experiment and print final standings
    await arena.evaluate_all()
    print_results(arena.get_experiment_results())

def print_results(results: Dict[str, Any]):
    logger.info("\nExperiment Results:")
    logger.info(f"Name: {results['experiment_name']}")
    logger.info(f"Total Games: {results['total_games']}")
    logger.info("\nFinal Results:")
    for name, rating in sorted(results['player_ratings'].items(), key=lambda x: x[1], reverse=True):
        concessions = results['player_concessions'][name]
        logger.info(f"{name}: {rating:.0f} ({concessions} concessions)")
    
    logger.info("\nGame History:")
    for game in results['games']:
        logger.info(f"Game {game['game_id']}: Winner - {game['winner']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
