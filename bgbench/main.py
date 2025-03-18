import argparse
import logging
import json
from typing import Any, Dict
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL
from bgbench.models import Experiment, GameMatch
from bgbench.export import export_experiment
from bgbench.logging_config import setup_logging
from bgbench.games import AVAILABLE_GAMES
from bgbench.arena import Arena

logger = logging.getLogger("bgbench")

load_dotenv()

async def main():
    parser = argparse.ArgumentParser(description='Run a game between LLM players')
    parser.add_argument('--game', choices=list(AVAILABLE_GAMES.keys()), help='The game to play')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    # Experiment management options
    parser.add_argument('--resume', type=int, help='Resume experiment by ID')
    parser.add_argument('--name', help='Name for new experiment')
    parser.add_argument('--export', type=int, help='Export results for experiment ID')
    parser.add_argument('--export-experiment', type=int, help='Export experiment results in schema.json format')
    parser.add_argument('--players', type=str, help='Path to player configuration JSON file')
    parser.add_argument('--list', action='store_true', help='List all experiments')
    
    # Parallel execution options
    parser.add_argument('--parallel-games', type=int, default=3, help='Number of games to run in parallel')
    parser.add_argument('--cost-budget', type=float, default=2.0, help='Maximum cost budget for the experiment in dollars')
    parser.add_argument('--confidence-threshold', type=float, default=0.70, help='Confidence threshold for Elo ratings')
    args = parser.parse_args()

    setup_logging(debug=args.debug)
    
    # Set up database session
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    db_session = Session()
    
    # Only load player configs if not using export-experiment or list flags
    player_configs = []
    if not args.export_experiment and not args.list:
        if not args.players and not args.export:
            parser.error("--players is required unless using --export-experiment or --list")
            
        if not args.game:
            parser.error("--game is required unless using --list")
            
        if args.players:
            with open(args.players, 'r') as f:
                player_configs = json.load(f)

            for entry in player_configs:
                model_conf = entry.get("model_config", {})
                logger.info(f"Player: {entry.get('name')} - Model: {model_conf.get('model')}, Temperature: {model_conf.get('temperature')}, Max Tokens: {model_conf.get('max_tokens')}, Response Style: {model_conf.get('response_style')}, Prompt Style: {entry.get('prompt_style')}")

    # Get game information if needed (not for --list)
    game = None
    if not args.list:
        if not args.game:
            parser.error("--game is required except when using --list")
        
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
        
    # Handle export-experiment flag
    if args.export_experiment:
        experiment = Experiment.resume_experiment(db_session, args.export_experiment)
        if not experiment:
            logger.error(f"No experiment found with ID {args.export_experiment}")
            return
            
        # Determine game name - use the one from args if provided, otherwise try to infer from experiment name
        game_name = args.game if args.game else experiment.name.split('_')[0]
        
        # Export the experiment data in schema format
        export_experiment(db_session, args.export_experiment, game_name)
        return

    if args.resume:
        if game is None:
            raise ValueError("--game is required when resuming an experiment")
        arena = Arena(
            game, 
            db_session, 
            experiment_id=args.resume,
            max_parallel_games=args.parallel_games,
            cost_budget=args.cost_budget,
            confidence_threshold=args.confidence_threshold
        )
    else:
        if game is None:
            raise ValueError("--game is required when creating a new experiment")
        arena = Arena(
            game, 
            db_session, 
            player_configs=player_configs,
            experiment_name=args.name,
            max_parallel_games=args.parallel_games,
            cost_budget=args.cost_budget,
            confidence_threshold=args.confidence_threshold
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
    logger.info(f"Completed Games: {results.get('completed_games', 0)}")
    logger.info(f"Draws: {results.get('draws', 0)}")
    logger.info("\nFinal Results:")
    for name, rating in sorted(results['player_ratings'].items(), key=lambda x: x[1], reverse=True):
        concessions = results['player_concessions'][name]
        logger.info(f"{name}: {rating:.0f} ({concessions} concessions)")
    
    logger.info("\nGame History:")
    for game in results['games']:
        if 'winner' in game:
            logger.info(f"Game {game['game_id']}: Winner - {game['winner']}")
        else:
            logger.info(f"Game {game['game_id']}: Draw")




if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
