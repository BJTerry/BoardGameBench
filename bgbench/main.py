import argparse
import logging
import json
import os
import datetime
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


def format_for_export(db_session: sessionmaker, experiment_id: int, game_name: str) -> Dict[str, Any]:
    """
    Format experiment data according to the schema.json format
    
    Args:
        db_session: Database session
        experiment_id: ID of experiment to export
        game_name: Name of the game played in this experiment
        
    Returns:
        Dictionary formatted according to schema.json
    """
    # Get experiment data
    experiment = Experiment.resume_experiment(db_session, experiment_id)
    if not experiment:
        raise ValueError(f"No experiment found with ID {experiment_id}")
    
    # Get player data
    players = experiment.get_players(db_session)
    
    # Get game data
    games = db_session.query(GameMatch).filter_by(experiment_id=experiment_id).all()
    
    # Count total games played
    total_games = len(games)
    
    # Get all data needed for export
    results_list = []
    
    from bgbench.bayes_rating import EloSystem
    # Initialize EloSystem to get confidence intervals
    elo_system = EloSystem()
    
    # Prepare data needed for ratings
    from bgbench.bayes_rating import GameResult
    match_history = []
    
    # Build match history for EloSystem
    for game in games:
        if game.winner_id is not None:
            player1 = next(p for p in players if p.id == game.player1_id)
            player2 = next(p for p in players if p.id == game.player2_id)
            winner = next(p for p in players if p.id == game.winner_id)
            
            match_history.append(GameResult(
                player_0=player1.name,
                player_1=player2.name,
                winner=winner.name
            ))
    
    # Get all player names
    player_names = [p.name for p in players]
    
    # Update ratings using Bayesian system if we have matches
    if match_history:
        # Calculate ratings and get confidence intervals
        _ = elo_system.update_ratings(match_history, player_names)
        intervals = elo_system.get_credible_intervals(player_names)
    else:
        # Default intervals if no match history
        intervals = {p: (1435, 1565) for p in player_names}  # Default 95% interval
    
    # Count games played and wins for each player
    games_played = {p.name: 0 for p in players}
    wins = {p.name: 0 for p in players}
    
    for game in games:
        if game.player1_id is not None and game.player2_id is not None:
            player1 = next(p for p in players if p.id == game.player1_id)
            player2 = next(p for p in players if p.id == game.player2_id)
            
            games_played[player1.name] += 1
            games_played[player2.name] += 1
            
            if game.winner_id is not None:
                winner = next(p for p in players if p.id == game.winner_id)
                wins[winner.name] += 1
    
    # Count concessions for each player
    concessions = {p.name: 0 for p in players}
    for game in games:
        if game.conceded and game.winner_id is not None:
            loser = None
            if game.player1_id == game.winner_id:
                loser = next(p for p in players if p.id == game.player2_id)
            else:
                loser = next(p for p in players if p.id == game.player1_id)
            
            if loser:
                concessions[loser.name] += 1
    
    # Calculate costs for each player
    from sqlalchemy import func
    from bgbench.models import LLMInteraction
    
    costs = {}
    for player in players:
        # Query total cost for this player across all games in this experiment
        total_cost = db_session.query(func.sum(LLMInteraction.cost)).join(
            GameMatch, LLMInteraction.game_id == GameMatch.id
        ).filter(
            GameMatch.experiment_id == experiment_id,
            LLMInteraction.player_id == player.id
        ).scalar() or 0.0
        
        # Store the total cost
        costs[player.name] = total_cost
    
    # Format results for each player
    for player in players:
        name = player.name
        rating_value = player.rating
        player_games = games_played[name]
        
        # Skip players with no games
        if player_games == 0:
            continue
        
        # Calculate win rate
        win_rate = wins[name] / player_games if player_games > 0 else 0.0
        
        # Get confidence interval
        if name in intervals and None not in intervals[name]:
            lower, upper = intervals[name]
        else:
            # Default values if no interval available
            lower = rating_value - 65
            upper = rating_value + 65
        
        # Calculate average cost per game
        cost_per_game = costs[name] / player_games if player_games > 0 else 0.0
        
        # Format player result according to schema
        player_result = {
            "modelName": name,
            "score": int(rating_value),
            "gamesPlayed": player_games,
            "winRate": round(win_rate, 2),
            "concessions": concessions[name],
            "costPerGame": round(cost_per_game, 4),
            "confidenceInterval": {
                "upper": int(upper),
                "lower": int(lower)
            }
        }
        
        results_list.append(player_result)
    
    # Sort results by score in descending order
    results_list.sort(key=lambda x: x["score"], reverse=True)
    
    # Format the final export data
    export_data = {
        "gameName": game_name,
        "results": results_list,
        "metadata": {
            "generatedAt": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "totalGamesPlayed": total_games
        }
    }
    
    return export_data


def export_experiment(db_session: sessionmaker, experiment_id: int, game_name: str) -> None:
    """
    Export experiment data to a JSON file according to schema.json format
    
    Args:
        db_session: Database session
        experiment_id: ID of experiment to export
        game_name: Name of the game played in this experiment
    """
    try:
        # Format the data
        export_data = format_for_export(db_session, experiment_id, game_name)
        
        # Create the export directory if it doesn't exist
        os.makedirs("exports", exist_ok=True)
        
        # Generate a filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exports/{game_name}_{experiment_id}_{timestamp}.json"
        
        # Write the data to a file
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported experiment data to {filename}")
        
    except Exception as e:
        logger.error(f"Error exporting experiment: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
