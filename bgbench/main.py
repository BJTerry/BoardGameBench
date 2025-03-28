import argparse
import logging
import json
import signal
from typing import Any, Dict, List
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL
from bgbench.models import Experiment, GameMatch
from bgbench.export import export_experiment, calculate_skill_comparison_data
from bgbench.rating import GameResult
from bgbench.logging_config import setup_logging
from bgbench.games import AVAILABLE_GAMES
from bgbench.arena import Arena

logger = logging.getLogger("bgbench")

load_dotenv()


async def main():
    parser = argparse.ArgumentParser(description="Run a game between LLM players")
    parser.add_argument(
        "--game", choices=list(AVAILABLE_GAMES.keys()), help="The game to play"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Experiment management options
    parser.add_argument("--resume", type=int, help="Resume experiment by ID")
    parser.add_argument("--name", help="Name for new experiment")
    parser.add_argument("--export", type=int, help="Export results for experiment ID")
    parser.add_argument(
        "--export-experiment",
        type=int,
        help="Export experiment results in schema.json format",
    )
    parser.add_argument(
        "--players", type=str, help="Path to player configuration JSON file"
    )
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument(
        "--selected-players",
        type=str,
        help="Comma-separated list of player names to focus matches on (games will only be played if they involve these players)",
    )

    # Parallel execution options
    parser.add_argument(
        "--parallel-games",
        type=int,
        default=3,
        help="Number of games to run in parallel",
    )
    parser.add_argument(
        "--cost-budget",
        type=float,
        default=2.0,
        help="Maximum cost budget for the experiment in dollars",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.70,
        help="Confidence threshold for Elo ratings",
    )
    parser.add_argument(
        "--max-games-per-pair",
        type=int,
        default=10,
        help="Maximum number of games played between each player pair",
    )
    args = parser.parse_args()

    setup_logging(debug=args.debug)

    # Set up database session with connection pooling for PostgreSQL
    if DATABASE_URL.startswith("postgresql"):
        # Connection pooling for PostgreSQL
        engine = create_engine(
            DATABASE_URL,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
        )
    else:
        # Default for SQLite
        engine = create_engine(DATABASE_URL)

    Session = sessionmaker(bind=engine)
    db_session = Session()

    # Check if we're resuming and can get game info from experiment
    game = None
    use_experiment_game = False

    if args.resume:
        # Try to get experiment info early to check for game_name
        experiment = Experiment.resume_experiment(db_session, args.resume)
        if (
            experiment
            and experiment.game_name
            and experiment.game_name in AVAILABLE_GAMES
        ):
            use_experiment_game = True

    # Only load player configs if not using export-experiment or list flags
    player_configs = []
    if not args.export_experiment and not args.list:
        if not args.players and not args.export:
            parser.error(
                "--players is required unless using --export-experiment or --list"
            )

        if not args.game and not use_experiment_game:
            parser.error(
                "--game is required unless using --list, --export-experiment, or resuming an experiment with stored game_name"
            )

        if args.players:
            with open(args.players, "r") as f:
                player_configs = json.load(f)

            for entry in player_configs:
                model_conf = entry.get("model_config", {})
                logger.info(
                    f"Player: {entry.get('name')} - Model: {model_conf.get('model')}, Temperature: {model_conf.get('temperature')}, Max Tokens: {model_conf.get('max_tokens')}, Response Style: {model_conf.get('response_style')}, Prompt Style: {entry.get('prompt_style')}"
                )

    # Get game information from args if provided
    if args.game:
        # Get the game class from our available games
        game_class = AVAILABLE_GAMES[args.game]
        game = game_class()

    # Handle experiment management commands
    if args.list:
        experiments = db_session.query(Experiment).all()
        logger.info("\nExisting Experiments:")
        for exp in experiments:
            logger.info(
                f"ID: {exp.id}, Name: {exp.name}, Description: {exp.description}"
            )
            games = db_session.query(GameMatch).filter_by(experiment_id=exp.id).count()
            logger.info(f"Games played: {games}")
        return

    # Handle export-experiment flag - doesn't require --game option
    if args.export_experiment:
        # Skip the game check for export-experiment
        experiment = Experiment.resume_experiment(db_session, args.export_experiment)
        if not experiment:
            logger.error(f"No experiment found with ID {args.export_experiment}")
            return

        # Use game_name from experiment if available, otherwise fall back to args.game or infer from name
        game_name = experiment.game_name or args.game or experiment.name.split("_")[0]

        # Export the experiment data in schema format
        export_experiment(db_session, args.export_experiment, game_name)
        return

    # Process selected players if provided
    selected_players = None
    if args.selected_players:
        selected_players = [name.strip() for name in args.selected_players.split(",")]
        logger.info(
            f"Focusing on games involving these players: {', '.join(selected_players)}"
        )

    if args.resume:
        # Retrieve the experiment to get its game name
        experiment = Experiment.resume_experiment(db_session, args.resume)
        if not experiment:
            logger.error(f"No experiment found with ID {args.resume}")
            return

        # Use game from args if specified, otherwise try to load game from experiment's game_name
        if game is None and experiment.game_name:
            # Get the game class from stored game_name
            if experiment.game_name in AVAILABLE_GAMES:
                game_class = AVAILABLE_GAMES[experiment.game_name]
                game = game_class()
                logger.info(f"Using game '{experiment.game_name}' from experiment")
            else:
                logger.warning(
                    f"Stored game name '{experiment.game_name}' not found in available games"
                )
                if args.game:
                    game_class = AVAILABLE_GAMES[args.game]
                    game = game_class()
                else:
                    # Still need a game
                    raise ValueError(
                        "Cannot determine game type from experiment. Please specify with --game"
                    )
        elif game is None:
            raise ValueError(
                "--game is required when resuming an experiment without stored game name"
            )

        arena = Arena(
            game,
            db_session,
            experiment_id=args.resume,
            player_configs=player_configs,  # Pass player_configs when resuming an experiment
            max_parallel_games=args.parallel_games,
            cost_budget=args.cost_budget,
            confidence_threshold=args.confidence_threshold,
            selected_players=selected_players,
            max_games_per_player_pair=args.max_games_per_pair,
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
            confidence_threshold=args.confidence_threshold,
            selected_players=selected_players,
            max_games_per_player_pair=args.max_games_per_pair,
        )

    if args.export:
        experiment = Experiment.resume_experiment(db_session, args.export)
        if not experiment:
            logger.error(f"No experiment found with ID {args.export}")
            return

        # If we're exporting via args.export and used the Arena, just print results
        if "arena" in locals():
            print_results(arena.get_experiment_results())
        else:
            # We need to create an Arena to export results
            game_name = experiment.game_name
            if game_name and game_name in AVAILABLE_GAMES:
                game_class = AVAILABLE_GAMES[game_name]
                game = game_class()
                arena = Arena(
                    game,
                    db_session,
                    experiment_id=args.export,
                    max_parallel_games=1,  # Doesn't matter for export
                    max_games_per_player_pair=args.max_games_per_pair,
                )
                print_results(arena.get_experiment_results())
            else:
                # If game_name isn't available, we still need args.game
                if not args.game:
                    logger.error(
                        "Cannot determine game type. Please specify with --game"
                    )
                    return
                game_class = AVAILABLE_GAMES[args.game]
                game = game_class()
                arena = Arena(
                    game,
                    db_session,
                    experiment_id=args.export,
                    max_parallel_games=1,  # Doesn't matter for export
                    max_games_per_player_pair=args.max_games_per_pair,
                )
                print_results(arena.get_experiment_results())
        return

    # Set up signal handlers for graceful termination
    def signal_handler(sig, frame):
        arena.handle_sigint()

    # Register signal handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

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
    for name, rating in sorted(
        results["player_ratings"].items(), key=lambda x: x[1], reverse=True
    ):
        concessions = results["player_concessions"][name]
        logger.info(f"{name}: {rating:.0f} ({concessions} concessions)")

    # Calculate skill comparison probabilities and records from game history
    # Sort players by rating (strongest to weakest)
    player_names = [
        name
        for name, _ in sorted(
            results["player_ratings"].items(), key=lambda x: x[1], reverse=True
        )
    ]
    if len(player_names) > 1:
        # Convert game dictionaries to GameResult objects
        game_results = convert_game_dicts_to_results(results["games"], player_names)
        # Calculate skill comparison data using the shared function from export.py
        skill_data = calculate_skill_comparison_data(game_results, player_names)
        print_skill_probability_table(skill_data, player_names)

    logger.info("\nGame History:")
    for game in results["games"]:
        if "winner" in game:
            logger.info(f"Game {game['game_id']}: Winner - {game['winner']}")
        else:
            logger.info(f"Game {game['game_id']}: Draw")


def convert_game_dicts_to_results(
    games: List[Dict[str, Any]], player_names: List[str]
) -> List[GameResult]:
    """
    Convert game dictionaries from experiment results to GameResult objects.

    Args:
        games: List of game result dictionaries
        player_names: List of player names

    Returns:
        List of GameResult objects for use with EloSystem
    """

    # Convert game results to GameResult objects for the EloSystem
    game_results = []
    for game in games:
        # Skip games without player information
        if not ("player1" in game and "player2" in game):
            continue

        player1 = game["player1"]
        player2 = game["player2"]

        # Skip games with unknown players
        if player1 not in player_names or player2 not in player_names:
            continue

        # Add this game to the results list for Bayesian rating
        if "winner" in game:
            winner = game["winner"]
            game_results.append(
                GameResult(player_0=player1, player_1=player2, winner=winner)
            )
        else:
            # It's a draw
            game_results.append(
                GameResult(
                    player_0=player1,
                    player_1=player2,
                    winner=None,  # None indicates a draw
                )
            )

    return game_results


def print_skill_probability_table(
    skill_probabilities_and_records,
    player_names: List[str],
):
    """
    Print a formatted table of skill comparison probabilities with win-loss-draw records.

    Args:
        skill_probabilities_and_records: Tuple containing skill probabilities and win-loss-draw records
        player_names: List of player names to include in the table
    """
    from tabulate import tabulate

    skill_probabilities, records = skill_probabilities_and_records

    # Shorten model names by taking only the part after the slash
    def shorten_name(name):
        if "/" in name:
            return name.split("/", 1)[1]
        return name

    short_names = [shorten_name(name) for name in player_names]

    # Create tabular data
    table_data = []
    for i, p1 in enumerate(player_names):
        row = [shorten_name(p1)]
        for j, p2 in enumerate(player_names):
            if p1 == p2:
                row.append("-")
            else:
                prob = skill_probabilities.get((p1, p2), 0.0)

                # Get record as (W-L-D) from p1's perspective
                if (p1, p2) in records:
                    wins, losses, draws = records[(p1, p2)]

                    # Format the record, omitting draws if zero
                    record_parts = []
                    record_parts.append(f"{wins}")
                    record_parts.append(f"{losses}")
                    if draws > 0:
                        record_parts.append(f"{draws}")

                    record_str = "-".join(record_parts)
                    row.append(f"{prob:.3f}\n({record_str})")
                else:
                    row.append(f"{prob:.3f}")
        table_data.append(row)

    # Create the formatted table
    table = tabulate(table_data, headers=["Player"] + short_names, tablefmt="grid")

    logger.info("\nSkill Comparison (P(row player skill > column player skill)):")
    logger.info("Format: probability (W-L-D) from row player's perspective")
    logger.info("\n" + table)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
