import argparse
import logging
import json
import signal
from typing import Any, Dict, List
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL
from bgbench.data.models import Experiment, GameMatch
from bgbench.experiment.export import export_experiment, calculate_skill_comparison_data
from bgbench.experiment.rating import GameResult
from bgbench.logging_config import setup_logging
from bgbench.games import AVAILABLE_GAMES
from bgbench.experiment.arena import Arena
import yaml # Import yaml for the except block
# Import the new config loader
from bgbench.config_loader import load_and_merge_config

logger = logging.getLogger("bgbench")

load_dotenv()


async def main():
    parser = argparse.ArgumentParser(
        description="Run a game evaluation experiment using LLM players.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )

    # Core arguments
    parser.add_argument(
        "--game", choices=list(AVAILABLE_GAMES.keys()), help="The game to play."
    )
    parser.add_argument(
        "--players", type=str, help="Path to player configuration JSON file."
    )
    parser.add_argument(
        "--config",
        dest="config_file", # Use dest to avoid conflict with config module name
        type=str,
        default=None,
        help="Path to a YAML configuration file for the experiment.",
    )

    # Experiment management (Action flags and resume/name)
    exp_group = parser.add_argument_group('Experiment Management')
    exp_group.add_argument("--resume", type=int, help="Resume experiment by ID.")
    exp_group.add_argument("--name", help="Name for a new experiment.")
    # Action flags - not typically set via YAML
    exp_group.add_argument("--export", type=int, help="Export results for experiment ID (action flag).")
    exp_group.add_argument(
        "--export-experiment",
        type=int,
        help="Export experiment results in schema.json format (action flag).",
    )
    exp_group.add_argument("--list", action="store_true", help="List all experiments (action flag).")


    # Arena configuration
    arena_group = parser.add_argument_group('Arena Configuration')
    arena_group.add_argument(
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
    parser.add_argument(
        "--max-concurrent-games-per-pair",
        type=int,
        default=1,
        help="Maximum number of games allowed to run concurrently between the same pair of players.",
    )
    arena_group.add_argument(
        "--selected-players",
        type=str,
        help="Comma-separated list of player names to focus matches on.",
    )

    # Other options
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")


    # Initial parse to get potential config file path and debug flag early
    args = parser.parse_args()

    # Setup logging based on initial debug flag from CLI
    # Logging level might be adjusted again if debug: true is in YAML/config
    setup_logging(debug=args.debug)

    # Load YAML config and merge with CLI args
    # Defaults defined in argparse will be used if not in YAML or CLI override
    try:
        config = load_and_merge_config(parser, args)
    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.error(f"Configuration Error: {e}")
        return # Exit if config loading fails

    # Re-apply logging setup in case debug flag was changed by config file
    if config.get('debug') != args.debug:
        setup_logging(debug=config.get('debug', False))
        logger.info(f"Logging level updated based on final config (debug={config.get('debug')}).")

    # --- Use 'config' dictionary for settings from here onwards ---
    # --- Use 'args' for action flags like list, export, export_experiment ---

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
    experiment = None # Initialize experiment to None
    experiment_id_to_resume = config.get('resume')

    if experiment_id_to_resume:
        # Try to get experiment info early to check for game_name
        experiment = Experiment.resume_experiment(db_session, experiment_id_to_resume)
        if (
            experiment
            and experiment.game_name
            and experiment.game_name in AVAILABLE_GAMES
        ):
            use_experiment_game = True
            logger.info(f"Identified game '{experiment.game_name}' from resumed experiment ID {experiment_id_to_resume}")


    # Only load player configs if not using action flags (list, export*)
    player_configs = []
    players_file = config.get('players') # Get players file path from config
    if not args.list and not args.export and not args.export_experiment:
        if not players_file:
            # Players file is required unless resuming (players loaded from DB)
            if not experiment_id_to_resume:
                 parser.error(
                    "A players file must be specified via --players or in the config file unless resuming an experiment or using action flags (--list, --export, --export-experiment)."
                 )
            else:
                 logger.warning("Resuming experiment without a players file specified. New players cannot be added.")
        else:
            # Load players if path is provided
            try:
                with open(players_file, "r") as f:
                    player_configs = json.load(f)
                logger.info(f"Loaded player configurations from: {players_file}")

                # Validate player names are unique
                from bgbench.experiment.arena import validate_unique_player_names

                is_valid, duplicates = validate_unique_player_names(player_configs)
                if not is_valid:
                    logger.error(
                        f"Error: Duplicate player names found: {', '.join(duplicates)}"
                    )
                    logger.error(
                        "All players must have unique names. Please update your player configuration."
                    )
                    return # Exit if validation fails

                # Log loaded player details
                for entry in player_configs:
                    model_conf = entry.get("model_config", {})
                    logger.info(
                        f"Player: {entry.get('name')} - Model: {model_conf.get('model')}, Temp: {model_conf.get('temperature')}, Max Tokens: {model_conf.get('max_tokens')}, Resp Style: {model_conf.get('response_style')}, Prompt Style: {entry.get('prompt_style')}"
                    )
            except FileNotFoundError:
                 logger.error(f"Player configuration file not found: {players_file}")
                 return # Exit if file not found
            except json.JSONDecodeError:
                 logger.error(f"Error decoding JSON from player file: {players_file}")
                 return # Exit on JSON error
            except Exception as e: # Catch other potential errors like permission issues
                 logger.error(f"Error reading player file {players_file}: {e}")
                 return

    # Determine the game to play
    game_name_from_config = config.get('game')
    # Ensure experiment is not None before accessing its attributes
    if not game_name_from_config and use_experiment_game and experiment:
        game_name_from_config = experiment.game_name # Use game from resumed experiment if not specified elsewhere
    elif not game_name_from_config and not args.list and not args.export and not args.export_experiment:
         # Game is required if not resuming or using action flags
         parser.error("A game must be specified via --game or in the config file unless resuming or using action flags.")

    if game_name_from_config:
        if game_name_from_config in AVAILABLE_GAMES:
            game_class = AVAILABLE_GAMES[game_name_from_config]
            game = game_class() # Instantiate the game object
            logger.info(f"Using game: {game_name_from_config}")
        else:
            logger.error(f"Game '{game_name_from_config}' specified is not available.")
            logger.info(f"Available games are: {', '.join(AVAILABLE_GAMES.keys())}")
            return # Exit if specified game not found

    # Handle experiment management commands
    if args.list:
        experiments = db_session.query(Experiment).all()
        logger.info("\nExisting Experiments:")
        for exp in experiments:
            logger.info( # Add missing logger.info call
                f"ID: {exp.id}, Name: {exp.name}, Game: {exp.game_name or 'N/A'}, Description: {exp.description}"
            )
            games = db_session.query(GameMatch).filter_by(experiment_id=exp.id).count()
            logger.info(f"  Games played: {games}")
        db_session.close()
        return

    # Handle export-experiment flag - use args here
    if args.export_experiment:
        # Use the ID from args for this action flag
        export_exp_id = args.export_experiment
        experiment = Experiment.resume_experiment(db_session, export_exp_id)
        if not experiment:
            logger.error(f"No experiment found with ID {export_exp_id}")
            db_session.close()
            return

        # Determine game name for export: Use experiment's game, then config, then infer
        export_game_name = experiment.game_name or config.get('game') or (experiment.name.split("_")[0] if experiment.name else None)

        if not export_game_name or export_game_name not in AVAILABLE_GAMES:
             logger.error(f"Cannot determine a valid game type ('{export_game_name}') for exporting experiment {export_exp_id}. Specify --game or ensure experiment has game_name.")
             db_session.close()
             return

        logger.info(f"Exporting experiment {export_exp_id} (Game: {export_game_name}) in schema format...")
        try:
            export_experiment(db_session, export_exp_id, export_game_name)
            logger.info("Export complete.")
        except Exception as e:
            logger.error(f"Error during schema export: {e}")
        finally:
            db_session.close()
        return

    # Process selected players from config
    selected_players_str = config.get('selected_players')
    selected_players_list = None
    if selected_players_str:
        selected_players_list = [name.strip() for name in selected_players_str.split(",")]
        logger.info(
            f"Focusing on games involving these players: {', '.join(selected_players_list)}"
        )

    arena_instance = None # Define arena_instance before try/finally block

    try:
        if experiment_id_to_resume:
            # Retrieve the experiment again (might have been fetched earlier)
            experiment = Experiment.resume_experiment(db_session, experiment_id_to_resume)
            if not experiment:
                logger.error(f"No experiment found with ID {experiment_id_to_resume}")
                return

            # Determine game: Use game from config first, then experiment, then fail
            if game is None: # Game wasn't set via --game or config['game']
                # Ensure experiment and experiment.game_name are valid before using as key
                if use_experiment_game and experiment and experiment.game_name:
                    game_class = AVAILABLE_GAMES[experiment.game_name]
                    game = game_class()
                    logger.info(f"Using game '{experiment.game_name}' from resumed experiment ID {experiment_id_to_resume}")
                else:
                     # Should not happen if checks above are correct, but safeguard
                     logger.error("Cannot determine game type for resuming experiment. Specify --game or ensure experiment has game_name.")
                     return

            logger.info(f"Resuming experiment ID: {experiment_id_to_resume}")
            arena_instance = Arena(
                game=game, # Must be determined by now
                db_session=db_session,
                experiment_id=experiment_id_to_resume,
                player_configs=player_configs, # Pass loaded configs (might be empty if resuming without adding)
                # Use .get() with defaults matching argparse defaults to satisfy pyright
                max_parallel_games=config.get('parallel_games', 3),
                cost_budget=config.get('cost_budget', 2.0),
                confidence_threshold=config.get('confidence_threshold', 0.70),
                selected_players=selected_players_list, # Use processed list
                max_games_per_player_pair=config.get('max_games_per_pair', 10),
                max_concurrent_games_per_pair=config.get('max_concurrent_games_per_pair', 1),
            )
        elif not args.export: # Don't create new if only exporting results
             # Create a new experiment
            if game is None:
                 # Should be caught by earlier checks, but safeguard
                 logger.error("Cannot create a new experiment without a game specified.")
                 return
            experiment_name = config.get('name')
            if not experiment_name:
                 logger.error("Cannot create a new experiment without a name specified via --name or in the config file.")
                 return
            if not player_configs:
                 # Need players to start a new experiment
                 logger.error("Cannot create a new experiment without players specified via --players or in the config file.")
                 return

            logger.info(f"Creating new experiment: {experiment_name}")
            arena_instance = Arena(
                game=game,
                db_session=db_session,
                player_configs=player_configs,
                experiment_name=experiment_name, # Use config value
                # Use .get() with defaults matching argparse defaults to satisfy pyright
                max_parallel_games=config.get('parallel_games', 3),
                cost_budget=config.get('cost_budget', 2.0),
                confidence_threshold=config.get('confidence_threshold', 0.70),
                selected_players=selected_players_list, # Use processed list
                max_games_per_player_pair=config.get('max_games_per_pair', 10),
                max_concurrent_games_per_pair=config.get('max_concurrent_games_per_pair', 1),
            )

        # Handle --export action flag (results printing)
        if args.export:
            export_id = args.export # Use ID from args
            experiment_to_export = Experiment.resume_experiment(db_session, export_id)
            if not experiment_to_export:
                logger.error(f"No experiment found with ID {export_id} to export results.")
                db_session.close()
                return

            # Determine game for export results calculation
            export_results_game_name = experiment_to_export.game_name or config.get('game')
            if not export_results_game_name or export_results_game_name not in AVAILABLE_GAMES:
                 logger.error(f"Cannot determine valid game type ('{export_results_game_name}') for exporting results of experiment {export_id}. Specify --game or ensure experiment has game_name.")
                 db_session.close()
                 return

            logger.info(f"Calculating and printing results for experiment ID: {export_id}")
            # Create a temporary Arena instance just for getting results
            # Check if the existing arena_instance corresponds to the export_id
            # Access experiment ID via arena_instance.experiment.id
            temp_arena = None
            if arena_instance and arena_instance.experiment and arena_instance.experiment.id == export_id:
                 temp_arena = arena_instance
                 logger.debug("Using existing Arena instance for results export.")

            if temp_arena is None:
                logger.debug(f"Creating temporary Arena instance for results export (ID: {export_id}).")
                temp_arena = Arena(
                    game=AVAILABLE_GAMES[export_results_game_name](), # Instantiate the correct game
                    db_session=db_session,
                    experiment_id=export_id,
                    # Pass other relevant config args, using .get() with defaults
                    max_parallel_games=1, # Not relevant for just getting results
                    max_games_per_player_pair=config.get('max_games_per_pair', 10),
                    max_concurrent_games_per_pair=config.get('max_concurrent_games_per_pair', 1),
                )
            print_results(temp_arena.get_experiment_results())
            db_session.close()
            return # Exit after printing export results

        # If we have an arena instance (new or resumed), and not just exporting, run it
        if arena_instance:
            # Set up signal handlers for graceful termination
            def signal_handler(sig, frame):
                logger.warning("Received termination signal. Attempting graceful shutdown...")
                if arena_instance:
                    arena_instance.handle_sigint()
            # arena.handle_sigint() # This call was misplaced

            # Register signal handler for SIGINT (Ctrl+C) and SIGTERM
            signal.signal(signal.SIGINT, signal_handler) # Correct indentation
            signal.signal(signal.SIGTERM, signal_handler) # Correct indentation

            logger.info("Starting experiment evaluation...")
            await arena_instance.evaluate_all()
            logger.info("Experiment evaluation finished.")
            print_results(arena_instance.get_experiment_results())
        else:
             # This case might occur if only --list or --export-experiment was used
             logger.info("No experiment run initiated (e.g., used --list or --export-experiment).")


    finally:
        # Ensure the database session is closed
        if db_session:
            db_session.close()
        logger.info("Database session closed.")


def print_results(results: Dict[str, Any]):
    if not results:
        logger.warning("No results dictionary provided to print.")
        return

    logger.info("\n" + "="*30 + " Experiment Results " + "="*30)
    logger.info(f"Experiment Name: {results.get('experiment_name', 'N/A')}")
    logger.info(f"Experiment ID: {results.get('experiment_id', 'N/A')}")
    logger.info(f"Game: {results.get('game_name', 'N/A')}") # Add game name to results printout
    logger.info(f"Total Matches Played: {results.get('total_matches', 0)}")
    logger.info(f"Completed Matches: {results.get('completed_matches', 0)}")
    logger.info(f"Draws: {results.get('draws', 0)}")
    logger.info(f"Total Cost: ${results.get('total_cost', 0.0):.4f}")

    player_ratings = results.get("player_ratings", {})
    player_concessions = results.get("player_concessions", {})
    player_costs = results.get("player_costs", {}) # Get player costs if available

    if player_ratings:
        # Sort players by rating (mu) descending
        sorted_players = sorted(
            player_ratings.items(), key=lambda item: item[1].mu, reverse=True
        )
        logger.info("\n--- Final Standings (Rating ± 95% CI) ---")
        for name, rating in sorted_players:
            concessions = player_concessions.get(name, 0)
            cost = player_costs.get(name, 0.0)
            # Display rating (mu) and uncertainty (sigma), cost, concessions
            logger.info(f"{name}: {rating.mu:.0f} ± {rating.sigma*1.96:.0f} (Cost: ${cost:.4f}, Concessions: {concessions})") # Show 95% CI

        # Calculate skill comparison probabilities and records from game history
        player_names_sorted = [name for name, _ in sorted_players]
        if len(player_names_sorted) > 1:
            game_dicts = results.get("matches", [])
            # Convert game dictionaries to GameResult objects
            game_results = convert_game_dicts_to_results(game_dicts, list(player_ratings.keys()))

            if game_results:
                 # Calculate skill comparison data using the shared function from export.py
                 # Pass the EloSystem instance if available in results, otherwise it creates a default one
                 elo_system_instance = results.get("elo_system")
                 skill_data = calculate_skill_comparison_data(game_results, player_names_sorted, elo_system=elo_system_instance)
                 print_skill_probability_table(skill_data, player_names_sorted)
            else:
                 logger.info("\nNo completed matches found to calculate skill comparison table.")

    else:
        logger.info("\nNo player ratings available.")

    # Optionally print detailed game history (can be verbose)
    # game_history = results.get("matches", [])
    # if game_history:
    #     logger.info("\n--- Game History ---")
    #     for game in game_history:
    #         p1 = game.get('player1', 'P1')
    #         p2 = game.get('player2', 'P2')
    #         outcome = "Draw"
    #         if "winner" in game and game["winner"] is not None:
    #             outcome = f"Winner: {game['winner']}"
    #         elif "winner" not in game and game.get("is_complete"): # Handle potential draws marked complete
    #              outcome = "Draw"
    #         elif not game.get("is_complete"):
    #              outcome = "Incomplete"

    #         cost = game.get('cost', 0.0)
    #         logger.info(f"Match {game.get('match_id', 'N/A')}: {p1} vs {p2} -> {outcome} (Cost: ${cost:.4f})")
    # else:
    #     logger.info("\nNo game history available.")

    logger.info("="*78 + "\n")


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
