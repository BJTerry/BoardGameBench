"""
Export module for BoardGameBench experiments

This module provides functionality to export experiment results in a standardized format.
"""

import os
import logging
import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from bgbench.models import Experiment, GameMatch, Player, LLMInteraction
from bgbench.rating import EloSystem, GameResult

logger = logging.getLogger("bgbench")


def is_game_complete(game: GameMatch) -> bool:
    """
    Determine if a game is complete.
    A game is complete if it's marked as complete (including both wins and draws).

    Args:
        game: The GameMatch to check

    Returns:
        True if the game is complete, False otherwise
    """
    # With PostgreSQL this is a native boolean, with SQLite it needs conversion
    return game.complete


def is_game_draw(game: GameMatch) -> bool:
    """
    Determine if a game ended in a draw.
    A game is a draw if it's complete but has no winner.

    Args:
        game: The GameMatch to check

    Returns:
        True if the game ended in a draw, False otherwise
    """
    return game.complete and game.winner_id is None


def count_complete_games(games: List[GameMatch]) -> int:
    """
    Count the number of complete games.

    Args:
        games: List of GameMatch objects

    Returns:
        Number of complete games
    """
    return len([g for g in games if is_game_complete(g)])


def count_draws(games: List[GameMatch]) -> int:
    """
    Count the number of games that ended in a draw.

    Args:
        games: List of GameMatch objects

    Returns:
        Number of draws
    """
    return len([g for g in games if is_game_draw(g)])


def build_match_history(
    games: List[GameMatch], players: List[Player]
) -> List[GameResult]:
    """
    Build match history for use with EloSystem from completed games.

    Args:
        games: List of GameMatch objects
        players: List of Player objects

    Returns:
        List of GameResult objects
    """
    player_name_map = {p.id: p.name for p in players}
    match_history = []

    for game in games:
        if not is_game_complete(game):
            continue

        if (
            game.player1_id not in player_name_map
            or game.player2_id not in player_name_map
        ):
            logger.warning(f"Game {game.id} references player IDs not in experiment")
            continue

        player_0 = player_name_map[game.player1_id]
        player_1 = player_name_map[game.player2_id]
        # For draws, winner_id will be None but we still include the game
        winner = player_name_map.get(game.winner_id) if game.winner_id else None

        match_history.append(
            GameResult(
                player_0=player_0,
                player_1=player_1,
                winner=winner,  # None indicates a draw in Bayesian rating system
            )
        )

    return match_history


def get_player_costs(
    session: Session, experiment_id: int, players: List[Player]
) -> Dict[str, float]:
    """
    Get total costs for each player in the experiment.

    Args:
        session: Database session
        experiment_id: ID of the experiment
        players: List of Player objects

    Returns:
        Dictionary mapping player names to their total costs
    """
    costs = {}
    for player in players:
        # Query total cost for this player across all games in this experiment
        total_cost = (
            session.query(func.sum(LLMInteraction.cost))
            .join(GameMatch, LLMInteraction.game_id == GameMatch.id)
            .filter(
                GameMatch.experiment_id == experiment_id,
                LLMInteraction.player_id == player.id,
            )
            .scalar()
            or 0.0
        )

        # Store the total cost
        costs[player.name] = total_cost

    return costs


def format_for_export(
    session: Session, experiment_id: int, game_name: str
) -> Dict[str, Any]:
    """
    Format experiment data according to the schema.json format

    Args:
        session: Database session
        experiment_id: ID of experiment to export
        game_name: Name of the game played in this experiment

    Returns:
        Dictionary formatted according to schema.json
    """
    # Get experiment data
    experiment = Experiment.resume_experiment(session, experiment_id)
    if not experiment:
        raise ValueError(f"No experiment found with ID {experiment_id}")

    # Get player data
    players = experiment.get_players(session)

    # Get game data
    games = session.query(GameMatch).filter_by(experiment_id=experiment_id).all()

    # Count total games and stats
    total_games = len(games)
    completed_games = count_complete_games(games)
    draws = count_draws(games)

    # Get all data needed for export
    results_list = []

    # Initialize EloSystem to get confidence intervals
    elo_system = EloSystem()

    # Build match history for EloSystem using the utility function
    match_history = build_match_history(games, players)

    # Get all player names
    player_names = [p.name for p in players]

    # Update ratings using Bayesian system if we have matches
    if match_history:
        # Calculate ratings and get confidence intervals
        player_ratings = elo_system.update_ratings(match_history, player_names)
        intervals = elo_system.get_credible_intervals(player_names)
    else:
        # Default intervals if no match history
        raise ValueError("Cannot export data: No match history available for the experiment.")

    # Count games played and wins for each player
    games_played = {p.name: 0 for p in players}
    wins = {p.name: 0 for p in players}

    for game in games:
        if not is_game_complete(game):
            continue

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

    # Get player costs
    costs = get_player_costs(session, experiment_id, players)

    # Format results for each player
    for player in players:
        name = player.name
        rating_value = player_ratings[name].rating
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
            "confidenceInterval": {"upper": int(upper), "lower": int(lower)},
        }

        results_list.append(player_result)

    # Sort results by score in descending order
    results_list.sort(key=lambda x: x["score"], reverse=True)

    # Assign ranks to players
    # We'll use the probability_stronger method from EloSystem to compare players
    # and implement the ranking logic as described
    if match_history:
        # Get sorted player names from results_list
        sorted_player_names = [result["modelName"] for result in results_list]

        # Initialize the ranks
        current_rank = 1
        reference_player = sorted_player_names[
            0
        ]  # First player is the reference for rank 1

        # Assign the first player rank 1
        results_list[0]["rank"] = current_rank

        # For each remaining player, compare to the reference player of the current rank
        for i in range(1, len(results_list)):
            player_name = results_list[i]["modelName"]

            # Calculate the probability that reference player is stronger than current player
            try:
                prob_ref_stronger = elo_system.probability_stronger(
                    reference_player, player_name
                )

                # If reference player is stronger with >95% probability, increase rank
                if prob_ref_stronger > 0.95:
                    current_rank += 1
                    # This player becomes the new reference for this rank
                    reference_player = player_name

                # Assign the current rank to this player
                results_list[i]["rank"] = current_rank

            except RuntimeError:
                # If we can't calculate the probability (e.g., no games between these players),
                # assign the same rank as the previous player
                logger.warning(
                    f"Could not calculate probability for {reference_player} vs {player_name}"
                )
                results_list[i]["rank"] = results_list[i - 1]["rank"]
    else:
        # If there are no matches, all players get rank 1
        for result in results_list:
            result["rank"] = 1

    # Format the final export data
    export_data = {
        "gameName": game_name,
        "results": results_list,
        "metadata": {
            "generatedAt": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "totalGamesPlayed": total_games,
            "completedGames": completed_games,
            "draws": draws,
            "experimentId": experiment_id,
        },
    }

    return export_data


def export_experiment(
    session: Session, experiment_id: int, game_name: str
) -> Optional[str]:
    """
    Export experiment data to a JSON file according to schema.json format

    Args:
        session: Database session
        experiment_id: ID of experiment to export
        game_name: Name of the game played in this experiment

    Returns:
        Path to the created export file or None if export failed
    """
    try:
        # Format the data
        export_data = format_for_export(session, experiment_id, game_name)

        # Create the export directory if it doesn't exist
        os.makedirs("exports", exist_ok=True)

        # Generate a filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exports/{game_name}_{experiment_id}_{timestamp}.json"

        # Write the data to a file
        import json

        with open(filename, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported experiment data to {filename}")
        return filename

    except Exception as e:
        logger.error(f"Error exporting experiment: {e}")
        return None
