import logging
import time
from dataclasses import dataclass
import random
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Set
from sqlalchemy import func
from sqlalchemy.orm import Session
from bgbench.match.state_manager import MatchStateManager
from bgbench.models import (
    Experiment,
    Player as DBPlayer,
    GameMatch,
    LLMInteraction,
)
from bgbench.llm_integration import ResponseStyle
from bgbench.game import Game  # Abstract ruleset
from bgbench.llm_player import LLMPlayer
from bgbench.match.view import PromptStyle
from bgbench.match.runner import MatchRunner
from bgbench.rating import PlayerRating, EloSystem, GameResult
from bgbench.export import (
    is_game_complete,
    is_game_draw,
    count_complete_games,
    count_draws,
    build_match_history,
)
from bgbench.scheduler import (
    MatchScheduler,
    SigmaMinimizationScheduler,
    MatchFilterSpec,
)

logger = logging.getLogger("bgbench")


def validate_unique_player_names(
    player_configs: List[Dict[str, Any]],
) -> Tuple[bool, Set[str]]:
    """
    Validate that all player configurations have unique names.

    Args:
        player_configs: List of player configuration dictionaries

    Returns:
        tuple: (is_valid, duplicates) where is_valid is a boolean and
               duplicates is a set of duplicate names (empty if valid)
    """
    # Process player names, considering None or missing names as invalid
    player_names: List[str] = []
    for config in player_configs:
        name = config.get("name")
        if name is None or name == "":
            return False, {""}  # Missing or None name is invalid
        player_names.append(str(name))

    # Check for duplicates
    if len(player_names) != len(set(player_names)):
        # Find duplicates for error message
        seen: Set[str] = set()
        duplicates: Set[str] = set()

        for name in player_names:
            if name in seen:
                duplicates.add(name)
            else:
                seen.add(name)

        return False, duplicates

    return True, set()


@dataclass
class ArenaPlayer:
    llm_player: LLMPlayer
    rating: PlayerRating
    player_model: DBPlayer


@dataclass
class OngoingMatch:
    player1_id: int
    player2_id: int
    task: asyncio.Task


class Arena:
    def __init__(
        self,
        game: Game,
        db_session: Session,
        player_configs: Optional[List[Dict[str, Any]]] = None,
        experiment_name: Optional[str] = None,
        experiment_id: Optional[int] = None,
        confidence_threshold: float = 0.70,
        max_parallel_games: int = 3,
        cost_budget: Optional[float] = 2.0,
        llm_factory=None,
        match_scheduler: Optional[MatchScheduler] = None,
        selected_players: Optional[List[str]] = None,
        max_games_per_player_pair: int = 10,
        max_concurrent_games_per_pair: int = 1,
    ):
        """
        Initialize Arena with either:
        - experiment_id to resume an existing experiment and its players
        - player_configs and experiment_name to start a new experiment

        Args:
            game: The game instance
            db_session: Database session
            player_configs: List of player configurations for new experiments
            experiment_name: Name for new experiment
            experiment_id: ID of experiment to resume
            confidence_threshold: Confidence threshold for Elo ratings
            max_parallel_games: Number of games to run in parallel
            cost_budget: Maximum cost budget for the experiment in dollars
            llm_factory: Optional function for testing that creates LLM instances
            match_scheduler: Optional scheduler for choosing player matchups (defaults to SigmaMinimizationScheduler)
            selected_players: Optional list of player names to focus on (only matches involving these players will be scheduled)
            max_games_per_player_pair: Maximum number of games played between each player pair
            max_concurrent_games_per_pair: Maximum number of games allowed to run concurrently between the same pair of players
        """
        self.game: Game = game
        self.players: List[ArenaPlayer] = []
        self.match_state_manager = MatchStateManager()
        self._resumable_matches = []  # Initialize resumable matches list
        self.confidence_threshold = confidence_threshold
        self.session = db_session
        self.max_parallel_games = max_parallel_games
        self.cost_budget = cost_budget
        self.max_games_per_player_pair = max_games_per_player_pair
        # Map (min_id, max_id) -> count of ongoing games for this pair
        self.ongoing_matches: Dict[Tuple[int, int], int] = {}
        self._scheduled_games_between: Dict[Tuple[int, int], int] = {}
        self._lock = asyncio.Lock()
        self.selected_players = selected_players
        self.max_concurrent_games_per_pair = max_concurrent_games_per_pair

        # For graceful termination
        self._stop_scheduling = False
        self._force_stop = False
        self._active_tasks: Set[asyncio.Task] = set()
        self._budget_exceeded = False

        # Set up the match scheduler (default to SigmaMinimizationScheduler if not provided)
        if match_scheduler is None:
            self.match_scheduler = SigmaMinimizationScheduler()
        else:
            self.match_scheduler = match_scheduler

        # Store completed matches history for Bayesian rating updates
        self.match_history: List[GameResult] = []

        if experiment_id is not None:
            self._resume_experiment(experiment_id, llm_factory, player_configs)
        else:
            if player_configs is None:
                raise ValueError("player_configs required when creating new experiment")

            # Use the validation function from this module

            # Validate player names are unique
            is_valid, duplicates = validate_unique_player_names(player_configs)
            if not is_valid:
                raise ValueError(
                    f"Duplicate player names found: {', '.join(duplicates)}. All players must have unique names."
                )

            self._create_new_experiment(experiment_name, player_configs, llm_factory)

    def _create_llm_player(
        self, name: str, model_config: Dict[str, Any], llm_factory=None, **kwargs
    ) -> LLMPlayer:
        """Create an LLM player with the specified configuration."""
        if llm_factory:
            # For testing purposes
            mock_llm = llm_factory(name)
            return LLMPlayer(
                name,
                model_config,
                prompt_style=PromptStyle[kwargs.get("prompt_style", "header").upper()],
                response_style=ResponseStyle[
                    kwargs.get("response_style", "direct").upper()
                ],
                _llm=mock_llm,
            )
        else:
            return LLMPlayer(
                name,
                model_config,
                prompt_style=PromptStyle[kwargs.get("prompt_style", "header").upper()],
                response_style=ResponseStyle[
                    kwargs.get("response_style", "direct").upper()
                ],
            )

    def _create_player_rating(
        self,
        name: str,
        player_ratings: Optional[Dict[str, PlayerRating]] = None,
        db_player: Optional[DBPlayer] = None,
        games_played: int = 0,
    ) -> PlayerRating:
        """Create a player rating object, either from existing ratings or as a new rating."""
        if player_ratings and name in player_ratings:
            return player_ratings[name]

        # Initialize matches_played with the input games_played as a default
        matches_played = games_played

        # For new players or players without ratings, try to get count from DB
        if db_player and not games_played:
            # Count matches played by this player
            matches_played = (  # Overwrite the default if query runs
                self.session.query(GameMatch)
                .filter(
                    GameMatch.experiment_id == self.experiment.id,
                    GameMatch.complete.is_(True),  # Count only completed matches
                    (GameMatch.player1_id == db_player.id)
                    | (GameMatch.player2_id == db_player.id),
                )
                .count()
            )

        # Use default 1500±400 prior for players with no match history
        return PlayerRating(
            name=name,
            rating=1500.0,  # Default initial rating
            sigma=400.0,  # Default uncertainty
            games_played=matches_played,  # Use matches_played count
        )

    def _create_player_from_config(
        self,
        config: Dict[str, Any],
        llm_factory=None,
        experiment_id: Optional[int] = None,
    ) -> ArenaPlayer:
        """Create a complete arena player from a configuration dictionary."""
        # Extract config values
        name = config["name"]
        model_config = config["model_config"]

        # Create LLM player
        llm_player = self._create_llm_player(
            name,
            model_config,
            llm_factory,
            prompt_style=config.get("prompt_style", "header"),
            response_style=config.get("response_style", "direct"),
        )

        # Create database player
        exp_id = experiment_id or self.experiment.id
        db_player = DBPlayer().create_player(self.session, name, model_config, exp_id)

        # Create player rating (starts with 0 matches played for new players)
        rating = self._create_player_rating(name, games_played=0)

        # Return the complete ArenaPlayer
        return ArenaPlayer(llm_player, rating, db_player)

    def _resume_experiment(
        self, experiment_id: int, llm_factory=None, player_configs=None
    ):
        """
        Resume an existing experiment and its players.

        If player_configs is provided, also add any new players not already in the experiment.
        """
        self.experiment = Experiment.resume_experiment(self.session, experiment_id)
        if not self.experiment:
            raise ValueError(f"No experiment found with ID {experiment_id}")

        # Load all completed matches for the match history - including draws
        completed_games = (
            self.session.query(GameMatch)
            .filter(
                GameMatch.experiment_id == self.experiment.id,
                GameMatch.complete.is_(
                    True
                ),  # Using the complete field from our utility
            )
            .all()
        )

        # Build match history using the utility function
        db_players = self.experiment.get_players(self.session)
        self.match_history = build_match_history(completed_games, db_players)

        elo, player_ratings = self.current_elo()

        # Track existing player names to later check for missing players
        existing_player_names = set()

        for db_player in db_players:
            existing_player_names.add(db_player.name)
            try:
                # Create LLM player
                llm_player = self._create_llm_player(
                    db_player.name, db_player.model_config, llm_factory
                )

                # Create player rating
                rating = self._create_player_rating(
                    db_player.name, player_ratings, db_player
                )

                self.players.append(ArenaPlayer(llm_player, rating, db_player))
            except Exception as e:
                logger.error(f"Could not recreate LLM for player {db_player.name}: {e}")
                continue

        incomplete_matches = (
            self.session.query(GameMatch)
            .filter(
                GameMatch.experiment_id == experiment_id,
                GameMatch.complete.is_(False),
            )
            .all()
        )

        # If player_configs is provided, check for and add new players
        if player_configs:
            # Validate all player configurations have unique names
            is_valid, duplicates = validate_unique_player_names(player_configs)
            if not is_valid:
                raise ValueError(
                    f"Duplicate player names found in config: {', '.join(duplicates)}. All players must have unique names."
                )

            new_player_configs = [
                config
                for config in player_configs
                if config["name"] not in existing_player_names
            ]

            if new_player_configs:
                logger.info(
                    f"Adding {len(new_player_configs)} new players to existing experiment"
                )

                for config in new_player_configs:
                    arena_player = self._create_player_from_config(config, llm_factory)
                    self.players.append(arena_player)

            # Check for players in database but not in configs and log a warning
            config_player_names = {config["name"] for config in player_configs}
            missing_players = existing_player_names - config_player_names

            if missing_players:
                logger.warning(
                    f"Players in database missing from provided configs: {', '.join(missing_players)}"
                )

        # Load incomplete matches for resumption
        for match in incomplete_matches:
            latest_state = self.match_state_manager.get_latest_state(
                self.session, match.id
            )
            if latest_state and latest_state.game_state:
                try:
                    # Let the game handle deserialization of the game_state field
                    deserialized_state = self.game.deserialize_state(
                        latest_state.game_state
                    )
                    player_a = next(
                        (
                            p
                            for p in self.players
                            if p.player_model.id == match.player1_id
                        ),
                        None,
                    )
                    player_b = next(
                        (
                            p
                            for p in self.players
                            if p.player_model.id == match.player2_id
                        ),
                        None,
                    )
                    if player_a and player_b:
                        self._resumable_matches.append(
                            (player_a, player_b, deserialized_state, match.id)
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to deserialize state for match {match.id}: {e}"
                    )

        logger.info(f"Resumed experiment {self.experiment.name} (id: {experiment_id})")

    def _create_new_experiment(
        self,
        experiment_name: Optional[str],
        player_configs: List[Dict[str, Any]],
        llm_factory=None,
    ):
        """Create a new experiment with the specified players."""
        game_class_name = self.game.__class__.__name__
        name = experiment_name or f"{game_class_name}_evaluation"
        self.experiment = Experiment().create_experiment(
            self.session,
            name=name,
            description=f"Evaluation of LLMs playing {game_class_name}",
            game_name=game_class_name,
        )

        for config in player_configs:
            arena_player = self._create_player_from_config(config, llm_factory)
            self.players.append(arena_player)

    def _matches_played_between(self, player_a: DBPlayer, player_b: DBPlayer) -> int:
        """Return the number of matches played between two players in this experiment."""
        # Get the count of completed matches from the database
        db_count = (
            self.session.query(
                func.count(GameMatch.id)
            )  # Use func.count for efficiency
            .filter(
                GameMatch.experiment_id == self.experiment.id,
                GameMatch.complete.is_(True),  # Only count completed matches
                GameMatch.experiment_id == self.experiment.id,
                (
                    (GameMatch.player1_id == player_a.id)
                    & (GameMatch.player2_id == player_b.id)
                )
                | (
                    (GameMatch.player1_id == player_b.id)
                    & (GameMatch.player2_id == player_a.id)
                ),
            )
            .scalar()
            or 0  # Use scalar() to get the count directly
        )

        # Get the count of currently scheduled (ongoing or pending) matches between these players
        pair_id = (min(player_a.id, player_b.id), max(player_a.id, player_b.id))
        # scheduled_count = self._scheduled_games_between.get(pair_id, 0) # _scheduled_games_between tracks total scheduled, not just pending
        ongoing_count = self.ongoing_matches.get(
            pair_id, 0
        )  # ongoing_matches tracks currently running

        # Count matches in self._resumable_matches if that list exists
        # For now, approximate total scheduled = completed + ongoing
        return db_count + ongoing_count

    async def find_next_available_match(
        self,
    ) -> Optional[Tuple[ArenaPlayer, ArenaPlayer]]:
        """
        Find the next match to run based on the configured match scheduling strategy.

        Uses the scheduler set during initialization (defaults to FullRankingScheduler)
        to determine the best matchup based on the scheduling strategy.

        If selected_players is set, only matches involving at least one of the selected players will be considered.
        """
        # Lock not needed here as we're just doing setup for the filter_spec

        # Create filter specification with all filtering criteria
        filter_spec = MatchFilterSpec(
            selected_player_names=self.selected_players,
            max_games_per_pairing=self.max_games_per_player_pair,
            confidence_threshold=self.confidence_threshold,
            max_concurrent_games_per_pair=self.max_concurrent_games_per_pair,
        )

        # Use the scheduler to find potential matches with filtering applied
        # This doesn't modify any shared state yet
        elo, _ = self.current_elo()
        match_pairs = self.match_scheduler.find_matches(
            self.players,
            self.match_history,
            elo,
            self.ongoing_matches,  # Pass the dict directly
            filter_spec=filter_spec,
            limit=100,
        )

        if not match_pairs:
            logger.debug("No candidate matches found by scheduler")
            return None

        # Try each match pair in order until we find a valid one
        for player_a, player_b in match_pairs:
            pair_ids = (
                min(player_a.player_model.id, player_b.player_model.id),
                max(player_a.player_model.id, player_b.player_model.id),
            )

            # Lock to update shared state
            async with self._lock:
                # Check if the pair has reached the total match limit (completed + ongoing)
                # Note: This check might slightly overcount if a match finishes between check and start
                matches_scheduled = self._matches_played_between(
                    player_a.player_model, player_b.player_model
                )
                if matches_scheduled < self.max_games_per_player_pair:
                    logger.debug(
                        f"Scheduling match between {player_a.llm_player.name} and {player_b.llm_player.name} "
                        f"(Scheduled: {matches_scheduled}/{self.max_games_per_player_pair})"
                    )
                    # Increment the count of *ongoing* matches for this pair
                    self.ongoing_matches[pair_ids] = (
                        self.ongoing_matches.get(pair_ids, 0) + 1
                    )
                    logger.debug(
                        f"Incremented ongoing match count for pair {pair_ids} to {self.ongoing_matches[pair_ids]}"
                    )
                    # No longer need _scheduled_games_between, rely on DB + ongoing_matches
                    # self._scheduled_games_between[pair_ids] = (
                    #     self._scheduled_games_between.get(pair_ids, 0) + 1
                    # )
                    return player_a, player_b
                else:
                    logger.debug(
                        f"Candidate match rejected - exceeded match limit ({matches_scheduled}/{self.max_games_per_player_pair}): {player_a.llm_player.name} vs {player_b.llm_player.name}"
                    )

        logger.debug("No valid matches found among candidates")
        return None

    def log_standings(self):
        """Log current ratings and costs for all players"""
        logger.info("\nCurrent Standings:")
        sorted_players = sorted(
            self.players, key=lambda p: p.rating.rating, reverse=True
        )

        total_cost = 0.0
        for player in sorted_players:
            # Count concessions by this player across all matches in the experiment
            concessions = self._get_player_concessions(player)

            player_cost = self._get_player_cost(player)
            total_cost += player_cost

            logger.info(
                f"{player.llm_player.name}: {player.rating.rating:.0f} "
                f"({player.rating.games_played} matches, {concessions} concessions, "  # Changed games -> matches
                f"${player_cost:.4f} cost)"
            )

        if self.cost_budget:
            logger.info(
                f"Total cost: ${total_cost:.4f} / ${self.cost_budget:.4f} "
                f"({total_cost / self.cost_budget * 100:.1f}% of budget)"
            )
        else:
            logger.info(f"Total cost: ${total_cost:.4f}")

    async def run_single_game(
        self,
        player_a: ArenaPlayer,
        player_b: ArenaPlayer,
        resumed_state: Optional[Any] = None,
        existing_match_id: Optional[int] = None,
    ):
        """Run a single match, potentially resuming from a state."""
        # Get database players at the start
        db_player_a = player_a.player_model
        db_player_b = player_b.player_model
        player_pair = (
            min(db_player_a.id, db_player_b.id),
            max(db_player_a.id, db_player_b.id),
        )

        match_id: int
        db_match: GameMatch

        try:
            if existing_match_id is not None:
                # Fetch existing match record
                db_match = self.session.get(GameMatch, existing_match_id)
                if not db_match:
                    logger.error(
                        f"Could not find existing match with ID {existing_match_id} to resume."
                    )
                    # Decrement ongoing count as this match won't run
                    async with self._lock:
                        current_count = self.ongoing_matches.get(player_pair, 0)
                        if current_count > 1:
                            self.ongoing_matches[player_pair] -= 1
                        elif current_count == 1:
                            del self.ongoing_matches[player_pair]
                    return  # Exit if match not found
                match_id = existing_match_id
                logger.info(
                    f"Resuming match {match_id} between {player_a.llm_player.name} and {player_b.llm_player.name}"
                )
                # Player order is determined by the existing record
                if db_match.player1_id == db_player_a.id:
                    p1_arena, p2_arena = player_a, player_b
                else:
                    p1_arena, p2_arena = player_b, player_a
            else:
                # Randomize player order for a new match
                p1_arena, p2_arena = random.sample([player_a, player_b], 2)
                # Create new match record
                try:
                    db_match = GameMatch(
                        experiment_id=self.experiment.id,
                        winner_id=None,
                        player1_id=p1_arena.player_model.id,
                        player2_id=p2_arena.player_model.id,
                        complete=False,  # Start as incomplete
                    )
                    self.session.add(db_match)
                    self.session.commit()
                    self.session.refresh(db_match)
                    match_id = int(db_match.id)
                    logger.info(
                        f"Starting new match {match_id} between {p1_arena.llm_player.name} and {p2_arena.llm_player.name}"
                    )
                except Exception as e:
                    self.session.rollback()
                    logger.error(f"Failed to create match record: {e}")
                    # Decrement ongoing count as this match won't run
                    async with self._lock:
                        current_count = self.ongoing_matches.get(player_pair, 0)
                        if current_count > 1:
                            self.ongoing_matches[player_pair] -= 1
                        elif current_count == 1:
                            del self.ongoing_matches[player_pair]
                    raise  # Re-raise exception

            runner = MatchRunner(
                game=self.game,  # Pass the abstract game ruleset
                player1=p1_arena.llm_player,
                player2=p2_arena.llm_player,
                db_session=self.session,
                game_id=match_id,  # Pass match_id to MatchRunner (renamed internally)
                player1_id=p1_arena.player_model.id,
                player2_id=p2_arena.player_model.id,
                experiment_name=self.experiment.name,
                initial_state=resumed_state,
                match_state_manager=self.match_state_manager,
            )

            winner, _, concession = await runner.play_game()

            # Mark the match as complete since it finished normally
            db_match.complete = True

            if winner:
                # Match has a winner
                winner_name = winner.name

                # Create a GameResult entry for this match result
                new_match_result = GameResult(
                    player_0=p1_arena.llm_player.name,  # Use the actual player order for the result
                    player_1=p2_arena.llm_player.name,
                    winner=winner_name,
                )

                # Update winner in database
                winner_db_player = (
                    p1_arena.player_model
                    if winner.name == p1_arena.llm_player.name
                    else p2_arena.player_model
                )
                db_match.winner_id = winner_db_player.id
                if concession:
                    db_match.conceded = True
                    db_match.concession_reason = concession
            else:
                # Match ended in a draw
                new_match_result = GameResult(
                    player_0=p1_arena.llm_player.name,
                    player_1=p2_arena.llm_player.name,
                    winner=None,  # None indicates a draw
                )
                # No winner_id is set, but the match is marked as complete

            self.match_history.append(new_match_result)
            try:
                self.session.commit()  # Commit winner, completion status, concession
            except Exception as e:
                self.session.rollback()
                logger.error(f"Failed to update match record {match_id}: {e}")
                raise

            # Current Elo looks up completed matches in the database, so commit first
            _, new_ratings = self.current_elo()

            # Update ratings for all players
            try:
                for arena_player in self.players:
                    name = arena_player.llm_player.name
                    if name in new_ratings:
                        # Update both memory and database ratings
                        arena_player.rating = new_ratings[name]
                        arena_player.player_model.update_rating(
                            self.session, new_ratings[name].rating
                        )

                # Commit updates to player ratings
                self.session.commit()
            except Exception as e:
                self.session.rollback()
                logger.error(
                    f"Failed to update player ratings after match {match_id}: {e}"
                )
                raise

            logger.info(
                f"Match {match_id} completed: {winner.name if winner else 'Draw'}"
            )

        finally:
            # Use lock to safely update tracking data
            async with self._lock:
                # Decrement the count of ongoing matches for this pair
                current_count = self.ongoing_matches.get(player_pair, 0)
                if current_count > 1:
                    self.ongoing_matches[player_pair] = current_count - 1
                    logger.debug(
                        f"Decremented ongoing match count for pair {player_pair} to {current_count - 1}"
                    )
                elif current_count == 1:
                    del self.ongoing_matches[player_pair]
                    logger.debug(f"Removed last ongoing match for pair {player_pair}")
                else:
                    # This case should ideally not happen if logic is correct
                    logger.warning(
                        f"Attempted to decrement ongoing match count for pair {player_pair}, but count was already {current_count}"
                    )

    def handle_sigint(self):
        """Handle SIGINT (Ctrl+C) signal"""
        if self._stop_scheduling:
            # Second Ctrl+C: Force immediate stop
            logger.warning("Second interrupt received. Stopping all tasks immediately.")
            self._force_stop = True

            # Cancel all active tasks
            for task in list(self._active_tasks):  # Iterate over a copy
                if not task.done():
                    task.cancel()
        else:
            # First Ctrl+C: Stop scheduling new matches, complete ongoing ones
            self._stop_scheduling = True
            logger.warning(
                "Scheduling stopped, completing ongoing matches. Press Ctrl+C again to exit immediately."
            )

    async def evaluate_all(self):
        """Run the experiment by scheduling and executing matches"""
        # Initialize tracking variables
        last_log_time = 0
        last_logged_cost = 0
        total_cost = 0.0  # Initialize total_cost
        MIN_LOG_INTERVAL = 10  # Only log costs at most every 10 seconds

        # Main evaluation loop
        while not self._force_stop:
            current_time = time.time()

            # Check total cost against budget if specified
            if self.cost_budget is not None:
                # Calculate current total cost
                total_cost = self._get_total_cost_for_all_players()

                # Log only when cost changes AND enough time has passed
                cost_changed = (
                    abs(total_cost - last_logged_cost) > 0.000001
                )  # Float comparison with small epsilon
                time_elapsed = current_time - last_log_time > MIN_LOG_INTERVAL

                if cost_changed and time_elapsed:
                    logger.info(
                        f"Current total cost: ${total_cost:.6f} / ${self.cost_budget:.6f}"
                    )
                    last_log_time = current_time
                    last_logged_cost = total_cost

                # Check if budget is exceeded
                if total_cost >= self.cost_budget:
                    logger.info(
                        f"Cost budget of ${self.cost_budget:.6f} reached. Stopping evaluation."
                    )
                    self._budget_exceeded = True
                    self._stop_scheduling = True  # Stop scheduling new/resumed matches
                    # Don't break immediately, let ongoing tasks finish unless Ctrl+C again

            # Track number of new tasks spawned in this iteration
            new_tasks_spawned = 0

            # Schedule new or resumed matches only if not stopping and below parallel limit
            if (
                not self._stop_scheduling
                and len(self._active_tasks) < self.max_parallel_games
            ):
                scheduled_something = False
                # Prioritize resumable matches
                if self._resumable_matches:
                    # Take the first resumable match
                    pA, pB, state, match_id = self._resumable_matches.pop(0)
                    pair_ids = (
                        min(pA.player_model.id, pB.player_model.id),
                        max(pA.player_model.id, pB.player_model.id),
                    )

                    # Check concurrency limits for this specific pair
                    if (
                        self.ongoing_matches.get(pair_ids, 0)
                        < self.max_concurrent_games_per_pair
                    ):
                        logger.info(
                            f"Attempting to resume match {match_id} between {pA.llm_player.name} and {pB.llm_player.name}"
                        )
                        # Increment ongoing count *before* creating task
                        self.ongoing_matches[pair_ids] = (
                            self.ongoing_matches.get(pair_ids, 0) + 1
                        )
                        task = asyncio.create_task(
                            self.run_single_game(
                                pA, pB, resumed_state=state, existing_match_id=match_id
                            ),
                            name=f"match-{match_id}-resume",
                        )
                        self._active_tasks.add(task)
                        new_tasks_spawned += 1
                        scheduled_something = True
                        # Add callback to remove task from set when done
                        task.add_done_callback(lambda t: self._active_tasks.discard(t))
                    else:
                        # Limit reached for this pair, put it back at the front
                        logger.debug(
                            f"Resuming match {match_id} deferred: concurrent limit for pair {pair_ids} reached."
                        )
                        self._resumable_matches.insert(0, (pA, pB, state, match_id))
                        scheduled_something = True

                # If no resumable match was scheduled, try scheduling a new one
                if not scheduled_something:
                    matchup = await self.find_next_available_match()
                    if matchup:
                        pA, pB = matchup
                        pair_names = f"{pA.llm_player.name}-vs-{pB.llm_player.name}"
                        # find_next_available_match already incremented ongoing_matches
                        task = asyncio.create_task(
                            self.run_single_game(pA, pB), name=f"match-new-{pair_names}"
                        )
                        self._active_tasks.add(task)
                        new_tasks_spawned += 1
                        scheduled_something = True
                        logger.info(
                            f"Starting new match between {pair_names} ({len(self._active_tasks)}/{self.max_parallel_games} active)"
                        )
                        # Add callback to remove task from set when done
                        task.add_done_callback(lambda t: self._active_tasks.discard(t))
                    else:
                        logger.debug("No new matches available to schedule right now.")

            # Check if we should exit the loop
            if not self._active_tasks and (
                self._stop_scheduling or new_tasks_spawned == 0
            ):
                # If stopping, or if we couldn't schedule anything and nothing is running
                logger.info(
                    "No active matches and no new matches scheduled. Exiting loop."
                )

                break

            # If we spawned new tasks, log current state before waiting
            if new_tasks_spawned > 0:
                # Update cost tracking when starting new matches
                if self.cost_budget is not None:
                    # Only log if cost changed significantly and enough time passed
                    cost_changed = abs(total_cost - last_logged_cost) > 0.000001
                    time_elapsed = current_time - last_log_time > MIN_LOG_INTERVAL
                    if cost_changed and time_elapsed:
                        logger.info(
                            f"Current total cost: ${total_cost:.6f} / ${self.cost_budget:.6f}"
                        )
                        last_log_time = current_time
                        last_logged_cost = total_cost
                self.log_standings()
                try:
                    self.log_pairwise_confidences()
                except RuntimeError as e:
                    logger.warning(
                        f"Could not log pairwise confidences: {e}"
                    )  # Log as warning

            # Wait for tasks to complete or timeout
            timeout = 1.0  # Use float for timeout

            if self._active_tasks:
                # Wait for any task to complete or timeout
                done, pending = await asyncio.wait(
                    self._active_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=timeout,
                )

                # Process completed tasks
                for finished in done:
                    try:
                        await finished
                        # Log completion using task name
                        task_name = (
                            finished.get_name()
                            if hasattr(finished, "get_name")
                            else "Unknown Task"
                        )
                        logger.info(f"Match task completed: {task_name}")
                    except asyncio.CancelledError:
                        task_name = (
                            finished.get_name()
                            if hasattr(finished, "get_name")
                            else "Unknown Task"
                        )
                        logger.warning(f"Match task cancelled: {task_name}")
                    except Exception as e:
                        task_name = (
                            finished.get_name()
                            if hasattr(finished, "get_name")
                            else "Unknown Task"
                        )
                        logger.error(
                            f"Match task failed: {task_name} - {e}", exc_info=True
                        )  # Log traceback
            elif not self._stop_scheduling:  # Only sleep if not stopping
                # No active tasks, wait before checking again
                await asyncio.sleep(timeout)
            # If stopping and no active tasks, the loop condition will handle exit

    def current_elo(self) -> Tuple[EloSystem, Dict[str, PlayerRating]]:
        """Calculates the current Elo ratings based on completed matches."""
        # Fetch completed matches for the current experiment
        completed_matches = (
            self.session.query(GameMatch)
            .filter(
                GameMatch.experiment_id == self.experiment.id,
                GameMatch.complete.is_(True),
            )
            .all()
        )
        # Get all players involved in the experiment
        players = self.experiment.get_players(self.session)
        # Build the history in the format required by EloSystem
        match_history = build_match_history(completed_matches, players)

        # Use a cache to store the latest computed ratings
        if not hasattr(self, "_latest_ratings_cache"):
            self._latest_ratings_cache = None

        # Create a key based on match_history and sorted player names
        cache_key = (
            tuple(
                sorted(
                    match_history,
                    key=lambda x: (x.player_0, x.player_1, x.winner or "None"),
                )
            ),
            tuple(sorted(p.name for p in players)),
        )

        # Check if the result is already cached
        if self._latest_ratings_cache and self._latest_ratings_cache[0] == cache_key:
            return self._latest_ratings_cache[1]
        else:
            elo = EloSystem(self.confidence_threshold)
            ratings = elo.update_ratings(match_history, [p.name for p in players])
            self._latest_ratings_cache = (cache_key, (elo, ratings))
            return (elo, ratings)

    def log_pairwise_confidences(self):
        """Log confidence levels between adjacent players sorted by rating"""
        logger.info("\nPairwise Confidences (Based on current Elo):")
        sorted_players = sorted(
            self.players, key=lambda p: p.rating.rating, reverse=True
        )

        try:
            elo, _ = self.current_elo()
        except ValueError:  # Handle case with no match history yet
            logger.warning(
                "Cannot calculate pairwise confidences: No completed matches yet."
            )
            return

        for i in range(len(sorted_players) - 1):
            player_a = sorted_players[i]
            player_b = sorted_players[i + 1]

            try:
                prob = elo.probability_stronger(
                    player_a.llm_player.name, player_b.llm_player.name
                )
                logger.info(
                    f"{player_a.llm_player.name} vs {player_b.llm_player.name}: "
                    f"{prob * 100:.1f}% confident that {player_a.llm_player.name} is stronger"
                )
            except RuntimeError as e:
                logger.warning(
                    f"Could not calculate confidence for {player_a.llm_player.name} vs {player_b.llm_player.name}: {e}"
                )

    def _get_player_concessions(self, player: ArenaPlayer) -> int:
        """Calculate the number of matches conceded by a player."""
        return (
            self.session.query(func.count(GameMatch.id))
            .filter(
                GameMatch.experiment_id == self.experiment.id,
                GameMatch.conceded.is_(True),
                # Player is either player1 or player2, but NOT the winner
                (
                    (GameMatch.player1_id == player.player_model.id)
                    | (GameMatch.player2_id == player.player_model.id)
                ),
                GameMatch.winner_id != player.player_model.id,
            )
            .scalar()
            or 0
        )

    def get_experiment_results(self) -> Dict[str, Any]:
        """Get summary of experiment results including matches played and final ratings."""
        # Fetch all matches for the experiment
        matches = (
            self.session.query(GameMatch)
            .filter_by(experiment_id=self.experiment.id)
            .all()
        )

        # Get final ratings from the player objects (which should be updated by run_single_game)
        player_ratings = {p.llm_player.name: p.rating.rating for p in self.players}

        # Include budget information if applicable
        budget_info = None
        if self.cost_budget is not None:
            total_cost = self._get_total_cost_for_all_players()
            budget_info = {
                "budget": self.cost_budget,
                "total_cost": total_cost,
                "budget_exceeded": self._budget_exceeded,
            }

        # Calculate concessions per player using the helper method
        player_concessions = {
            p.llm_player.name: self._get_player_concessions(p) for p in self.players
        }

        # Use utility functions for counts
        draws = count_draws(matches)
        completed_matches = count_complete_games(matches)  # Renamed function internally

        results = {
            "experiment_id": self.experiment.id,
            "experiment_name": self.experiment.name,
            "total_matches": len(matches),
            "completed_matches": completed_matches,
            "draws": draws,
            "player_ratings": player_ratings,
            "player_concessions": player_concessions,
            "budget_info": budget_info,
            "matches": [],  # Changed key from 'games' to 'matches'
        }

        # Add details for each completed match
        for match in matches:
            if is_game_complete(match):  # Use utility function
                # Find corresponding ArenaPlayers to get names easily
                p1_arena = next(
                    (p for p in self.players if p.player_model.id == match.player1_id),
                    None,
                )
                p2_arena = next(
                    (p for p in self.players if p.player_model.id == match.player2_id),
                    None,
                )
                winner_arena = (
                    next(
                        (
                            p
                            for p in self.players
                            if p.player_model.id == match.winner_id
                        ),
                        None,
                    )
                    if match.winner_id
                    else None
                )

                match_entry = {
                    "match_id": match.id,
                    "player1": p1_arena.llm_player.name
                    if p1_arena
                    else f"ID:{match.player1_id}",
                    "player2": p2_arena.llm_player.name
                    if p2_arena
                    else f"ID:{match.player2_id}",
                    "winner": winner_arena.llm_player.name if winner_arena else None,
                    "is_draw": is_game_draw(match),  # Use utility function
                    "conceded": match.conceded,
                    "concession_reason": match.concession_reason,
                    # Optionally add final ratings snapshot if needed, but it's complex
                    # "final_ratings_snapshot": {p.llm_player.name: p.rating.rating for p in self.players} # This reflects ratings *after* this match potentially
                }
                results["matches"].append(match_entry)

        return results

    def _get_total_cost_for_all_players(self) -> float:
        """Calculate total cost of LLM interactions for all players in this experiment."""
        total_cost = (
            self.session.query(func.sum(LLMInteraction.cost))
            .join(GameMatch, LLMInteraction.game_id == GameMatch.id)
            .filter(GameMatch.experiment_id == self.experiment.id)
            .scalar()
            or 0.0
        )

        return total_cost

    def _get_player_cost(self, player: ArenaPlayer) -> float:
        """Calculate total cost of LLM interactions for a player in this experiment."""
        total_cost = (
            self.session.query(func.sum(LLMInteraction.cost))
            .join(GameMatch, LLMInteraction.game_id == GameMatch.id)
            .filter(
                GameMatch.experiment_id == self.experiment.id,
                LLMInteraction.player_id == player.player_model.id,
            )
            .scalar()
            or 0.0
        )

        return total_cost
