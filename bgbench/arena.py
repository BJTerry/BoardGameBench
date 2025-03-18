import logging
import time
from dataclasses import dataclass
import random
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Set
from sqlalchemy import func
from sqlalchemy.orm import Session
from bgbench.models import Experiment, Player as DBPlayer, GameMatch, LLMInteraction
from bgbench.llm_integration import ResponseStyle
from bgbench.game import Game
from bgbench.llm_player import LLMPlayer
from bgbench.game_view import PromptStyle
from bgbench.game_runner import GameRunner
from bgbench.bayes_rating import PlayerRating, EloSystem, GameResult
from bgbench.export import is_game_complete, is_game_draw, count_complete_games, count_draws, build_match_history
from bgbench.scheduler import MatchScheduler, FullRankingScheduler

logger = logging.getLogger("bgbench")

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

class Arena():
    def __init__(self, game: Game, db_session: Session, 
                 player_configs: Optional[List[Dict[str, Any]]] = None,
                 experiment_name: Optional[str] = None,
                 experiment_id: Optional[int] = None, 
                 confidence_threshold: float = 0.70,
                 max_parallel_games: int = 3,
                 cost_budget: Optional[float] = 2.0,
                 llm_factory=None,
                 match_scheduler: Optional[MatchScheduler] = None):
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
            match_scheduler: Optional scheduler for choosing player matchups (defaults to FullRankingScheduler)
        """
        self.game: Game = game
        self.players: List[ArenaPlayer] = []
        self.confidence_threshold = confidence_threshold
        self.session = db_session
        self.max_parallel_games = max_parallel_games
        self.cost_budget = cost_budget
        self.ongoing_matches: Set[Tuple[int, int]] = set()
        self._scheduled_games_between: Dict[Tuple[int, int], int] = {}
        self._lock = asyncio.Lock()
        
        # Set up the match scheduler (default to FullRankingScheduler if not provided)
        if match_scheduler is None:
            self.match_scheduler = FullRankingScheduler()
        else:
            self.match_scheduler = match_scheduler
        
        # Store completed matches history for Bayesian rating updates
        self.match_history: List[GameResult] = []

        if experiment_id is not None:
            self._resume_experiment(experiment_id, llm_factory)
        else:
            if player_configs is None:
                raise ValueError("player_configs required when creating new experiment")
            self._create_new_experiment(experiment_name, player_configs, llm_factory)

    def _resume_experiment(self, experiment_id: int, llm_factory=None):
        """Resume an existing experiment and its players."""
        self.experiment = Experiment.resume_experiment(self.session, experiment_id)
        if not self.experiment:
            raise ValueError(f"No experiment found with ID {experiment_id}")
        
        # Clean up incomplete games and their states
        incomplete_games = self.session.query(GameMatch).filter(
            GameMatch.experiment_id == experiment_id,
            GameMatch.complete.is_(False)  # Using the complete field to determine if a game is completed
        ).all()
        
        for game in incomplete_games:
            # Delete associated game states first
            if game.state:
                self.session.delete(game.state)
            self.session.delete(game)
            
        self.session.commit()
        
        logger.info(f"Resumed experiment {self.experiment.name} (id: {experiment_id})")
        logger.info(f"Cleaned up {len(incomplete_games)} incomplete games and their states")
        
        # Load all completed games for the match history - including draws
        completed_games = self.session.query(GameMatch).filter(
            GameMatch.experiment_id == self.experiment.id,
            GameMatch.complete.is_(True)  # Using the complete field from our utility
        ).all()
        
        # Build match history using the utility function
        players = self.experiment.get_players(self.session)
        self.match_history = build_match_history(completed_games, players)
        
        elo, player_ratings = self.current_elo()
        
        for db_player in self.experiment.players:
            if llm_factory:
                llm_player = LLMPlayer(
                    db_player.name,
                    db_player.model_config,
                    _llm=llm_factory(db_player.name),
                )
            else:
                try:
                    llm_player = LLMPlayer(db_player.name, db_player.model_config)
                except Exception as e:
                    logger.error(f"Could not recreate LLM for player {db_player.name}: {e}")
                    continue
            
            # Always use the freshly calculated Bayesian rating for consistency
            # This ensures we use the same calculation process as in run_single_game
            if db_player.name in player_ratings:
                rating = player_ratings[db_player.name]
            else:
                # Count games for this player
                games_played = self.session.query(GameMatch).filter(
                    GameMatch.experiment_id == self.experiment.id,
                    GameMatch.winner_id.isnot(None),
                    (GameMatch.player1_id == db_player.id) | 
                    (GameMatch.player2_id == db_player.id)
                ).count()
                
                # Use default 1500±400 prior for players with no history
                rating = PlayerRating(
                    name=db_player.name,
                    rating=1500.0,  # Default initial rating
                    sigma=400.0,    # Default uncertainty
                    games_played=games_played
                )
            
            self.players.append(ArenaPlayer(llm_player, rating, db_player))

    def _create_new_experiment(self, experiment_name: Optional[str], 
                             player_configs: List[Dict[str, Any]], 
                             llm_factory=None):
        """Create a new experiment with the specified players."""
        name = experiment_name or f"{self.game.__class__.__name__}_evaluation"
        self.experiment = Experiment().create_experiment(
            self.session,
            name=name,
            description=f"Evaluation of LLMs playing {self.game.__class__.__name__}"
        )

        for config in player_configs:
            if llm_factory:
                # For testing purposes
                # Create LLM player with mock for testing
                mock_llm = llm_factory(config["name"])
                llm_player = LLMPlayer(
                    config["name"],
                    config["model_config"],
                    prompt_style=PromptStyle[config.get("prompt_style", "header").upper()],
                    response_style=ResponseStyle[config.get("response_style", "direct").upper()],
                    _llm=mock_llm,
                )
            else:
                llm_player = LLMPlayer(
                    config["name"],
                    config["model_config"],
                    prompt_style=PromptStyle[config.get("prompt_style", "header").upper()],
                    response_style=ResponseStyle[config.get("response_style", "direct").upper()]
                )
            
            # Create database player
            db_player = DBPlayer().create_player(
                self.session,
                config["name"],
                config["model_config"],
                self.experiment.id
            )
            
            # Create arena player with initial values
            rating = PlayerRating(
                name=config["name"],
                rating=1500.0,  # Default initial rating
                sigma=400.0,    # Default uncertainty
                games_played=0
            )
            self.players.append(ArenaPlayer(llm_player, rating, db_player))
        

    def _games_played_between(self, player_a: DBPlayer, player_b: DBPlayer) -> int:
        """Return the number of games played between two players in this experiment."""
        # Get the count from the database
        db_count = self.session.query(GameMatch).filter(
            GameMatch.experiment_id == self.experiment.id,
            ((GameMatch.player1_id == player_a.id) & (GameMatch.player2_id == player_b.id)) |
            ((GameMatch.player1_id == player_b.id) & (GameMatch.player2_id == player_a.id))
        ).count()
        
        # Get the count of scheduled games between these players
        pair_id = (min(player_a.id, player_b.id), max(player_a.id, player_b.id))
        scheduled_count = self._scheduled_games_between.get(pair_id, 0)
        
        return db_count + scheduled_count

    async def find_next_available_match(self) -> Optional[Tuple[ArenaPlayer, ArenaPlayer]]:
        """
        Find the next match to run based on the configured match scheduling strategy.
        
        Uses the scheduler set during initialization (defaults to FullRankingScheduler)
        to determine the best matchup based on the scheduling strategy.
        """
        # Get a snapshot of ongoing matches for the scheduler
        async with self._lock:
            ongoing = self.ongoing_matches.copy()
        
        # Use the scheduler to find the next match
        # This doesn't modify any shared state yet
        elo, _ = self.current_elo()
        match_pair = self.match_scheduler.find_match(
            self.players, 
            self.match_history, 
            elo,
            ongoing,
            max_games_per_pairing=10
        )
        
        if match_pair is None:
            logger.debug("No candidate matches found by scheduler")
            return None
            
        player_a, player_b = match_pair
        pair_ids = (min(player_a.player_model.id, player_b.player_model.id), 
                     max(player_a.player_model.id, player_b.player_model.id))
        
        # Now lock to update shared state
        async with self._lock:
            # Double check match is still valid (could have changed while we were processing)
            if pair_ids not in self.ongoing_matches:
                games = self._games_played_between(player_a.player_model, player_b.player_model)
                if games < 10:
                    logger.debug(f"Scheduling match between {player_a.llm_player.name} and {player_b.llm_player.name}")
                    self.ongoing_matches.add(pair_ids)
                    # Track this scheduled game
                    self._scheduled_games_between[pair_ids] = self._scheduled_games_between.get(pair_ids, 0) + 1
                    return match_pair
                else:
                    logger.debug(f"Candidate match rejected - exceeded game limit ({games}/10): {player_a.llm_player.name} vs {player_b.llm_player.name}")
            else:
                logger.debug(f"Candidate match became invalid due to ongoing match: {player_a.llm_player.name} vs {player_b.llm_player.name}")
        
        return None

    def log_standings(self):
        """Log current ratings and costs for all players"""
        logger.info("\nCurrent Standings:")
        sorted_players = sorted(self.players, key=lambda p: p.rating.rating, reverse=True)
        
        total_cost = 0.0
        for player in sorted_players:
            # Count concessions by this player
            concessions = self.session.query(GameMatch).filter(
                (GameMatch.experiment_id == self.experiment.id) &
                (GameMatch.conceded) &
                (GameMatch.winner_id != player.player_model.id) &
                ((GameMatch.player1_id == player.player_model.id) | 
                 (GameMatch.player2_id == player.player_model.id))
            ).count()
            
            player_cost = self._get_player_cost(player)
            total_cost += player_cost
            
            logger.info(f"{player.llm_player.name}: {player.rating.rating:.0f} "
                       f"({player.rating.games_played} games, {concessions} concessions, "
                       f"${player_cost:.4f} cost)")
        
        if self.cost_budget:
            logger.info(f"Total cost: ${total_cost:.4f} / ${self.cost_budget:.4f} " 
                       f"({total_cost/self.cost_budget*100:.1f}% of budget)")
        else:
            logger.info(f"Total cost: ${total_cost:.4f}")

    async def run_single_game(self, player_a: ArenaPlayer, player_b: ArenaPlayer):
        """Run a single game and update ratings."""
        # Get database players at the start
        db_player_a = player_a.player_model
        db_player_b = player_b.player_model
        player_pair = (min(db_player_a.id, db_player_b.id), max(db_player_a.id, db_player_b.id))
        
        try:
            # Randomize player order
            player_a, player_b = random.sample([player_a, player_b], 2)

            # Create game record
            db_game = GameMatch(
                experiment_id=self.experiment.id,
                winner_id=None,
                player1_id=player_a.player_model.id,
                player2_id=player_b.player_model.id
            )
            self.session.add(db_game)
            self.session.commit()
            self.session.refresh(db_game)
            game_id = int(db_game.id)

            runner = GameRunner(
                self.game, 
                player_a.llm_player, 
                player_b.llm_player, 
                self.session,
                game_id,
                player_a.player_model.id,  # Pass the correct player IDs
                player_b.player_model.id
            )
            
            winner, _, concession = await runner.play_game()
            
            # Mark the game as complete since it finished normally (either with a winner or a draw)
            db_game.complete = True
            
            if winner:
                # Game has a winner
                winner_name = winner.name
                
                # Create a GameResult entry for this match
                new_game = GameResult(
                    player_0=player_a.llm_player.name,
                    player_1=player_b.llm_player.name, 
                    winner=winner_name
                )
                
                # Update winner in database
                winner_db_player = db_player_a if winner.name == player_a.llm_player.name else db_player_b
                db_game.winner_id = winner_db_player.id
                if concession:
                    db_game.conceded = True
                    db_game.concession_reason = concession
            else:
                # Game ended in a draw
                new_game = GameResult(
                    player_0=player_a.llm_player.name,
                    player_1=player_b.llm_player.name, 
                    winner=None  # None indicates a draw
                )
                # No winner_id is set, but the game is marked as complete
            
            self.match_history.append(new_game)
            self.session.commit()
            
            # Current Elo looks up completed matches in the database, so you have to commit first
            _, new_ratings = self.current_elo()
            
            # Update ratings for all players
            for arena_player in self.players:
                name = arena_player.llm_player.name
                if name in new_ratings:
                    # Update both memory and database ratings
                    arena_player.rating = new_ratings[name]
                    arena_player.player_model.update_rating(self.session, new_ratings[name].rating)
            
            # Commit updates to player ratings
            self.session.commit() 

            logger.info(f"Game {game_id} completed: {winner.name if winner else 'Draw'}")
            
        finally:
            # Use lock to safely update tracking data
            async with self._lock:
                # Remove from ongoing matches
                self.ongoing_matches.discard(player_pair)

    async def evaluate_all(self):
        active_tasks: Set[asyncio.Task] = set()
        
        # Initialize cost tracking variables
        last_log_time = 0
        last_logged_cost = 0
        total_cost = 0.0  # Initialize total_cost
        MIN_LOG_INTERVAL = 10  # Only log costs at most every 10 seconds
        
        while True:
            current_time = time.time()
            
            # Check total cost against budget if specified, but don't log on every iteration
            if self.cost_budget is not None:
                # Calculate current total cost
                total_cost = self._get_total_cost_for_all_players()
                
                # Log only when cost changes AND enough time has passed
                cost_changed = abs(total_cost - last_logged_cost) > 0.000001  # Float comparison with small epsilon
                time_elapsed = current_time - last_log_time > MIN_LOG_INTERVAL
                
                if cost_changed and time_elapsed:
                    logger.info(f"Current total cost: ${total_cost:.6f} / ${self.cost_budget:.6f}")
                    last_log_time = current_time
                    last_logged_cost = total_cost
                
                # Check if budget is exceeded
                if total_cost >= self.cost_budget:
                    logger.info(f"Cost budget of ${self.cost_budget:.6f} reached. Stopping evaluation.")
                    break
                    
            # Track number of new tasks spawned in this iteration
            new_tasks_spawned = 0
            timeout = 0 
            # Fill all available slots up to max_parallel_games
            while len(active_tasks) < self.max_parallel_games:
                matchup = await self.find_next_available_match()
                if not matchup:
                    logger.debug("No more matches available to schedule right now")
                    break
                pA, pB = matchup
                
                # Create a task for this match - using unique name for better logging
                pair_names = f"{pA.llm_player.name}-vs-{pB.llm_player.name}"
                task = asyncio.create_task(
                    self.run_single_game(pA, pB), 
                    name=f"game-{pair_names}"
                )
                
                
                # Add to active tasks and track that we spawned a new task
                active_tasks.add(task)
                new_tasks_spawned += 1

                # Log that we're starting a new game to help diagnose parallelism
                logger.info(f"Starting game between {pair_names} ({len(active_tasks)}/{self.max_parallel_games} active)")
                
                # Create a proper callback that captures the task properly
                def create_done_callback(t):
                    def callback(future):
                        active_tasks.discard(t)
                    return callback
                
                task.add_done_callback(create_done_callback(task))
            
            if not active_tasks and new_tasks_spawned == 0:
                # Nothing currently running and we didn't just launch one
                break
            
            # If we spawned new tasks, log current state before waiting
            if new_tasks_spawned > 0:
                # Update cost tracking when starting new tasks
                if self.cost_budget is not None:
                    # Only log if cost changed significantly and enough time passed
                    cost_changed = abs(total_cost - last_logged_cost) > 0.000001
                    time_elapsed = current_time - last_log_time > MIN_LOG_INTERVAL
                    if cost_changed and time_elapsed:
                        logger.info(f"Current total cost: ${total_cost:.6f} / ${self.cost_budget:.6f}")
                        last_log_time = current_time
                        last_logged_cost = total_cost
                self.log_standings()
                self.log_pairwise_confidences()
            
            # Wait for at least one task to complete so we can fill its slot
            # Use a shorter timeout if we have capacity to schedule more games
            if len(active_tasks) == self.max_parallel_games:
                timeout = None
            elif len(active_tasks) < self.max_parallel_games and timeout is None:
                # If we have capacity but couldn't schedule, wait briefly and retry
                timeout = 0.1
            else:
                timeout = min(timeout * 2, 5)
            
            done, pending = await asyncio.wait(
                active_tasks,
                return_when=asyncio.FIRST_COMPLETED,
                timeout=timeout
            )
            
            # Process completed tasks
            for finished in done:
                try:
                    await finished
                    # Log completion
                    logger.info(f"Game completed: {finished.get_name() if hasattr(finished, 'get_name') else 'Unknown'}")
                except Exception as e:
                    logger.error(f"Game failed: {e}")


    def current_elo(self) -> Tuple[EloSystem, Dict[str, PlayerRating]]:
        completed_games = self.session.query(GameMatch).filter(
            GameMatch.experiment_id == self.experiment.id,
            GameMatch.complete.is_(True)  # Using the complete field from our utility
        ).all()
        players = self.experiment.get_players(self.session)
        match_history = build_match_history(completed_games, players)

        # Use a cache to store the latest computed ratings
        if not hasattr(self, '_latest_ratings_cache'):
            self._latest_ratings_cache = None

        # Create a key based on match_history and sorted player names
        cache_key = (tuple(sorted(match_history, key=lambda x: (x.player_0, x.player_1, x.winner or "None"))),
                     tuple(sorted(p.name for p in players)))

        # Check if the result is already cached
        if self._latest_ratings_cache and self._latest_ratings_cache[0] == cache_key:
            return self._latest_ratings_cache[1]
        else:
            elo = EloSystem(self.confidence_threshold)
            ratings = elo.update_ratings(
                match_history, [p.name for p in players]
            )
            self._latest_ratings_cache = (cache_key, (elo, ratings))
            return (elo, ratings)

    def log_pairwise_confidences(self):
        """Log confidence levels between adjacent players sorted by rating"""
        logger.info("\nPairwise Confidences:")
        sorted_players = sorted(self.players, key=lambda p: p.rating.rating, reverse=True)
        
        elo, _ = self.current_elo()
        
        for i in range(len(sorted_players) - 1):
            player_a = sorted_players[i]
            player_b = sorted_players[i + 1]
            
            prob = elo.probability_stronger(player_a.llm_player.name, player_b.llm_player.name)
            logger.info(f"{player_a.llm_player.name} vs {player_b.llm_player.name}: "
                        f"{prob*100:.1f}% confident")

    def get_experiment_results(self) -> Dict[str, Any]:
        """Get summary of experiment results including games played and final ratings."""
        games = self.session.query(GameMatch).filter_by(experiment_id=self.experiment.id).all()
        
        # Get all players associated with this experiment
        db_players = self.experiment.get_players(self.session)
        player_ratings = {p.name: p.rating for p in db_players}
        
        # Calculate concessions per player
        player_concessions = {}
        for player in db_players:
            concessions = self.session.query(GameMatch).filter(
                (GameMatch.experiment_id == self.experiment.id) &
                (GameMatch.conceded) &
                (GameMatch.winner_id != player.id) &
                ((GameMatch.player1_id == player.id) | 
                 (GameMatch.player2_id == player.id))
            ).count()
            player_concessions[player.name] = concessions

        draws = count_draws(games)
        completed_games = count_complete_games(games)
        
        results = {
            "experiment_id": self.experiment.id,
            "experiment_name": self.experiment.name,
            "total_games": len(games),
            "completed_games": completed_games,
            "draws": draws,
            "player_ratings": player_ratings,
            "player_concessions": player_concessions,
            "games": []
        }
        
        for game in games:
            if is_game_complete(game):
                result_entry = {
                    "game_id": game.id,
                    "final_ratings": {p.name: p.rating for p in db_players}
                }
                
                if game.winner_id is not None:  # Has winner
                    winner = self.session.query(DBPlayer).filter_by(id=game.winner_id).first()
                    if winner:
                        result_entry["winner"] = winner.name
                elif is_game_draw(game):  # Using the utility function to check if it's a draw
                    result_entry["result"] = "draw"
                    
                results["games"].append(result_entry)
        
        return results

    def _get_total_cost_for_all_players(self) -> float:
        """Calculate total cost of LLM interactions for all players in this experiment."""
        total_cost = self.session.query(
            func.sum(LLMInteraction.cost)
        ).join(
            GameMatch, LLMInteraction.game_id == GameMatch.id
        ).filter(
            GameMatch.experiment_id == self.experiment.id
        ).scalar() or 0.0
        
        return total_cost

    def _get_player_cost(self, player: ArenaPlayer) -> float:
        """Calculate total cost of LLM interactions for a player in this experiment."""
        total_cost = self.session.query(
            func.sum(LLMInteraction.cost)
        ).join(
            GameMatch, LLMInteraction.game_id == GameMatch.id
        ).filter(
            GameMatch.experiment_id == self.experiment.id,
            LLMInteraction.player_id == player.player_model.id
        ).scalar() or 0.0
        
        return total_cost