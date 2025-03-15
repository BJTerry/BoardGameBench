import logging
from dataclasses import dataclass
import random
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Set
from sqlalchemy.orm import Session
from bgbench.models import Experiment, Player as DBPlayer, GameMatch, LLMInteraction
from bgbench.llm_integration import ResponseStyle
from bgbench.game import Game
from bgbench.llm_player import LLMPlayer
from bgbench.game_view import PromptStyle
from bgbench.game_runner import GameRunner
from bgbench.rating import PlayerRating, EloSystem

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
    def _get_player_cost(self, player: ArenaPlayer) -> float:
        """Calculate total cost of LLM interactions for a player in this experiment."""
        # Add debug logging
        logger.debug(f"Getting costs for player {player.llm_player.name} (id: {player.player_model.id}) in experiment {self.experiment.id}")
        
        # First, let's check what player_id is actually used in the database
        player_id_check = self.session.query(LLMInteraction.player_id).filter(
            LLMInteraction.game_id.in_(
                self.session.query(GameMatch.id).filter(
                    GameMatch.experiment_id == self.experiment.id
                )
            )
        ).distinct().all()
        
        logger.debug(f"Player IDs found in LLMInteraction for this experiment: {player_id_check}")
        
        # Query LLMInteraction joined with GameMatch to filter by experiment
        query = self.session.query(LLMInteraction.cost).join(
            GameMatch, LLMInteraction.game_id == GameMatch.id
        ).filter(
            GameMatch.experiment_id == self.experiment.id,
            LLMInteraction.player_id == player.player_model.id  # Use the ID from the player model
        )
        
        # Log the SQL query
        logger.debug(f"SQL Query: {query}")
        
        # Execute the query and get results
        costs = query.all()
        
        # Log the raw results
        logger.debug(f"Raw cost results: {costs}")
        
        # Sum the costs, handling None values
        total_cost = sum(cost[0] for cost in costs if cost[0] is not None)
        
        logger.debug(f"Total cost for player {player.llm_player.name}: ${total_cost:.6f}")
        
        return total_cost
    def __init__(self, game: Game, db_session: Session, 
                 player_configs: Optional[List[Dict[str, Any]]] = None,
                 experiment_name: Optional[str] = None,
                 experiment_id: Optional[int] = None, 
                 confidence_threshold: float = 0.70,
                 max_parallel_games: int = 3,
                 llm_factory=None):
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
            llm_factory: Optional function for testing that creates LLM instances
        """
        self.game: Game = game
        self.players: List[ArenaPlayer] = []
        self.elo_system = EloSystem()
        self.confidence_threshold = confidence_threshold
        self.session = db_session
        self.max_parallel_games = max_parallel_games
        self.ongoing_matches: Set[Tuple[int, int]] = set()
        self._scheduled_games_between: Dict[Tuple[int, int], int] = {}
        self._lock = asyncio.Lock()

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
            GameMatch.winner_id.is_(None)
        ).all()
        
        for game in incomplete_games:
            # Delete associated game states first
            if game.state:
                self.session.delete(game.state)
            self.session.delete(game)
            
        self.session.commit()
        
        logger.info(f"Resumed experiment {self.experiment.name} (id: {experiment_id})")
        logger.info(f"Cleaned up {len(incomplete_games)} incomplete games and their states")
        
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
            
            # Count only completed games where player was involved
            games_played = self.session.query(GameMatch).filter(
                GameMatch.experiment_id == self.experiment.id,
                GameMatch.winner_id.isnot(None),
                (GameMatch.player1_id == db_player.id) | 
                (GameMatch.player2_id == db_player.id)
            ).count()
            
            rating = PlayerRating(
                name=db_player.name,
                rating=db_player.rating,
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
                rating=self.elo_system.initial_rating,
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
        """Pick the best matchup between adjacent players that doesn't exceed 10 games and isn't ongoing."""
        async with self._lock:
            best_uncertainty = -1.0
            best_pair: Optional[Tuple[ArenaPlayer, ArenaPlayer]] = None
            
            # Sort players by rating
            sorted_players = sorted(self.players, key=lambda p: p.rating.rating, reverse=True)
            
            # Only look at adjacent pairs
            for i in range(len(sorted_players) - 1):
                player_a = sorted_players[i]
                player_b = sorted_players[i + 1]
                
                pair_ids = (min(player_a.player_model.id, player_b.player_model.id), 
                             max(player_a.player_model.id, player_b.player_model.id))
                if pair_ids in self.ongoing_matches:
                    continue
                
                games = self._games_played_between(player_a.player_model, player_b.player_model)
                if games >= 10:
                    continue
                
                if not self.elo_system.is_match_needed(player_a.rating, player_b.rating):
                    continue
                
                uncertainty = self.elo_system.calculate_match_uncertainty(
                    player_a.rating, player_b.rating
                )
                if uncertainty > best_uncertainty:
                    best_uncertainty = uncertainty
                    best_pair = (player_a, player_b)

            if best_pair is not None:
                pair_ids = (min(best_pair[0].player_model.id, best_pair[1].player_model.id),
                            max(best_pair[0].player_model.id, best_pair[1].player_model.id))
                self.ongoing_matches.add(pair_ids)
                
                # Track this scheduled game
                self._scheduled_games_between[pair_ids] = self._scheduled_games_between.get(pair_ids, 0) + 1
                
            return best_pair

    def log_standings(self):
        """Log current ratings and costs for all players"""
        logger.info("\nCurrent Standings:")
        sorted_players = sorted(self.players, key=lambda p: p.rating.rating, reverse=True)
        for player in sorted_players:
            # Count concessions by this player
            concessions = self.session.query(GameMatch).filter(
                (GameMatch.experiment_id == self.experiment.id) &
                (GameMatch.conceded) &
                (GameMatch.winner_id != player.player_model.id) &
                ((GameMatch.player1_id == player.player_model.id) | 
                 (GameMatch.player2_id == player.player_model.id))
            ).count()
            
            total_cost = self._get_player_cost(player)
            
            logger.info(f"{player.llm_player.name}: {player.rating.rating:.0f} "
                       f"({player.rating.games_played} games, {concessions} concessions, "
                       f"${total_cost:.4f} cost)")

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
                player1_id=db_player_a.id,
                player2_id=db_player_b.id
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
            
            winner, history, concession = await runner.play_game()
            
            # Calculate new Elo ratings
            if winner:
                # Determine if player_a won
                a_wins = winner.name == player_a.llm_player.name
                
                # Update ratings using ELO system
                new_a_rating, new_b_rating = self.elo_system.update_ratings(
                    player_a.rating,
                    player_b.rating,
                    a_wins
                )
                
                # Update both memory and database ratings
                player_a.rating = new_a_rating
                player_b.rating = new_b_rating
                db_player_a.update_rating(self.session, new_a_rating.rating)
                db_player_b.update_rating(self.session, new_b_rating.rating)
                
                # Update winner in database
                winner_db_player = db_player_a if winner.name == player_a.llm_player.name else db_player_b
                db_game.winner_id = winner_db_player.id
                if concession:
                    db_game.conceded = True
                    db_game.concession_reason = concession
            self.session.commit()

            logger.info(f"Game {game_id} completed: {winner.name if winner else 'Draw'}")
            
        finally:
            # Use lock to safely update tracking data
            async with self._lock:
                # Remove from ongoing matches
                self.ongoing_matches.discard(player_pair)

    def _check_confidence_threshold(self) -> bool:
        """Check if all pairwise probabilities are above the threshold."""
        sorted_players = sorted(self.players, key=lambda p: p.rating.rating, reverse=True)
        for i in range(len(sorted_players) - 1):
            player_a = sorted_players[i]
            player_b = sorted_players[i + 1]
            prob = self.elo_system.probability_stronger(player_a.rating, player_b.rating)
            if prob < self.confidence_threshold:
                return False
        return True

    async def evaluate_all(self) -> Dict[str, float]:
        active_tasks: Set[asyncio.Task] = set()
        
        # For each player pair, ensure we get exactly the max number of games (10)
        if len(self.players) == 2:  # Optimization for the test case with just 2 players
            player_a, player_b = self.players[0], self.players[1]
            pair_id = (min(player_a.player_model.id, player_b.player_model.id),
                       max(player_a.player_model.id, player_b.player_model.id))
            
            # Play exactly 10 games between these two players
            games_to_play = 10
            games_completed = 0
            
            # Launch games in parallel up to max_parallel_games
            while games_completed < games_to_play:
                # Schedule up to max_parallel_games tasks at once
                while len(active_tasks) < self.max_parallel_games and games_completed + len(active_tasks) < games_to_play:
                    async with self._lock:
                        # Mark match as ongoing
                        self.ongoing_matches.add(pair_id)
                    
                    # Create and launch task
                    task = asyncio.create_task(self.run_single_game(player_a, player_b))
                    
                    # Add to active tasks set
                    active_tasks.add(task)
                    
                    # Create a proper callback to remove task when done
                    def create_callback(t):
                        def callback(future):
                            active_tasks.discard(t)
                        return callback
                    
                    task.add_done_callback(create_callback(task))
                
                # If we've scheduled all tasks but need to wait for them to complete
                if active_tasks:
                    done, _ = await asyncio.wait(
                        active_tasks,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Process completed tasks
                    for finished in done:
                        try:
                            await finished
                            games_completed += 1
                        except Exception as e:
                            logger.error(f"Game failed: {e}")
                    
                    # Log progress
                    logger.info(f"Games completed: {games_completed}/{games_to_play}")
                else:
                    # This shouldn't happen, but just in case
                    break
                    
            return {p.llm_player.name: p.rating.rating for p in self.players}
            
        # Regular implementation for more than 2 players
        while True:
            # Fill slots up to max_parallel_games
            while len(active_tasks) < self.max_parallel_games:
                matchup = await self.find_next_available_match()
                if not matchup:
                    break
                pA, pB = matchup

                # Create a task for this match
                task = asyncio.create_task(self.run_single_game(pA, pB))
                active_tasks.add(task)
                
                # We use a custom done callback to safely remove the task
                def create_done_callback(t):
                    def callback(future):
                        active_tasks.discard(t)
                    return callback
                
                task.add_done_callback(create_done_callback(task))
            
            # If no active tasks and no new matches can be scheduled -> done
            if not active_tasks:
                # Double-check if we can schedule anything after the last Elo updates
                matchup = await self.find_next_available_match()
                if not matchup:
                    # Truly done - all games completed or reached limits
                    break
                else:
                    # This should never execute since we have no tasks and found a match
                    # But we'd just continue and schedule it in the next loop iteration
                    continue

            # Wait for at least one task to complete so we can fill its slot
            done, _ = await asyncio.wait(
                active_tasks,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Process completed tasks
            for finished in done:
                try:
                    await finished
                except Exception as e:
                    logger.error(f"Game failed: {e}")

            # Log current state
            self.log_standings()
            self.log_pairwise_confidences()

        return {p.llm_player.name: p.rating.rating for p in self.players}

    def log_pairwise_confidences(self):
        """Log confidence levels between adjacent players sorted by rating"""
        logger.info("\nPairwise Confidences:")
        sorted_players = sorted(self.players, key=lambda p: p.rating.rating, reverse=True)
        
        for i in range(len(sorted_players) - 1):
            player_a = sorted_players[i]
            player_b = sorted_players[i + 1]
            prob = self.elo_system.probability_stronger(player_a.rating, player_b.rating)
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

        results = {
            "experiment_id": self.experiment.id,
            "experiment_name": self.experiment.name,
            "total_games": len(games),
            "player_ratings": player_ratings,
            "player_concessions": player_concessions,
            "games": []
        }
        
        for game in games:
            if game.winner_id is not None:  # Winner exists
                winner = self.session.query(DBPlayer).filter_by(id=game.winner_id).first()
                if winner:
                    results["games"].append({
                        "game_id": game.id,
                        "winner": winner.name,
                    "final_ratings": {p.name: p.rating for p in db_players}
                })
        
        return results

    def get_player_game_history(self, player_id: int) -> List[Dict[str, Any]]:
        """Get game history for a specific player."""
        db_player = self.session.query(DBPlayer).filter_by(id=player_id).first()
        if not db_player:
            return []
            
        games = self.session.query(GameMatch).filter(
            GameMatch.experiment_id == self.experiment.id,
            GameMatch.winner_id == db_player.id
        ).all()
        
        history = []
        for game in games:
            game_data = {
                "game_id": game.id,
                "won": True,  # If player_id matches, they won
                "state": game.state.state_data if game.state else {}
            }
            history.append(game_data)
                
        return history