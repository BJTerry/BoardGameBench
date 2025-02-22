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
        """Calculate total cost of LLM interactions for a player."""
        costs = self.session.query(LLMInteraction.cost).join(
            GameMatch, LLMInteraction.game_id == GameMatch.id
        ).filter(
            GameMatch.experiment_id == self.experiment.id,
            LLMInteraction.player_id == player.player_model.id
        ).all()
        return sum(cost[0] for cost in costs if cost[0] is not None)
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
        count = self.session.query(GameMatch).filter(
            GameMatch.experiment_id == self.experiment.id,
            ((GameMatch.player1_id == player_a.id) & (GameMatch.player2_id == player_b.id)) |
            ((GameMatch.player1_id == player_b.id) & (GameMatch.player2_id == player_a.id))
        ).count()
        return count

    async def find_next_available_match(self) -> Optional[Tuple[ArenaPlayer, ArenaPlayer]]:
        """Pick the best matchup between adjacent players that doesn't exceed 10 games and isn't ongoing."""
        best_uncertainty = -1.0
        best_pair: Optional[Tuple[ArenaPlayer, ArenaPlayer]] = None
        
        # Sort players by rating
        sorted_players = sorted(self.players, key=lambda p: p.rating.rating, reverse=True)
        
        # Only look at adjacent pairs
        for i in range(len(sorted_players) - 1):
            player_a = sorted_players[i]
            player_b = sorted_players[i + 1]
            
            pair_ids = tuple(sorted([player_a.player_model.id, player_b.player_model.id]))
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
            pair_ids = tuple(sorted([best_pair[0].player_model.id, best_pair[1].player_model.id]))
            self.ongoing_matches.add(pair_ids)
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
        player_pair = tuple(sorted([db_player_a.id, db_player_b.id]))
        
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
                game_id
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
            # Remove from ongoing matches without lock
            self.ongoing_matches.discard(player_pair)  # discard is safe if pair isn't present

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
                # Remove from active set on completion
                task.add_done_callback(active_tasks.discard)
            
            # If no active tasks, no new match -> done
            if not active_tasks:
                # Double-check if we can schedule anything after the last Elo updates
                # If find_next_available_match is empty again, we are truly done
                matchup = await self.find_next_available_match()
                if not matchup:
                    break
                else:
                    # matchup found, but no slot is free => continue 
                    # Actually, this shouldn’t happen because we have no tasks, so we do have a slot. 
                    # But keep it for safety:
                    pair_ids = tuple(sorted([matchup[0].player_model.id, matchup[1].player_model.id]))
                    self.ongoing_matches.discard(pair_ids)
                    continue

            # Wait for just one task to complete so we can fill its slot
            done, _ = await asyncio.wait(
                active_tasks,
                return_when=asyncio.FIRST_COMPLETED
            )
            for finished in done:
                try:
                    await finished
                except Exception as e:
                    logger.error(f"Game failed: {e}")

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
