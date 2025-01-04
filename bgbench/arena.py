import logging
from dataclasses import dataclass
import random
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic_ai import Agent
from sqlalchemy.orm import Session
from bgbench.models import Experiment, Player as DBPlayer, GameMatch
from bgbench.llm_integration import ResponseStyle, create_llm
from bgbench.game import Game
from bgbench.llm_player import LLMPlayer
from bgbench.game_view import PromptStyle
from bgbench.game_runner import GameRunner
from bgbench.moves import ChainOfThoughtMove
from bgbench.rating import PlayerRating, EloSystem

logger = logging.getLogger("bgbench")

@dataclass
class ArenaPlayer:
    llm_player: LLMPlayer
    rating: PlayerRating
    player_model: DBPlayer

class Arena():
    def __init__(self, game: Game, db_session: Session, 
                 player_configs: Optional[List[Dict[str, Any]]] = None,
                 experiment_name: Optional[str] = None,
                 experiment_id: Optional[int] = None, 
                 confidence_threshold: float = 0.70,
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
        
        logger.info(f"Resumed experiment {self.experiment.name} (id: {experiment_id})")
        
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
            # Count games where player was involved
            games_played = self.session.query(GameMatch).filter(
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
                llm_player = LLMPlayer(
                    config["name"],
                    config["model_config"],
                    prompt_style=PromptStyle[config.get("prompt_style", "header").upper()],
                    response_style=ResponseStyle[config.get("response_style", "direct").upper()],
                    _llm=llm_factory(config["name"])
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
        

    def find_best_match(self) -> Optional[Tuple[ArenaPlayer, ArenaPlayer]]:
        """Find the match with highest uncertainty between players."""
        if len(self.players) < 2:
            return None
            
        best_uncertainty = -1
        best_pair = None
        
        for i, player_a in enumerate(self.players):
            for player_b in self.players[i+1:]:
                # Only consider matches where we need more games
                if self.elo_system.is_match_needed(player_a.rating, player_b.rating):
                    uncertainty = self.elo_system.calculate_match_uncertainty(
                        player_a.rating, player_b.rating
                    )
                    if uncertainty > best_uncertainty:
                        best_uncertainty = uncertainty
                        best_pair = (player_a, player_b)
        
        return best_pair

    def log_standings(self):
        """Log current ratings for all players"""
        logger.info("\nCurrent Standings:")
        sorted_players = sorted(self.players, key=lambda p: p.rating.rating, reverse=True)
        for player in sorted_players:
            # Count concessions by this player
            concessions = self.session.query(GameMatch).filter(
                (GameMatch.experiment_id == self.experiment.id) &
                (GameMatch.conceded == True) &
                (GameMatch.winner_id != player.player_model.id) &
                ((GameMatch.player1_id == player.player_model.id) | 
                 (GameMatch.player2_id == player.player_model.id))
            ).count()
            
            logger.info(f"{player.llm_player.name}: {player.rating.rating:.0f} "
                       f"({player.rating.games_played} games, {concessions} concessions)")

    async def evaluate_all(self) -> Dict[str, float]:
        while True:
            match = self.find_best_match()
            if not match:
                break  # All pairs have reached confidence threshold
                
            # Randomize the order to avoid first-player advantage
            player_a, player_b = random.sample(match, 2)
            # Get database players
            db_player_a = player_a.player_model
            db_player_b = player_b.player_model            

            # Create game record first to get game_id
            db_game: GameMatch = GameMatch(
                experiment_id=self.experiment.id,
                winner_id=None,  # Will update this after we know the winner
                player1_id=db_player_a.id,
                player2_id=db_player_b.id
            )
            self.session.add(db_game)
            self.session.commit()

            # Commit and get the integer ID value
            self.session.refresh(db_game)
            game_id = int(db_game.id)  # Convert Column to int

            runner = GameRunner(
                # Impossible to get this to typecheck without a cast to Any because of generic type
                self.game, 
                player_a.llm_player, 
                player_b.llm_player, 
                self.session,
                game_id
            )
            winner, history, concession = await runner.play_game()
            
            # Update ratings in memory and database
            # If no winner, treat as a draw (False)
            a_wins = bool(winner and winner.name == player_a.llm_player.name)
            new_rating_a, new_rating_b = self.elo_system.update_ratings(
                player_a.rating, player_b.rating, 
                a_wins
            )
            player_a.rating = new_rating_a
            player_b.rating = new_rating_b
            
            # Update player ratings in database
            db_player_a.update_rating(self.session, new_rating_a.rating)
            db_player_b.update_rating(self.session, new_rating_b.rating)
            
            # Update game record with winner
            if winner:
                winner_db_player = db_player_a if winner.name == player_a.llm_player.name else db_player_b
                db_game.winner_id = winner_db_player.id
                if concession:
                    db_game.conceded = True
                    db_game.concession_reason = concession
            self.session.commit()
            
            logger.info(f"Game completed: {winner.name if winner else 'Draw'}")
            
            # Log current standings and pairwise confidences
            self.log_standings()
            self.log_pairwise_confidences()
            
            # Check if all pairwise probabilities are above the threshold
            sorted_players = sorted(self.players, key=lambda p: p.rating.rating, reverse=True)
            all_above_threshold = True
            for i in range(len(sorted_players) - 1):
                player_a = sorted_players[i]
                player_b = sorted_players[i + 1]
                prob = self.elo_system.probability_stronger(player_a.rating, player_b.rating)
                if prob < self.confidence_threshold:
                    all_above_threshold = False
                    break
            
            if all_above_threshold:
                break

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
                (GameMatch.conceded == True) &
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
