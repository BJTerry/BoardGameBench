import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from bgbench.models import Experiment, Player as DBPlayer, Game as DBGame
from bgbench.game import Game
from bgbench.llm_player import LLMPlayer
from bgbench.game_runner import GameRunner
from bgbench.rating import PlayerRating, EloSystem

logger = logging.getLogger("bgbench")

@dataclass
class ArenaPlayer:
    llm_player: LLMPlayer
    rating: PlayerRating

class Arena:
    def __init__(self, game: Game, db_session: Session, experiment_name: Optional[str] = None, 
                 experiment_id: Optional[int] = None, confidence_threshold: float = 0.70):
        self.game = game
        self.players: List[ArenaPlayer] = []
        self.elo_system = EloSystem()
        self.confidence_threshold = confidence_threshold
        self.session = db_session

        if experiment_id is not None:
            self.experiment = Experiment.resume_experiment(self.session, experiment_id)
            logger.info(f"Resumed experiment {self.experiment.name} (id: {experiment_id})")
            # Get existing players from the experiment
            db_players = self.experiment.get_players(self.session)
            if not db_players:
                logger.warning(f"No players found in experiment {experiment_id}")
            else:
                logger.info(f"Found {len(db_players)} players in experiment")
                for db_player in db_players:
                    # Create LLMPlayer with same configuration
                    llm_player = LLMPlayer(db_player.name, create_llm(db_player.name))
                    # Create PlayerRating from database
                    rating = PlayerRating(name=db_player.name, 
                                        rating=db_player.rating,
                                        games_played=len(db_player.games))
                    # Add to arena players
                    self.players.append(ArenaPlayer(llm_player, rating))
        else:
            name = experiment_name or f"{game.__class__.__name__}_evaluation"
            self.experiment = Experiment().create_experiment(
                self.session,
                name=name,
                description=f"Evaluation of LLMs playing {game.__class__.__name__}"
            )
        
    def add_player(self, player: LLMPlayer, initial_rating: float = 1500):
        # For resumed experiments, only allow adding players that were in the original experiment
        if hasattr(self, 'experiment') and self.experiment.id is not None:
            db_players = self.experiment.get_players(self.session)
            if any(db_player.name == player.name for db_player in db_players):
                logger.warning(f"Player {player.name} already exists in experiment {self.experiment.id}")
                return
            logger.warning(f"Cannot add new player {player.name} to existing experiment {self.experiment.id}")
            return

        # Check if player already exists in database
        existing_player = self.session.query(DBPlayer).filter_by(name=player.name).first()
        
        if existing_player:
            db_player = existing_player
            logger.info(f"Found existing player {player.name} with rating {existing_player.rating}")
        else:
            # Create new player in database
            db_player = DBPlayer(name=player.name, rating=initial_rating)
            self.session.add(db_player)
            self.session.commit()
            logger.info(f"Created new player {player.name} with initial rating {initial_rating}")
        
        # Create ArenaPlayer with PlayerRating
        rating = PlayerRating(name=player.name, rating=db_player.rating, games_played=0)
        arena_player = ArenaPlayer(player, rating)
        self.players.append(arena_player)

    def calculate_match_uncertainty(self, player_a: ArenaPlayer, player_b: ArenaPlayer) -> float:
        prob = self.elo_system.probability_stronger(player_a.rating, player_b.rating)
        # Uncertainty is highest when prob is close to 0.5
        return 1.0 - abs(prob - 0.5) * 2

    def find_best_match(self) -> Optional[Tuple[ArenaPlayer, ArenaPlayer]]:
        if len(self.players) < 2:
            return None
            
        best_uncertainty = -1
        best_pair = None
        
        for i, player_a in enumerate(self.players):
            for player_b in self.players[i+1:]:
                prob = self.elo_system.probability_stronger(
                    player_a.rating, player_b.rating)
                if prob < self.confidence_threshold:
                    uncertainty = self.calculate_match_uncertainty(player_a, player_b)
                    if uncertainty > best_uncertainty:
                        best_uncertainty = uncertainty
                        best_pair = (player_a, player_b)
        
        return best_pair

    def log_standings(self):
        """Log current ratings for all players"""
        logger.info("\nCurrent Standings:")
        sorted_players = sorted(self.players, key=lambda p: p.rating.rating, reverse=True)
        for player in sorted_players:
            logger.info(f"{player.llm_player.name}: {player.rating.rating:.0f} "
                       f"({player.rating.games_played} games)")

    async def evaluate_all(self) -> Dict[str, float]:
        while True:
            match = self.find_best_match()
            if not match:
                break  # All pairs have reached confidence threshold
                
            player_a, player_b = match
            # Create game record first to get game_id
            db_game = DBGame(
                experiment_id=self.experiment.id,
                player_id=None  # Will update this after we know the winner
            )
            self.session.add(db_game)
            self.session.commit()

            # Commit and get the integer ID value
            self.session.refresh(db_game)
            game_id = int(db_game.id)  # Convert Column to int

            runner = GameRunner(
                self.game, 
                player_a.llm_player, 
                player_b.llm_player, 
                self.session,
                game_id
            )
            winner, _ = await runner.play_game()
            
            # Update ratings in memory and database
            new_rating_a, new_rating_b = self.elo_system.update_ratings(
                player_a.rating, player_b.rating, 
                winner.name == player_a.llm_player.name
            )
            player_a.rating = new_rating_a
            player_b.rating = new_rating_b
            
            try:
                # Update database players
                db_player_a = self.session.query(DBPlayer).filter_by(name=player_a.llm_player.name).first()
                db_player_b = self.session.query(DBPlayer).filter_by(name=player_b.llm_player.name).first()
                
                if db_player_a is None or db_player_b is None:
                    logger.error("Could not find players in database")
                    self.session.rollback()
                    continue
                
                db_player_a.update_rating(self.session, new_rating_a.rating)
                db_player_b.update_rating(self.session, new_rating_b.rating)
            except Exception as e:
                logger.error(f"Error updating player ratings: {e}")
                self.session.rollback()
                continue
            
            # Update game record with winner
            winner_db_player = db_player_a if winner.name == player_a.llm_player.name else db_player_b
            db_game.player_id = winner_db_player.id
            self.session.commit()
            
            logger.info(f"Game completed: {winner.name} won")
            
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
        games = self.session.query(DBGame).filter_by(experiment_id=self.experiment.id).all()
        
        results = {
            "experiment_id": self.experiment.id,
            "experiment_name": self.experiment.name,
            "total_games": len(games),
            "player_ratings": {p.llm_player.name: p.rating.rating for p in self.players},
            "games": []
        }
        
        for game in games:
            if game.player is not None:  # Winner exists
                results["games"].append({
                    "game_id": game.id,
                    "winner": game.player.name,
                    "final_ratings": {
                        p.name: p.rating for p in self.session.query(DBPlayer).all()
                    }
                })
        
        return results

    def get_player_game_history(self, player_name: str) -> List[Dict[str, Any]]:
        """Get game history for a specific player."""
        db_player = self.session.query(DBPlayer).filter_by(name=player_name).first()
        if not db_player:
            return []
            
        games = self.session.query(DBGame).filter_by(
            experiment_id=self.experiment.id
        ).all()
        
        history = []
        for game in games:
            if game.state:  # Game has state data
                game_data = {
                    "game_id": game.id,
                    "won": game.player_id == db_player.id if game.player_id else None,
                    "state": game.state.state_data
                }
                history.append(game_data)
                
        return history
