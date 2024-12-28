import logging
from typing import Optional, List, Dict, Any
from sqlalchemy import Integer, String, ForeignKey, JSON, Float, select, event
from sqlalchemy.orm import relationship, Session, Mapped, mapped_column, declarative_base
from bgbench.serialization import serialize_value

logger = logging.getLogger(__name__)
Base = declarative_base()


class Experiment(Base):
    def create_experiment(self, session: Session, name: str, description: str = "") -> 'Experiment':
        new_experiment = Experiment(name=name, description=description)
        session.add(new_experiment)
        session.commit()
        logger.info(f"Created new experiment: {name} (id: {new_experiment.id})")
        return new_experiment

    @staticmethod
    def resume_experiment(session: Session, experiment_id: int) -> 'Experiment':
        stmt = select(Experiment).where(Experiment.id == experiment_id)
        return session.execute(stmt).scalar_one()
    
    def get_players(self, session: Session) -> List['Player']:
        """Get all players associated with this experiment."""
        # Get players directly associated with the experiment
        direct_players = self.players
        
        # Get players from games (both player1 and player2)
        games = session.query(Game).filter_by(experiment_id=self.id).all()
        game_player_ids = set()
        for game in games:
            if game.player1_id is not None:
                game_player_ids.add(game.player1_id)
            if game.player2_id is not None:
                game_player_ids.add(game.player2_id)
            if game.winner_id is not None:
                game_player_ids.add(game.winner_id)
        game_players = session.query(Player).filter(Player.id.in_(game_player_ids)).all() if game_player_ids else []
        
        # Combine and deduplicate players
        all_players = list(set(direct_players + game_players))
        return all_players

    def get_game_summary(self, session: Session) -> Dict[str, Any]:
        """Get summary statistics for games in this experiment."""
        games = session.query(Game).filter_by(experiment_id=self.id).all()
        total_games = len(games)
        completed_games = len([g for g in games if g.winner_id is not None and not g.conceded])
        ongoing_games = len([g for g in games if g.winner_id is None])
        conceded_games = len([g for g in games if g.conceded])
        
        return {
            "total_games": total_games,
            "completed_games": completed_games,
            "ongoing_games": ongoing_games,
            "conceded_games": conceded_games,
            "completion_rate": (completed_games + conceded_games) / total_games if total_games > 0 else 0
        }

    def get_win_matrix(self, session: Session) -> Dict[str, Dict[str, int]]:
        """Generate a matrix of wins between players."""
        games = session.query(Game).filter_by(experiment_id=self.id).all()
        players = self.get_players(session)
        
        # Initialize win matrix
        win_matrix = {p.name: {op.name: 0 for op in players} for p in players}
        
        # Count wins
        for game in games:
            if game.winner_id is not None:
                winner = next(p for p in players if p.id == game.winner_id)
                loser = next(p for p in players if p.id == (
                    game.player2_id if game.winner_id == game.player1_id else game.player1_id
                ))
                win_matrix[winner.name][loser.name] += 1
                
        return win_matrix
    
    __tablename__ = 'experiments'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String)
    games: Mapped[list["Game"]] = relationship("Game", back_populates="experiment")
    players: Mapped[list["Player"]] = relationship("Player", back_populates="experiment")

class Player(Base):
    def update_rating(self, session: Session, new_rating: float):
        old_rating = self.rating
        self.rating = new_rating
        session.commit()
        logger.info(f"Updated player {self.name} rating: {old_rating} -> {new_rating}")

    def get_statistics(self, session: Session) -> Dict[str, Any]:
        """Get comprehensive statistics for this player."""
        # Get all games where this player participated
        games_as_p1 = session.query(Game).filter_by(player1_id=self.id).all()
        games_as_p2 = session.query(Game).filter_by(player2_id=self.id).all()
        
        total_games = len(games_as_p1) + len(games_as_p2)
        if total_games == 0:
            return {
                "total_games": 0,
                "games_completed": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "concession_wins": 0,
                "concession_losses": 0
            }
        
        # Count different game outcomes
        wins = len([g for g in games_as_p1 + games_as_p2 if g.winner_id == self.id])
        completed_games = len([g for g in games_as_p1 + games_as_p2 if g.winner_id is not None])
        concession_wins = len([g for g in games_as_p1 + games_as_p2 
                             if g.winner_id == self.id and g.conceded])
        concession_losses = len([g for g in games_as_p1 + games_as_p2 
                               if g.winner_id is not None and g.winner_id != self.id and g.conceded])
        
        return {
            "total_games": total_games,
            "games_completed": completed_games,
            "wins": wins,
            "losses": completed_games - wins,
            "win_rate": wins / completed_games if completed_games > 0 else 0.0,
            "concession_wins": concession_wins,
            "concession_losses": concession_losses
        }

    @classmethod
    def create_player(cls, session: Session, name: str, model_config: dict, experiment_id: int) -> 'Player':
        """Create a new player with model configuration."""
        player = cls(name=name, model_config=model_config, experiment_id=experiment_id)
        session.add(player)
        session.commit()
        return player

    __tablename__ = 'players'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    rating: Mapped[float] = mapped_column(Float, default=1500.0)
    model_config: Mapped[dict] = mapped_column(JSON, nullable=False)
    experiment_id: Mapped[int] = mapped_column(Integer, ForeignKey('experiments.id'))
    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="players")
    games_as_player1: Mapped[list["Game"]] = relationship("Game", back_populates="player1", foreign_keys="[Game.player1_id]")
    games_as_player2: Mapped[list["Game"]] = relationship("Game", back_populates="player2", foreign_keys="[Game.player2_id]")
    games_won: Mapped[list["Game"]] = relationship("Game", back_populates="winner", foreign_keys="[Game.winner_id]")

class Game(Base):
    __tablename__ = 'games'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(Integer, ForeignKey('experiments.id'))
    player1_id: Mapped[int] = mapped_column(Integer, ForeignKey('players.id'))
    player2_id: Mapped[int] = mapped_column(Integer, ForeignKey('players.id'))
    winner_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey('players.id'), nullable=True)
    conceded: Mapped[bool] = mapped_column(Integer, default=False)
    concession_reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    state: Mapped["GameState"] = relationship("GameState", uselist=False, back_populates="game")
    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="games")
    player1: Mapped["Player"] = relationship("Player", back_populates="games_as_player1", foreign_keys=[player1_id])
    player2: Mapped["Player"] = relationship("Player", back_populates="games_as_player2", foreign_keys=[player2_id])
    winner: Mapped[Optional["Player"]] = relationship("Player", back_populates="games_won", foreign_keys=[winner_id])

class GameState(Base):
    def _serialize_state(self, state_data: dict) -> dict:
        """Convert state data to JSON-serializable format."""
        try:
            result = serialize_value(state_data)
            if not isinstance(result, dict):
                raise ValueError("Serialized state must be a dictionary")
            return result
        except (TypeError, ValueError) as e:
            raise ValueError(f"State data must be JSON-serializable: {str(e)}")

    def update_state(self, session: Session, new_state_data: dict):
        self.state_data = self._serialize_state(new_state_data)
        session.commit()
        logger.debug(f"Updated game state for game {self.game_id}: {self.state_data}")

    def record_state(self, session: Session):
        self.state_data = self._serialize_state(self.state_data)
        session.add(self)
        session.commit()

    __tablename__ = 'game_states'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[int] = mapped_column(Integer, ForeignKey('games.id'))
    state_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    game: Mapped["Game"] = relationship("Game", back_populates="state")

@event.listens_for(Game, 'before_insert')
def validate_players(mapper, connection, target):
    if target.player1_id == target.player2_id:
        raise ValueError("player1_id and player2_id must be different")

class LLMInteraction(Base):
    def log_interaction(self, session: Session, prompt: dict, response: str, 
                       start_time: float, end_time: float, 
                       prompt_tokens: Optional[int] = None,
                       completion_tokens: Optional[int] = None,
                       total_tokens: Optional[int] = None):
        self.prompt = prompt
        self.response = response
        self.start_time = start_time
        self.end_time = end_time
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        session.add(self)
        session.commit()
        
        duration = end_time - start_time
        logger.debug(f"Logged LLM interaction for game {self.game_id}")
        logger.debug(f"Duration: {duration:.2f}s")
        if total_tokens:
            logger.debug(f"Total tokens: {total_tokens}")

    __tablename__ = 'llm_interactions'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[int] = mapped_column(Integer, ForeignKey('games.id'))
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey('players.id'))
    prompt: Mapped[dict] = mapped_column(JSON, nullable=False)
    response: Mapped[str] = mapped_column(String, nullable=False)
    start_time: Mapped[float] = mapped_column(Float, nullable=False)  # Unix timestamp
    end_time: Mapped[float] = mapped_column(Float, nullable=False)  # Unix timestamp
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True) 
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    player: Mapped["Player"] = relationship("Player")
