import logging
from typing import Optional, List
from sqlalchemy import Integer, String, ForeignKey, JSON, Float, select, Table, Column
from sqlalchemy.orm import relationship, Session, Mapped, mapped_column, declarative_base
from bgbench.serialization import serialize_value

logger = logging.getLogger(__name__)
Base = declarative_base()

experiment_players = Table(
    'experiment_players',
    Base.metadata,
    Column('experiment_id', Integer, ForeignKey('experiments.id')),
    Column('player_id', Integer, ForeignKey('players.id')),
)

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
        
        # Get players from games
        games = session.query(Game).filter_by(experiment_id=self.id).all()
        game_player_ids = set(game.player_id for game in games if game.player_id is not None)
        game_players = session.query(Player).filter(Player.id.in_(game_player_ids)).all() if game_player_ids else []
        
        # Combine and deduplicate players
        all_players = list(set(direct_players + game_players))
        return all_players
    
    __tablename__ = 'experiments'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String)
    games: Mapped[list["Game"]] = relationship("Game", back_populates="experiment")
    players: Mapped[list["Player"]] = relationship(
        "Player",
        secondary=experiment_players,
        backref="experiments"
    )

class Player(Base):
    def update_rating(self, session: Session, new_rating: float):
        old_rating = self.rating
        self.rating = new_rating
        session.commit()
        logger.info(f"Updated player {self.name} rating: {old_rating} -> {new_rating}")

    @classmethod
    def create_player(cls, session: Session, name: str, model_config: dict) -> 'Player':
        """Create a new player with model configuration."""
        player = cls(name=name, model_config=model_config)
        session.add(player)
        session.commit()
        return player

    __tablename__ = 'players'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    rating: Mapped[float] = mapped_column(Float, default=1500.0)
    model_config: Mapped[dict] = mapped_column(JSON, nullable=False)
    games: Mapped[list["Game"]] = relationship("Game", back_populates="player")

class Game(Base):
    __tablename__ = 'games'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(Integer, ForeignKey('experiments.id'))
    player_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey('players.id'), nullable=True)
    state: Mapped["GameState"] = relationship("GameState", uselist=False, back_populates="game")
    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="games")
    player: Mapped["Player"] = relationship("Player", back_populates="games")

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

class LLMInteraction(Base):
    def log_interaction(self, session: Session, prompt: dict, response: str):
        self.prompt = prompt
        self.response = response
        session.add(self)
        session.commit()
        logger.debug(f"Logged LLM interaction for game {self.game_id}")
        logger.debug(f"Prompt: {prompt}")
        logger.debug(f"Response: {response}")

    __tablename__ = 'llm_interactions'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[int] = mapped_column(Integer, ForeignKey('games.id'))
    prompt: Mapped[dict] = mapped_column(JSON, nullable=False)
    response: Mapped[str] = mapped_column(String, nullable=False)
