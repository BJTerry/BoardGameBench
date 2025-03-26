import json
import logging
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Integer,
    String,
    ForeignKey,
    JSON,
    Float,
    select,
    event,
    Boolean,
    func,
    text,
)
from sqlalchemy.orm import (
    relationship,
    Session,
    Mapped,
    mapped_column,
    declarative_base,
)
from sqlalchemy.types import TypeDecorator
from bgbench.serialization import serialize_value


# Define a custom type that works with both PostgreSQL and SQLite
class JsonType(TypeDecorator):
    """Platform-independent JSON type.

    Uses PostgreSQL's JSONB when available, otherwise falls back to SQLite's JSON type.
    """

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import JSONB

            return dialect.type_descriptor(JSONB())
        else:
            return dialect.type_descriptor(JSON())


logger = logging.getLogger(__name__)
Base = declarative_base()


class Experiment(Base):
    def create_experiment(
        self, session: Session, name: str, description: str = "", game_name: str = ""
    ) -> "Experiment":
        # Check if we need to adjust the sequence for PostgreSQL
        if session.bind is not None and session.bind.dialect.name == "postgresql":
            # Get the highest experiment ID to ensure proper sequence
            stmt = select(func.max(Experiment.id))
            max_id = session.execute(stmt).scalar()

            if max_id is not None:
                # Set the sequence to start from max_id + 1
                session.execute(
                    text(f"SELECT setval('experiments_id_seq', {max_id}, true)")
                )
                logger.debug(f"Adjusted PostgreSQL sequence to start after ID {max_id}")

        new_experiment = Experiment(
            name=name, description=description, game_name=game_name
        )
        session.add(new_experiment)
        session.commit()
        logger.info(f"Created new experiment: {name} (id: {new_experiment.id})")
        return new_experiment

    @staticmethod
    def resume_experiment(session: Session, experiment_id: int) -> "Experiment":
        stmt = select(Experiment).where(Experiment.id == experiment_id)
        return session.execute(stmt).scalar_one()

    def get_players(self, session: Session) -> List["Player"]:
        """Get all players associated with this experiment."""
        # Get players directly associated with the experiment
        direct_players = self.players

        # Get players from games (both player1 and player2)
        games = session.query(GameMatch).filter_by(experiment_id=self.id).all()
        game_player_ids = set()
        for game in games:
            if game.player1_id is not None:
                game_player_ids.add(game.player1_id)
            if game.player2_id is not None:
                game_player_ids.add(game.player2_id)
            if game.winner_id is not None:
                game_player_ids.add(game.winner_id)
        game_players = (
            session.query(Player).filter(Player.id.in_(game_player_ids)).all()
            if game_player_ids
            else []
        )

        # Combine and deduplicate players
        all_players = list(set(direct_players + game_players))
        return all_players

    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String)
    game_name: Mapped[Optional[str]] = mapped_column(String)
    games: Mapped[list["GameMatch"]] = relationship(
        "GameMatch", back_populates="experiment"
    )
    players: Mapped[list["Player"]] = relationship(
        "Player", back_populates="experiment"
    )


class Player(Base):
    @classmethod
    def create_player(
        cls, session: Session, name: str, model_config: dict, experiment_id: int
    ) -> "Player":
        """Create a new player with model configuration."""
        # Check if we need to adjust the sequence for PostgreSQL
        if session.bind is not None and session.bind.dialect.name == "postgresql":
            # Get the highest player ID to ensure proper sequence
            stmt = select(func.max(Player.id))
            max_id = session.execute(stmt).scalar()

            if max_id is not None:
                # Set the sequence to start from max_id + 1
                session.execute(
                    text(f"SELECT setval('players_id_seq', {max_id}, true)")
                )
                logger.debug(f"Adjusted PostgreSQL sequence to start after ID {max_id}")

        player = cls(name=name, model_config=model_config, experiment_id=experiment_id)
        session.add(player)
        session.commit()
        return player

    def update_rating(self, session: Session, new_rating: float) -> None:
        """Update the player's rating in the database."""
        self.rating = new_rating
        session.commit()

    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    rating: Mapped[float] = mapped_column(Float, default=1500.0)
    model_config: Mapped[dict] = mapped_column(JsonType, nullable=False)
    experiment_id: Mapped[int] = mapped_column(Integer, ForeignKey("experiments.id"))
    experiment: Mapped["Experiment"] = relationship(
        "Experiment", back_populates="players"
    )
    games_as_player1: Mapped[list["GameMatch"]] = relationship(
        "GameMatch", back_populates="player1", foreign_keys="[GameMatch.player1_id]"
    )
    games_as_player2: Mapped[list["GameMatch"]] = relationship(
        "GameMatch", back_populates="player2", foreign_keys="[GameMatch.player2_id]"
    )
    games_won: Mapped[list["GameMatch"]] = relationship(
        "GameMatch", back_populates="winner", foreign_keys="[GameMatch.winner_id]"
    )


class GameMatch(Base):
    __tablename__ = "games"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(Integer, ForeignKey("experiments.id"))
    player1_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.id"))
    player2_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.id"))
    winner_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("players.id"), nullable=True
    )
    conceded: Mapped[bool] = mapped_column(Boolean, default=False)
    concession_reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    complete: Mapped[bool] = mapped_column(
        Boolean, default=False
    )  # Indicates if game finished normally (including draws)
    state: Mapped["GameState"] = relationship(
        "GameState", uselist=False, back_populates="game", cascade="all, delete-orphan"
    )
    experiment: Mapped["Experiment"] = relationship(
        "Experiment", back_populates="games"
    )
    player1: Mapped["Player"] = relationship(
        "Player", back_populates="games_as_player1", foreign_keys=[player1_id]
    )
    player2: Mapped["Player"] = relationship(
        "Player", back_populates="games_as_player2", foreign_keys=[player2_id]
    )
    winner: Mapped[Optional["Player"]] = relationship(
        "Player", back_populates="games_won", foreign_keys=[winner_id]
    )


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

    __tablename__ = "game_states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("games.id", ondelete="CASCADE")
    )
    state_data: Mapped[dict] = mapped_column(JsonType, nullable=False)
    game: Mapped["GameMatch"] = relationship("GameMatch", back_populates="state")


@event.listens_for(GameMatch, "before_insert")
def validate_players(mapper, connection, target):
    if target.player1_id == target.player2_id:
        raise ValueError("player1_id and player2_id must be different")


class LLMInteraction(Base):
    def log_interaction(
        self,
        session: Session,
        prompt: List[Dict[str, Any]],
        response: str,
        start_time: float,
        end_time: float,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        cost: Optional[float] = None,
    ):
        self.prompt = json.loads(json.dumps(prompt))  # Ensure proper JSON serialization
        self.response = response
        self.start_time = start_time
        self.end_time = end_time
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.cost = cost

        # Add debug logging
        logger.debug(
            f"Setting cost={cost} for interaction in game_id={self.game_id}, player_id={self.player_id}"
        )

        session.add(self)
        session.commit()

        duration = end_time - start_time
        logger.debug(f"Logged LLM interaction for game {self.game_id}")
        logger.debug(f"Duration: {duration:.2f}s")
        if total_tokens:
            logger.debug(f"Total tokens: {total_tokens}")
        if cost:
            logger.debug(f"Cost: ${cost:.6f}")

    __tablename__ = "llm_interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("games.id", ondelete="CASCADE")
    )
    player_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("players.id", ondelete="CASCADE")
    )
    prompt: Mapped[List[dict]] = mapped_column(JsonType, nullable=False)
    response: Mapped[str] = mapped_column(String, nullable=False)
    start_time: Mapped[float] = mapped_column(Float, nullable=False)  # Unix timestamp
    end_time: Mapped[float] = mapped_column(Float, nullable=False)  # Unix timestamp
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    player: Mapped["Player"] = relationship("Player")
