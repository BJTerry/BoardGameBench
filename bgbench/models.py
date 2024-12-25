import logging
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, JSON, Float, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session, Mapped, mapped_column

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
    
    __tablename__ = 'experiments'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String)
    games: Mapped[list["Game"]] = relationship("Game", back_populates="experiment")

class Player(Base):
    def update_rating(self, session: Session, new_rating: float):
        old_rating = self.rating
        self.rating = new_rating
        session.commit()
        logger.info(f"Updated player {self.name} rating: {old_rating} -> {new_rating}")
    __tablename__ = 'players'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    rating: Mapped[float] = mapped_column(Float, default=1500.0)
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
    def update_state(self, session: Session, new_state_data: dict):
        self.state_data = new_state_data
        session.commit()
        logger.debug(f"Updated game state for game {self.game_id}: {new_state_data}")
    def record_state(self, session: Session):
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
