import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError
from bgbench.models import Base, Experiment, Player, Game, GameState, LLMInteraction

@pytest.fixture
def db_session():
    """Provide a database session for testing"""
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)

class TestExperiment:
    def test_create_experiment(self, db_session):
        """Test creating a new experiment"""
        experiment = Experiment().create_experiment(db_session, "Test Experiment", "Test Description")
        assert experiment.id is not None
        assert experiment.name == "Test Experiment"
        assert experiment.description == "Test Description"

    def test_resume_experiment(self, db_session):
        """Test resuming an existing experiment"""
        # Create experiment
        experiment = Experiment().create_experiment(db_session, "Test Experiment")
        exp_id = experiment.id
        
        # Resume experiment
        resumed = Experiment.resume_experiment(db_session, exp_id)
        assert resumed.id == exp_id
        assert resumed.name == "Test Experiment"

    def test_experiment_not_found(self, db_session):
        """Test handling non-existent experiment"""
        with pytest.raises(Exception):
            Experiment.resume_experiment(db_session, 999)

class TestPlayer:
    def test_update_player_rating(self, db_session):
        """Test updating player rating"""
        player = Player(name="Test Player", rating=1500.0)
        db_session.add(player)
        db_session.commit()
        
        old_rating = player.rating
        new_rating = 1600.0
        player.update_rating(db_session, new_rating)
        
        # Verify rating update
        assert player.rating == new_rating
        # Verify player still exists
        updated = db_session.query(Player).filter_by(id=player.id).first()
        assert updated.rating == new_rating

class TestGameState:
    def test_game_state_lifecycle(self, db_session):
        """Test complete game state lifecycle"""
        # Create experiment and game
        experiment = Experiment().create_experiment(db_session, "Test Experiment")
        game = Game(experiment_id=experiment.id)
        db_session.add(game)
        db_session.commit()

        # Initial state
        initial_state = {"phase": "setup", "turn": 0}
        game_state = GameState(game_id=game.id, state_data=initial_state)
        game_state.record_state(db_session)

        # Update state
        new_state = {"phase": "play", "turn": 1}
        game_state.update_state(db_session, new_state)

        # Verify final state
        saved_state = db_session.query(GameState).filter_by(game_id=game.id).first()
        assert saved_state.state_data["phase"] == "play"
        assert saved_state.state_data["turn"] == 1

    def test_invalid_state_data(self, db_session):
        """Test handling invalid state data"""
        game = Game(experiment_id=1)
        db_session.add(game)
        db_session.commit()

        game_state = GameState(game_id=game.id, state_data={})
        
        # Create a class that doesn't implement to_dict
        class Unserializable:
            pass
        
        # Try to update with non-serializable data
        invalid_state = Unserializable()  # Will fail because Unserializable doesn't implement to_dict
        with pytest.raises(ValueError, match="State data must be JSON-serializable"):
            game_state.update_state(db_session, invalid_state)

class TestLLMInteraction:
    def test_log_interaction(self, db_session):
        """Test logging LLM interactions"""
        # Setup
        experiment = Experiment().create_experiment(db_session, "Test Experiment")
        game = Game(experiment_id=experiment.id)
        db_session.add(game)
        db_session.commit()

        # Log interaction
        interaction = LLMInteraction(game_id=game.id)
        prompt = {"role": "user", "content": "Make a move"}
        response = "I choose to take 3 objects"
        interaction.log_interaction(db_session, prompt, response)

        # Verify
        saved = db_session.query(LLMInteraction).filter_by(game_id=game.id).first()
        assert saved.prompt == prompt
        assert saved.response == response

    def test_multiple_interactions(self, db_session):
        """Test logging multiple interactions for same game"""
        experiment = Experiment().create_experiment(db_session, "Test Experiment")
        game = Game(experiment_id=experiment.id)
        db_session.add(game)
        db_session.commit()

        # Log multiple interactions
        for i in range(3):
            interaction = LLMInteraction(game_id=game.id)
            interaction.log_interaction(
                db_session,
                {"turn": i},
                f"Response {i}"
            )

        # Verify all interactions saved
        interactions = db_session.query(LLMInteraction).filter_by(game_id=game.id).all()
        assert len(interactions) == 3
