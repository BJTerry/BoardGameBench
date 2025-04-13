import os
import pytest
from typing import cast
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from bgbench.data.models import (
    Base,
    Experiment,
    Player,
    GameMatch,
    MatchState, # Renamed from GameState
    LLMInteraction,
)


@pytest.fixture
def db_session():
    """Provide a database session for testing"""
    # Use in-memory SQLite for tests by default
    # Test-specific PostgreSQL connection can be set with TEST_DATABASE_URL env var
    test_db_url = os.getenv("TEST_DATABASE_URL", "sqlite:///:memory:")

    engine = create_engine(test_db_url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    Base.metadata.drop_all(engine)


class TestExperiment:
    def test_create_experiment(self, db_session):
        """Test creating a new experiment"""
        experiment = Experiment().create_experiment(
            db_session, "Test Experiment", "Test Description"
        )
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


class TestMatchState: # Renamed class
    def test_match_state_lifecycle(self, db_session): # Renamed test method
        """Test complete match state lifecycle"""
        # Create experiment and players first
        experiment = Experiment().create_experiment(db_session, "Test Experiment")
        player1 = Player(name="Player 1", model_config={}, experiment_id=experiment.id)
        player2 = Player(name="Player 2", model_config={}, experiment_id=experiment.id)
        db_session.add_all([player1, player2])
        db_session.flush()  # Get IDs without committing

        # Create game with required player relationships
        game = GameMatch(
            experiment_id=experiment.id, player1_id=player1.id, player2_id=player2.id
        )
        db_session.add(game)
        db_session.commit()

        # Initial state
        initial_state = {"phase": "setup", "turn": 0}
        # Use MatchState and match_id
        match_state = MatchState(match_id=game.id, state_data=initial_state)
        match_state.record_state(db_session)

        # Update state
        new_state = {"phase": "play", "turn": 1}
        match_state.update_state(db_session, new_state)

        # Verify final state
        # Query MatchState using match_id
        saved_state = db_session.query(MatchState).filter_by(match_id=game.id).first()
        assert saved_state is not None # Ensure state was found
        assert saved_state.state_data["phase"] == "play"
        assert saved_state.state_data["turn"] == 1

    def test_invalid_state_data(self, db_session):
        """Test handling invalid state data"""
        # Create experiment and players first
        experiment = Experiment().create_experiment(db_session, "Test Experiment")
        player1 = Player(name="Player 1", model_config={}, experiment_id=experiment.id)
        player2 = Player(name="Player 2", model_config={}, experiment_id=experiment.id)
        db_session.add_all([player1, player2])
        db_session.flush()

        game = GameMatch(
            experiment_id=experiment.id, player1_id=player1.id, player2_id=player2.id
        )
        db_session.add(game)
        db_session.commit()

        # game_state = GameState(game_id=game.id, state_data={}) # This line is unused and refers to old name

        # Create a class that doesn't implement to_dict
        class Unserializable:
            pass
        invalid_state = Unserializable()
        # Update the expected error message regex to match the actual raised error
        with pytest.raises(ValueError, match="Match state data must be JSON-serializable: .*"):
            match_state = MatchState(match_id=game.id, state_data={}) # Use MatchState and match_id
            match_state.update_state(db_session, cast(dict, invalid_state))  # type: ignore


class TestGameMatch: # Renamed class for clarity
    def test_match_player_validation(self, db_session): # Renamed test method
        """Test game creation with invalid player combinations"""
        experiment = Experiment().create_experiment(db_session, "Test Experiment")
        player1 = Player(name="Player 1", model_config={}, experiment_id=experiment.id)
        db_session.add(player1)
        db_session.flush()

        # Test creating game with missing player2
        with pytest.raises(Exception):  # Should fail due to NOT NULL constraint
            game = GameMatch(
                experiment_id=experiment.id, player1_id=player1.id, player2_id=None
            )
            db_session.add(game)
            try:
                db_session.commit()
            except:
                db_session.rollback()
                raise

        # Create second player
        player2 = Player(name="Player 2", model_config={}, experiment_id=experiment.id)
        db_session.add(player2)
        db_session.commit()

        # Test creating game with same player for both positions
        with pytest.raises(ValueError):
            game = GameMatch(
                experiment_id=experiment.id,
                player1_id=player1.id,
                player2_id=player1.id,
            )
            db_session.add(game)
            try:
                db_session.commit()
            except:
                db_session.rollback()
                raise
            # db_session.add(game) # No need to add again
            db_session.commit()

    def test_match_experiment_isolation(self, db_session): # Renamed test method
        """Test that matches are properly isolated by experiment"""
        # Create two experiments
        exp1 = Experiment().create_experiment(db_session, "Experiment 1")
        exp2 = Experiment().create_experiment(db_session, "Experiment 2")

        # Create players in different experiments
        player1 = Player(name="Player 1", model_config={}, experiment_id=exp1.id)
        player2 = Player(name="Player 2", model_config={}, experiment_id=exp1.id)
        player3 = Player(name="Player 3", model_config={}, experiment_id=exp2.id)
        db_session.add_all([player1, player2, player3])
        db_session.flush()

        # Create game in experiment 1
        game1 = GameMatch(
            experiment_id=exp1.id, player1_id=player1.id, player2_id=player2.id
        )
        db_session.add(game1)
        db_session.commit()

        # Verify match retrieval by experiment
        exp1_matches = db_session.query(GameMatch).filter_by(experiment_id=exp1.id).all()
        exp2_matches = db_session.query(GameMatch).filter_by(experiment_id=exp2.id).all()
        assert len(exp1_matches) == 1
        assert len(exp2_matches) == 0


class TestPlayerExperiment:
    def test_player_experiment_constraints(self, db_session):
        """Test player creation and experiment constraints"""
        # Test creating player without experiment_id (should fail NOT NULL constraint)
        player = Player(name="Test Player", model_config={})
        db_session.add(player)
        with pytest.raises(Exception):
            db_session.commit()
        db_session.rollback()

        # Test creating player with valid experiment (should succeed)
        experiment = Experiment().create_experiment(db_session, "Test Experiment")
        player = Player(
            name="Test Player", model_config={}, experiment_id=experiment.id
        )
        db_session.add(player)
        db_session.commit()

        # Verify player was created with correct experiment
        saved_player = db_session.query(Player).filter_by(id=player.id).first()
        assert saved_player is not None
        assert saved_player.experiment_id == experiment.id

    def test_experiment_player_lookup(self, db_session):
        """Test player lookup methods in experiment"""
        experiment = Experiment().create_experiment(db_session, "Test Experiment")

        # Create players
        players = [
            Player(name=f"Player {i}", model_config={}, experiment_id=experiment.id)
            for i in range(3)
        ]
        db_session.add_all(players)
        db_session.flush()

        # Create some games
        game1 = GameMatch(
            experiment_id=experiment.id,
            player1_id=players[0].id,
            player2_id=players[1].id,
        )
        game2 = GameMatch(
            experiment_id=experiment.id,
            player1_id=players[1].id,
            player2_id=players[2].id,
        )
        db_session.add_all([game1, game2])
        db_session.commit()

        # Test get_players method
        exp_players = experiment.get_players(db_session)
        assert len(exp_players) == 3
        player_names = {p.name for p in exp_players}
        assert player_names == {"Player 0", "Player 1", "Player 2"}


class TestLLMInteraction: # Keep this class name
    def test_log_interaction(self, db_session):
        """Test logging LLM interactions"""
        # Setup
        experiment = Experiment().create_experiment(db_session, "Test Experiment")
        player1 = Player(name="Player 1", model_config={}, experiment_id=experiment.id)
        player2 = Player(name="Player 2", model_config={}, experiment_id=experiment.id)
        db_session.add_all([player1, player2])
        db_session.flush()

        game = GameMatch(
            experiment_id=experiment.id, player1_id=player1.id, player2_id=player2.id
        )
        db_session.add(game)
        db_session.commit()

        # Log interaction
        interaction = LLMInteraction(game_id=game.id, player_id=player1.id)
        prompt = [{"role": "user", "content": "Make a move"}]
        response = "I choose to take 3 objects"
        interaction.log_interaction(
            db_session,
            prompt,
            response,
            start_time=1234567890.0,  # Example timestamp
            end_time=1234567891.0,  # Example timestamp
        )

        # Verify
        saved = db_session.query(LLMInteraction).filter_by(game_id=game.id).first()
        assert saved is not None # Ensure interaction was found
        assert saved.prompt == prompt
        assert saved.response == response

    def test_multiple_interactions(self, db_session):
        """Test logging multiple interactions for same game"""
        experiment = Experiment().create_experiment(db_session, "Test Experiment")
        player1 = Player(name="Player 1", model_config={}, experiment_id=experiment.id)
        player2 = Player(name="Player 2", model_config={}, experiment_id=experiment.id)
        db_session.add_all([player1, player2])
        db_session.flush()

        game = GameMatch(
            experiment_id=experiment.id, player1_id=player1.id, player2_id=player2.id
        )
        db_session.add(game)
        db_session.commit()

        # Log multiple interactions
        for i in range(3):
            interaction = LLMInteraction(game_id=game.id, player_id=player1.id)
            interaction.log_interaction(
                db_session,
                [{"turn": i}],
                f"Response {i}",
                start_time=1234567890.0 + i,  # Different timestamp for each interaction
                end_time=1234567891.0 + i,
            )

        # Verify all interactions saved
        interactions = db_session.query(LLMInteraction).filter_by(game_id=game.id).all()
        assert len(interactions) == 3


class TestGameMatchOutcomes: # Renamed class
    def test_match_winner(self, db_session): # Renamed test method
        """Test setting and retrieving match winner"""
        experiment = Experiment().create_experiment(db_session, "Test Experiment")
        player1 = Player(name="Player 1", model_config={}, experiment_id=experiment.id)
        player2 = Player(name="Player 2", model_config={}, experiment_id=experiment.id)
        db_session.add_all([player1, player2])
        db_session.flush()

        game = GameMatch(
            experiment_id=experiment.id, player1_id=player1.id, player2_id=player2.id
        )
        db_session.add(game)
        db_session.commit()

        # Set winner
        game.winner_id = player1.id
        db_session.commit()

        # Verify winner
        saved_game = db_session.query(GameMatch).filter_by(id=game.id).first()
        assert saved_game is not None # Ensure match was found
        assert saved_game.winner_id == player1.id
        assert saved_game.winner is not None # Ensure winner relationship loaded
        assert saved_game.winner.name == "Player 1"
        assert not saved_game.conceded

    def test_match_concession(self, db_session): # Renamed test method
        """Test handling match concessions"""
        experiment = Experiment().create_experiment(db_session, "Test Experiment")
        player1 = Player(name="Player 1", model_config={}, experiment_id=experiment.id)
        player2 = Player(name="Player 2", model_config={}, experiment_id=experiment.id)
        db_session.add_all([player1, player2])
        db_session.flush()

        game = GameMatch(
            experiment_id=experiment.id, player1_id=player1.id, player2_id=player2.id
        )
        db_session.add(game)
        db_session.commit()

        # Record concession
        game.conceded = True
        game.concession_reason = "Invalid moves exceeded"
        game.winner_id = player2.id  # Other player wins
        db_session.commit()

        # Verify concession
        saved_game = db_session.query(GameMatch).filter_by(id=game.id).first()
        assert saved_game.conceded
        assert saved_game is not None # Ensure match was found
        assert saved_game.conceded
        assert saved_game.concession_reason == "Invalid moves exceeded"
        assert saved_game.winner_id == player2.id

    def test_match_completion_queries(self, db_session): # Renamed test method
        """Test querying completed and ongoing matches"""
        experiment = Experiment().create_experiment(db_session, "Test Experiment")
        player1 = Player(name="Player 1", model_config={}, experiment_id=experiment.id)
        player2 = Player(name="Player 2", model_config={}, experiment_id=experiment.id)
        db_session.add_all([player1, player2])
        db_session.flush()
