import pytest
from typing import cast
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from bgbench.models import Base, Experiment, Player, GameMatch, GameState, LLMInteraction

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
        # Create experiment first
        experiment = Experiment().create_experiment(db_session, "Test Experiment")
        player = Player(
            name="Test Player",
            rating=1500.0,
            model_config={"model": "test-model"},
            experiment_id=experiment.id
        )
        db_session.add(player)
        db_session.commit()
        
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
        # Create experiment and players first
        experiment = Experiment().create_experiment(db_session, "Test Experiment")
        player1 = Player(name="Player 1", model_config={}, experiment_id=experiment.id)
        player2 = Player(name="Player 2", model_config={}, experiment_id=experiment.id)
        db_session.add_all([player1, player2])
        db_session.flush()  # Get IDs without committing
        
        # Create game with required player relationships
        game = GameMatch(
            experiment_id=experiment.id,
            player1_id=player1.id,
            player2_id=player2.id
        )
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
        # Create experiment and players first
        experiment = Experiment().create_experiment(db_session, "Test Experiment")
        player1 = Player(name="Player 1", model_config={}, experiment_id=experiment.id)
        player2 = Player(name="Player 2", model_config={}, experiment_id=experiment.id)
        db_session.add_all([player1, player2])
        db_session.flush()
        
        game = GameMatch(
            experiment_id=experiment.id,
            player1_id=player1.id,
            player2_id=player2.id
        )
        db_session.add(game)
        db_session.commit()

        game_state = GameState(game_id=game.id, state_data={})
        
        # Create a class that doesn't implement to_dict
        class Unserializable:
            pass
        
        invalid_state = Unserializable()
        with pytest.raises(ValueError, match="State data must be JSON-serializable"):
            game_state.update_state(db_session, cast(dict, invalid_state))  # type: ignore

class TestGame:
    def test_game_player_validation(self, db_session):
        """Test game creation with invalid player combinations"""
        experiment = Experiment().create_experiment(db_session, "Test Experiment")
        player1 = Player(name="Player 1", model_config={}, experiment_id=experiment.id)
        db_session.add(player1)
        db_session.flush()

        # Test creating game with missing player2
        with pytest.raises(Exception):  # Should fail due to NOT NULL constraint
            game = GameMatch(
                experiment_id=experiment.id,
                player1_id=player1.id,
                player2_id=None
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
                player2_id=player1.id
            )
            db_session.add(game)
            try:
                db_session.commit()
            except:
                db_session.rollback()
                raise
            db_session.add(game)
            db_session.commit()

    def test_game_experiment_isolation(self, db_session):
        """Test that games are properly isolated by experiment"""
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
            experiment_id=exp1.id,
            player1_id=player1.id,
            player2_id=player2.id
        )
        db_session.add(game1)
        db_session.commit()

        # Verify game retrieval by experiment
        exp1_games = db_session.query(GameMatch).filter_by(experiment_id=exp1.id).all()
        exp2_games = db_session.query(GameMatch).filter_by(experiment_id=exp2.id).all()
        assert len(exp1_games) == 1
        assert len(exp2_games) == 0

class TestPlayerExperiment:
    def test_player_experiment_constraints(self, db_session):
        """Test player creation and experiment constraints"""
        # Test creating player without experiment_id (should fail NOT NULL constraint)
        player = Player(name="Test Player", model_config={})
        db_session.add(player)
        with pytest.raises(Exception) as exc_info:
            db_session.commit()
        db_session.rollback()
        
        # Test creating player with valid experiment (should succeed)
        experiment = Experiment().create_experiment(db_session, "Test Experiment")
        player = Player(name="Test Player", model_config={}, experiment_id=experiment.id)
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
            player2_id=players[1].id
        )
        game2 = GameMatch(
            experiment_id=experiment.id,
            player1_id=players[1].id,
            player2_id=players[2].id
        )
        db_session.add_all([game1, game2])
        db_session.commit()

        # Test get_players method
        exp_players = experiment.get_players(db_session)
        assert len(exp_players) == 3
        player_names = {p.name for p in exp_players}
        assert player_names == {"Player 0", "Player 1", "Player 2"}

class TestLLMInteraction:
    def test_log_interaction(self, db_session):
        """Test logging LLM interactions"""
        # Setup
        experiment = Experiment().create_experiment(db_session, "Test Experiment")
        player1 = Player(name="Player 1", model_config={}, experiment_id=experiment.id)
        player2 = Player(name="Player 2", model_config={}, experiment_id=experiment.id)
        db_session.add_all([player1, player2])
        db_session.flush()
        
        game = GameMatch(
            experiment_id=experiment.id,
            player1_id=player1.id,
            player2_id=player2.id
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
            end_time=1234567891.0  # Example timestamp
        )

        # Verify
        saved = db_session.query(LLMInteraction).filter_by(game_id=game.id).first()
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
            experiment_id=experiment.id,
            player1_id=player1.id,
            player2_id=player2.id
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
                end_time=1234567891.0 + i
            )

        # Verify all interactions saved
        interactions = db_session.query(LLMInteraction).filter_by(game_id=game.id).all()
        assert len(interactions) == 3

class TestGameOutcomes:
    def test_game_winner(self, db_session):
        """Test setting and retrieving game winner"""
        experiment = Experiment().create_experiment(db_session, "Test Experiment")
        player1 = Player(name="Player 1", model_config={}, experiment_id=experiment.id)
        player2 = Player(name="Player 2", model_config={}, experiment_id=experiment.id)
        db_session.add_all([player1, player2])
        db_session.flush()
        
        game = GameMatch(
            experiment_id=experiment.id,
            player1_id=player1.id,
            player2_id=player2.id
        )
        db_session.add(game)
        db_session.commit()

        # Set winner
        game.winner_id = player1.id
        db_session.commit()

        # Verify winner
        saved_game = db_session.query(GameMatch).filter_by(id=game.id).first()
        assert saved_game.winner_id == player1.id
        assert saved_game.winner.name == "Player 1"
        assert not saved_game.conceded

    def test_game_concession(self, db_session):
        """Test handling game concessions"""
        experiment = Experiment().create_experiment(db_session, "Test Experiment")
        player1 = Player(name="Player 1", model_config={}, experiment_id=experiment.id)
        player2 = Player(name="Player 2", model_config={}, experiment_id=experiment.id)
        db_session.add_all([player1, player2])
        db_session.flush()
        
        game = GameMatch(
            experiment_id=experiment.id,
            player1_id=player1.id,
            player2_id=player2.id
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
        assert saved_game.concession_reason == "Invalid moves exceeded"
        assert saved_game.winner_id == player2.id

    def test_game_completion_queries(self, db_session):
        """Test querying completed and ongoing games"""
        experiment = Experiment().create_experiment(db_session, "Test Experiment")
        player1 = Player(name="Player 1", model_config={}, experiment_id=experiment.id)
        player2 = Player(name="Player 2", model_config={}, experiment_id=experiment.id)
        db_session.add_all([player1, player2])
        db_session.flush()
        
    def test_experiment_game_summary(self, db_session):
        """Test getting experiment game summary statistics"""
        experiment = Experiment().create_experiment(db_session, "Test Experiment")
        player1 = Player(name="Player 1", model_config={}, experiment_id=experiment.id)
        player2 = Player(name="Player 2", model_config={}, experiment_id=experiment.id)
        db_session.add_all([player1, player2])
        db_session.flush()
        
        # Create mix of completed, ongoing, and conceded games
        games = [
            # Completed game
            GameMatch(experiment_id=experiment.id, player1_id=player1.id, player2_id=player2.id, winner_id=player1.id),
            # Ongoing game
            GameMatch(experiment_id=experiment.id, player1_id=player2.id, player2_id=player1.id),
            # Conceded game
            GameMatch(experiment_id=experiment.id, player1_id=player1.id, player2_id=player2.id, 
                winner_id=player2.id, conceded=True, concession_reason="Time limit")
        ]
        db_session.add_all(games)
        db_session.commit()
        
        # Get summary
        summary = experiment.get_game_summary(db_session)
        
        # Verify summary statistics
        assert summary["total_games"] == 3
        assert summary["completed_games"] == 1
        assert summary["ongoing_games"] == 1
        assert summary["conceded_games"] == 1
        assert summary["completion_rate"] == pytest.approx(0.667, rel=0.01)  # 2/3 games completed or conceded
        
    def test_player_statistics(self, db_session):
        """Test getting player statistics"""
        experiment = Experiment().create_experiment(db_session, "Test Experiment")
        player1 = Player(name="Player 1", model_config={}, experiment_id=experiment.id)
        player2 = Player(name="Player 2", model_config={}, experiment_id=experiment.id)
        db_session.add_all([player1, player2])
        db_session.flush()
        
        # Create games with various outcomes
        games = [
            # Player 1 wins
            GameMatch(experiment_id=experiment.id, player1_id=player1.id, player2_id=player2.id, winner_id=player1.id),
            # Player 2 wins
            GameMatch(experiment_id=experiment.id, player1_id=player1.id, player2_id=player2.id, winner_id=player2.id),
            # Player 2 wins by concession
            GameMatch(experiment_id=experiment.id, player1_id=player1.id, player2_id=player2.id, 
                winner_id=player2.id, conceded=True, concession_reason="Invalid moves"),
            # Ongoing game
            GameMatch(experiment_id=experiment.id, player1_id=player2.id, player2_id=player1.id)
        ]
        db_session.add_all(games)
        db_session.commit()
        
        # Get player stats
        player1_stats = player1.get_statistics(db_session)
        player2_stats = player2.get_statistics(db_session)
        
        # Verify Player 1 stats
        assert player1_stats["total_games"] == 4
        assert player1_stats["games_completed"] == 3
        assert player1_stats["wins"] == 1
        assert player1_stats["losses"] == 2
        assert player1_stats["win_rate"] == pytest.approx(0.333, rel=0.01)
        assert player1_stats["concession_losses"] == 1
        
        # Verify Player 2 stats
        assert player2_stats["total_games"] == 4
        assert player2_stats["games_completed"] == 3
        assert player2_stats["wins"] == 2
        assert player2_stats["losses"] == 1
        assert player2_stats["win_rate"] == pytest.approx(0.667, rel=0.01)
        assert player2_stats["concession_wins"] == 1
        
    def test_experiment_win_matrix(self, db_session):
        """Test generating win/loss matrix between players"""
        experiment = Experiment().create_experiment(db_session, "Test Experiment")
        players = [
            Player(name=f"Player {i}", model_config={}, experiment_id=experiment.id)
            for i in range(3)
        ]
        db_session.add_all(players)
        db_session.flush()
        
        # Create games with various outcomes between players
        games = [
            # Player 0 vs Player 1 (Player 0 wins)
            GameMatch(experiment_id=experiment.id, player1_id=players[0].id, player2_id=players[1].id, winner_id=players[0].id),
            # Player 1 vs Player 2 (Player 2 wins)
            GameMatch(experiment_id=experiment.id, player1_id=players[1].id, player2_id=players[2].id, winner_id=players[2].id),
            # Player 2 vs Player 0 (Player 2 wins)
            GameMatch(experiment_id=experiment.id, player1_id=players[2].id, player2_id=players[0].id, winner_id=players[2].id)
        ]
        db_session.add_all(games)
        db_session.commit()
        
        # Get win matrix
        win_matrix = experiment.get_win_matrix(db_session)
        
        # Verify matrix entries
        # Format: win_matrix[winner][loser] = number of wins
        assert win_matrix["Player 0"]["Player 1"] == 1
        assert win_matrix["Player 0"]["Player 2"] == 0
        assert win_matrix["Player 2"]["Player 0"] == 1
        assert win_matrix["Player 2"]["Player 1"] == 1
        
        # Create completed game
        completed_game = GameMatch(
            experiment_id=experiment.id,
            player1_id=players[0].id,
            player2_id=players[1].id,
            winner_id=players[0].id
        )
        
        # Create ongoing game
        ongoing_game = GameMatch(
            experiment_id=experiment.id,
            player1_id=players[0].id,
            player2_id=players[1].id
        )
        
        # Create conceded game
        conceded_game = GameMatch(
            experiment_id=experiment.id,
            player1_id=players[0].id,
            player2_id=players[1].id,
            winner_id=players[1].id,
            conceded=True,
            concession_reason="Time limit exceeded"
        )
        
        db_session.add_all([completed_game, ongoing_game, conceded_game])
        db_session.commit()

        # Query completed games (either won or conceded) from the new batch only
        completed = db_session.query(GameMatch).filter(
            GameMatch.winner_id.isnot(None),
            GameMatch.id.in_([completed_game.id, conceded_game.id])
        ).all()
        assert len(completed) == 2  # Only counting the new completed and conceded games
        
        # Query ongoing games (no winner yet)
        ongoing = db_session.query(GameMatch).filter(GameMatch.winner_id.is_(None)).all()
        assert len(ongoing) == 1
        
        # Query conceded games
        conceded = db_session.query(GameMatch).filter(GameMatch.conceded.is_(True)).all()
        assert len(conceded) == 1
