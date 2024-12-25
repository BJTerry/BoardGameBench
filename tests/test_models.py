import unittest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from bgbench.models import Base, Experiment, Player, Game, GameState, LLMInteraction
from config import DATABASE_URL

class TestModels(unittest.TestCase):
    def setUp(self):
        # Set up an in-memory SQLite database for testing
        self.engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

    def tearDown(self):
        self.session.close()
        Base.metadata.drop_all(self.engine)

    def test_create_experiment(self):
        experiment = Experiment(name="Test Experiment")
        self.session.add(experiment)
        self.session.commit()
        self.assertEqual(self.session.query(Experiment).count(), 1)

    def test_update_player_rating(self):
        player = Player(name="Test Player", rating=1500.0)
        self.session.add(player)
        self.session.commit()
        player.update_rating(self.session, 1600.0)
        self.assertEqual(player.rating, 1600.0)

    def test_record_game_state(self):
        game = Game(experiment_id=1, player_id=1)
        self.session.add(game)
        self.session.commit()
        game_state = GameState(game_id=game.id, state_data={"score": 10})
        game_state.record_state(self.session)
        self.assertEqual(self.session.query(GameState).count(), 1)

    def test_log_llm_interaction(self):
        # Create required game record first
        experiment = Experiment(name="Test Experiment")
        self.session.add(experiment)
        self.session.commit()
        
        game = Game(experiment_id=experiment.id, player_id=None)
        self.session.add(game)
        self.session.commit()

        # Now create and test LLM interaction
        llm_interaction = LLMInteraction(game_id=game.id)
        test_prompt = {"question": "What is 2+2?"}
        test_response = "4"
        llm_interaction.log_interaction(self.session, test_prompt, test_response)
        self.assertEqual(self.session.query(LLMInteraction).count(), 1)

if __name__ == '__main__':
    unittest.main()
