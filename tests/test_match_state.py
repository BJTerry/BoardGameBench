import unittest
from datetime import datetime
from typing import Dict, Any

from bgbench.match_state import MatchStateData


class TestMatchStateData(unittest.TestCase):
    """Tests for the MatchStateData class."""

    def test_init(self):
        """Test that MatchStateData can be properly instantiated."""
        # Arrange
        turn = 3
        current_player_id = 1
        timestamp = datetime.now()
        game_state = {"board": [1, 2, 3], "scores": [10, 20]}
        metadata = {"history": ["move1", "move2", "move3"]}

        # Act
        state_data = MatchStateData(
            turn=turn,
            current_player_id=current_player_id,
            timestamp=timestamp,
            game_state=game_state,
            metadata=metadata
        )

        # Assert
        self.assertEqual(state_data.turn, turn)
        self.assertEqual(state_data.current_player_id, current_player_id)
        self.assertEqual(state_data.timestamp, timestamp)
        self.assertEqual(state_data.game_state, game_state)
        self.assertEqual(state_data.metadata, metadata)

    def test_to_dict(self):
        """Test that MatchStateData can be converted to a dictionary."""
        # Arrange
        timestamp = datetime(2023, 1, 15, 12, 30, 45)
        state_data = MatchStateData(
            turn=7,
            current_player_id=3,
            timestamp=timestamp,
            game_state={"board": [7, 8, 9], "scores": [50, 60]},
            metadata={"history": ["moveX", "moveY", "moveZ"]}
        )

        # Act
        result = state_data.to_dict()

        # Assert
        self.assertEqual(result["turn"], 7)
        self.assertEqual(result["current_player_id"], 3)
        self.assertEqual(result["timestamp"], "2023-01-15T12:30:45")
        self.assertEqual(result["game_state"], {"board": [7, 8, 9], "scores": [50, 60]})
        self.assertEqual(result["metadata"], {"history": ["moveX", "moveY", "moveZ"]})

    def test_from_dict_with_iso_timestamp(self):
        """Test that MatchStateData can be created from a dictionary with ISO timestamp string."""
        # Arrange
        data = {
            "turn": 5,
            "current_player_id": 2,
            "timestamp": "2023-01-15T12:30:45",
            "game_state": {"board": [4, 5, 6], "scores": [30, 40]},
            "metadata": {"history": ["moveA", "moveB"]}
        }

        # Act
        state_data = MatchStateData.from_dict(data)

        # Assert
        self.assertEqual(state_data.turn, 5)
        self.assertEqual(state_data.current_player_id, 2)
        self.assertEqual(state_data.timestamp, datetime(2023, 1, 15, 12, 30, 45))
        self.assertEqual(state_data.game_state, {"board": [4, 5, 6], "scores": [30, 40]})
        self.assertEqual(state_data.metadata, {"history": ["moveA", "moveB"]})

    def test_from_dict_with_datetime_object(self):
        """Test that MatchStateData can be created from a dictionary with datetime object."""
        # Arrange
        timestamp = datetime(2023, 1, 15, 12, 30, 45)
        data = {
            "turn": 5,
            "current_player_id": 2,
            "timestamp": timestamp,
            "game_state": {"board": [4, 5, 6], "scores": [30, 40]},
            "metadata": {"history": ["moveA", "moveB"]}
        }

        # Act
        state_data = MatchStateData.from_dict(data)

        # Assert
        self.assertEqual(state_data.turn, 5)
        self.assertEqual(state_data.current_player_id, 2)
        self.assertEqual(state_data.timestamp, timestamp)
        self.assertEqual(state_data.game_state, {"board": [4, 5, 6], "scores": [30, 40]})
        self.assertEqual(state_data.metadata, {"history": ["moveA", "moveB"]})

    def test_default_metadata(self):
        """Test that metadata defaults to None if not provided."""
        # Arrange & Act
        state_data = MatchStateData(
            turn=1,
            current_player_id=0,
            timestamp=datetime.now(),
            game_state={"board": []}
        )

        # Assert
        self.assertIsNone(state_data.metadata)

    def test_to_dict_without_metadata(self):
        """Test that to_dict works correctly when metadata is None."""
        # Arrange
        timestamp = datetime(2023, 1, 15, 12, 30, 45)
        state_data = MatchStateData(
            turn=1,
            current_player_id=0,
            timestamp=timestamp,
            game_state={"board": []}
        )

        # Act
        result = state_data.to_dict()

        # Assert
        self.assertEqual(result["turn"], 1)
        self.assertEqual(result["current_player_id"], 0)
        self.assertEqual(result["timestamp"], "2023-01-15T12:30:45")
        self.assertEqual(result["game_state"], {"board": []})
        self.assertNotIn("metadata", result)

    def test_from_dict_missing_metadata(self):
        """Test that from_dict handles missing metadata."""
        # Arrange
        data = {
            "turn": 2,
            "current_player_id": 1,
            "timestamp": "2023-01-15T12:30:45",
            "game_state": {"board": [1, 2]}
        }

        # Act
        state_data = MatchStateData.from_dict(data)

        # Assert
        self.assertIsNone(state_data.metadata)


if __name__ == "__main__":
    unittest.main()
