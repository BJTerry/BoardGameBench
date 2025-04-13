import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from typing import Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import select, desc
from sqlalchemy.sql.selectable import Select

from bgbench.data.models import MatchState
from bgbench.match.state_manager import MatchStateManager
from bgbench.match.match_state import MatchStateData


# Test Fixture for the manager
@pytest.fixture
def manager() -> MatchStateManager:
    return MatchStateManager()

# Test Fixture for a mock session
@pytest.fixture
def mock_session() -> MagicMock:
    return MagicMock(spec=Session)

# Test Fixture for sample MatchStateData
@pytest.fixture
def sample_state_data() -> MatchStateData:
    return MatchStateData(
        turn=3,
        current_player_id=1,
        timestamp=datetime(2023, 1, 15, 12, 30, 45),
        game_state={"board": [1, 2, 3], "scores": [10, 20]},
        metadata={"history": ["move1", "move2", "move3"]}
    )


def test_save_state_success(manager: MatchStateManager, mock_session: MagicMock, sample_state_data: MatchStateData):
    """Test successfully saving a MatchStateData object."""
    match_id = 1
    expected_dict = sample_state_data.to_dict()

    manager.save_state(mock_session, match_id, sample_state_data)

    # Assert that session.add was called once
    mock_session.add.assert_called_once()
    # Get the object passed to session.add
    added_object = mock_session.add.call_args[0][0]

    # Assert it's a MatchState instance with correct data
    assert isinstance(added_object, MatchState)
    assert added_object.match_id == match_id
    assert added_object.state_data == expected_dict
    
    # Assert that commit was called
    mock_session.commit.assert_called_once()


def test_save_state_error_handling(manager: MatchStateManager, mock_session: MagicMock, sample_state_data: MatchStateData):
    """Test error handling when saving a state fails."""
    match_id = 2
    
    # Mock to_dict to raise an error
    sample_state_data.to_dict = MagicMock(side_effect=ValueError("Serialization error"))

    with pytest.raises(ValueError, match=f"Failed to serialize state for match {match_id}"):
        manager.save_state(mock_session, match_id, sample_state_data)

    # Ensure session.add was not called
    mock_session.add.assert_not_called()
    
    # Ensure rollback was called
    mock_session.rollback.assert_called_once()


@patch('bgbench.match.state_manager.select')
def test_get_latest_state_found(mock_select, manager: MatchStateManager, mock_session: MagicMock, sample_state_data: MatchStateData):
    """Test retrieving the latest state when one exists."""
    match_id = 3
    state_dict = sample_state_data.to_dict()
    
    # Set up the mock select chain
    mock_stmt = MagicMock()
    mock_select.return_value = mock_stmt
    mock_stmt.where.return_value = mock_stmt
    mock_stmt.order_by.return_value = mock_stmt
    mock_stmt.limit.return_value = mock_stmt
    
    # Configure the mock session's execute chain
    mock_execute = mock_session.execute.return_value
    mock_execute.scalar_one_or_none.return_value = state_dict

    # Patch MatchStateData.from_dict to return our sample data
    with patch('bgbench.match.state_manager.MatchStateData.from_dict', return_value=sample_state_data):
        result = manager.get_latest_state(mock_session, match_id)

    # Assert the correct query structure was used
    mock_session.execute.assert_called_once()
    
    # Assert the result is correct
    assert result == sample_state_data
    mock_execute.scalar_one_or_none.assert_called_once()


@patch('bgbench.match.state_manager.select')
def test_get_latest_state_not_found(mock_select, manager: MatchStateManager, mock_session: MagicMock):
    """Test retrieving the latest state when none exists."""
    match_id = 4
    
    # Set up the mock select chain
    mock_stmt = MagicMock()
    mock_select.return_value = mock_stmt
    mock_stmt.where.return_value = mock_stmt
    mock_stmt.order_by.return_value = mock_stmt
    mock_stmt.limit.return_value = mock_stmt
    
    # Configure the mock session's execute chain to return None
    mock_execute = mock_session.execute.return_value
    mock_execute.scalar_one_or_none.return_value = None

    result = manager.get_latest_state(mock_session, match_id)

    # Assert the correct query structure was used
    mock_session.execute.assert_called_once()
    
    # Assert the result is None
    assert result is None
    mock_execute.scalar_one_or_none.assert_called_once()


@patch('bgbench.match.state_manager.select')
def test_get_latest_state_db_error(mock_select, manager: MatchStateManager, mock_session: MagicMock):
    """Test that database errors during retrieval are propagated."""
    match_id = 5
    db_error = Exception("Database connection failed")
    
    # Set up the mock select chain
    mock_stmt = MagicMock()
    mock_select.return_value = mock_stmt
    mock_stmt.where.return_value = mock_stmt
    mock_stmt.order_by.return_value = mock_stmt
    mock_stmt.limit.return_value = mock_stmt

    # Configure the mock session's execute to raise an error
    mock_session.execute.side_effect = db_error

    with pytest.raises(Exception, match="Database connection failed"):
        manager.get_latest_state(mock_session, match_id)

    mock_session.execute.assert_called_once()
def test_save_state_database_error(manager: MatchStateManager, mock_session: MagicMock, sample_state_data: MatchStateData):
    """Test handling of database errors when saving a state."""
    match_id = 6
    
    # Mock commit to raise a database error
    mock_session.commit.side_effect = Exception("Database error")

    with pytest.raises(Exception, match="Database error"):
        manager.save_state(mock_session, match_id, sample_state_data)

    # Ensure session.add was called
    mock_session.add.assert_called_once()
    
    # Ensure rollback was called
    mock_session.rollback.assert_called_once()
