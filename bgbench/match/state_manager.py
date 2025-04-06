import logging
from typing import Optional, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import select, desc

from bgbench.models import MatchState
from bgbench.match_state import MatchStateData

logger = logging.getLogger(__name__)


class MatchStateManager:
    """Handles saving and retrieving match state snapshots from the database."""

    def save_state(
        self, session: Session, match_id: int, state_data: MatchStateData
    ) -> int:
        """
        Saves the given match state to the database as a new record and returns its ID.
        This creates a new historical snapshot rather than updating an existing record,
        allowing for a complete history of the match state.

        Args:
            session: The SQLAlchemy session.
            match_id: The ID of the match this state belongs to.
            state_data: MatchStateData object to save.
        """
        try:
            # Ensure we have a MatchStateData object
            if not isinstance(state_data, MatchStateData):
                raise TypeError("state_data must be a MatchStateData object")
            
            match_state_data = state_data
            
            # Convert MatchStateData to dictionary for storage
            state_data_dict = match_state_data.to_dict()
            # Create a new MatchState record
            new_match_state = MatchState(
                match_id=match_id,
                state_data=state_data_dict
            )
            session.add(new_match_state)
            # Flush to get the ID before committing
            session.flush()
            saved_state_id = new_match_state.id
            # Commit the transaction
            session.commit()
            logger.debug(f"Saved new match state record {saved_state_id} for match {match_id}")
            return saved_state_id
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize or prepare state for match {match_id}: {e}")
            session.rollback()
            raise ValueError(f"Failed to serialize state for match {match_id}: {e}") from e
        except Exception as e:
            logger.error(f"Database error while saving state for match {match_id}: {e}")
            session.rollback()
            raise


    def get_latest_state(
        self, session: Session, match_id: int
    ) -> Optional[MatchStateData]:
        """
        Retrieves the most recent state for a given match.

        Args:
            session: The SQLAlchemy session.
            match_id: The ID of the match to retrieve the state for.

        Returns:
            The most recent MatchStateData, or None if no state is found.
        """
        try:
            stmt = (
                select(MatchState.state_data)
                .where(MatchState.match_id == match_id)
                .order_by(desc(MatchState.timestamp))
                .limit(1)
            )
            result = session.execute(stmt).scalar_one_or_none()
            if result:
                logger.debug(f"Retrieved latest state data for match {match_id}")
                # Convert the dictionary back to a MatchStateData object
                return MatchStateData.from_dict(result)
            else:
                logger.warning(f"No state data found for match {match_id}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving latest state data for match {match_id}: {e}")
            # Depending on desired behavior, re-raise or return None
            raise # Re-raise by default
            


# Example Usage (Conceptual - actual usage will be in Arena/MatchRunner)
# if __name__ == '__main__':
#     from bgbench.models import get_session, GameMatch # Assuming these exist
#     from bgbench.games.azul_game import AzulState # Example state type
#     from datetime import datetime
#
#     session = next(get_session())
#     manager = MatchStateManager()
#
#     # --- Saving Example ---
#     # Assume match_id = 1 exists
#     match_id_to_save = 1
#     
#     # Create a MatchStateData object with the current game state
#     state_data = MatchStateData(
#         turn=3,
#         current_player_id=0,
#         timestamp=datetime.now(),
#         game_state={"board": [1, 2, 3], "scores": [10, 20]},
#         metadata={"history": ["move1", "move2", "move3"]}
#     )
#     try:
#         # Transaction is managed internally by save_state
#         manager.save_state(session, match_id_to_save, state_data)
#         print(f"Saved state for match {match_id_to_save}")
#     except Exception as e:
#         print(f"Error saving state: {e}")
#
#     # --- Loading Example ---
#     match_id_to_load = 1
#     try:
#         latest_state = manager.get_latest_state(session, match_id_to_load)
#         if latest_state:
#             print(f"Loaded latest state for match {match_id_to_load}:")
#             print(f"  Turn: {latest_state.turn}")
#             print(f"  Current Player: {latest_state.current_player_id}")
#             print(f"  Timestamp: {latest_state.timestamp}")
#             print(f"  Game State: {latest_state.game_state}")
#             print(f"  Metadata: {latest_state.metadata}")
#         else:
#             print(f"No state found for match {match_id_to_load}")
#     except Exception as e:
#         print(f"Error loading state: {e}")
#
#     session.close()
