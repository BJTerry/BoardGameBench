"""Rename GameState to MatchState and related columns

Revision ID: b20f910fb6ce
Revises: 3bc91ed9e74d
Create Date: 2025-03-29 18:43:38.483166

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b20f910fb6ce'
down_revision: Union[str, None] = '3bc91ed9e74d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


import bgbench # Import needed for JsonType resolution if used directly
import sqlalchemy as sa # Import sa

def upgrade() -> None:
    # Rename the table from game_states to match_states
    op.rename_table('game_states', 'match_states')

    # Rename the column game_id to match_id
    # Note: Need to handle dialect specifics for renaming columns and constraints
    dialect = op.get_context().dialect.name
    if dialect == 'postgresql':
        # Rename constraints first
        op.execute('ALTER TABLE match_states RENAME CONSTRAINT game_states_pkey TO match_states_pkey')
        op.execute('ALTER TABLE match_states RENAME CONSTRAINT game_states_game_id_fkey TO match_states_match_id_fkey')
        # Rename the index associated with the primary key - This is often implicit when renaming the constraint in PG, removing explicit command.
        # op.execute('ALTER INDEX game_states_pkey RENAME TO match_states_pkey')

        # Now rename the column
        op.alter_column('match_states', 'game_id', new_column_name='match_id', existing_type=sa.Integer(), nullable=False)

        # Create the standard index for the renamed column (match_id)
        op.create_index(op.f('ix_match_states_match_id'), 'match_states', ['match_id'], unique=False)
    elif dialect == 'sqlite':
        # SQLite requires batch operations for altering tables
        with op.batch_alter_table('match_states', schema=None) as batch_op:
            # SQLite doesn't easily support renaming constraints/indexes directly in batch mode
            # It often involves recreating the table. Assuming simple rename for now.
            batch_op.alter_column('game_id', new_column_name='match_id', existing_type=sa.Integer(), nullable=False)
            batch_op.create_index(batch_op.f('ix_match_states_match_id'), ['match_id'], unique=False) # Create new index
    else:
        # Fallback or raise error for unsupported dialects - may need manual constraint/index renaming
        op.alter_column('match_states', 'game_id', new_column_name='match_id', existing_type=sa.Integer(), nullable=False)
        op.create_index(op.f('ix_match_states_match_id'), 'match_states', ['match_id'], unique=False) # Create new index

    # Add the new timestamp column
    op.add_column('match_states', sa.Column('timestamp', sa.DateTime(), server_default=sa.text('now()'), nullable=False))


def downgrade() -> None:
    # Drop the timestamp column
    op.drop_column('match_states', 'timestamp')

    # Rename the column match_id back to game_id
    dialect = op.get_context().dialect.name
    if dialect == 'postgresql':
        # Drop the standard named index created during upgrade
        op.drop_index(op.f('ix_match_states_match_id'), table_name='match_states')

        # Rename the column back
        op.alter_column('match_states', 'match_id', new_column_name='game_id', existing_type=sa.Integer(), nullable=False)

        # Rename constraints back
        op.execute('ALTER TABLE match_states RENAME CONSTRAINT match_states_pkey TO game_states_pkey')
        op.execute('ALTER TABLE match_states RENAME CONSTRAINT match_states_match_id_fkey TO game_states_game_id_fkey')
        # Rename the index back - This is often implicit when renaming the constraint in PG, removing explicit command.
        # op.execute('ALTER INDEX match_states_pkey RENAME TO game_states_pkey')

    elif dialect == 'sqlite':
        with op.batch_alter_table('match_states', schema=None) as batch_op:
            batch_op.drop_index(batch_op.f('ix_match_states_match_id')) # Drop the standard named index
            batch_op.alter_column('match_id', new_column_name='game_id', existing_type=sa.Integer(), nullable=False)
    else:
        op.drop_index(op.f('ix_match_states_match_id'), table_name='match_states') # Drop the standard named index
        op.alter_column('match_states', 'match_id', new_column_name='game_id', existing_type=sa.Integer(), nullable=False)

    # Rename the table back to game_states
    op.rename_table('match_states', 'game_states')
