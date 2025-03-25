"""Add cascade delete to foreign keys

Revision ID: 50285f1003de
Revises: 
Create Date: 2025-03-23 18:33:13.606526

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '50285f1003de'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop existing foreign key constraints
    op.drop_constraint('game_states_game_id_fkey', 'game_states', type_='foreignkey')
    op.drop_constraint('llm_interactions_game_id_fkey', 'llm_interactions', type_='foreignkey')
    op.drop_constraint('llm_interactions_player_id_fkey', 'llm_interactions', type_='foreignkey')
    
    # Add new foreign key constraints with CASCADE
    op.create_foreign_key(
        'game_states_game_id_fkey', 'game_states', 'games',
        ['game_id'], ['id'], ondelete='CASCADE'
    )
    op.create_foreign_key(
        'llm_interactions_game_id_fkey', 'llm_interactions', 'games',
        ['game_id'], ['id'], ondelete='CASCADE'
    )
    op.create_foreign_key(
        'llm_interactions_player_id_fkey', 'llm_interactions', 'players',
        ['player_id'], ['id'], ondelete='CASCADE'
    )


def downgrade() -> None:
    # Drop CASCADE foreign key constraints
    op.drop_constraint('game_states_game_id_fkey', 'game_states', type_='foreignkey')
    op.drop_constraint('llm_interactions_game_id_fkey', 'llm_interactions', type_='foreignkey')
    op.drop_constraint('llm_interactions_player_id_fkey', 'llm_interactions', type_='foreignkey')
    
    # Add back standard foreign key constraints without CASCADE
    op.create_foreign_key(
        'game_states_game_id_fkey', 'game_states', 'games',
        ['game_id'], ['id']
    )
    op.create_foreign_key(
        'llm_interactions_game_id_fkey', 'llm_interactions', 'games',
        ['game_id'], ['id']
    )
    op.create_foreign_key(
        'llm_interactions_player_id_fkey', 'llm_interactions', 'players',
        ['player_id'], ['id']
    )