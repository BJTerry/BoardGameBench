"""Add cascade delete to foreign keys

Revision ID: add_cascade_deletion
Revises: 
Create Date: 2023-03-23

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_cascade_deletion'
down_revision = None  # Adjust this to the appropriate previous migration if needed
branch_labels = None
depends_on = None


def upgrade():
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


def downgrade():
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