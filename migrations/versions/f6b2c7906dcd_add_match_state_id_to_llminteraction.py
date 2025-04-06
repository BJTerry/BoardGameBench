"""Add match_state_id to LLMInteraction

Revision ID: f6b2c7906dcd
Revises: b20f910fb6ce
Create Date: 2025-04-05 18:37:18.544817

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f6b2c7906dcd'
down_revision: Union[str, None] = 'b20f910fb6ce'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('llm_interactions', sa.Column('match_state_id', sa.Integer(), nullable=True))
    op.create_index(op.f('ix_llm_interactions_match_state_id'), 'llm_interactions', ['match_state_id'], unique=False)
    op.create_foreign_key(
        'fk_llm_interactions_match_state_id_match_states',  # Explicit constraint name
        'llm_interactions', 'match_states',
        ['match_state_id'], ['id']
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(
        'fk_llm_interactions_match_state_id_match_states',  # Use the explicit name
        'llm_interactions', type_='foreignkey'
    )
    op.drop_index(op.f('ix_llm_interactions_match_state_id'), table_name='llm_interactions')
    op.drop_column('llm_interactions', 'match_state_id')
    # ### end Alembic commands ###
