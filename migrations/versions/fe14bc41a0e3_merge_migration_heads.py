"""Merge migration heads

Revision ID: fe14bc41a0e3
Revises: 50285f1003de, add_cascade_deletion
Create Date: 2025-03-23 18:35:12.190687

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fe14bc41a0e3'
down_revision: Union[str, None] = ('50285f1003de', 'add_cascade_deletion')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass