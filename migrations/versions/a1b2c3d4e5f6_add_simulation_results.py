"""add_simulation_results

Revision ID: a1b2c3d4e5f6
Revises: c7628b0d7489
Create Date: 2026-06-14 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'c7628b0d7489'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'simulation_results',
        sa.Column('team', sa.String(length=100), nullable=False),
        sa.Column('elo', sa.Float(), nullable=True),
        sa.Column('p_qualify', sa.Float(), nullable=False),
        sa.Column('p_reach_r16', sa.Float(), nullable=False),
        sa.Column('p_reach_qf', sa.Float(), nullable=False),
        sa.Column('p_reach_sf', sa.Float(), nullable=False),
        sa.Column('p_reach_final', sa.Float(), nullable=False),
        sa.Column('p_champion', sa.Float(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('team'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('simulation_results')
