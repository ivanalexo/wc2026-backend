"""add_bracket_slots_to_matches
Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-06-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('matches', sa.Column('match_number', sa.Integer(), nullable=True))
    op.add_column('matches', sa.Column('home_slot', sa.String(length=10), nullable=True))
    op.add_column('matches', sa.Column('away_slot', sa.String(length=10), nullable=True))

    # unique=True + index=True en el modelo => índice único
    op.create_index('ix_matches_match_number', 'matches', ['match_number'], unique=True)

    op.alter_column('matches', 'home_team', existing_type=sa.String(length=100), nullable=True)
    op.alter_column('matches', 'away_team', existing_type=sa.String(length=100), nullable=True)


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column('matches', 'away_team', existing_type=sa.String(length=100), nullable=False)
    op.alter_column('matches', 'home_team', existing_type=sa.String(length=100), nullable=False)

    op.drop_index('ix_matches_match_number', table_name='matches')

    op.drop_column('matches', 'away_slot')
    op.drop_column('matches', 'home_slot')
    op.drop_column('matches', 'match_number')
