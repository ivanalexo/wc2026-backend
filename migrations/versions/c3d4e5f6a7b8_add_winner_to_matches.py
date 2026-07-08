"""add_winner_to_matches

Agrega `winner` a `matches`: lado ganador ("HOME"/"AWAY") cuando un partido de
eliminatoria se define por prórroga/penales (fullTime empatado). NULL en el resto.
Necesario para propagar W<n>/L<n> en empates, donde el marcador no basta.

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-06-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c3d4e5f6a7b8'
down_revision: Union[str, Sequence[str], None] = 'b2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('matches', sa.Column('winner', sa.String(length=4), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('matches', 'winner')
