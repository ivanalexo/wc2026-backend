"""add_download_events

Tabla de métricas de descargas del calendario (.ics): navegador + geo por descarga.

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-07-08 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'd4e5f6a7b8c9'
down_revision: Union[str, Sequence[str], None] = 'c3d4e5f6a7b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'download_events',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('round', sa.String(length=30), nullable=False),
        sa.Column('browser', sa.String(length=40), nullable=True),
        sa.Column('os', sa.String(length=40), nullable=True),
        sa.Column('user_agent', sa.String(length=300), nullable=True),
        sa.Column('country', sa.String(length=80), nullable=True),
        sa.Column('city', sa.String(length=120), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
    )
    op.create_index('ix_download_events_round', 'download_events', ['round'])
    op.create_index('ix_download_events_country', 'download_events', ['country'])
    op.create_index('ix_download_events_created_at', 'download_events', ['created_at'])


def downgrade() -> None:
    op.drop_index('ix_download_events_created_at', table_name='download_events')
    op.drop_index('ix_download_events_country', table_name='download_events')
    op.drop_index('ix_download_events_round', table_name='download_events')
    op.drop_table('download_events')
