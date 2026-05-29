# =============================================================================
# app/db/base.py — Base declarativa de SQLAlchemy
# =============================================================================
# Define la clase Base de la que heredan todos los modelos ORM.
#
# IMPORTANTE — para Alembic:
# El env.py de Alembic debe importar Base Y todos los modelos para que
# autogenerate detecte los cambios de esquema. Ese import se hace en
# migrations/env.py, no aquí, para evitar imports circulares.
# =============================================================================

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass