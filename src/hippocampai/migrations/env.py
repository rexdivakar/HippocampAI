"""Alembic environment — synchronous engine supporting SQLite and PostgreSQL.

Database selection is driven by the DB_TYPE environment variable (default:
"sqlite"). Connection parameters are read from the same env vars used by
``hippocampai.config.Config``, so no extra configuration is needed beyond
what the application already requires.

Environment variables
---------------------
DB_TYPE          "sqlite" (default) or "postgres"
SQLITE_PATH      Path to the SQLite file (default: data/hippocampai_auth.db)
POSTGRES_HOST    Postgres hostname (default: localhost)
POSTGRES_PORT    Postgres port (default: 5432)
POSTGRES_DB      Postgres database name (default: hippocampai)
POSTGRES_USER    Postgres user (default: hippocampai)
POSTGRES_PASSWORD Postgres password (default: hippocampai_secret)

The ``target_metadata`` is set to the SQLAlchemy metadata from
``hippocampai.auth.sa_models`` so that ``alembic revision --autogenerate``
can detect schema drift.
"""

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# ---------------------------------------------------------------------------
# SQLAlchemy metadata — required for autogenerate support
# ---------------------------------------------------------------------------
from hippocampai.auth.sa_models import metadata as target_metadata  # noqa: E402

# ---------------------------------------------------------------------------
# Alembic config object (exposes the .ini file values)
# ---------------------------------------------------------------------------
config = context.config

# Wire up Python logging from alembic.ini if available
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _build_database_url() -> str:
    """Construct the database URL from environment variables.

    Reads the same env vars as ``hippocampai.config.Config`` so that the
    migration environment is always consistent with the application config.
    """
    db_type = os.environ.get("DB_TYPE", "sqlite").lower()

    if db_type == "postgres":
        host = os.environ.get("POSTGRES_HOST", "localhost")
        port = os.environ.get("POSTGRES_PORT", "5432")
        db = os.environ.get("POSTGRES_DB", "hippocampai")
        user = os.environ.get("POSTGRES_USER", "hippocampai")
        password = os.environ.get("POSTGRES_PASSWORD", "hippocampai_secret")
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"

    # SQLite — three slashes for a relative path, four for absolute.
    sqlite_path = os.environ.get("SQLITE_PATH", "data/hippocampai_auth.db")
    if os.path.isabs(sqlite_path):
        return f"sqlite:///{sqlite_path}"
    return f"sqlite:///{sqlite_path}"


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — emits SQL to stdout, no DB connection needed.

    Useful for generating migration SQL to review or apply manually.
    """
    url = _build_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Render migration operations using the full type system rather than raw SQL
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    url = _build_database_url()

    # Override the URL from alembic.ini with the one we constructed
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = url

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # render_as_batch enables ALTER TABLE emulation for SQLite, which
            # does not natively support column addition/removal.
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
