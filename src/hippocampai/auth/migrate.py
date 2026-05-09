"""Migration helpers for HippocampAI auth/audit database.

Provides programmatic access to Alembic migration commands so that
application startup code and CLI commands can apply or inspect migrations
without shelling out to the ``alembic`` CLI.

Usage from application startup
-------------------------------
    from hippocampai.auth.migrate import run_migrations
    run_migrations()

Usage from CLI / scripts
-------------------------
    from hippocampai.auth.migrate import get_migration_status
    status = get_migration_status()
    print(status)

The functions locate ``alembic.ini`` relative to this file's location in
the installed package, so they work regardless of the working directory from
which the application is launched.
"""

import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alembic.config import Config

logger = logging.getLogger(__name__)

# alembic.ini sits at the project root, four levels above this file:
# src/hippocampai/auth/migrate.py → src/hippocampai/auth/ → src/hippocampai/ → src/ → project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ALEMBIC_INI = _PROJECT_ROOT / "alembic.ini"


def _get_alembic_config() -> "Config":
    """Return an Alembic ``Config`` object pointing at the project ini file.

    Raises
    ------
    FileNotFoundError
        If ``alembic.ini`` cannot be found at the expected location.
    """
    from alembic.config import Config

    if not _ALEMBIC_INI.exists():
        raise FileNotFoundError(
            f"alembic.ini not found at {_ALEMBIC_INI}. "
            "Ensure the project root is correctly detected."
        )

    cfg = Config(str(_ALEMBIC_INI))
    # Ensure the project src directory is on sys.path so that env.py can
    # import hippocampai packages during migration execution.
    cfg.set_main_option("prepend_sys_path", str(_PROJECT_ROOT / "src"))
    return cfg


def run_migrations(revision: str = "head") -> None:
    """Apply all pending Alembic migrations up to ``revision``.

    Parameters
    ----------
    revision:
        Alembic revision target. Defaults to ``"head"`` (latest). Pass a
        specific revision ID or ``"-1"`` to upgrade/downgrade selectively.

    Raises
    ------
    FileNotFoundError
        If ``alembic.ini`` cannot be located.
    RuntimeError
        If Alembic raises an error during migration execution.
    """
    from alembic import command

    cfg = _get_alembic_config()
    logger.info("Running database migrations to revision '%s'", revision)
    try:
        command.upgrade(cfg, revision)
        logger.info("Database migrations completed successfully.")
    except Exception as exc:
        logger.error("Database migration failed: %s", exc, exc_info=True)
        raise RuntimeError(f"Alembic upgrade to '{revision}' failed: {exc}") from exc


def downgrade_migrations(revision: str = "-1") -> None:
    """Roll back migrations to ``revision``.

    Parameters
    ----------
    revision:
        Alembic revision target. Defaults to ``"-1"`` (one step back).

    Raises
    ------
    FileNotFoundError
        If ``alembic.ini`` cannot be located.
    RuntimeError
        If Alembic raises an error during downgrade execution.
    """
    from alembic import command

    cfg = _get_alembic_config()
    logger.info("Rolling back database migrations to revision '%s'", revision)
    try:
        command.downgrade(cfg, revision)
        logger.info("Database downgrade completed successfully.")
    except Exception as exc:
        logger.error("Database downgrade failed: %s", exc, exc_info=True)
        raise RuntimeError(f"Alembic downgrade to '{revision}' failed: {exc}") from exc


def get_migration_status() -> str:
    """Return the current Alembic revision as a human-readable string.

    Returns
    -------
    str
        The current revision ID(s), or ``"(no migrations applied)"`` if the
        revision table does not exist yet.

    Raises
    ------
    FileNotFoundError
        If ``alembic.ini`` cannot be located.
    """
    from alembic import command

    cfg = _get_alembic_config()

    # Capture alembic's output by providing a StringIO buffer as the
    # config's output stream via the Config constructor argument.
    buf = io.StringIO()
    from alembic.config import Config

    capturing_cfg = Config(str(_ALEMBIC_INI), stdout=buf)
    capturing_cfg.set_main_option("prepend_sys_path", str(_PROJECT_ROOT / "src"))

    try:
        command.current(capturing_cfg, verbose=True)
        output = buf.getvalue().strip()
        return output if output else "(no migrations applied)"
    except Exception as exc:
        logger.warning("Could not determine migration status: %s", exc)
        return f"(error retrieving status: {exc})"


def stamp_revision(revision: str) -> None:
    """Stamp the database with ``revision`` without running any migrations.

    Use this when the schema already exists (e.g. created by a Docker
    entrypoint script) and you want to record the baseline revision without
    re-executing the migration's ``upgrade()`` function.

    Parameters
    ----------
    revision:
        Alembic revision to stamp. Typically ``"head"`` or a specific ID such
        as ``"a1b2c3d4e5f6"`` for the initial schema.

    Raises
    ------
    FileNotFoundError
        If ``alembic.ini`` cannot be located.
    RuntimeError
        If Alembic raises an error during stamping.
    """
    from alembic import command

    cfg = _get_alembic_config()
    logger.info("Stamping database with revision '%s'", revision)
    try:
        command.stamp(cfg, revision)
        logger.info("Database stamped successfully.")
    except Exception as exc:
        logger.error("Database stamp failed: %s", exc, exc_info=True)
        raise RuntimeError(f"Alembic stamp to '{revision}' failed: {exc}") from exc
