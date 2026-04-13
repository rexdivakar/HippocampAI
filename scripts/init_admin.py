#!/usr/bin/env python3
"""Create the initial admin user.

Reads credentials from environment variables, generates a bcrypt hash,
and inserts the admin user via the create_default_admin() SQL function.

Required env vars:
    ADMIN_PASSWORD   Plain-text password for admin@hippocampai.com (min 12 chars)

Optional env vars (default to docker-compose values):
    POSTGRES_HOST     (default: localhost)
    POSTGRES_PORT     (default: 5432)
    POSTGRES_DB       (default: hippocampai)
    POSTGRES_USER     (default: hippocampai)
    POSTGRES_PASSWORD (default: hippocampai_secret)
    ADMIN_EMAIL       (default: admin@hippocampai.com)

Usage:
    ADMIN_PASSWORD=<strong-password> python scripts/init_admin.py
"""

import asyncio
import os
import sys


async def main() -> None:
    try:
        import asyncpg
        import bcrypt
    except ImportError as exc:
        print(f"ERROR: missing dependency — {exc}")
        print("Install with: pip install asyncpg bcrypt")
        sys.exit(1)

    password = os.environ.get("ADMIN_PASSWORD", "").strip()
    if not password:
        print("ERROR: ADMIN_PASSWORD env var is required and must not be empty.")
        sys.exit(1)
    if len(password) < 12:
        print("ERROR: ADMIN_PASSWORD must be at least 12 characters.")
        sys.exit(1)

    admin_email = os.environ.get("ADMIN_EMAIL", "admin@hippocampai.com").strip()

    hashed: str = bcrypt.hashpw(password.encode(), bcrypt.gensalt(12)).decode()

    dsn = {
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": int(os.environ.get("POSTGRES_PORT", "5432")),
        "database": os.environ.get("POSTGRES_DB", "hippocampai"),
        "user": os.environ.get("POSTGRES_USER", "hippocampai"),
        "password": os.environ.get("POSTGRES_PASSWORD", "hippocampai_secret"),
    }

    conn = await asyncpg.connect(**dsn)
    try:
        await conn.execute("SELECT create_default_admin($1)", hashed)
        # Update email if a custom ADMIN_EMAIL was supplied
        if admin_email != "admin@hippocampai.com":
            await conn.execute(
                "UPDATE users SET email = $1 WHERE email = 'admin@hippocampai.com'",
                admin_email,
            )
        print(f"Admin user created (or already exists): {admin_email}")
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
