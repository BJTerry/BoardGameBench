"""
Script to generate Alembic migration for initial PostgreSQL schema.
"""

import os
import sys
from pathlib import Path
from alembic import command
from alembic.config import Config


def generate_migration():
    """Generate an Alembic migration for the current schema."""
    base_dir = Path(__file__).resolve().parent

    # Set up Alembic configuration
    alembic_cfg = Config(str(base_dir / "alembic.ini"))

    # Set the PostgreSQL URL - required for autogenerate to work
    if "DB_HOST" not in os.environ:
        print(
            "Error: DB_HOST environment variable not set. Create a .env file with PostgreSQL configuration."
        )
        sys.exit(1)

    from urllib.parse import quote_plus

    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "bgbench")
    db_user = os.getenv("DB_USER", "bgbench_user")
    db_password = os.getenv("DB_PASSWORD", "bgbench_password")

    postgres_url = (
        f"postgresql://{quote_plus(db_user)}:{quote_plus(db_password)}@"
        f"{db_host}:{db_port}/{db_name}"
    )

    alembic_cfg.set_main_option("sqlalchemy.url", postgres_url)

    # Generate the migration
    command.revision(
        alembic_cfg, message="Initial PostgreSQL schema", autogenerate=True
    )
    print(f"Migration generated successfully.")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    generate_migration()
