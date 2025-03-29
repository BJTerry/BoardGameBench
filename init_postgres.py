#!/usr/bin/env python3
"""
Initialize PostgreSQL database for BoardGameBench.
This script will create all necessary tables in the PostgreSQL database.
"""

import os
import argparse
import logging
from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker
from urllib.parse import quote_plus
from dotenv import load_dotenv
from bgbench.models import (
    Base,
    Experiment,
    Player,
    GameMatch,
    MatchState, # Renamed from GameState
    LLMInteraction,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def init_postgres_db():
    """Initialize the PostgreSQL database with the BoardGameBench schema."""
    # Load environment variables
    load_dotenv()

    # Get PostgreSQL connection details from environment
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "bgbench")
    db_user = os.getenv("DB_USER", "bgbench_user")
    db_password = os.getenv("DB_PASSWORD", "bgbench_password")

    # URL-encode username and password to handle special characters
    postgres_url = (
        f"postgresql://{quote_plus(db_user)}:{quote_plus(db_password)}@"
        f"{db_host}:{db_port}/{db_name}"
    )

    try:
        # Create engine and initialize tables
        engine = create_engine(postgres_url)

        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info(f"Successfully connected to PostgreSQL at {db_host}:{db_port}")

        # Create tables
        Base.metadata.create_all(engine)
        logger.info(f"Database schema created successfully in {db_name}")

        # Create a session to verify tables
        Session = sessionmaker(bind=engine)
        session = Session()

        # Get list of tables
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            )
            tables = [row[0] for row in result]

        logger.info(f"Created tables: {', '.join(tables)}")

        # Initialize PostgreSQL sequences
        if engine.dialect.name == "postgresql":
            logger.info("Initializing PostgreSQL sequences...")
            table_map = {
                "experiments": Experiment,
                "players": Player,
                "games": GameMatch,
                "match_states": MatchState, # Use MatchState and table name
                "llm_interactions": LLMInteraction,
            }

            for table_name, model_class in table_map.items():
                # Set sequence to start at 1 (or after existing max ID if data already exists)
                max_id = session.query(func.max(model_class.id)).scalar() or 0
                session.execute(
                    text(f"SELECT setval('{table_name}_id_seq', {max_id}, true)")
                )
                logger.info(f"Initialized {table_name}_id_seq to start after {max_id}")

            session.commit()

        session.close()

    except Exception as e:
        logger.error(f"Error initializing PostgreSQL database: {e}")
        return 1

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Initialize PostgreSQL database for BoardGameBench"
    )
    args = parser.parse_args()

    exit(init_postgres_db())
