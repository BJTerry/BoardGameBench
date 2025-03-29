"""
Data migration utility to transfer data from SQLite to PostgreSQL.
"""

import os
import argparse
import logging
from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import sessionmaker
from bgbench.models import (
    Base,
    Experiment,
    Player,
    GameMatch,
    MatchState, # Renamed from GameState
    LLMInteraction,
)
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_postgres_connection(args):
    """Set up PostgreSQL connection from environment or arguments."""
    db_host = args.host or os.getenv("DB_HOST", "localhost")
    db_port = args.port or os.getenv("DB_PORT", "5432")
    db_name = args.dbname or os.getenv("DB_NAME", "bgbench")
    db_user = args.user or os.getenv("DB_USER", "bgbench_user")
    db_password = args.password or os.getenv("DB_PASSWORD", "bgbench_password")

    # URL-encode username and password to handle special characters
    postgres_url = (
        f"postgresql://{quote_plus(db_user)}:{quote_plus(db_password)}@"
        f"{db_host}:{db_port}/{db_name}"
    )

    return postgres_url


def setup_sqlite_connection():
    """Set up SQLite connection from the standard location."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    sqlite_path = os.path.join(data_dir, "bgbench.db")

    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(f"SQLite database not found at {sqlite_path}")

    sqlite_url = f"sqlite:///{sqlite_path}"
    return sqlite_url


def migrate_data(sqlite_url, postgres_url, drop_existing=False):
    """Migrate data from SQLite to PostgreSQL."""
    # Connect to SQLite (source)
    sqlite_engine = create_engine(sqlite_url)
    SQLiteSession = sessionmaker(bind=sqlite_engine)
    sqlite_session = SQLiteSession()

    # Connect to PostgreSQL (target)
    postgres_engine = create_engine(postgres_url)

    # Optionally drop existing tables
    if drop_existing:
        logger.info("Dropping existing tables in PostgreSQL...")
        Base.metadata.drop_all(postgres_engine)

    # Create schema in PostgreSQL
    logger.info("Creating schema in PostgreSQL...")
    Base.metadata.create_all(postgres_engine)

    PostgresSession = sessionmaker(bind=postgres_engine)
    postgres_session = PostgresSession()

    try:
        # Migrate Experiments
        logger.info("Migrating experiments...")
        experiments = sqlite_session.query(Experiment).all()
        for exp in experiments:
            postgres_exp = Experiment(
                id=exp.id, name=exp.name, description=exp.description
            )
            postgres_session.add(postgres_exp)
        postgres_session.commit()

        # Migrate Players
        logger.info("Migrating players...")
        players = sqlite_session.query(Player).all()
        for player in players:
            postgres_player = Player(
                id=player.id,
                name=player.name,
                rating=player.rating,
                model_config=player.model_config,
                experiment_id=player.experiment_id,
            )
            postgres_session.add(postgres_player)
        postgres_session.commit()

        # Migrate Games
        logger.info("Migrating games...")
        games = sqlite_session.query(GameMatch).all()
        for game in games:
            postgres_game = GameMatch(
                id=game.id,
                experiment_id=game.experiment_id,
                player1_id=game.player1_id,
                player2_id=game.player2_id,
                winner_id=game.winner_id,
                conceded=bool(game.conceded),  # Convert SQLite integers to booleans
                concession_reason=game.concession_reason,
                complete=bool(game.complete),  # Convert SQLite integers to booleans
            )
            postgres_session.add(postgres_game)
        postgres_session.commit()

        # Create index of valid game IDs in PostgreSQL
        valid_game_ids = set()
        for game in postgres_session.query(GameMatch.id).all():
            valid_game_ids.add(game[0])

        # Create index of valid player IDs in PostgreSQL
        valid_player_ids = set()
        for player in postgres_session.query(Player.id).all():
            valid_player_ids.add(player[0])

        # Migrate Match States with FK validation
        logger.info("Migrating match states...")
        states = sqlite_session.query(MatchState).all() # Use MatchState
        valid_states = 0
        invalid_states = 0

        for state in states:
            # Check if match_id exists in PostgreSQL
            if state.match_id in valid_game_ids: # Use match_id
                postgres_state = MatchState( # Use MatchState
                    id=state.id, match_id=state.match_id, state_data=state.state_data # Use match_id
                )
                postgres_session.add(postgres_state)
                valid_states += 1
            else:
                invalid_states += 1
                logger.warning(
                    f"Skipping match state {state.id} - references non-existent match_id {state.match_id}" # Use match_id
                )

        logger.info(
            f"Match states: {valid_states} valid, {invalid_states} skipped due to invalid foreign keys"
        )
        postgres_session.commit()

        # Migrate LLM Interactions with FK validation
        logger.info("Migrating LLM interactions...")
        interactions = sqlite_session.query(LLMInteraction).all()
        valid_interactions = 0
        invalid_interactions = 0

        for interaction in interactions:
            # Check if both game_id and player_id exist in PostgreSQL
            if (
                interaction.game_id in valid_game_ids
                and interaction.player_id in valid_player_ids
            ):
                postgres_interaction = LLMInteraction(
                    id=interaction.id,
                    game_id=interaction.game_id,
                    player_id=interaction.player_id,
                    prompt=interaction.prompt,
                    response=interaction.response,
                    start_time=interaction.start_time,
                    end_time=interaction.end_time,
                    prompt_tokens=interaction.prompt_tokens,
                    completion_tokens=interaction.completion_tokens,
                    total_tokens=interaction.total_tokens,
                    cost=interaction.cost,
                )
                postgres_session.add(postgres_interaction)
                valid_interactions += 1
            else:
                invalid_interactions += 1
                if interaction.game_id not in valid_game_ids:
                    logger.warning(
                        f"Skipping LLM interaction {interaction.id} - references non-existent game_id {interaction.game_id}"
                    )
                if interaction.player_id not in valid_player_ids:
                    logger.warning(
                        f"Skipping LLM interaction {interaction.id} - references non-existent player_id {interaction.player_id}"
                    )

        logger.info(
            f"LLM interactions: {valid_interactions} valid, {invalid_interactions} skipped due to invalid foreign keys"
        )
        postgres_session.commit()

        # Verify migration with counts and report status
        total_sqlite_records = 0
        total_postgres_records = 0

        logger.info("=== Migration Summary ===")

        for table_name, model_class in [
            ("experiments", Experiment),
            ("players", Player),
            ("games", GameMatch),
            ("match_states", MatchState), # Use MatchState and table name
            ("llm_interactions", LLMInteraction),
        ]:
            sqlite_count = sqlite_session.query(model_class).count()
            postgres_count = postgres_session.query(model_class).count()
            total_sqlite_records += sqlite_count
            total_postgres_records += postgres_count

            if sqlite_count == postgres_count:
                status = "✓ Complete"
            else:
                skipped = sqlite_count - postgres_count
                percent = (
                    (postgres_count / sqlite_count * 100) if sqlite_count > 0 else 0
                )
                status = f"⚠ {skipped} records skipped ({percent:.1f}% migrated)"

            logger.info(
                f"Table {table_name}: SQLite={sqlite_count}, PostgreSQL={postgres_count} - {status}"
            )

        # Overall migration stats
        overall_percent = (
            (total_postgres_records / total_sqlite_records * 100)
            if total_sqlite_records > 0
            else 0
        )
        logger.info(
            f"Overall: {total_postgres_records}/{total_sqlite_records} records migrated ({overall_percent:.1f}%)"
        )

        if total_postgres_records < total_sqlite_records:
            logger.info(
                "Note: Missing records are likely due to foreign key constraints that were not enforced in SQLite"
            )
            logger.info(
                "This is normal and expected. You may need to recreate some records manually if necessary."
            )

        # Synchronize PostgreSQL sequences with current max IDs
        if postgres_engine.dialect.name == "postgresql":
            logger.info("Synchronizing PostgreSQL sequences...")
            table_map = {
                "experiments": Experiment,
                "players": Player,
                "games": GameMatch,
                "match_states": MatchState, # Use MatchState and table name
                "llm_interactions": LLMInteraction,
            }

            for table_name, model_class in table_map.items():
                # Get max ID for this table
                max_id = postgres_session.query(func.max(model_class.id)).scalar() or 0

                # Update sequence
                postgres_session.execute(
                    text(f"SELECT setval('{table_name}_id_seq', {max_id}, true)")
                )
                logger.info(f"Set {table_name}_id_seq to start after {max_id}")

            postgres_session.commit()

        logger.info("Migration completed.")

    except SQLAlchemyError as e:
        postgres_session.rollback()
        logger.error(f"Error during migration: {str(e)}")
        raise
    finally:
        sqlite_session.close()
        postgres_session.close()


def main():
    parser = argparse.ArgumentParser(
        description="Migrate data from SQLite to PostgreSQL"
    )
    parser.add_argument("--host", help="PostgreSQL host")
    parser.add_argument("--port", help="PostgreSQL port")
    parser.add_argument("--dbname", help="PostgreSQL database name")
    parser.add_argument("--user", help="PostgreSQL username")
    parser.add_argument("--password", help="PostgreSQL password")
    parser.add_argument(
        "--drop", action="store_true", help="Drop existing tables in PostgreSQL"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting migration from SQLite to PostgreSQL")
    logger.info("Note: Some items may be skipped due to foreign key constraints")
    logger.info("This is normal and expected when migrating from SQLite to PostgreSQL")

    try:
        # Set up connections
        sqlite_url = setup_sqlite_connection()
        logger.info(f"Using SQLite database: {sqlite_url}")

        postgres_url = setup_postgres_connection(args)
        # Mask password in log
        masked_url = (
            postgres_url.replace(
                quote_plus(
                    args.password or os.getenv("DB_PASSWORD", "bgbench_password")
                ),
                "********",
            )
            if args.password or os.getenv("DB_PASSWORD")
            else postgres_url
        )
        logger.info(f"Using PostgreSQL database: {masked_url}")

        # Perform migration
        migrate_data(sqlite_url, postgres_url, drop_existing=args.drop)

        logger.info(
            "Migration completed successfully, though some records may have been skipped"
        )
        logger.info(
            "If you encounter any issues with missing data, please run with --debug for more information"
        )

    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
