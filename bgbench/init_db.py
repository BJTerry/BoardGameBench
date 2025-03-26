from sqlalchemy import create_engine, text
from bgbench.models import Base
from config import DATABASE_URL


def init_db():
    engine = create_engine(DATABASE_URL)

    # Create all tables
    Base.metadata.create_all(engine)

    # Initialize sequences for PostgreSQL
    if engine.dialect.name == "postgresql":
        with engine.connect() as conn:
            # List of tables with sequences to initialize
            tables = [
                "experiments",
                "players",
                "games",
                "game_states",
                "llm_interactions",
            ]

            for table in tables:
                # Ensure sequence starts at 1 (if no records exist yet)
                conn.execute(text(f"SELECT setval('{table}_id_seq', 1, false)"))

            conn.commit()
            print("Initialized PostgreSQL sequences.")

    print("Database initialized successfully.")


if __name__ == "__main__":
    init_db()
