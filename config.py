import os
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Default to SQLite if PostgreSQL environment variables are not set
if os.getenv("DB_HOST"):
    # PostgreSQL configuration from environment variables
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "bgbench")
    DB_USER = os.getenv("DB_USER", "bgbench_user")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "bgbench_password")

    # URL-encode username and password to handle special characters
    DATABASE_URL = (
        f"postgresql://{quote_plus(DB_USER)}:{quote_plus(DB_PASSWORD)}@"
        f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
else:
    # SQLite fallback
    DATABASE_URL = f"sqlite:///{os.path.join(DATA_DIR, 'bgbench.db')}"
