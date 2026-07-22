"""Application configuration and environment variable handling."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_LLM_PROVIDER = "google"
DEFAULT_MODEL_NAME = "gemini-flash-latest"
DEFAULT_DATABASE_FILENAME = "chinook.db"
DEFAULT_CSV_TABLE_NAME = "uploaded_data"
DEFAULT_TOP_K = 5
DEFAULT_SAMPLE_ROWS = 3

UPLOAD_DIR = PROJECT_ROOT / "uploads"

GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"
LOG_FILE = PROJECT_ROOT / "text2sql-agent.log"


def load_environment() -> None:
    """Load environment variables from a local .env file if one exists."""
    load_dotenv()
    configure_logging()


def configure_logging() -> None:
    """Configure quiet file logging for operational events."""
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def get_required_env(name: str) -> str:
    """Return a required environment variable or raise a helpful error."""
    value = os.getenv(name)

    if not value:
        raise ValueError(
            f"Missing required environment variable: {name}. "
            "Create a .env file or set it in your shell."
        )

    return value


def get_google_api_key() -> str:
    """Return the Google API key required by the current Gemini model."""
    return get_required_env(GOOGLE_API_KEY_ENV)


def get_model_name() -> str:
    """Return the configured chat model name."""
    return os.getenv("LLM_MODEL_NAME", DEFAULT_MODEL_NAME)


def get_database_path() -> Path:
    """Return the default SQLite database path."""
    return PROJECT_ROOT / DEFAULT_DATABASE_FILENAME
