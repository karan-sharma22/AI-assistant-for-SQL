"""Database connection helpers for the SQL agent."""

import logging
import re
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

from langchain_community.utilities import SQLDatabase

from app.config import (
    DEFAULT_CSV_TABLE_NAME,
    DEFAULT_SAMPLE_ROWS,
    get_database_path,
)

logger = logging.getLogger(__name__)


def get_database(csv_path: str | Path | None = None) -> SQLDatabase:
    """Create and return a configured SQLite database connection."""
    database_path = (
        create_sqlite_database_from_csv(csv_path)
        if csv_path
        else get_database_path()
    )

    return _create_sql_database(database_path)


def create_sqlite_database_from_csv(csv_path: str | Path) -> Path:
    """Create a temporary SQLite database from a CSV file."""
    source_path = _validate_csv_path(csv_path)
    dataframe = _read_csv(source_path)
    dataframe.columns = _clean_column_names(dataframe.columns)

    temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    database_path = Path(temp_file.name)
    temp_file.close()

    with sqlite3.connect(database_path) as connection:
        dataframe.to_sql(
            DEFAULT_CSV_TABLE_NAME,
            connection,
            if_exists="replace",
            index=False,
        )

    logger.info(
        "Imported CSV into SQLite database: csv=%s database=%s rows=%s columns=%s",
        source_path,
        database_path,
        len(dataframe),
        len(dataframe.columns),
    )

    return database_path


def _create_sql_database(database_path: Path) -> SQLDatabase:
    """Create a LangChain SQLDatabase from a SQLite file path."""
    sqlite_uri = f"sqlite:///{database_path.as_posix()}"
    logger.info("Creating SQLite database connection: %s", database_path)

    return SQLDatabase.from_uri(
        sqlite_uri,
        sample_rows_in_table_info=DEFAULT_SAMPLE_ROWS,
    )


def _validate_csv_path(csv_path: str | Path) -> Path:
    """Validate the CSV path before attempting to import it."""
    source_path = Path(csv_path).expanduser()

    if not source_path.exists():
        raise FileNotFoundError(f"CSV file not found: {source_path}")

    if not source_path.is_file():
        raise ValueError(f"CSV path is not a file: {source_path}")

    if source_path.suffix.lower() != ".csv":
        raise ValueError("Unsupported file type. The --csv option requires a .csv file.")

    return source_path


def _read_csv(source_path: Path) -> pd.DataFrame:
    """Read and validate a CSV file."""
    try:
        dataframe = pd.read_csv(source_path)
    except pd.errors.EmptyDataError as error:
        raise ValueError("CSV file is empty or does not contain headers.") from error
    except pd.errors.ParserError as error:
        raise ValueError(
            "CSV file could not be parsed. Check for mismatched quotes or columns."
        ) from error
    except UnicodeDecodeError as error:
        raise ValueError("CSV file encoding is not readable as text.") from error

    if dataframe.empty:
        raise ValueError("CSV file has headers but no data rows.")

    if dataframe.columns.empty:
        raise ValueError("CSV file must contain at least one column.")

    return dataframe


def _clean_column_names(columns: pd.Index) -> list[str]:
    """Normalize column names and make duplicates unique."""
    cleaned_names: list[str] = []
    seen: dict[str, int] = {}

    for index, column in enumerate(columns, start=1):
        base_name = str(column).strip().lower()
        base_name = re.sub(r"\W+", "_", base_name).strip("_")
        base_name = base_name or f"column_{index}"

        occurrence = seen.get(base_name, 0)
        seen[base_name] = occurrence + 1

        name = base_name if occurrence == 0 else f"{base_name}_{occurrence + 1}"
        cleaned_names.append(name)

    return cleaned_names
