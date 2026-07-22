"""SQL agent assembly."""

from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from app.config import DEFAULT_TOP_K
from app.database import get_database
from app.llm import get_llm
from app.prompts import SQL_AGENT_SYSTEM_PROMPT


def create_sql_agent(csv_path: str | Path | None = None) -> Any:
    """Create and return a text-to-SQL agent."""
    database = get_database(csv_path=csv_path)
    llm = get_llm()

    toolkit = SQLDatabaseToolkit(db=database, llm=llm)
    tools = toolkit.get_tools()

    return create_agent(
        llm,
        tools,
        system_prompt=SQL_AGENT_SYSTEM_PROMPT.format(
            dialect=database.dialect,
            top_k=DEFAULT_TOP_K,
        ),
    )
