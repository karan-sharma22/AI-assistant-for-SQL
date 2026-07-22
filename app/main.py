"""Command-line entry point for the Text-to-SQL agent."""

import argparse
import logging
import sys

from rich.console import Console
from rich.panel import Panel

from app.config import load_environment
from app.utils import extract_final_answer, extract_sql_queries


console = Console()
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the SQL Agent CLI."""
    load_environment()

    parser = argparse.ArgumentParser(
        description="Text-to-SQL Agent powered by LangChain and Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py "What are the top 5 best-selling artists?"
  python agent.py "Which employee generated the most revenue?"
  python agent.py "How many customers are from Canada?"
  python agent.py --csv sample_sales.csv "Top 5 customers by revenue"
        """,
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to a CSV file to query instead of the Chinook database",
    )
    parser.add_argument(
        "--show-sql",
        action="store_true",
        help="Display generated SQL when it is available in the agent trace",
    )
    parser.add_argument(
        "question",
        type=str,
        help="Natural language question to answer using the database",
    )

    args = parser.parse_args()

    try:
        load_environment()

        console.print(
            Panel(
                f"[bold cyan]Question:[/bold cyan] {args.question}",
                border_style="cyan",
            )
        )
        console.print()

        if args.csv:
            console.print("[dim]Loading CSV...[/dim]")

        console.print("[dim]Creating SQL Agent...[/dim]")
        from app.sql_agent import create_sql_agent

        agent = create_sql_agent(csv_path=args.csv)

        console.print("[dim]Generating SQL...[/dim]")
        console.print("[dim]Executing SQL...[/dim]\n")
        result = agent.invoke(
            {"messages": [{"role": "user", "content": args.question}]}
        )
        answer = extract_final_answer(result)
        sql_queries = extract_sql_queries(result)

        if args.show_sql and sql_queries:
            console.print(
                Panel(
                    "\n\n".join(sql_queries),
                    title="SQL",
                    border_style="yellow",
                )
            )

        console.print(
            Panel(
                f"[bold green]Answer:[/bold green]\n\n{answer}",
                border_style="green",
            )
        )
    except Exception as error:
        logger.exception("CLI execution failed")
        console.print(
            Panel(
                f"[bold red]Error:[/bold red]\n\n{_friendly_error_message(error)}",
                border_style="red",
            )
        )
        sys.exit(1)


def _friendly_error_message(error: Exception) -> str:
    """Return a friendly, actionable CLI error message."""
    error_text = str(error)
    normalized = error_text.lower()

    if "google_api_key" in normalized or "missing required environment variable" in normalized:
        return (
            "Google API key is not configured. Add GOOGLE_API_KEY to your .env "
            "file or export it in your shell."
        )

    if isinstance(error, ImportError):
        return (
            "A required dependency is not installed in this Python environment. "
            "Activate the project virtualenv or run `pip install -e .`."
        )

    if "api_key_invalid" in normalized or "invalid api key" in normalized:
        return (
            "Google rejected the API key. Check GOOGLE_API_KEY in your .env file "
            "and make sure the key is active."
        )

    if "quota" in normalized or "resource_exhausted" in normalized or "429" in normalized:
        return (
            "The Gemini quota or rate limit was exceeded. Wait a bit, check your "
            "Google AI quota, or switch to another available model."
        )

    if "model" in normalized and (
        "not found" in normalized
        or "unavailable" in normalized
        or "unsupported" in normalized
    ):
        return (
            "The configured Gemini model is unavailable. Check LLM_MODEL_NAME in "
            "your .env file and choose a model your account can access."
        )

    if "503" in normalized or "overloaded" in normalized or "unavailable" in normalized:
        return (
            "Gemini is temporarily overloaded or unavailable. Please retry in a "
            "moment."
        )

    if "timeout" in normalized or "timed out" in normalized or "deadline" in normalized:
        return "The request timed out. Check your network connection and retry."

    if (
        "connecterror" in normalized
        or "connection refused" in normalized
        or "no connection could be made" in normalized
        or "network" in normalized
    ):
        return (
            "The agent could not reach Gemini. Check your network connection, "
            "proxy settings, and Google API access, then retry."
        )

    if isinstance(error, (FileNotFoundError, ValueError)):
        return error_text

    if any(
        marker in normalized
        for marker in ("sqlite", "sqlalchemy", "operationalerror", "syntax error")
    ):
        return (
            "The generated SQL could not be executed successfully. Try rephrasing "
            "the question or inspect the database schema."
        )

    return (
        "Something went wrong while running the agent. See text2sql-agent.log for "
        "details."
    )


if __name__ == "__main__":
    main()
