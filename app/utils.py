"""Shared helper functions."""

from collections.abc import Mapping, Sequence
from typing import Any


def extract_final_answer(agent_result: Any) -> str:
    """Extract readable final assistant text from a LangChain agent response."""
    try:
        final_message = _get_final_message(agent_result)
        content = getattr(final_message, "content", final_message)
        return _format_message_content(content)
    except Exception:
        return _safe_string(agent_result)


def extract_sql_queries(agent_result: Any) -> list[str]:
    """Extract SQL queries from agent tool calls when available."""
    queries: list[str] = []

    try:
        messages = (
            agent_result.get("messages")
            if isinstance(agent_result, Mapping)
            else getattr(agent_result, "messages", None)
        )
        if not messages:
            return queries

        for message in messages:
            tool_calls = getattr(message, "tool_calls", None) or []
            for tool_call in tool_calls:
                query = _extract_query_from_tool_call(tool_call)
                if query and query not in queries:
                    queries.append(query)
    except Exception:
        return queries

    return queries


def _get_final_message(agent_result: Any) -> Any:
    """Return the last message-like object from an agent result."""
    if isinstance(agent_result, Mapping):
        messages = agent_result.get("messages")
        if messages:
            return messages[-1]
        return agent_result.get("output", agent_result)

    messages = getattr(agent_result, "messages", None)
    if messages:
        return messages[-1]

    return agent_result


def _format_message_content(content: Any) -> str:
    """Format provider-specific message content as readable text."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, Mapping):
        text = _extract_text_from_block(content)
        return text if text else _safe_string(content)

    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        text_parts = []
        for item in content:
            text = (
                _extract_text_from_block(item)
                if isinstance(item, Mapping)
                else _format_message_content(item)
            )
            if text:
                text_parts.append(text)
        return "\n".join(text_parts).strip() or _safe_string(content)

    return _safe_string(content)


def _extract_text_from_block(block: Mapping[str, Any]) -> str:
    """Extract text from common Gemini/LangChain structured content blocks."""
    for key in ("text", "content", "output_text"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            text = _format_message_content(value)
            if text:
                return text

    nested = block.get("parts")
    if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes, bytearray)):
        return _format_message_content(nested)

    return ""


def _extract_query_from_tool_call(tool_call: Any) -> str:
    """Return the SQL query from a LangChain SQL tool call if present."""
    if not isinstance(tool_call, Mapping):
        return ""

    name = str(tool_call.get("name", "")).lower()
    if "query" not in name and "sql" not in name:
        return ""

    args = tool_call.get("args", {})
    if isinstance(args, Mapping):
        query = args.get("query") or args.get("sql") or args.get("statement")
        return query.strip() if isinstance(query, str) else ""

    return args.strip() if isinstance(args, str) else ""


def _safe_string(value: Any) -> str:
    """Return a readable string for any value without allowing repr failures."""
    try:
        return str(value)
    except Exception:
        return "No readable answer was returned."
