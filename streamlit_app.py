"""Premium Streamlit frontend for the existing Text-to-SQL agent."""

from __future__ import annotations

import os
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import plotly.express as px
import streamlit as st
from google import genai
from google.genai import types

from app.config import get_database_path, get_google_api_key, get_model_name, load_environment
from app.database import create_sqlite_database_from_csv, get_database
from app.sql_agent import create_sql_agent
from app.utils import extract_final_answer, extract_sql_queries


PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
UPLOAD_CACHE_DIR = FRONTEND_DIR / "uploads"
STYLE_PATH = FRONTEND_DIR / "styles.css"
SAMPLE_CSV_PATH = PROJECT_ROOT / "sample_sales.csv"
AGENT_CLIENT_CONFIG_VERSION = "direct-http-v1"
FALLBACK_GEMINI_MODELS = [
    {
        "name": "gemini-3.1-flash-lite",
        "description": "Lightweight Flash model for fast text generation.",
        "input_token_limit": None,
        "output_token_limit": None,
    },
    {
        "name": "gemini-2.5-flash-lite",
        "description": "Lightweight Flash model for fast text generation.",
        "input_token_limit": None,
        "output_token_limit": None,
    },
    {
        "name": "gemini-2.5-flash",
        "description": "Flash model for general text generation.",
        "input_token_limit": None,
        "output_token_limit": None,
    },
    {
        "name": "gemini-3.1-flash",
        "description": "Flash model for general text generation.",
        "input_token_limit": None,
        "output_token_limit": None,
    },
]
PREFERRED_MODEL_ORDER = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.1-flash",
]


def main() -> None:
    """Render and run the Streamlit application."""
    st.set_page_config(
        page_title="AI Data Analytics Assistant",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _remove_dead_loopback_proxies()
    load_environment()
    _ensure_session_state()
    _load_css()

    settings = _render_sidebar()
    _render_top_nav(settings)
    _render_hero(settings)
    _render_history()


def _ensure_session_state() -> None:
    defaults = {
        "uploaded_csv_path": None,
        "uploaded_csv_name": None,
        "csv_signature": None,
        "history": [],
        "pending_question": "",
        "latest_result": None,
        "database_label": "Chinook SQLite",
        "selected_history_idx": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _remove_dead_loopback_proxies() -> None:
    """Prevent local Streamlit proxy settings from hijacking Gemini HTTP calls."""
    proxy_keys = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    )
    loopback_hosts = {"127.0.0.1", "localhost", "::1"}
    dead_proxy_ports = {9}

    for key in proxy_keys:
        value = os.environ.get(key)
        if not value:
            continue

        parsed = urlparse(value)
        if parsed.hostname in loopback_hosts and parsed.port in dead_proxy_ports:
            os.environ.pop(key, None)


@st.cache_data(show_spinner=False, ttl=3600)
def _get_available_gemini_models() -> tuple[list[dict[str, Any]], str | None]:
    """Return compatible Gemini chat models, cached across Streamlit reruns."""
    try:
        client = genai.Client(
            api_key=get_google_api_key(),
            http_options=types.HttpOptions(clientArgs={"trust_env": False}),
        )
        models = [
            _model_to_option(model)
            for model in client.models.list()
            if _is_compatible_chat_model(model)
        ]
        if not models:
            return FALLBACK_GEMINI_MODELS, "No compatible Gemini chat models were returned. Using fallback models."
        return sorted(models, key=_model_sort_key), None
    except Exception as error:
        return (
            FALLBACK_GEMINI_MODELS,
            f"Could not fetch Gemini models right now. Using fallback models. Details: {error}",
        )


def _model_to_option(model: Any) -> dict[str, Any]:
    """Convert a Google GenAI model object into sidebar display metadata."""
    name = _normalize_model_name(getattr(model, "name", ""))
    return {
        "name": name,
        "description": getattr(model, "description", "") or getattr(model, "display_name", "") or "",
        "input_token_limit": getattr(model, "input_token_limit", None),
        "output_token_limit": getattr(model, "output_token_limit", None),
    }


def _is_compatible_chat_model(model: Any) -> bool:
    """Keep Gemini text/chat generation models and exclude other model families."""
    name = _normalize_model_name(getattr(model, "name", ""))
    description = str(getattr(model, "description", "") or "")
    supported_actions = {str(action).lower() for action in getattr(model, "supported_actions", []) or []}
    searchable = f"{name} {description}".lower()
    excluded_terms = (
        "aqa",
        "audio",
        "deprecated",
        "embedding",
        "imagen",
        "image",
        "live",
        "tts",
        "veo",
        "vision",
    )

    return (
        name.startswith("gemini-")
        and "generatecontent" in supported_actions
        and not any(term in searchable for term in excluded_terms)
    )


def _normalize_model_name(name: str) -> str:
    """Use the model id expected by LangChain instead of the SDK resource path."""
    return name.removeprefix("models/")


def _model_sort_key(model: dict[str, Any]) -> tuple[int, int, str]:
    """Sort lightweight Flash models first, then other compatible models."""
    name = model["name"]
    if name in PREFERRED_MODEL_ORDER:
        return (0, PREFERRED_MODEL_ORDER.index(name), name)
    if "flash-lite" in name:
        return (1, 0, name)
    if "flash" in name:
        return (2, 0, name)
    return (3, 0, name)


def _format_model_label(model: dict[str, Any], recommended_model: str) -> str:
    """Render a concise selectbox label with capability badges."""
    name = model["name"]
    badges = []
    if name == recommended_model:
        badges.append("⭐ Recommended")
    if "flash" in name:
        badges.append("⚡ Flash")
    if "lite" in name:
        badges.append("🟢 Lite")
    return f"{name}   {'  '.join(badges)}".rstrip()


def _render_model_details(model: dict[str, Any]) -> None:
    """Show available model metadata without changing backend behavior."""
    details = []
    if model.get("description"):
        details.append(str(model["description"]).strip())
    if model.get("input_token_limit"):
        details.append(f"Input limit: {int(model['input_token_limit']):,} tokens")
    if model.get("output_token_limit"):
        details.append(f"Output limit: {int(model['output_token_limit']):,} tokens")
    if details:
        st.caption(" • ".join(details))


def _load_css() -> None:
    if STYLE_PATH.exists():
        st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.markdown('<div class="side-kicker">Workspace</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

        if uploaded_file is not None:
            _handle_upload(uploaded_file)

        dataset = _load_dataset_for_preview(st.session_state.uploaded_csv_path)
        if dataset is not None:
            _render_dataset_profile(dataset)
        else:
            st.markdown(
                '<div class="empty-panel">Using the bundled SQLite database. Upload a CSV to create a fresh analytics table.</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-label">Settings</div>', unsafe_allow_html=True)
        current_model = get_model_name()
        model_options, model_warning = _get_available_gemini_models()
        if model_warning:
            st.warning(model_warning)

        model_names = [model["name"] for model in model_options]
        selected_index = model_names.index(current_model) if current_model in model_names else 0
        selected_model = st.selectbox(
            "Model",
            model_options,
            index=selected_index,
            format_func=lambda model: _format_model_label(model, model_options[0]["name"]),
            help="Compatible Gemini text-generation models available to your API key.",
        )
        model_name = selected_model["name"]
        _render_model_details(selected_model)

        if model_name != current_model:
            os.environ["LLM_MODEL_NAME"] = model_name
            _clear_agent_cache()

        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05, help="Displayed as a frontend preference. The current backend initializes Gemini at 0.")
        max_rows = st.slider("Max rows returned", 5, 500, 50, 5)
        show_sql = st.toggle("Show SQL", value=True)
        show_charts = st.toggle("Show charts", value=True)

        if st.button("Reset Session", use_container_width=True, type="secondary"):
            _reset_session()
            st.rerun()

        st.markdown('<div class="theme-pill">Dark analytics theme active</div>', unsafe_allow_html=True)

    return {
        "model_name": model_name or current_model,
        "temperature": temperature,
        "max_rows": max_rows,
        "show_sql": show_sql,
        "show_charts": show_charts,
    }


def _handle_upload(uploaded_file: Any) -> None:
    signature = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.csv_signature == signature:
        return

    UPLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", uploaded_file.name).strip("_") or "uploaded.csv"
    csv_path = UPLOAD_CACHE_DIR / f"{int(time.time())}_{safe_name}"
    csv_path.write_bytes(uploaded_file.getbuffer())

    try:
        dataset = pd.read_csv(csv_path)
        if dataset.empty:
            raise ValueError("CSV file has headers but no data rows.")
        create_sqlite_database_from_csv(csv_path)
    except Exception as error:
        _render_error(error)
        return

    st.session_state.uploaded_csv_path = str(csv_path)
    st.session_state.uploaded_csv_name = uploaded_file.name
    st.session_state.csv_signature = signature
    st.session_state.database_label = uploaded_file.name
    st.session_state.latest_result = None
    _clear_agent_cache()
    st.toast("CSV loaded and indexed for SQL querying.", icon=":material/check_circle:")


def _render_dataset_profile(dataset: pd.DataFrame) -> None:
    st.markdown('<div class="upload-success">CSV connected</div>', unsafe_allow_html=True)
    cols = len(dataset.columns)
    rows = len(dataset)
    numeric_cols = dataset.select_dtypes(include="number").columns.tolist()
    categorical_cols = dataset.select_dtypes(exclude="number").columns.tolist()

    metric_cols = st.columns(2)
    metric_cols[0].metric("Rows", f"{rows:,}")
    metric_cols[1].metric("Columns", f"{cols:,}")

    with st.expander("Column names", expanded=False):
        st.caption(", ".join(map(str, dataset.columns)))

    with st.expander("Dataset preview", expanded=True):
        st.dataframe(dataset.head(20), use_container_width=True, height=220)

    with st.expander("Summary", expanded=False):
        st.write(
            {
                "missing_values": int(dataset.isna().sum().sum()),
                "numeric_columns": len(numeric_cols),
                "categorical_columns": len(categorical_cols),
            }
        )


def _render_top_nav(settings: dict[str, Any]) -> None:
    api_ready = bool(os.getenv("GOOGLE_API_KEY"))
    status_class = "status-ok" if api_ready else "status-warn"
    status_text = "Connected" if api_ready else "API key needed"
    database = st.session_state.database_label
    st.markdown(
        f"""
        <header class="top-nav">
            <div class="brand-block">
                <div class="logo-mark">SQL</div>
                <div>
                    <div class="brand-title">AI Data Analytics Assistant</div>
                    <div class="brand-subtitle">Natural language insight over CSV and SQLite</div>
                </div>
            </div>
            <div class="nav-meta">
                <span class="status-chip {status_class}">{status_text}</span>
                <span>{settings["model_name"]}</span>
                <span>{database}</span>
            </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def _render_hero(settings: dict[str, Any]) -> None:
    csv_path = st.session_state.uploaded_csv_path
    st.markdown(
        """
        <section class="hero-shell">
            <div class="hero-eyebrow">Premium text-to-SQL workspace</div>
            <h1>Talk to your Data</h1>
            <p>Ask questions in natural language using AI.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    suggestions = [
        "Top 5 customers by revenue",
        "Average revenue",
        "Revenue by region",
        "Highest sales",
        "Lowest sales",
        "Monthly trends",
    ]
    if st.session_state.uploaded_csv_path:
        suggestion_cols = st.columns(6)
        for col, suggestion in zip(suggestion_cols, suggestions, strict=False):
            if col.button(suggestion, use_container_width=True):
                st.session_state.pending_question = suggestion

    with st.form("ask_form", clear_on_submit=False):
        question = st.text_area(
            "Question",
            value=st.session_state.pending_question,
            placeholder="What were the top 5 customers by revenue?",
            label_visibility="collapsed",
            height=112,
        )
        ask = st.form_submit_button("Ask", use_container_width=True, type="primary")

    if ask:
        st.session_state.pending_question = question.strip()
        if not question.strip():
            st.warning("Ask a question to start the analysis.")
        elif not csv_path and _default_database_is_empty():
            _render_error(
                ValueError(
                    "The default Chinook database is empty. Upload a CSV or add a "
                    "valid chinook.db file before querying the default database."
                )
            )
        else:
            _run_query(question.strip(), settings)

    if st.session_state.latest_result:
        _render_result(st.session_state.latest_result, settings)


def _run_query(question: str, settings: dict[str, Any]) -> None:
    started_at = time.perf_counter()
    csv_path = st.session_state.uploaded_csv_path

    try:
        with st.status("Analyzing your data", expanded=True) as status:
            st.write("Loading CSV" if csv_path else "Loading SQLite database")
            _get_database(csv_path)
            st.write("Initializing AI")
            agent = _get_agent(
                csv_path,
                settings["model_name"],
                AGENT_CLIENT_CONFIG_VERSION,
            )
            st.write("Generating SQL")
            st.write("Executing Query")
            result = agent.invoke({"messages": [{"role": "user", "content": question}]})
            st.write("Preparing Answer")
            status.update(label="Analysis complete", state="complete", expanded=False)

        answer = extract_final_answer(result)
        sql_queries = extract_sql_queries(result)
        table = _query_dataframe(sql_queries[-1] if sql_queries else "", csv_path, settings["max_rows"])
        elapsed = time.perf_counter() - started_at
        payload = {
            "question": question,
            "answer": answer or "No readable answer was returned.",
            "sql_queries": sql_queries,
            "table": table,
            "elapsed": elapsed,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
        st.session_state.latest_result = payload
        st.session_state.history.insert(0, payload)
    except Exception as error:
        import traceback

        st.exception(error)
        st.code(traceback.format_exc(), language="python")

        _render_error(error)

@st.cache_resource(show_spinner=False)
def _get_agent(
    csv_path: str | None,
    model_name: str,
    client_config_version: str,
) -> Any:
    os.environ["LLM_MODEL_NAME"] = model_name
    return create_sql_agent(csv_path=csv_path)


@st.cache_resource(show_spinner=False)
def _get_database(csv_path: str | None) -> Any:
    return get_database(csv_path=csv_path)


@st.cache_data(show_spinner=False)
def _get_csv_database_path(csv_path: str) -> str:
    return str(create_sqlite_database_from_csv(csv_path))


@st.cache_data(show_spinner=False)
def _load_dataset_for_preview(csv_path: str | None) -> pd.DataFrame | None:
    if not csv_path:
        return None
    return pd.read_csv(csv_path)


def _query_dataframe(query: str, csv_path: str | None, max_rows: int) -> pd.DataFrame | None:
    if not query:
        return None

    try:
        database_path = (
            Path(_get_csv_database_path(csv_path)) if csv_path else get_database_path()
        )
        if not database_path.exists() or database_path.stat().st_size == 0:
            return None
        display_query = _limit_query(query, max_rows)
        with sqlite3.connect(database_path) as connection:
            return pd.read_sql_query(display_query, connection)
    except Exception:
        return None


def _limit_query(query: str, max_rows: int) -> str:
    cleaned = query.strip().rstrip(";")
    if re.search(r"\blimit\s+\d+\b", cleaned, re.IGNORECASE):
        return cleaned
    if not cleaned.lower().startswith("select"):
        return cleaned
    return f"{cleaned} LIMIT {max_rows}"


def _render_result(result: dict[str, Any], settings: dict[str, Any]) -> None:
    st.markdown('<div class="results-grid">', unsafe_allow_html=True)
    st.markdown("#### Natural language answer")
    answer = _format_answer_for_display(result["answer"])
    st.markdown(f'<div class="answer-card">{_escape_html(answer)}</div>', unsafe_allow_html=True)

    sql_queries = result["sql_queries"]
    if settings["show_sql"]:
        with st.expander("Generated SQL", expanded=bool(sql_queries)):
            if sql_queries:
                for query in sql_queries:
                    st.code(query, language="sql")
            else:
                st.info("SQL was not exposed by the agent trace for this response.")

    table = result["table"]
    stat_cols = st.columns(3)
    stat_cols[0].metric("Rows returned", 0 if table is None else f"{len(table):,}")
    stat_cols[1].metric("Columns returned", 0 if table is None else f"{len(table.columns):,}")
    stat_cols[2].metric("Execution time", f"{result['elapsed']:.2f}s")

    if table is not None and not table.empty:
        st.markdown("#### Results table")
        st.dataframe(table, use_container_width=True, height=360)
        if settings["show_charts"]:
            chart = _build_chart(table)
            if chart is not None:
                st.markdown("#### Automatic visualization")
                st.plotly_chart(chart, use_container_width=True, theme=None)
    else:
        st.markdown('<div class="empty-panel">No tabular result was available for this answer.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _build_chart(dataframe: pd.DataFrame) -> Any | None:
    numeric = dataframe.select_dtypes(include="number").columns.tolist()
    categorical = dataframe.select_dtypes(exclude="number").columns.tolist()
    if not numeric:
        return None

    template = "plotly_dark"
    if categorical and len(numeric) == 1:
        category = categorical[0]
        value = numeric[0]
        unique_count = dataframe[category].nunique(dropna=True)
        if 2 <= unique_count <= 8:
            return px.pie(dataframe, names=category, values=value, template=template)
        return px.bar(dataframe.head(30), x=category, y=value, template=template)

    if len(numeric) >= 2:
        return px.scatter(dataframe, x=numeric[0], y=numeric[1], template=template)

    value = numeric[0]
    if pd.api.types.is_datetime64_any_dtype(dataframe.index):
        return px.line(dataframe, y=value, template=template)
    return px.histogram(dataframe, x=value, template=template)


def _render_history() -> None:
    if not st.session_state.history:
        return

    st.markdown("#### Query history")
    for index, item in enumerate(st.session_state.history[:8]):
        label = f"{item['timestamp']} - {item['question'][:72]}"
        if st.button(label, key=f"history_{index}", use_container_width=True):
            st.session_state.latest_result = item
            st.session_state.pending_question = item["question"]
            st.session_state.selected_history_idx = index
            st.rerun()


def _render_error(error: Exception) -> None:
    st.markdown(
        f"""
        <div class="error-card">
            <div class="error-title">Analysis paused</div>
            <div>{_escape_html(_friendly_error_message(error))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _friendly_error_message(error: Exception) -> str:
    error_text = str(error)
    normalized = error_text.lower()

    if "google_api_key" in normalized or "missing required environment variable" in normalized:
        return "Google API key is not configured. Add GOOGLE_API_KEY to your .env file and retry."
    if isinstance(error, ImportError):
        return "A required dependency is not installed. Activate the project environment and install the project."
    if "api_key_invalid" in normalized or "invalid api key" in normalized:
        return "Google rejected the API key. Check GOOGLE_API_KEY and make sure it is active."
    if "quota" in normalized or "resource_exhausted" in normalized or "429" in normalized:
        return "Gemini quota or rate limit was exceeded. Wait a bit or switch to another available model."
    if "model" in normalized and any(marker in normalized for marker in ("not found", "unavailable", "unsupported")):
        return "The configured Gemini model is unavailable. Check LLM_MODEL_NAME and choose an accessible model."
    if "503" in normalized or "overloaded" in normalized or "unavailable" in normalized:
        return "Gemini is temporarily overloaded or unavailable. Please retry in a moment."
    if "timeout" in normalized or "timed out" in normalized or "deadline" in normalized:
        return "The request timed out. Check your network connection and retry."
    if any(marker in normalized for marker in ("connecterror", "connection refused", "network")):
        return "The agent could not reach Gemini. Check network, proxy, and Google API access."
    if isinstance(error, (FileNotFoundError, ValueError)):
        return error_text
    if any(marker in normalized for marker in ("sqlite", "sqlalchemy", "operationalerror", "syntax error")):
        return "The generated SQL could not be executed. Try rephrasing the question or inspect the schema."
    return "Something went wrong while running the agent. See text2sql-agent.log for details."


def _reset_session() -> None:
    for key in ("uploaded_csv_path", "uploaded_csv_name", "csv_signature", "latest_result", "selected_history_idx"):
        st.session_state[key] = None
    st.session_state.history = []
    st.session_state.pending_question = ""
    st.session_state.database_label = "Chinook SQLite"
    _clear_agent_cache()


def _clear_agent_cache() -> None:
    _get_agent.clear()
    _get_database.clear()
    _get_csv_database_path.clear()
    _load_dataset_for_preview.clear()


def _default_database_is_empty() -> bool:
    database_path = get_database_path()
    return not database_path.exists() or database_path.stat().st_size == 0


def _escape_html(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _format_answer_for_display(value: Any) -> str:
    """Keep the answer card compact and leave tabular output to st.dataframe."""
    text = str(value).strip()
    text = re.sub(r"(?:[ \t]*\n){3,}", "\n\n", text)
    text = re.split(r"\n\s*\|.+\|\s*\n\s*\|[-:\s|]+\|", text, maxsplit=1)[0]
    text = re.split(r"\n\s*<table\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    return text.strip()


if __name__ == "__main__":
    main()
