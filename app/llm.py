"""LLM initialization for the SQL agent."""

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_google_api_key, get_model_name

logger = logging.getLogger(__name__)


def get_llm() -> BaseChatModel:
    """Create and return the configured LangChain chat model."""
    model_name = get_model_name()
    logger.info("Initializing Gemini LLM: %s", model_name)

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=get_google_api_key(),
        temperature=0,
        client_args={"trust_env": False},
    )
