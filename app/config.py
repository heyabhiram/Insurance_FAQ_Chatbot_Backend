import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator

# Base directory pointing to backend root
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """Application configuration settings."""

    # Project Info
    PROJECT_NAME: str = "Vehicle Insurance FAQ Chatbot"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    # TCS GenAI Lab LLM Settings
    TCS_API_KEY: str = "a3703LKtv9gvCJzoQIyQfQ"
    TCS_BASE_URL: str = "https://genailab.tcs.in"
    TCS_MODEL: str = "genailab-maas-gpt-4o"
    TCS_VERIFY_SSL: bool = False

    # Embedding Model (Exact required model)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Data and Vector Store Paths
    FAQ_DATA_PATH: Path = BASE_DIR / "data" / "vehicle_insurance_faqs.json"
    CHROMA_PERSIST_DIRECTORY: Path = BASE_DIR / "chroma_db"
    CHROMA_COLLECTION_NAME: str = "vehicle_insurance_faqs"

    # Retrieval Threshold (Cosine similarity threshold)
    SIMILARITY_THRESHOLD: float = 0.35
    TOP_K_RESULTS: int = 3

    # SQLite Database for Chat History
    SQLITE_DB_PATH: Path = BASE_DIR / "chat_history.db"

    # Server Settings
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    @field_validator("FAQ_DATA_PATH", "CHROMA_PERSIST_DIRECTORY", "SQLITE_DB_PATH", mode="before")
    @classmethod
    def resolve_paths(cls, value):
        if value is None:
            return value
        p = Path(value)
        if not p.is_absolute():
            return BASE_DIR / p
        return p

    model_config = SettingsConfigDict(
        env_file=[str(BASE_DIR / ".env"), ".env"],
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
