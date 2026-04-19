"""Configuration settings for semantic search API using Pydantic."""

import logging
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings following the same pattern as llm-chat-api.
    
    Configuration is loaded from .env file only, not from system environment variables.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        # Only load from .env file, not system environment
        env_file_encoding="utf-8"
    )
    
    @classmethod  
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Customize settings sources to only use .env file."""
        # Only use .env file and init settings, ignore system environment
        return (init_settings, dotenv_settings, file_secret_settings)
    
    # === Provider Selection ===
    embedding_provider: str = "openai"
    
    # === OpenAI Configuration ===
    openai_api_key: Optional[str] = None
    openai_default_model: str = "text-embedding-3-small"
    openai_timeout: float = 30.0
    
    # === Gemini Configuration ===  
    gemini_api_key: Optional[str] = None
    gemini_default_model: str = "models/embedding-001"
    gemini_timeout: float = 30.0
    
    # === Vector Database Configuration ===
    chroma_persist_directory: str = "./chroma_db"
    chroma_collection_name: str = "documents"
    
    # === Search Configuration ===
    default_search_results: int = 5
    max_search_results: int = 50
    similarity_threshold: float = 0.0
    
    # === Indexing Configuration ===
    max_chunk_size: int = 1000
    chunk_overlap: int = 200
    batch_size: int = 10
    
    # === Application Settings ===
    app_name: str = "Semantic Search API"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # === API Configuration ===
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["*"]
    max_retries: int = 3
    request_timeout: float = 300.0
    
    # === Development Settings ===
    use_mock_embeddings: bool = False
    mock_embedding_dimension: int = 1536
    
    def get_default_model(self) -> str:
        """Get the default embedding model for the selected provider.
        
        Returns:
            Default model identifier for the active provider
        """
        provider = self.embedding_provider.lower()
        
        if provider == "openai":
            return self.openai_default_model
        elif provider == "gemini":
            return self.gemini_default_model
        else:
            return "text-embedding-3-small"  # Fallback
    
    def get_provider_timeout(self) -> float:
        """Get timeout setting for the selected provider.
        
        Returns:
            Timeout in seconds
        """
        provider = self.embedding_provider.lower()
        
        if provider == "openai":
            return self.openai_timeout
        elif provider == "gemini":
            return self.gemini_timeout
        else:
            return 30.0  # Fallback
    
    def mask_sensitive_config(self) -> dict:
        """Get configuration with sensitive values masked.
        
        Returns:
            Configuration dictionary with API keys masked
        """
        config = self.model_dump()
        
        # Mask API keys
        for key in config:
            if "api_key" in key and config[key]:
                config[key] = f"{config[key][:8]}..." if len(config[key]) > 8 else "***"
        
        return config


# Global settings instance
settings = Settings()

# Configure logging with UTC timestamps (matching llm-chat-api pattern)
import time
logging.Formatter.converter = time.gmtime

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s UTC - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
logger.info(f"Loaded configuration for {settings.app_name} v{settings.app_version}")
logger.info(f"Active embedding provider: {settings.embedding_provider}")
logger.info(f"Debug mode: {settings.debug}")