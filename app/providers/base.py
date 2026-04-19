"""Base embedding provider interface following the same pattern as LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseEmbeddingProvider(ABC):
    """Abstract base class for all embedding providers.
    
    Following the same architectural pattern as BaseLLMProvider from llm-chat-api.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the provider with configuration.
        
        Args:
            config: Configuration dictionary containing provider-specific settings
        """
        self.config = config
        self.validate_config()
    
    @abstractmethod
    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate embeddings for given texts.
        
        Args:
            texts: List of texts to embed
            model: Optional model override
            **kwargs: Provider-specific parameters
            
        Returns:
            Standardized response dictionary with structure:
            {
                "embeddings": List[List[float]],  # List of embedding vectors
                "model": str,                     # Model used for embeddings
                "usage": {                        # Token/request usage info
                    "prompt_tokens": int,
                    "total_tokens": int
                },
                "provider": str                   # Provider name
            }
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available embedding models.
        
        Returns:
            List of model identifiers available for this provider
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate provider configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    def get_provider_name(self) -> str:
        """Get the provider name.
        
        Returns:
            Provider identifier (e.g., 'openai', 'gemini')
        """
        return self.__class__.__name__.lower().replace('embeddingprovider', '')
    
    def get_max_input_tokens(self) -> int:
        """Get maximum input tokens for this provider.
        
        Returns:
            Maximum number of input tokens supported
        """
        # Default implementation, providers can override
        return 8192
    
    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """Get the embedding dimension for the specified model.
        
        Args:
            model: Model name, if None uses default model
            
        Returns:
            Embedding vector dimension
        """
        # Default implementation, providers should override
        return 1536


class EmbeddingError(Exception):
    """Base exception for embedding provider errors."""
    
    def __init__(self, message: str, provider: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.provider = provider
        self.error_code = error_code


class EmbeddingConfigError(EmbeddingError):
    """Exception for configuration-related errors."""
    pass


class EmbeddingAPIError(EmbeddingError):
    """Exception for API-related errors."""
    pass


class EmbeddingRateLimitError(EmbeddingError):
    """Exception for rate limiting errors."""
    pass