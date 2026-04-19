"""Provider factory for embedding providers following llm-chat-api pattern."""

import logging
from typing import Any, Dict

from .base import BaseEmbeddingProvider, EmbeddingConfigError

logger = logging.getLogger(__name__)

# Import providers with graceful handling of missing dependencies
try:
    from .openai_provider import OpenAIEmbeddingProvider
except ImportError as e:
    logger.warning(f"OpenAI provider not available: {e}")
    OpenAIEmbeddingProvider = None

try:
    from .gemini_provider import GeminiEmbeddingProvider
except ImportError as e:
    logger.warning(f"Gemini provider not available: {e}")
    GeminiEmbeddingProvider = None

logger = logging.getLogger(__name__)


class ProviderFactory:
    """Factory for creating embedding providers based on configuration.
    
    Follows the same pattern as the LLM provider factory.
    """
    
    # Registry of available providers (lowercase names)
    _providers = {}
    
    # Dynamically register available providers
    if OpenAIEmbeddingProvider is not None:
        _providers["openai"] = OpenAIEmbeddingProvider
    if GeminiEmbeddingProvider is not None:
        _providers["gemini"] = GeminiEmbeddingProvider
    
    @classmethod
    def create_provider(
        cls, 
        provider_name: str, 
        config: Dict[str, Any]
    ) -> BaseEmbeddingProvider:
        """Create an embedding provider instance.
        
        Args:
            provider_name: Name of the provider to create (case-insensitive)
            config: Configuration dictionary for the provider
            
        Returns:
            Initialized provider instance
            
        Raises:
            EmbeddingConfigError: If provider is not supported or config is invalid
        """
        # Normalize provider name to lowercase
        provider_name = provider_name.lower().strip()
        
        if provider_name not in cls._providers:
            available_providers = ', '.join(cls._providers.keys())
            raise EmbeddingConfigError(
                f"Unsupported embedding provider: '{provider_name}'. "
                f"Available providers: {available_providers}",
                provider=provider_name
            )
        
        provider_class = cls._providers[provider_name]
        
        try:
            logger.info(f"Creating {provider_name} embedding provider")
            provider = provider_class(config)
            logger.info(f"Successfully created {provider_name} embedding provider")
            return provider
        except Exception as e:
            logger.error(f"Failed to create {provider_name} provider: {str(e)}")
            raise EmbeddingConfigError(
                f"Failed to initialize {provider_name} provider: {str(e)}",
                provider=provider_name
            ) from e
    
    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of available provider names.
        
        Returns:
            List of supported provider identifiers
        """
        return list(cls._providers.keys())
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """Register a new provider class.
        
        Args:
            name: Provider name (will be converted to lowercase)
            provider_class: Provider class that inherits from BaseEmbeddingProvider
        """
        name = name.lower().strip()
        if not issubclass(provider_class, BaseEmbeddingProvider):
            raise ValueError(
                f"Provider class must inherit from BaseEmbeddingProvider, "
                f"got {provider_class.__name__}"
            )
        
        cls._providers[name] = provider_class
        logger.info(f"Registered new embedding provider: {name}")


class ProviderManager:
    """Manages multiple embedding providers (future extension point).
    
    Similar to the LLM ProviderManager for multi-provider scenarios.
    """
    
    def __init__(self):
        self._providers: Dict[str, BaseEmbeddingProvider] = {}
        self._active_provider: str = None
    
    def add_provider(self, name: str, provider: BaseEmbeddingProvider):
        """Add a provider to the manager.
        
        Args:
            name: Provider identifier
            provider: Initialized provider instance
        """
        self._providers[name] = provider
        if self._active_provider is None:
            self._active_provider = name
    
    def set_active_provider(self, name: str):
        """Set the active provider.
        
        Args:
            name: Provider identifier
            
        Raises:
            EmbeddingConfigError: If provider is not registered
        """
        if name not in self._providers:
            available = ', '.join(self._providers.keys())
            raise EmbeddingConfigError(
                f"Provider '{name}' not found. Available: {available}",
                provider=name
            )
        self._active_provider = name
    
    def get_active_provider(self) -> BaseEmbeddingProvider:
        """Get the currently active provider.
        
        Returns:
            Active provider instance
            
        Raises:
            EmbeddingConfigError: If no provider is active
        """
        if not self._active_provider:
            raise EmbeddingConfigError("No active provider set", provider="none")
        
        return self._providers[self._active_provider]
    
    def get_provider(self, name: str) -> BaseEmbeddingProvider:
        """Get a specific provider by name.
        
        Args:
            name: Provider identifier
            
        Returns:
            Provider instance
            
        Raises:
            EmbeddingConfigError: If provider is not found
        """
        if name not in self._providers:
            available = ', '.join(self._providers.keys())
            raise EmbeddingConfigError(
                f"Provider '{name}' not found. Available: {available}",
                provider=name
            )
        
        return self._providers[name]
    
    def list_providers(self) -> list[str]:
        """List all registered providers.
        
        Returns:
            List of provider names
        """
        return list(self._providers.keys())