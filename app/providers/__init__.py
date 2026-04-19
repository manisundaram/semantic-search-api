# Embedding providers module
from .base import BaseEmbeddingProvider
from .factory import ProviderFactory

# Import available providers
_available_providers = []
try:
    from .openai_provider import OpenAIEmbeddingProvider
    _available_providers.append("OpenAIEmbeddingProvider")
except ImportError:
    OpenAIEmbeddingProvider = None

try:
    from .gemini_provider import GeminiEmbeddingProvider
    _available_providers.append("GeminiEmbeddingProvider")
except ImportError:
    GeminiEmbeddingProvider = None

__all__ = [
    "BaseEmbeddingProvider",
    "ProviderFactory"
] + _available_providers