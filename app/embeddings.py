"""Embedding utilities following the same pattern as llm-chat-api utils."""

import logging
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import settings
from .providers import BaseEmbeddingProvider, ProviderFactory
from .providers.base import EmbeddingError

logger = logging.getLogger(__name__)

# Global provider instance (singleton pattern)
_provider: Optional[BaseEmbeddingProvider] = None


def get_provider_config() -> Dict[str, Any]:
    """Get configuration for the selected embedding provider.
    
    Maps the EMBEDDING_PROVIDER setting to the appropriate credentials and settings.
    
    Returns:
        Configuration dictionary for the provider
        
    Raises:
        ValueError: If provider configuration is incomplete
    """
    provider_name = settings.embedding_provider.lower()
    
    if provider_name == "openai":
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        
        return {
            "api_key": settings.openai_api_key,
            "default_model": settings.openai_default_model,
            "timeout": settings.openai_timeout
        }
    
    elif provider_name == "gemini":
        if not settings.gemini_api_key:
            raise ValueError("Gemini API key not configured")
        
        return {
            "api_key": settings.gemini_api_key,
            "default_model": settings.gemini_default_model,
            "timeout": settings.gemini_timeout
        }
    
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")


def get_embedding_provider() -> BaseEmbeddingProvider:
    """Get the configured embedding provider instance (singleton pattern).
    
    Creates the provider on first call and reuses it for subsequent calls.
    
    Returns:
        Initialized embedding provider instance
        
    Raises:
        EmbeddingError: If provider cannot be initialized
    """
    global _provider
    
    if _provider is None:
        try:
            provider_config = get_provider_config()
            _provider = ProviderFactory.create_provider(
                settings.embedding_provider, 
                provider_config
            )
            logger.info(f"Initialized {settings.embedding_provider} embedding provider")
        except Exception as e:
            logger.error(f"Failed to initialize embedding provider: {str(e)}")
            raise EmbeddingError(
                f"Provider initialization failed: {str(e)}",
                provider=settings.embedding_provider
            ) from e
    
    return _provider


def reset_provider():
    """Reset the global provider instance.
    
    Useful for testing or configuration changes.
    """
    global _provider
    _provider = None
    logger.info("Reset embedding provider instance")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
async def generate_embeddings(
    texts: List[str],
    model: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Generate embeddings with retry logic.
    
    High-level wrapper around provider embed() method with retry and error handling.
    
    Args:
        texts: List of texts to embed
        model: Optional model override
        **kwargs: Provider-specific parameters
        
    Returns:
        Standardized embedding response
        
    Raises:
        EmbeddingError: If embedding generation fails after retries
    """
    if not texts:
        raise ValueError("No texts provided for embedding")
    
    provider = get_embedding_provider()
    
    try:
        logger.info(f"Generating embeddings for {len(texts)} texts")
        result = await provider.embed(texts, model=model, **kwargs)
        logger.info(f"Successfully generated {len(result['embeddings'])} embeddings")
        return result
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        if isinstance(e, EmbeddingError):
            raise
        else:
            raise EmbeddingError(
                f"Embedding generation failed: {str(e)}",
                provider=settings.embedding_provider
            ) from e


async def generate_single_embedding(
    text: str,
    model: Optional[str] = None,
    **kwargs
) -> List[float]:
    """Generate embedding for a single text.
    
    Convenience function for single text embedding.
    
    Args:
        text: Text to embed
        model: Optional model override
        **kwargs: Provider-specific parameters
        
    Returns:
        Embedding vector as list of floats
        
    Raises:
        EmbeddingError: If embedding generation fails
    """
    if not text.strip():
        raise ValueError("Text cannot be empty")
    
    result = await generate_embeddings([text], model=model, **kwargs)
    return result["embeddings"][0]


def get_embedding_dimension(model: Optional[str] = None) -> int:
    """Get the embedding dimension for the current provider and model.
    
    Args:
        model: Optional model name, uses default if None
        
    Returns:
        Embedding vector dimension
    """
    provider = get_embedding_provider()
    return provider.get_embedding_dimension(model)


def get_available_models() -> List[str]:
    """Get list of available embedding models for the current provider.
    
    Returns:
        List of model identifiers
    """
    provider = get_embedding_provider()
    return provider.get_available_models()


def get_max_input_tokens() -> int:
    """Get maximum input tokens for the current provider.
    
    Returns:
        Maximum number of input tokens
    """
    provider = get_embedding_provider()
    return provider.get_max_input_tokens()


def estimate_token_count(text: str) -> int:
    """Estimate token count for a text.
    
    Uses a simple approximation based on word count.
    
    Args:
        text: Text to analyze
        
    Returns:
        Estimated token count
    """
    # Simple approximation: ~0.75 tokens per word for English text
    word_count = len(text.split())
    return int(word_count * 0.75)


def chunk_texts_by_tokens(
    texts: List[str],
    max_tokens: Optional[int] = None
) -> List[List[str]]:
    """Chunk texts into batches that fit within token limits.
    
    Args:
        texts: List of texts to chunk
        max_tokens: Maximum tokens per batch (uses provider default if None)
        
    Returns:
        List of text batches
    """
    if max_tokens is None:
        max_tokens = get_max_input_tokens()
    
    batches = []
    current_batch = []
    current_tokens = 0
    
    for text in texts:
        text_tokens = estimate_token_count(text)
        
        # If single text exceeds limit, process it alone (will likely fail, but let provider handle it)
        if text_tokens > max_tokens:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            batches.append([text])
            continue
        
        # If adding this text would exceed limit, start new batch
        if current_tokens + text_tokens > max_tokens:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
        
        current_batch.append(text)
        current_tokens += text_tokens
    
    # Add final batch if not empty
    if current_batch:
        batches.append(current_batch)
    
    return batches


# Mock embedding functionality for development/testing
def create_mock_embedding(text: str, dimension: Optional[int] = None) -> List[float]:
    """Create a mock embedding vector for testing.
    
    Args:
        text: Text to create mock embedding for
        dimension: Embedding dimension (uses config default if None)
        
    Returns:
        Mock embedding vector
    """
    import hashlib
    import random
    
    if dimension is None:
        dimension = settings.mock_embedding_dimension
    
    # Use text hash as seed for reproducible results
    seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    random.seed(seed)
    
    # Generate normalized random vector
    vector = [random.uniform(-1, 1) for _ in range(dimension)]
    
    # Normalize to unit length
    magnitude = sum(x*x for x in vector) ** 0.5
    if magnitude > 0:
        vector = [x / magnitude for x in vector]
    
    return vector


async def generate_mock_embeddings(
    texts: List[str],
    model: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Generate mock embeddings for testing.
    
    Args:
        texts: List of texts to embed
        model: Model name (ignored for mocks)
        **kwargs: Additional parameters (ignored for mocks)
        
    Returns:
        Mock embedding response in standardized format
    """
    embeddings = [create_mock_embedding(text) for text in texts]
    
    return {
        "embeddings": embeddings,
        "model": model or "mock-embedding-model",
        "usage": {
            "prompt_tokens": sum(estimate_token_count(text) for text in texts),
            "total_tokens": sum(estimate_token_count(text) for text in texts)
        },
        "provider": "mock"
    }