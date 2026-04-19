"""OpenAI embedding provider implementation."""

import logging
from typing import Any, Dict, List, Optional

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseEmbeddingProvider, EmbeddingAPIError, EmbeddingConfigError, EmbeddingRateLimitError

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider using the OpenAI API."""
    
    # Available OpenAI embedding models
    AVAILABLE_MODELS = [
        "text-embedding-3-small",
        "text-embedding-3-large", 
        "text-embedding-ada-002"
    ]
    
    # Model dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI embedding provider.
        
        Args:
            config: Configuration containing 'api_key' and optional settings
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI provider requires 'openai' package. "
                "Install it with: pip install openai"
            )
        
        super().__init__(config)
        
        # Initialize OpenAI client
        self.client = openai.AsyncOpenAI(
            api_key=config["api_key"],
            timeout=config.get("timeout", 30.0)
        )
        
        # Set default model
        self.default_model = config.get("default_model", "text-embedding-3-small")
        
        logger.info(f"Initialized OpenAI embedding provider with model: {self.default_model}")
    
    def validate_config(self) -> bool:
        """Validate OpenAI provider configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            EmbeddingConfigError: If configuration is invalid
        """
        if not self.config.get("api_key"):
            raise EmbeddingConfigError(
                "OpenAI API key is required",
                provider="openai"
            )
        
        # Validate default model if specified
        default_model = self.config.get("default_model")
        if default_model and default_model not in self.AVAILABLE_MODELS:
            raise EmbeddingConfigError(
                f"Invalid OpenAI model: {default_model}. "
                f"Available models: {', '.join(self.AVAILABLE_MODELS)}",
                provider="openai"
            )
        
        return True
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate embeddings using OpenAI API.
        
        Args:
            texts: List of texts to embed
            model: Model to use (defaults to configured default)
            **kwargs: Additional OpenAI-specific parameters
            
        Returns:
            Standardized embedding response
            
        Raises:
            EmbeddingAPIError: If API request fails
            EmbeddingRateLimitError: If rate limited
        """
        if not texts:
            raise EmbeddingAPIError("No texts provided for embedding", provider="openai")
        
        # Use provided model or default
        embedding_model = model or self.default_model
        
        if embedding_model not in self.AVAILABLE_MODELS:
            raise EmbeddingAPIError(
                f"Unsupported OpenAI model: {embedding_model}. "
                f"Available: {', '.join(self.AVAILABLE_MODELS)}",
                provider="openai"
            )
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts using {embedding_model}")
            
            # Make API request
            response = await self.client.embeddings.create(
                input=texts,
                model=embedding_model,
                **kwargs
            )
            
            # Extract embeddings
            embeddings = [item.embedding for item in response.data]
            
            # Calculate usage
            usage_info = {
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            logger.info(f"Generated {len(embeddings)} embeddings, tokens used: {usage_info['total_tokens']}")
            
            # Return standardized response
            return {
                "embeddings": embeddings,
                "model": embedding_model,
                "usage": usage_info,
                "provider": "openai"
            }
            
        except openai.RateLimitError as e:
            logger.warning(f"OpenAI rate limit exceeded: {str(e)}")
            raise EmbeddingRateLimitError(
                f"OpenAI rate limit exceeded: {str(e)}",
                provider="openai"
            ) from e
            
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI authentication failed: {str(e)}")
            raise EmbeddingConfigError(
                f"OpenAI authentication failed: {str(e)}",
                provider="openai"
            ) from e
            
        except openai.BadRequestError as e:
            logger.error(f"OpenAI bad request: {str(e)}")
            raise EmbeddingAPIError(
                f"OpenAI request failed: {str(e)}",
                provider="openai"
            ) from e
            
        except Exception as e:
            logger.error(f"Unexpected OpenAI API error: {str(e)}")
            raise EmbeddingAPIError(
                f"OpenAI API error: {str(e)}",
                provider="openai"
            ) from e
    
    def get_available_models(self) -> List[str]:
        """Get list of available OpenAI embedding models.
        
        Returns:
            List of model identifiers
        """
        return self.AVAILABLE_MODELS.copy()
    
    def get_max_input_tokens(self) -> int:
        """Get maximum input tokens for OpenAI embeddings.
        
        Returns:
            Maximum input tokens (8191 for most embedding models)
        """
        return 8191
    
    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """Get embedding dimension for the specified model.
        
        Args:
            model: Model name, uses default if None
            
        Returns:
            Embedding vector dimension
        """
        embedding_model = model or self.default_model
        return self.MODEL_DIMENSIONS.get(embedding_model, 1536)