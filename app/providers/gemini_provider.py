"""Gemini embedding provider implementation."""

import logging
from typing import Any, Dict, List, Optional

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseEmbeddingProvider, EmbeddingAPIError, EmbeddingConfigError, EmbeddingRateLimitError

logger = logging.getLogger(__name__)


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Gemini embedding provider using Google's Generative AI API."""
    
    # Available Gemini embedding models
    AVAILABLE_MODELS = [
        "models/gemini-embedding-001",
        "models/gemini-embedding-2-preview"
    ]
    
    # Model dimensions (Gemini embeddings are typically 768-dimensional)
    MODEL_DIMENSIONS = {
        "models/gemini-embedding-001": 768,
        "models/gemini-embedding-2-preview": 768
    }
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Gemini embedding provider.
        
        Args:
            config: Configuration containing 'api_key' and optional settings
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Gemini provider requires 'google-generativeai' package. "
                "Install it with: pip install google-generativeai"
            )
        
        super().__init__(config)
        
        # Configure Gemini client
        genai.configure(api_key=config["api_key"])
        
        # Set default model
        self.default_model = config.get("default_model", "models/gemini-embedding-001")
        
        logger.info(f"Initialized Gemini embedding provider with model: {self.default_model}")
    
    def validate_config(self) -> bool:
        """Validate Gemini provider configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            EmbeddingConfigError: If configuration is invalid
        """
        if not self.config.get("api_key"):
            raise EmbeddingConfigError(
                "Gemini API key is required",
                provider="gemini"
            )
        
        # Validate default model if specified
        default_model = self.config.get("default_model")
        if default_model and default_model not in self.AVAILABLE_MODELS:
            raise EmbeddingConfigError(
                f"Invalid Gemini model: {default_model}. "
                f"Available models: {', '.join(self.AVAILABLE_MODELS)}",
                provider="gemini"
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
        """Generate embeddings using Gemini API.
        
        Args:
            texts: List of texts to embed
            model: Model to use (defaults to configured default)
            **kwargs: Additional Gemini-specific parameters
            
        Returns:
            Standardized embedding response
            
        Raises:
            EmbeddingAPIError: If API request fails
            EmbeddingRateLimitError: If rate limited
        """
        if not texts:
            raise EmbeddingAPIError("No texts provided for embedding", provider="gemini")
        
        # Use provided model or default
        embedding_model = model or self.default_model
        
        if embedding_model not in self.AVAILABLE_MODELS:
            raise EmbeddingAPIError(
                f"Unsupported Gemini model: {embedding_model}. "
                f"Available: {', '.join(self.AVAILABLE_MODELS)}",
                provider="gemini"
            )
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts using {embedding_model}")
            
            embeddings = []
            total_tokens = 0
            
            # Process each text (Gemini API processes one at a time)
            for text in texts:
                try:
                    # Generate embedding for single text
                    result = genai.embed_content(
                        model=embedding_model,
                        content=text,
                        **kwargs
                    )
                    
                    embeddings.append(result["embedding"])
                    
                    # Estimate token usage (rough approximation)
                    total_tokens += len(text.split()) * 1.3  # Approximate token count
                    
                except Exception as e:
                    logger.error(f"Failed to embed text chunk: {str(e)}")
                    raise EmbeddingAPIError(
                        f"Failed to embed text: {str(e)}",
                        provider="gemini"
                    ) from e
            
            # Calculate usage info
            usage_info = {
                "prompt_tokens": int(total_tokens),
                "total_tokens": int(total_tokens)
            }
            
            logger.info(f"Generated {len(embeddings)} embeddings, estimated tokens: {usage_info['total_tokens']}")
            
            # Return standardized response
            return {
                "embeddings": embeddings,
                "model": embedding_model,
                "usage": usage_info,
                "provider": "gemini"
            }
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Handle rate limiting
            if "quota" in error_msg or "rate" in error_msg:
                logger.warning(f"Gemini rate limit exceeded: {str(e)}")
                raise EmbeddingRateLimitError(
                    f"Gemini rate limit exceeded: {str(e)}",
                    provider="gemini"
                ) from e
            
            # Handle authentication errors
            elif "authentication" in error_msg or "api key" in error_msg:
                logger.error(f"Gemini authentication failed: {str(e)}")
                raise EmbeddingConfigError(
                    f"Gemini authentication failed: {str(e)}",
                    provider="gemini"
                ) from e
            
            # Handle other API errors
            else:
                logger.error(f"Unexpected Gemini API error: {str(e)}")
                raise EmbeddingAPIError(
                    f"Gemini API error: {str(e)}",
                    provider="gemini"
                ) from e
    
    def get_available_models(self) -> List[str]:
        """Get list of available Gemini embedding models.
        
        Returns:
            List of model identifiers
        """
        return self.AVAILABLE_MODELS.copy()
    
    def get_max_input_tokens(self) -> int:
        """Get maximum input tokens for Gemini embeddings.
        
        Returns:
            Maximum input tokens
        """
        return 2048  # Gemini embedding models typically support 2048 tokens
    
    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """Get embedding dimension for the specified model.
        
        Args:
            model: Model name, uses default if None
            
        Returns:
            Embedding vector dimension
        """
        embedding_model = model or self.default_model
        return self.MODEL_DIMENSIONS.get(embedding_model, 768)