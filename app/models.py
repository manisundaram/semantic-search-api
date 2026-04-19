"""Pydantic models for request/response validation."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


# === Request Models ===

class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""
    
    texts: List[str] = Field(..., min_items=1, max_items=100, description="Texts to embed")
    model: Optional[str] = Field(None, description="Override default embedding model")
    
    @validator('texts')
    def validate_texts(cls, v):
        """Validate text inputs."""
        if not v:
            raise ValueError("At least one text is required")
        
        for text in v:
            if not text.strip():
                raise ValueError("Empty or whitespace-only texts are not allowed")
            if len(text) > 8000:  # Reasonable limit
                raise ValueError(f"Text too long: {len(text)} chars (max 8000)")
        
        return v


class IndexRequest(BaseModel):
    """Request model for indexing documents."""
    
    documents: List[Dict[str, Any]] = Field(
        ..., 
        min_items=1,
        description="Documents to index with content and metadata"
    )
    collection_name: Optional[str] = Field(
        None, 
        description="Collection name (uses default if not specified)"
    )
    chunk_size: Optional[int] = Field(
        None, 
        ge=100, 
        le=2000, 
        description="Text chunk size for splitting"
    )
    chunk_overlap: Optional[int] = Field(
        None, 
        ge=0, 
        le=500, 
        description="Overlap between chunks"
    )
    
    @validator('documents')
    def validate_documents(cls, v):
        """Validate document structure."""
        for i, doc in enumerate(v):
            if not isinstance(doc, dict):
                raise ValueError(f"Document {i} must be a dictionary")
            if 'content' not in doc:
                raise ValueError(f"Document {i} missing required 'content' field")
            if not isinstance(doc['content'], str):
                raise ValueError(f"Document {i} 'content' must be a string")
            if not doc['content'].strip():
                raise ValueError(f"Document {i} has empty content")
        
        return v


class SearchRequest(BaseModel):
    """Request model for semantic search."""
    
    query: str = Field(..., min_length=1, description="Search query")
    k: Optional[int] = Field(5, ge=1, le=50, description="Number of results to return")
    collection_name: Optional[str] = Field(
        None, 
        description="Collection to search (uses default if not specified)"
    )
    filter_metadata: Optional[Dict[str, Any]] = Field(
        None, 
        description="Metadata filters for search results"
    )
    similarity_threshold: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Minimum similarity score threshold"
    )
    
    @validator('query')
    def validate_query(cls, v):
        """Validate search query."""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace")
        if len(v) > 2000:
            raise ValueError(f"Query too long: {len(v)} chars (max 2000)")
        return v.strip()


# === Response Models ===

class EmbeddingUsage(BaseModel):
    """Usage information for embedding requests."""
    
    prompt_tokens: int = Field(..., description="Number of prompt tokens used")
    total_tokens: int = Field(..., description="Total tokens used")


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    
    embeddings: List[List[float]] = Field(..., description="Generated embedding vectors")
    model: str = Field(..., description="Model used for embedding")
    usage: EmbeddingUsage = Field(..., description="Token usage information")
    provider: str = Field(..., description="Embedding provider used")
    dimension: int = Field(..., description="Embedding vector dimension")


class IndexedDocument(BaseModel):
    """Model for indexed document information."""
    
    id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    chunk_index: Optional[int] = Field(None, description="Chunk index if document was split")


class IndexResponse(BaseModel):
    """Response model for document indexing."""
    
    message: str = Field(..., description="Operation status message")
    indexed_count: int = Field(..., description="Number of documents indexed")
    chunk_count: int = Field(..., description="Number of chunks created")
    collection_name: str = Field(..., description="Collection used for indexing")
    documents: List[IndexedDocument] = Field(..., description="Indexed document details")


class SearchResult(BaseModel):
    """Model for a single search result."""
    
    id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    chunk_index: Optional[int] = Field(None, description="Chunk index if document was split")


class SearchResponse(BaseModel):
    """Response model for semantic search."""
    
    query: str = Field(..., description="Original search query")
    results: List[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results found")
    collection_name: str = Field(..., description="Collection searched")
    embedding_model: str = Field(..., description="Model used for query embedding")
    search_time_ms: float = Field(..., description="Search execution time in milliseconds")


# === Health Check Models ===

class StartupHealthCheck(BaseModel):
    """Startup health check results."""
    
    chromadb_available: bool = Field(..., description="ChromaDB can be imported")
    openai_imported: bool = Field(..., description="OpenAI client can be imported")
    config_loaded: bool = Field(..., description="Configuration loaded successfully")
    env_file_found: bool = Field(..., description="Environment file exists")
    required_vars_present: bool = Field(..., description="Required environment variables present")
    chroma_dir_writable: bool = Field(..., description="ChromaDB directory is writable")


class RuntimeHealthCheck(BaseModel):
    """Runtime health check results."""
    
    chromadb_connected: bool = Field(..., description="ChromaDB client connected")
    collections_count: int = Field(..., description="Number of collections")
    provider_initialized: bool = Field(..., description="Embedding provider initialized")
    last_operation: Optional[str] = Field(None, description="Last successful operation timestamp")
    vector_store_ready: bool = Field(..., description="Vector store is operational")


class MockHealthCheck(BaseModel):
    """Mock functionality health check results."""
    
    embeddings_working: bool = Field(..., description="Mock embeddings can be generated")
    search_working: bool = Field(..., description="Mock search functionality works")
    dimensions_correct: bool = Field(..., description="Mock embeddings have correct dimensions")
    performance_acceptable: bool = Field(..., description="Mock operations are fast enough")


class ConfigurationInfo(BaseModel):
    """Configuration information (no secrets)."""
    
    provider: str = Field(..., description="Active embedding provider")
    mock_mode: bool = Field(..., description="Whether mock mode is enabled")
    debug_mode: bool = Field(..., description="Whether debug mode is enabled")
    cors_enabled: bool = Field(..., description="Whether CORS is enabled")
    openai_key: Optional[str] = Field(None, description="OpenAI API key status (masked)")
    gemini_key: Optional[str] = Field(None, description="Gemini API key status (masked)")
    default_models: Dict[str, str] = Field(default_factory=dict, description="Default models per provider")
    chromadb_settings: Dict[str, Any] = Field(default_factory=dict, description="ChromaDB configuration")


class HealthResponse(BaseModel):
    """Comprehensive health check response."""
    
    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Response timestamp (UTC)")
    response_time_ms: int = Field(..., description="Health check response time")
    
    startup: StartupHealthCheck = Field(..., description="Startup health checks")
    runtime: RuntimeHealthCheck = Field(..., description="Runtime health checks")
    mock: MockHealthCheck = Field(..., description="Mock functionality checks")
    configuration: ConfigurationInfo = Field(..., description="Configuration information")


class APITestResult(BaseModel):
    """Individual API test result."""
    
    status: str = Field(..., description="Test status (ok/error/warning)")
    duration_ms: int = Field(..., description="Test duration in milliseconds")
    error: Optional[str] = Field(None, description="Error message if failed")
    models_available: List[str] = Field(default_factory=list, description="Available models")
    test_embedding_dimensions: Optional[int] = Field(None, description="Dimensions of test embedding")


class FunctionalTestResult(BaseModel):
    """Functional test result."""
    
    status: str = Field(..., description="Test status")
    duration_ms: int = Field(..., description="Test duration")
    error: Optional[str] = Field(None, description="Error message if failed")
    test_steps: Dict[str, str] = Field(default_factory=dict, description="Individual test steps")
    similarity_score: Optional[float] = Field(None, description="Test similarity score")


class DiagnosticsResponse(BaseModel):
    """Comprehensive diagnostics response."""
    
    status: str = Field(..., description="Diagnostics status")
    timestamp: str = Field(..., description="Response timestamp (UTC)")
    total_duration_ms: int = Field(..., description="Total diagnostics duration")
    
    api_tests: Dict[str, APITestResult] = Field(default_factory=dict, description="API test results")
    functional_tests: Dict[str, FunctionalTestResult] = Field(default_factory=dict, description="Functional test results")
    performance_benchmarks: Dict[str, str] = Field(default_factory=dict, description="Performance benchmark results")
    
    summary: Dict[str, Any] = Field(default_factory=dict, description="Test summary")


class MetricsResponse(BaseModel):
    """Metrics response."""
    
    timestamp: str = Field(..., description="Response timestamp")
    collection_period: str = Field(..., description="Metrics collection period")
    
    performance: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    usage: Dict[str, Any] = Field(default_factory=dict, description="Usage metrics")
    reliability: Dict[str, Any] = Field(default_factory=dict, description="Reliability metrics")
    errors: Dict[str, Any] = Field(default_factory=dict, description="Error metrics")


# === Legacy Health Model for Compatibility ===

class SimpleHealthResponse(BaseModel):
    """Simple health check response (legacy compatibility)."""
    
    status: str = Field("healthy", description="Service health status")
    version: str = Field(..., description="API version")
    provider: str = Field(..., description="Active embedding provider")
    timestamp: str = Field(..., description="Response timestamp (UTC)")


class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    provider: Optional[str] = Field(None, description="Provider that caused the error")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class ConfigResponse(BaseModel):
    """Configuration debug response."""
    
    provider: str = Field(..., description="Active embedding provider")
    model: str = Field(..., description="Default embedding model")
    available_models: List[str] = Field(..., description="Available models for provider")
    config: Dict[str, Any] = Field(..., description="Masked configuration")
    chroma_config: Dict[str, Any] = Field(..., description="Vector database configuration")


# === Collection Management Models ===

class CollectionInfo(BaseModel):
    """Information about a vector collection."""
    
    name: str = Field(..., description="Collection name")
    document_count: int = Field(..., description="Number of documents in collection")
    embedding_dimension: int = Field(..., description="Embedding vector dimension")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Collection metadata")


class CollectionsResponse(BaseModel):
    """Response for listing collections."""
    
    collections: List[CollectionInfo] = Field(..., description="Available collections")
    total_count: int = Field(..., description="Total number of collections")