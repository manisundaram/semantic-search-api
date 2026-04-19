"""Main FastAPI application for semantic search API."""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

from .config import settings
from .embeddings import (
    generate_embeddings,
    generate_mock_embeddings,
    get_available_models,
    get_embedding_dimension,
    get_embedding_provider,
    reset_provider
)
from .models import (
    ConfigResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorResponse,
    HealthResponse,
    DiagnosticsResponse,
    MetricsResponse,
    SimpleHealthResponse,
    IndexRequest,
    IndexResponse,
    SearchRequest,
    SearchResponse,
    CollectionsResponse,
    IndexedDocument,
    CollectionInfo
)
from .providers.base import EmbeddingError, EmbeddingRateLimitError, EmbeddingConfigError
from .vectorstore import get_vector_store, VectorStore
from .health import check_health, get_diagnostics, get_metrics, record_operation_time

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.
    
    Handles startup and shutdown tasks.
    """
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    # Initialize embedding provider
    try:
        provider = get_embedding_provider()
        logger.info(f"Embedding provider initialized: {provider.get_provider_name()}")
    except Exception as e:
        logger.error(f"Failed to initialize embedding provider: {e}")
        if not settings.use_mock_embeddings:
            raise
    
    # Initialize vector store
    try:
        vector_store = get_vector_store()
        logger.info("Vector store initialized")
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info(f"Shutting down {settings.app_name}")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Semantic search API with embedding-based document retrieval",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Request ID middleware
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Add request ID and logging middleware."""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Add request ID to request state
    request.state.request_id = request_id
    
    # Log request
    logger.info(
        f"Request {request_id}: {request.method} {request.url.path} "
        f"from {request.client.host if request.client else 'unknown'}"
    )
    
    # Process request
    response = await call_next(request)
    
    # Log response
    duration = time.time() - start_time
    logger.info(
        f"Request {request_id}: {response.status_code} "
        f"completed in {duration:.3f}s"
    )
    
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    
    return response


# Exception handlers
@app.exception_handler(EmbeddingRateLimitError)
async def handle_rate_limit_error(request: Request, exc: EmbeddingRateLimitError):
    """Handle rate limiting errors."""
    logger.warning(f"Rate limit error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=ErrorResponse(
            error="rate_limit_exceeded",
            message=str(exc),
            provider=exc.provider,
            request_id=getattr(request.state, 'request_id', None)
        ).model_dump()
    )


@app.exception_handler(EmbeddingConfigError)
async def handle_config_error(request: Request, exc: EmbeddingConfigError):
    """Handle configuration errors."""
    logger.error(f"Configuration error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content=ErrorResponse(
            error="provider_configuration_error",
            message=str(exc),
            provider=exc.provider,
            request_id=getattr(request.state, 'request_id', None)
        ).model_dump()
    )


@app.exception_handler(EmbeddingError)
async def handle_embedding_error(request: Request, exc: EmbeddingError):
    """Handle general embedding errors."""
    logger.error(f"Embedding error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content=ErrorResponse(
            error="embedding_provider_error",
            message=str(exc),
            provider=exc.provider,
            request_id=getattr(request.state, 'request_id', None)
        ).model_dump()
    )


@app.exception_handler(Exception)
async def handle_general_error(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_server_error",
            message="An unexpected error occurred",
            request_id=getattr(request.state, 'request_id', None)
        ).model_dump()
    )


# Root endpoint
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with service information."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "provider": settings.embedding_provider,
        "endpoints": {
            "health": "/health",
            "diagnostics": "/diagnostics",
            "metrics": "/metrics",
            "embed": "/embed",
            "index": "/index",
            "search": "/search",
            "collections": "/collections",
            "models": "/models",
            "debug": "/debug/config"
        }
    }


# Health check endpoints
@app.get("/health", response_model=HealthResponse)
async def health():
    """Comprehensive health check endpoint."""
    return check_health()


@app.get("/diagnostics", response_model=DiagnosticsResponse)
async def diagnostics():
    """Comprehensive diagnostics with API validation, functional testing, and performance benchmarks.
    
    This endpoint performs complete system diagnostics including real API calls and performance testing.
    Use the /health endpoint for quick status checks.
    """
    return await get_diagnostics()


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """Performance and usage metrics."""
    return get_metrics()


# Legacy health endpoint for compatibility
@app.get("/health/simple", response_model=SimpleHealthResponse)
async def simple_health():
    """Simple health check endpoint (legacy compatibility)."""
    try:
        # Test provider initialization
        if settings.use_mock_embeddings:
            provider_status = "mock"
        else:
            provider = get_embedding_provider()
            provider_status = provider.get_provider_name()
        
        return SimpleHealthResponse(
            status="healthy",
            version=settings.app_version,
            provider=provider_status,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return SimpleHealthResponse(
            status="unhealthy",
            version=settings.app_version,
            provider="error",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Health monitoring dashboard - static HTML with live data updates."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Semantic Search API - Health Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1100px;
            margin: 0 auto;
            padding: 15px;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 20px;
        }
        
        .header h1 {
            font-size: 2rem;
            margin-bottom: 5px;
        }
        
        .header p {
            font-size: 1rem;
            opacity: 0.9;
        }
        
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .status-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            padding: 18px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            transition: transform 0.2s ease;
        }
        
        .status-card:hover {
            transform: translateY(-3px);
        }
        
        .card-header {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .status-icon {
            font-size: 1.5rem;
            margin-right: 10px;
        }
        
        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #2c3e50;
        }
        
        .status-value {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 8px;
        }
        
        .status-healthy { color: #27ae60; }
        .status-degraded { color: #f39c12; }
        .status-critical { color: #e74c3c; }
        .status-error { color: #c0392b; }
        
        .metric-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 0;
            border-bottom: 1px solid #ecf0f1;
        }
        
        .metric-item:last-child {
            border-bottom: none;
        }
        
        .metric-label {
            color: #7f8c8d;
            font-size: 0.85rem;
        }
        
        .metric-value {
            font-weight: 600;
            color: #2c3e50;
            font-size: 0.9rem;
        }
        
        .api-tests {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        
        .api-test {
            background: #f8f9fa;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }
        
        .api-name {
            font-weight: 600;
            margin-bottom: 6px;
            text-transform: capitalize;
            font-size: 0.9rem;
        }
        
        .api-duration {
            font-size: 1rem;
            font-weight: 700;
            color: #3498db;
            margin-bottom: 4px;
        }
        
        .api-dimensions {
            color: #7f8c8d;
            font-size: 0.8rem;
        }
        
        .loading {
            text-align: center;
            color: #7f8c8d;
            font-style: italic;
        }
        
        .error {
            color: #e74c3c;
            text-align: center;
            font-weight: 600;
        }
        
        .refresh-info {
            text-align: center;
            color: rgba(255, 255, 255, 0.8);
            margin-top: 15px;
            font-size: 0.85rem;
        }
        
        .timestamp {
            text-align: center;
            color: rgba(255, 255, 255, 0.7);
            margin-top: 5px;
            font-size: 0.75rem;
        }
        
        @media (max-width: 768px) {
            .container { padding: 10px; }
            .header h1 { font-size: 1.8rem; }
            .status-grid { grid-template-columns: 1fr; gap: 10px; }
            .status-card { padding: 15px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Semantic Search API</h1>
            <p>Health Monitoring Dashboard</p>
        </div>
        
        <div class="status-grid">
            <!-- System Health Card -->
            <div class="status-card">
                <div class="card-header">
                    <div class="status-icon">🏥</div>
                    <div class="card-title">System Health</div>
                </div>
                <div id="system-status" class="loading">Loading...</div>
            </div>
            
            <!-- Runtime Metrics Card -->
            <div class="status-card">
                <div class="card-header">
                    <div class="status-icon">📊</div>
                    <div class="card-title">Runtime Metrics</div>
                </div>
                <div id="runtime-metrics" class="loading">Loading...</div>
            </div>
            
            <!-- API Tests Card -->
            <div class="status-card">
                <div class="card-header">
                    <div class="status-icon">🧪</div>
                    <div class="card-title">API Performance</div>
                </div>
                <div id="api-tests" class="loading">Loading...</div>
            </div>
            
            <!-- Performance Benchmarks Card -->
            <div class="status-card">
                <div class="card-header">
                    <div class="status-icon">⚡</div>
                    <div class="card-title">Performance Benchmarks</div>
                </div>
                <div id="benchmarks" class="loading">Loading...</div>
            </div>
        </div>
        
        <div class="refresh-info">
            🔄 Auto-refreshing every 30 seconds
        </div>
        <div id="last-update" class="timestamp"></div>
    </div>
    
    <script>
        let healthData = null;
        let diagnosticsData = null;
        
        function formatUptime(seconds) {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = seconds % 60;
            return `${hours}h ${minutes}m ${secs}s`;
        }
        
        function getStatusClass(status) {
            return `status-${status.toLowerCase()}`;
        }
        
        function getStatusEmoji(status) {
            const emojis = {
                'healthy': '🟢',
                'degraded': '🟡', 
                'critical': '🔴',
                'error': '⚫'
            };
            return emojis[status.toLowerCase()] || '⚪';
        }
        
        async function fetchHealthData() {
            try {
                const healthResponse = await fetch('/health');
                healthData = await healthResponse.json();
                
                const diagnosticsResponse = await fetch('/diagnostics');
                diagnosticsData = await diagnosticsResponse.json();
                
                updateDashboard();
            } catch (error) {
                console.error('Failed to fetch health data:', error);
                showError('Failed to load health data');
            }
        }
        
        function updateDashboard() {
            updateSystemStatus();
            updateRuntimeMetrics();
            updateApiTests();
            updateBenchmarks();
            updateTimestamp();
        }
        
        function updateSystemStatus() {
            const container = document.getElementById('system-status');
            if (!healthData) return;
            
            const status = healthData.status;
            const statusClass = getStatusClass(status);
            const statusEmoji = getStatusEmoji(status);
            
            container.innerHTML = `
                <div class="status-value ${statusClass}">${statusEmoji} ${status.toUpperCase()}</div>
                <div class="metric-item">
                    <span class="metric-label">API Keys</span>
                    <span class="metric-value">${healthData.startup?.required_vars_present ? '✅' : '❌'}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">ChromaDB</span>
                    <span class="metric-value">${healthData.runtime?.chromadb_connected ? '✅' : '❌'}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Provider Ready</span>
                    <span class="metric-value">${healthData.runtime?.provider_initialized ? '✅' : '❌'}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Provider</span>
                    <span class="metric-value">${healthData.configuration?.provider || 'N/A'}</span>
                </div>
            `;
        }
        
        function updateRuntimeMetrics() {
            const container = document.getElementById('runtime-metrics');
            if (!healthData) return;
            
            const runtime = healthData.runtime || {};
            const config = healthData.configuration || {};
            
            container.innerHTML = `
                <div class="metric-item">
                    <span class="metric-label">Collections</span>
                    <span class="metric-value">${runtime.collections_count || 0}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Response Time</span>
                    <span class="metric-value">${healthData.response_time_ms || 'N/A'}ms</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Vector Store</span>
                    <span class="metric-value">${runtime.vector_store_ready ? '✅ Ready' : '❌ Not Ready'}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Mock Mode</span>
                    <span class="metric-value">${config.mock_mode ? '🧪 Enabled' : '🔥 Live'}</span>
                </div>
            `;
        }
        
        function updateApiTests() {
            const container = document.getElementById('api-tests');
            if (!diagnosticsData || !diagnosticsData.api_tests) return;
            
            const apiTests = diagnosticsData.api_tests;
            let html = '<div class="api-tests">';
            
            Object.entries(apiTests).forEach(([provider, test]) => {
                const statusEmoji = getStatusEmoji(test.status);
                html += `
                    <div class="api-test">
                        <div class="api-name">${statusEmoji} ${provider}</div>
                        <div class="api-duration">${test.duration_ms}ms</div>
                        <div class="api-dimensions">${test.test_embedding_dimensions || 'N/A'} dimensions</div>
                    </div>
                `;
            });
            
            html += '</div>';
            container.innerHTML = html;
        }
        
        function updateBenchmarks() {
            const container = document.getElementById('benchmarks');
            if (!diagnosticsData || !diagnosticsData.performance_benchmarks) return;
            
            const benchmarks = diagnosticsData.performance_benchmarks;
            
            container.innerHTML = `
                <div class="metric-item">
                    <span class="metric-label">Single Embedding</span>
                    <span class="metric-value">${benchmarks.single_embedding_ms || 'N/A'}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Batch Processing</span>
                    <span class="metric-value">${benchmarks.batch_embedding_ms || 'N/A'}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Throughput</span>
                    <span class="metric-value">${benchmarks.embedding_throughput || 'N/A'}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Vector Search</span>
                    <span class="metric-value">${benchmarks.vector_search_ms || 'N/A'}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Status</span>
                    <span class="metric-value">${getStatusEmoji(benchmarks.status)} ${benchmarks.status}</span>
                </div>
            `;
        }
        
        function updateTimestamp() {
            const container = document.getElementById('last-update');
            container.textContent = `Last updated: ${new Date().toLocaleString()}`;
        }
        
        function showError(message) {
            ['system-status', 'runtime-metrics', 'api-tests', 'benchmarks'].forEach(id => {
                document.getElementById(id).innerHTML = `<div class="error">${message}</div>`;
            });
        }
        
        // Initial load
        fetchHealthData();
        
        // Auto-refresh every 30 seconds
        setInterval(fetchHealthData, 30000);
    </script>
</body>
</html>
    """
    return html_content


# Embedding endpoint
@app.post("/embed", response_model=EmbeddingResponse)
async def embed_texts(request: EmbeddingRequest):
    """Generate embeddings for input texts."""
    start_time = time.time()
    
    try:
        # Use mock embeddings if configured
        if settings.use_mock_embeddings:
            result = await generate_mock_embeddings(
                request.texts,
                model=request.model
            )
            provider_name = "mock"
        else:
            result = await generate_embeddings(
                request.texts,
                model=request.model
            )
            provider_name = result.get("provider", "unknown")
        
        # Record metrics
        duration_ms = int((time.time() - start_time) * 1000)
        record_operation_time("embedding", duration_ms, provider=provider_name)
        
        return EmbeddingResponse(
            embeddings=result["embeddings"],
            model=result["model"],
            usage=result["usage"],
            provider=result["provider"],
            dimension=len(result["embeddings"][0]) if result["embeddings"] else 0
        )
    
    except Exception as e:
        # Record error
        from .health.metrics import get_metrics_collector
        get_metrics_collector().record_error("embedding_failed", provider_name if 'provider_name' in locals() else "unknown")
        
        logger.error(f"Embedding request failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(e)}"
        )


# Document indexing endpoint
@app.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest):
    """Index documents into the vector store."""
    start_time = time.time()
    
    try:
        vector_store = get_vector_store()
        
        result = await vector_store.index_documents(
            documents=request.documents,
            collection_name=request.collection_name,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        
        # Record metrics
        duration_ms = int((time.time() - start_time) * 1000)
        record_operation_time(
            "indexing", 
            duration_ms, 
            doc_count=result["indexed_count"],
            collection=result["collection_name"]
        )
        
        # Create indexed document details
        indexed_docs = []
        for i, doc in enumerate(request.documents):
            indexed_docs.append(IndexedDocument(
                id=f"doc_{i}",
                content=doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                metadata=doc.get("metadata", {}),
                chunk_index=None  # TODO: Add chunk mapping
            ))
        
        return IndexResponse(
            message=f"Successfully indexed {result['indexed_count']} documents",
            indexed_count=result["indexed_count"],
            chunk_count=result["chunk_count"],
            collection_name=result["collection_name"],
            documents=indexed_docs
        )
    
    except Exception as e:
        # Record error
        from .health.metrics import get_metrics_collector
        get_metrics_collector().record_error("indexing_failed")
        
        logger.error(f"Indexing request failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document indexing failed: {str(e)}"
        )


# Search endpoint
@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search for similar documents."""
    start_time = time.time()
    
    try:
        vector_store = get_vector_store()
        
        result = await vector_store.search(
            query=request.query,
            k=request.k,
            collection_name=request.collection_name,
            filter_metadata=request.filter_metadata,
            similarity_threshold=request.similarity_threshold
        )
        
        # Record metrics
        duration_ms = int((time.time() - start_time) * 1000)
        record_operation_time(
            "search", 
            duration_ms, 
            query=request.query,
            collection=request.collection_name or "default"
        )
        
        return SearchResponse(**result)
    
    except Exception as e:
        # Record error
        from .health.metrics import get_metrics_collector
        get_metrics_collector().record_error("search_failed")
        
        logger.error(f"Search request failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document search failed: {str(e)}"
        )


# Collections management endpoint
@app.get("/collections", response_model=CollectionsResponse)
async def list_collections():
    """List all collections in the vector store."""
    try:
        vector_store = get_vector_store()
        collections_info = vector_store.list_collections()
        
        collections = [
            CollectionInfo(
                name=info["name"],
                document_count=info["document_count"],
                embedding_dimension=info["embedding_dimension"],
                metadata=info["metadata"]
            )
            for info in collections_info
        ]
        
        return CollectionsResponse(
            collections=collections,
            total_count=len(collections)
        )
    
    except Exception as e:
        logger.error(f"Failed to list collections: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list collections: {str(e)}"
        )


# Models endpoint
@app.get("/models", response_model=dict)
async def list_models():
    """List available embedding models for the current provider."""
    try:
        if settings.use_mock_embeddings:
            return {
                "provider": "mock",
                "models": ["mock-embedding-model"],
                "default_model": "mock-embedding-model"
            }
        
        models = get_available_models()
        provider = get_embedding_provider()
        
        return {
            "provider": provider.get_provider_name(),
            "models": models,
            "default_model": settings.get_default_model()
        }
    
    except Exception as e:
        logger.error(f"Failed to get models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get available models: {str(e)}"
        )


# Debug configuration endpoint
@app.get("/debug/config", response_model=ConfigResponse)
async def debug_config():
    """Debug endpoint to view current configuration (with sensitive data masked)."""
    try:
        if settings.use_mock_embeddings:
            available_models = ["mock-embedding-model"]
            provider_name = "mock"
        else:
            available_models = get_available_models()
            provider = get_embedding_provider()
            provider_name = provider.get_provider_name()
        
        return ConfigResponse(
            provider=provider_name,
            model=settings.get_default_model(),
            available_models=available_models,
            config=settings.mask_sensitive_config(),
            chroma_config={
                "persist_directory": settings.chroma_persist_directory,
                "collection_name": settings.chroma_collection_name
            }
        )
    
    except Exception as e:
        logger.error(f"Debug config failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration debug failed: {str(e)}"
        )


@app.get("/debug/chromadb-status")
async def debug_chromadb_status():
    """Debug endpoint to check ChromaDB availability status"""
    from .vectorstore import CHROMADB_AVAILABLE
    try:
        import chromadb
        chroma_version = chromadb.__version__
        chroma_importable = True
    except ImportError:
        chroma_version = "Not available"
        chroma_importable = False
    
    return {
        "chromadb_available_flag": CHROMADB_AVAILABLE,
        "chromadb_importable": chroma_importable,
        "chromadb_version": chroma_version,
        "flag_type": type(CHROMADB_AVAILABLE).__name__,
        "flag_truthiness": bool(CHROMADB_AVAILABLE)
    }


# Development endpoint for resetting providers (debug only)
if settings.debug:
    @app.post("/debug/reset")
    async def reset_providers():
        """Reset all provider instances (debug only)."""
        reset_provider()
        logger.info("Reset embedding provider instances")
        return {"message": "Providers reset successfully"}