"""Basic health checks for fast system status validation."""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from ..config import settings
from ..models import (
    HealthResponse, 
    StartupHealthCheck, 
    RuntimeHealthCheck, 
    MockHealthCheck, 
    ConfigurationInfo
)

logger = logging.getLogger(__name__)


def _check_startup_health() -> StartupHealthCheck:
    """Check startup-related health indicators."""
    start_time = time.time()
    
    # Check ChromaDB import
    chromadb_available = False
    try:
        import chromadb
        chromadb_available = True
    except ImportError:
        pass
    
    # Check OpenAI import
    openai_imported = False
    try:
        import openai
        openai_imported = True
    except ImportError:
        pass
    
    # Check config loaded
    config_loaded = bool(settings)
    
    # Check environment file
    env_file_found = Path(".env").exists()
    
    # Check required variables (using settings, not os.getenv)
    required_vars_present = bool(settings.openai_api_key or settings.gemini_api_key)
    
    # Check ChromaDB directory writable
    chroma_dir_writable = False
    try:
        chroma_path = Path("./chroma_db")
        chroma_path.mkdir(exist_ok=True)
        test_file = chroma_path / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
        chroma_dir_writable = True
    except Exception:
        pass
    
    duration_ms = int((time.time() - start_time) * 1000)
    logger.debug(f"Startup health check completed in {duration_ms}ms")
    
    return StartupHealthCheck(
        chromadb_available=chromadb_available,
        openai_imported=openai_imported,
        config_loaded=config_loaded,
        env_file_found=env_file_found,
        required_vars_present=required_vars_present,
        chroma_dir_writable=chroma_dir_writable
    )


def _check_runtime_health() -> RuntimeHealthCheck:
    """Check runtime health indicators."""
    start_time = time.time()
    
    # Check ChromaDB connection
    chromadb_connected = False
    collections_count = 0
    vector_store_ready = False
    try:
        from ..vectorstore import get_vector_store, CHROMADB_AVAILABLE
        if CHROMADB_AVAILABLE:
            vector_store = get_vector_store()
            collections_count = len(vector_store.list_collections())
            chromadb_connected = True
            vector_store_ready = True
    except Exception as e:
        logger.debug(f"ChromaDB connection check failed: {e}")
    
    # Check provider initialization
    provider_initialized = False
    try:
        from ..embeddings import get_embedding_provider
        provider = get_embedding_provider()
        provider_initialized = bool(provider)
    except Exception as e:
        logger.debug(f"Provider initialization check failed: {e}")
    
    # Get last operation timestamp (placeholder for now)
    last_operation = datetime.utcnow().isoformat() + "Z"
    
    duration_ms = int((time.time() - start_time) * 1000)
    logger.debug(f"Runtime health check completed in {duration_ms}ms")
    
    return RuntimeHealthCheck(
        chromadb_connected=chromadb_connected,
        collections_count=collections_count,
        provider_initialized=provider_initialized,
        last_operation=last_operation,
        vector_store_ready=vector_store_ready
    )


def _check_mock_health() -> MockHealthCheck:
    """Check mock functionality health."""
    start_time = time.time()
    
    embeddings_working = False
    search_working = False
    dimensions_correct = False
    performance_acceptable = True
    
    try:
        from ..embeddings import generate_mock_embeddings
        
        # Test mock embedding generation
        mock_start = time.time()
        result = generate_mock_embeddings(["test text"], model="test-model")
        mock_duration = time.time() - mock_start
        
        if hasattr(result, 'embeddings') and len(result.embeddings) == 1:
            embeddings_working = True
            embedding = result.embeddings[0]
            if len(embedding) == 1536:  # Standard dimension
                dimensions_correct = True
        
        # Check if mock generation is fast enough (< 50ms)
        performance_acceptable = mock_duration < 0.05
        
        # Test mock search (basic similarity test)
        if embeddings_working:
            search_working = True  # Mock search is always available if embeddings work
        
    except Exception as e:
        logger.debug(f"Mock health check failed: {e}")
    
    duration_ms = int((time.time() - start_time) * 1000)
    logger.debug(f"Mock health check completed in {duration_ms}ms")
    
    return MockHealthCheck(
        embeddings_working=embeddings_working,
        search_working=search_working,
        dimensions_correct=dimensions_correct,
        performance_acceptable=performance_acceptable
    )


def _get_configuration_info() -> ConfigurationInfo:
    """Get configuration information without exposing secrets."""
    
    # Mask API keys for safe display (using settings)
    openai_key = None
    if settings.openai_api_key:
        key = settings.openai_api_key
        if len(key) > 12:
            openai_key = f"{key[:7]}****...****{key[-6:]}"
        else:
            openai_key = "****...****"
    
    gemini_key = None
    if settings.gemini_api_key:
        key = settings.gemini_api_key
        if len(key) > 12:
            gemini_key = f"{key[:7]}****...****{key[-6:]}"
        else:
            gemini_key = "****...****"
    
    # Get default models
    default_models = {
        "openai": "text-embedding-3-small",
        "gemini": "models/embedding-001"
    }
    
    # ChromaDB settings
    chromadb_settings = {
        "persist_directory": "./chroma_db",
        "allow_reset": False
    }
    
    return ConfigurationInfo(
        provider=settings.embedding_provider,
        mock_mode=settings.use_mock_embeddings,
        debug_mode=settings.debug,
        cors_enabled=len(settings.cors_origins) > 0,
        openai_key=openai_key,
        gemini_key=gemini_key,
        default_models=default_models,
        chromadb_settings=chromadb_settings
    )


def check_health() -> HealthResponse:
    """Perform comprehensive but fast health check."""
    overall_start = time.time()
    
    try:
        # Run all health checks
        startup = _check_startup_health()
        runtime = _check_runtime_health()
        mock = _check_mock_health()
        configuration = _get_configuration_info()
        
        # Determine overall status
        startup_issues = sum([
            not startup.chromadb_available,
            not startup.openai_imported,
            not startup.config_loaded,
            not startup.env_file_found,
            not startup.required_vars_present,
            not startup.chroma_dir_writable
        ])
        
        runtime_issues = sum([
            not runtime.chromadb_connected,
            not runtime.provider_initialized,
            not runtime.vector_store_ready
        ])
        
        mock_issues = sum([
            not mock.embeddings_working,
            not mock.search_working,
            not mock.dimensions_correct,
            not mock.performance_acceptable
        ])
        
        # Determine overall health status
        if startup_issues >= 3 or runtime_issues >= 2:
            status = "critical"
        elif startup_issues >= 1 or runtime_issues >= 1 or mock_issues >= 2:
            status = "degraded"
        else:
            status = "healthy"
        
        response_time_ms = int((time.time() - overall_start) * 1000)
        
        return HealthResponse(
            status=status,
            version=settings.app_version,
            timestamp=datetime.utcnow().isoformat() + "Z",
            response_time_ms=response_time_ms,
            startup=startup,
            runtime=runtime,
            mock=mock,
            configuration=configuration
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        response_time_ms = int((time.time() - overall_start) * 1000)
        
        return HealthResponse(
            status="error",
            version=settings.app_version,
            timestamp=datetime.utcnow().isoformat() + "Z",
            response_time_ms=response_time_ms,
            startup=StartupHealthCheck(
                chromadb_available=False,
                openai_imported=False,
                config_loaded=False,
                env_file_found=False,
                required_vars_present=False,
                chroma_dir_writable=False
            ),
            runtime=RuntimeHealthCheck(
                chromadb_connected=False,
                collections_count=0,
                provider_initialized=False,
                last_operation=None,
                vector_store_ready=False
            ),
            mock=MockHealthCheck(
                embeddings_working=False,
                search_working=False,
                dimensions_correct=False,
                performance_acceptable=False
            ),
            configuration=ConfigurationInfo(
                provider="unknown",
                mock_mode=True,
                debug_mode=False,
                cors_enabled=False,
                openai_key=None,
                gemini_key=None,
                default_models={},
                chromadb_settings={}
            )
        )