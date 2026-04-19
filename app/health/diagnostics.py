"""Fast diagnostics with optional real API testing."""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any

from ..config import settings
from ..models import (
    DiagnosticsResponse,
    APITestResult, 
    FunctionalTestResult
)

logger = logging.getLogger(__name__)


async def _quick_api_check(provider_name: str) -> APITestResult:
    """Quick API readiness check without actual calls."""
    start_time = time.time()
    
    try:
        # Check if API key is available using settings (not os.getenv)
        if provider_name == "openai" and not settings.openai_api_key:
            return APITestResult(
                status="warning",
                duration_ms=int((time.time() - start_time) * 1000),
                error="API key not configured"
            )
        elif provider_name == "gemini" and not settings.gemini_api_key:
            return APITestResult(
                status="warning", 
                duration_ms=int((time.time() - start_time) * 1000),
                error="API key not configured"
            )
        
        # Just check if we can import and initialize the provider
        from ..providers import ProviderFactory
        
        # Get the appropriate config for this provider
        provider_config = {}
        if provider_name.lower() == "openai":
            if settings.openai_api_key:
                provider_config = {
                    "api_key": settings.openai_api_key,
                    "default_model": settings.openai_default_model,
                    "timeout": settings.openai_timeout
                }
        elif provider_name.lower() == "gemini":
            if settings.gemini_api_key:
                provider_config = {
                    "api_key": settings.gemini_api_key,
                    "default_model": settings.gemini_default_model,
                    "timeout": settings.gemini_timeout
                }
        
        provider = ProviderFactory.create_provider(provider_name, provider_config)
        duration_ms = int((time.time() - start_time) * 1000)
        
        return APITestResult(
            status="healthy",
            duration_ms=duration_ms,
            error=None
        )
        
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        return APITestResult(
            status="error",
            duration_ms=duration_ms,
            error=f"Provider initialization failed: {str(e)}"
        )


async def _quick_functional_test() -> FunctionalTestResult:
    """Quick functional readiness check."""
    start_time = time.time()
    
    try:
        # Check vector store availability
        from ..vectorstore import get_vector_store
        vector_store = get_vector_store()
        
        if vector_store is None:
            return FunctionalTestResult(
                status="error",
                duration_ms=0,
                similarity_score=None,
                error="Vector store not available"
            )
        
        # Check if we have collections
        collections = vector_store.list_collections()
        duration_ms = int((time.time() - start_time) * 1000)
        
        return FunctionalTestResult(
            status="healthy",
            duration_ms=duration_ms,
            similarity_score=None,  # Not calculated in quick mode
            error=None
        )
        
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        return FunctionalTestResult(
            status="error",
            duration_ms=duration_ms,
            similarity_score=None,
            error=f"Functional test failed: {str(e)}"
        )


async def _run_performance_benchmarks() -> Dict[str, Any]:
    """Run performance benchmark tests with proper status reporting."""
    start_time = time.time()
    benchmarks = {}
    
    try:
        from ..embeddings import generate_embeddings
        
        # Benchmark 1: Single embedding generation
        single_start = time.time()
        await generate_embeddings(["Test text for performance benchmark."])
        single_duration = (time.time() - single_start) * 1000
        benchmarks["single_embedding_ms"] = f"{single_duration:.1f}ms"
        
        # Benchmark 2: Batch embedding generation
        test_texts = [f"Test text number {i} for batch performance test." for i in range(5)]
        batch_start = time.time()
        await generate_embeddings(test_texts)
        batch_duration = (time.time() - batch_start) * 1000
        benchmarks["batch_embedding_ms"] = f"{batch_duration:.1f}ms"
        
        # Calculate throughput
        throughput = len(test_texts) / (batch_duration / 1000)
        benchmarks["embedding_throughput"] = f"{throughput:.1f} texts/sec"
        
        # Benchmark 3: Vector search (if we have data)
        try:
            from ..vectorstore import get_vector_store
            vector_store = get_vector_store()
            collections = vector_store.list_collections()
            
            if collections:
                # Test search performance on existing collection
                query_embedding = await generate_embeddings(["Performance test query"])
                search_start = time.time()
                results = vector_store.search(
                    query_embedding=query_embedding[0],
                    collection_name=collections[0].name,
                    num_results=5
                )
                search_duration = (time.time() - search_start) * 1000
                benchmarks["vector_search_ms"] = f"{search_duration:.1f}ms"
            else:
                benchmarks["vector_search_ms"] = "no collections available"
                
        except Exception as e:
            benchmarks["vector_search_ms"] = f"error: {str(e)}"
        
        # Add overall status and timing
        total_time = int((time.time() - start_time) * 1000)
        benchmarks["status"] = "completed"
        benchmarks["total_benchmark_time_ms"] = f"{total_time}ms"
        
        return benchmarks
        
    except Exception as e:
        logger.error(f"Performance benchmark failed: {e}")
        return {
            "error": f"Benchmarks failed: {str(e)}",
            "status": "failed"
        }


async def get_diagnostics() -> DiagnosticsResponse:
    """Get comprehensive diagnostics including performance benchmarks."""
    start_time = time.time()
    
    try:
        api_tests = {}
        functional_tests = {}
        
        # Real API tests with actual API calls
        openai_result = await _test_openai_api()
        api_tests["openai"] = openai_result
        
        try:
            gemini_result = await _test_gemini_api()
            api_tests["gemini"] = gemini_result
        except Exception as e:
            api_tests["gemini"] = APITestResult(
                status="error",
                duration_ms=0,
                error=f"Gemini API test failed: {str(e)}"
            )
        
        # Functional test
        functional_result = await _quick_functional_test()
        functional_tests["vector_operations"] = functional_result
        
        # Always run performance benchmarks
        performance_benchmarks = await _run_performance_benchmarks()
        
        # Calculate summary
        total_tests = len(api_tests) + len(functional_tests)
        passed = sum(1 for test in api_tests.values() if test.status == "healthy")
        passed += sum(1 for test in functional_tests.values() if test.status == "healthy")
        failed = sum(1 for test in api_tests.values() if test.status == "error") 
        failed += sum(1 for test in functional_tests.values() if test.status == "error")
        warnings = total_tests - passed - failed
        
        total_duration_ms = int((time.time() - start_time) * 1000)
        
        return DiagnosticsResponse(
            status="healthy" if failed == 0 else "degraded" if passed > 0 else "critical",
            timestamp=datetime.utcnow().isoformat() + "Z",
            total_duration_ms=total_duration_ms,
            api_tests=api_tests,
            functional_tests=functional_tests,
            performance_benchmarks=performance_benchmarks,
            summary={
                "total_tests": total_tests,
                "passed": passed,
                "failed": failed,
                "warnings": warnings
            }
        )
        
    except Exception as e:
        logger.error(f"Diagnostics failed: {e}")
        total_duration_ms = int((time.time() - start_time) * 1000)
        
        return DiagnosticsResponse(
            status="error",
            timestamp=datetime.utcnow().isoformat() + "Z", 
            total_duration_ms=total_duration_ms,
            api_tests={},
            functional_tests={},
            performance_benchmarks={"error": f"Diagnostics failed: {str(e)}"},
            summary={
                "total_tests": 0,
                "passed": 0,
                "failed": 1,
                "warnings": 0
            }
        )
        
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error(f"OpenAI API test failed: {e}")
        return APITestResult(
            status="error",
            duration_ms=duration_ms,
            error=str(e)
        )


async def _test_openai_api() -> APITestResult:
    """Test OpenAI API functionality with real API calls."""
    start_time = time.time()
    
    try:
        from ..providers.factory import ProviderFactory
        
        # Create OpenAI provider with proper config
        provider_config = {
            "api_key": settings.openai_api_key,
            "default_model": settings.openai_default_model,
            "timeout": settings.openai_timeout
        } if settings.openai_api_key else {}
        test_provider = ProviderFactory.create_provider("openai", provider_config)
        
        # Test embedding generation with real API call
        test_text = "This is a test for OpenAI API validation and performance measurement."
        result = await test_provider.embed([test_text])
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Validate result - handle different response formats
        if not result:
            return APITestResult(
                status="error",
                duration_ms=duration_ms,
                error="No embeddings returned from OpenAI API"
            )
        
        # Handle OpenAI response format
        embedding = None
        dimensions = None
        
        if isinstance(result, list) and len(result) > 0:
            embedding = result[0]
            dimensions = len(embedding) if embedding else None
        elif isinstance(result, dict):
            # Handle potential dict response
            if 'data' in result and result['data'] and len(result['data']) > 0:
                embedding = result['data'][0].get('embedding')
                dimensions = len(embedding) if embedding else None
            elif 'embeddings' in result and result['embeddings'] and len(result['embeddings']) > 0:
                embedding = result['embeddings'][0] 
                dimensions = len(embedding) if embedding else None
        
        # Get available models
        models = test_provider.get_available_models()
        
        return APITestResult(
            status="healthy",
            duration_ms=duration_ms,
            error=None,
            models_available=models,
            test_embedding_dimensions=dimensions
        )
        
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error(f"OpenAI API test failed: {e}")
        return APITestResult(
            status="error",
            duration_ms=duration_ms,
            error=f"OpenAI API call failed: {str(e)}"
        )


async def _test_gemini_api() -> APITestResult:
    """Test Gemini API functionality with real API calls."""
    start_time = time.time()
    
    try:
        from ..providers.factory import ProviderFactory
        
        # Create Gemini provider with proper config
        provider_config = {
            "api_key": settings.gemini_api_key,
            "default_model": settings.gemini_default_model,
            "timeout": settings.gemini_timeout  
        } if settings.gemini_api_key else {}
        test_provider = ProviderFactory.create_provider("gemini", provider_config)
        
        # Test embedding generation with real API call
        test_text = "This is a test for Gemini API validation and performance measurement."
        result = await test_provider.embed([test_text])
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Validate result - Gemini returns a dict format
        if not result:
            return APITestResult(
                status="error",
                duration_ms=duration_ms,
                error="No embeddings returned from Gemini API"
            )
        
        # Handle Gemini's dict response format  
        embedding = None
        dimensions = None
        
        # Gemini provider typically returns dict with 'embedding' key
        if isinstance(result, dict):
            if 'embedding' in result:
                embedding = result['embedding']
                dimensions = len(embedding) if embedding else None
            elif 'embeddings' in result:
                embeddings = result['embeddings']
                if embeddings and len(embeddings) > 0:
                    embedding = embeddings[0]
                    dimensions = len(embedding) if embedding else None
        elif isinstance(result, list) and len(result) > 0:
            embedding = result[0]
            dimensions = len(embedding) if embedding else None
        
        # Get available models
        models = test_provider.get_available_models()
        
        return APITestResult(
            status="healthy",
            duration_ms=duration_ms,
            error=None,
            models_available=models,
            test_embedding_dimensions=dimensions
        )
        
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Gemini API test failed: {e}")
        return APITestResult(
            status="error",
            duration_ms=duration_ms,
            error=f"Gemini API call failed: {str(e)}"
        )


async def _run_functional_test() -> FunctionalTestResult:
    """Run end-to-end functional test."""
    start_time = time.time()
    test_steps = {}
    similarity_score = 0.0
    
    try:
        from ..vectorstore import get_vector_store
        from ..embeddings import generate_embeddings
        
        vector_store = get_vector_store()
        collection_name = "diagnostics_test"
        
        # Step 1: Clean up any existing test collection
        try:
            existing_collections = vector_store.list_collections()
            if collection_name in existing_collections:
                vector_store.delete_collection(collection_name)
            test_steps["cleanup_collection"] = "passed"
        except Exception as e:
            test_steps["cleanup_collection"] = f"warning: {str(e)}"
        
        # Step 2: Index test documents
        test_docs = [
            {
                "content": "FastAPI is a modern web framework for building APIs with Python.",
                "metadata": {"framework": "fastapi", "language": "python", "test_id": "doc1"}
            },
            {
                "content": "ChromaDB is a vector database for storing and searching embeddings.",
                "metadata": {"database": "chromadb", "type": "vector", "test_id": "doc2"}
            },
            {
                "content": "OpenAI provides powerful embedding models for semantic search.",
                "metadata": {"provider": "openai", "service": "embeddings", "test_id": "doc3"}
            }
        ]
        
        # Generate embeddings for test documents
        texts = [doc["content"] for doc in test_docs]
        embeddings = await generate_embeddings(texts)
        
        # Index documents
        vector_store.index_documents(
            embeddings=embeddings,
            documents=test_docs,
            collection_name=collection_name
        )
        test_steps["index_documents"] = "passed"
        
        # Step 3: Search test
        query = "Python web framework for APIs"
        query_embedding = await generate_embeddings([query])
        
        search_results = vector_store.search(
            query_embedding=query_embedding[0],
            collection_name=collection_name,
            num_results=3
        )
        
        if not search_results:
            test_steps["search_documents"] = "error: no results"
        else:
            test_steps["search_documents"] = "passed"
            # Get the best similarity score
            similarity_score = max(result.get("score", 0.0) for result in search_results)
        
        # Step 4: Verify results
        if similarity_score > 0.3:  # Reasonable threshold
            test_steps["verify_similarity"] = "passed"
            status = "ok"
        else:
            test_steps["verify_similarity"] = f"warning: low similarity {similarity_score:.3f}"
            status = "warning"
        
        # Step 5: Cleanup
        try:
            vector_store.delete_collection(collection_name)
            test_steps["cleanup"] = "passed"
        except Exception as e:
            test_steps["cleanup"] = f"warning: {str(e)}"
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        return FunctionalTestResult(
            status=status,
            duration_ms=duration_ms,
            test_steps=test_steps,
            similarity_score=similarity_score
        )
        
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Functional test failed: {e}")
        test_steps["error"] = str(e)
        
        return FunctionalTestResult(
            status="error",
            duration_ms=duration_ms,
            error=str(e),
            test_steps=test_steps,
            similarity_score=similarity_score
        )


async def _run_api_tests() -> Dict[str, APITestResult]:
    """Run all API tests concurrently."""
    api_tests = {}
    
    # Test APIs based on configuration using settings (not os.getenv)
    test_tasks = []
    
    # Always test OpenAI if key is available
    if settings.openai_api_key:
        test_tasks.append(("openai", _test_openai_api()))
    
    # Test Gemini if key is available
    if settings.gemini_api_key:
        test_tasks.append(("gemini", _test_gemini_api()))
    
    if not test_tasks:
        return api_tests
    
    # Run tests concurrently
    results = await asyncio.gather(
        *[task for _, task in test_tasks],
        return_exceptions=True
    )
    
    # Collect results
    for i, (name, _) in enumerate(test_tasks):
        result = results[i]
        if isinstance(result, Exception):
            api_tests[name] = APITestResult(
                status="error",
                duration_ms=0,
                error=str(result)
            )
        else:
            api_tests[name] = result
    
    return api_tests


async def run_diagnostics() -> DiagnosticsResponse:
    """Run comprehensive diagnostics with API validation and functional testing."""
    overall_start = time.time()
    
    try:
        # Run all diagnostic tests concurrently for speed
        api_tests_task = asyncio.create_task(_run_api_tests())
        functional_test_task = asyncio.create_task(_run_functional_test())
        benchmarks_task = asyncio.create_task(_run_performance_benchmarks())
        
        # Wait for all tests to complete
        api_tests, functional_test, benchmarks = await asyncio.gather(
            api_tests_task,
            functional_test_task,
            benchmarks_task,
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(api_tests, Exception):
            logger.error(f"API tests failed: {api_tests}")
            api_tests = {}
            
        if isinstance(functional_test, Exception):
            logger.error(f"Functional test failed: {functional_test}")
            functional_test = FunctionalTestResult(
                status="error",
                duration_ms=0,
                error=str(functional_test),
                test_steps={},
                similarity_score=None
            )
            
        if isinstance(benchmarks, Exception):
            logger.error(f"Benchmarks failed: {benchmarks}")
            benchmarks = {"error": str(benchmarks)}
        
        # Calculate summary
        total_tests = len(api_tests) + 1  # API tests + functional test
        passed_tests = sum(1 for test in api_tests.values() if test.status == "ok")
        if functional_test.status == "ok":
            passed_tests += 1
        
        failed_tests = sum(1 for test in api_tests.values() if test.status == "error")
        if functional_test.status == "error":
            failed_tests += 1
            
        warnings = sum(1 for test in api_tests.values() if test.status == "warning")
        if functional_test.status == "warning":
            warnings += 1
        
        # Determine overall status
        if failed_tests > 0:
            status = "failed"
        elif warnings > 0:
            status = "warning"
        else:
            status = "passed"
        
        total_duration_ms = int((time.time() - overall_start) * 1000)
        
        return DiagnosticsResponse(
            status=status,
            timestamp=datetime.utcnow().isoformat() + "Z",
            total_duration_ms=total_duration_ms,
            api_tests=api_tests,
            functional_tests={"end_to_end": functional_test},
            performance_benchmarks=benchmarks,
            summary={
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "warnings": warnings
            }
        )
        
    except Exception as e:
        logger.error(f"Diagnostics run failed: {e}")
        total_duration_ms = int((time.time() - overall_start) * 1000)
        
        return DiagnosticsResponse(
            status="error",
            timestamp=datetime.utcnow().isoformat() + "Z",
            total_duration_ms=total_duration_ms,
            api_tests={},
            functional_tests={},
            performance_benchmarks={"error": str(e)},
            summary={
                "total_tests": 0,
                "passed": 0,
                "failed": 1,
                "warnings": 0
            }
        )