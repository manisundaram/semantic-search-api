"""Basic tests for the semantic search API."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings

# Override settings for testing
settings.use_mock_embeddings = True
settings.debug = True

client = TestClient(app)


class TestBasicEndpoints:
    """Test basic API endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint returns service info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == settings.app_name
        assert "endpoints" in data
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "version" in data
        assert "runtime" in data
        assert "configuration" in data
    
    def test_models_endpoint(self):
        """Test models listing endpoint."""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "provider" in data
        assert "models" in data
        assert isinstance(data["models"], list)


class TestEmbeddingEndpoint:
    """Test embedding generation endpoint."""
    
    def test_embed_single_text(self):
        """Test embedding single text."""
        request_data = {
            "texts": ["Hello world, this is a test."]
        }
        response = client.post("/embed", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "embeddings" in data
        assert len(data["embeddings"]) == 1
        assert len(data["embeddings"][0]) > 0  # Should have embedding dimensions
        assert "model" in data
        assert "provider" in data
    
    def test_embed_multiple_texts(self):
        """Test embedding multiple texts."""
        request_data = {
            "texts": [
                "First test document.",
                "Second test document.",
                "Third test document."
            ]
        }
        response = client.post("/embed", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert len(data["embeddings"]) == 3
        # All embeddings should have same dimension
        assert all(
            len(emb) == len(data["embeddings"][0]) 
            for emb in data["embeddings"]
        )
    
    def test_embed_empty_texts(self):
        """Test embedding with empty texts list."""
        request_data = {"texts": []}
        response = client.post("/embed", json=request_data)
        assert response.status_code == 422  # Validation error


class TestIndexEndpoint:
    """Test document indexing endpoint."""
    
    def test_index_documents(self):
        """Test indexing documents."""
        request_data = {
            "documents": [
                {
                    "content": "This is the first test document for indexing.",
                    "metadata": {"source": "test", "category": "example"}
                },
                {
                    "content": "This is the second test document with different content.",
                    "metadata": {"source": "test", "category": "demo"}
                }
            ]
        }
        response = client.post("/index", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["indexed_count"] == 2
        assert "chunk_count" in data
        assert "collection_name" in data
        assert len(data["documents"]) == 2


class TestSearchEndpoint:
    """Test document search endpoint."""
    
    def test_search_documents(self):
        """Test searching for documents."""
        # First index some documents
        index_data = {
            "documents": [
                {
                    "content": "FastAPI is a modern web framework for building APIs with Python.",
                    "metadata": {"topic": "web development"}
                },
                {
                    "content": "Vector databases store high-dimensional vectors for similarity search.",
                    "metadata": {"topic": "databases"}
                }
            ]
        }
        client.post("/index", json=index_data)
        
        # Now search for similar documents
        search_data = {
            "query": "Python web frameworks",
            "k": 5
        }
        response = client.post("/search", json=search_data)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total_results" in data
        assert data["query"] == search_data["query"]
    
    def test_search_empty_query(self):
        """Test search with empty query."""
        search_data = {"query": "", "k": 5}
        response = client.post("/search", json=search_data)
        assert response.status_code == 422  # Validation error


class TestCollectionsEndpoint:
    """Test collections management endpoint."""
    
    def test_list_collections(self):
        """Test listing collections."""
        response = client.get("/collections")
        assert response.status_code == 200
        data = response.json()
        assert "collections" in data
        assert "total_count" in data
        assert isinstance(data["collections"], list)


# TODO: Add more comprehensive tests
# - Test error handling
# - Test different providers
# - Test chunking behavior
# - Test metadata filtering
# - Test similarity thresholds
# - Test collection management
# - Integration tests with real providers