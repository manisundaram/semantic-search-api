"""Pytest configuration for semantic search API tests."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment with mock embeddings."""
    # Enable mock mode for all tests
    settings.use_mock_embeddings = True
    settings.debug = True
    settings.log_level = "INFO"
    
    # Use in-memory Chroma for tests
    settings.chroma_persist_directory = ":memory:"
    
    yield
    
    # Cleanup after tests
    settings.use_mock_embeddings = False


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "content": "FastAPI is a modern, fast web framework for building APIs with Python.",
            "metadata": {"source": "docs", "topic": "web development"}
        },
        {
            "content": "Vector databases enable semantic search over high-dimensional data.",
            "metadata": {"source": "blog", "topic": "databases"}
        },
        {
            "content": "Machine learning embeddings capture semantic meaning of text.",
            "metadata": {"source": "tutorial", "topic": "machine learning"}
        }
    ]


@pytest.fixture
def sample_texts():
    """Sample texts for embedding tests."""
    return [
        "Hello world, this is a test.",
        "Machine learning is transforming technology.",
        "Vector databases store embeddings efficiently."
    ]