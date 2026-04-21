"""Pytest configuration for semantic search API tests."""

import pytest
from fastapi.testclient import TestClient

from app import embeddings as embeddings_module
from app import vectorstore as vectorstore_module
from app.main import app
from app.config import settings
from app.vectorstore import reset_vector_store


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment(tmp_path_factory):
    """Set up test environment with mock embeddings."""
    original_use_mock_embeddings = settings.use_mock_embeddings
    original_debug = settings.debug
    original_log_level = settings.log_level
    original_chroma_persist_directory = settings.chroma_persist_directory
    original_vectorstore_generate_embeddings = vectorstore_module.generate_embeddings

    # Enable mock mode for all tests
    settings.use_mock_embeddings = True
    settings.debug = True
    settings.log_level = "INFO"

    # Use a real temporary directory for Chroma during tests.
    test_chroma_dir = tmp_path_factory.mktemp("chroma-test-db")
    settings.chroma_persist_directory = str(test_chroma_dir)

    # Ensure vector store paths also use mock embeddings during search and indexing.
    vectorstore_module.generate_embeddings = embeddings_module.generate_mock_embeddings
    reset_vector_store()

    yield

    # Cleanup after tests
    vectorstore_module.generate_embeddings = original_vectorstore_generate_embeddings
    settings.use_mock_embeddings = original_use_mock_embeddings
    settings.debug = original_debug
    settings.log_level = original_log_level
    settings.chroma_persist_directory = original_chroma_persist_directory
    reset_vector_store()


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