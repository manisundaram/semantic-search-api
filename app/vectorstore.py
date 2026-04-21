"""Vector store implementation using Chroma for document indexing and search."""

# Force reload to detect ChromaDB after installation
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
    print(f"✅ DEBUG: ChromaDB imported successfully! Version: {chromadb.__version__}")
except ImportError as e:
    chromadb = None
    ChromaSettings = None
    CHROMADB_AVAILABLE = False
    print(f"❌ DEBUG: ChromaDB import failed: {e}")
except Exception as e:
    chromadb = None  
    ChromaSettings = None
    CHROMADB_AVAILABLE = False
    print(f"❌ DEBUG: ChromaDB other error: {e}")

print(f"DEBUG: CHROMADB_AVAILABLE = {CHROMADB_AVAILABLE}")

from .config import settings
from .embeddings import generate_embeddings, get_embedding_dimension

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store implementation using Chroma database.
    
    Handles document indexing, embedding storage, and semantic search.
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        """Initialize vector store.
        
        Args:
            persist_directory: Directory to persist Chroma database
            collection_name: Default collection name
        """
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available. Using mock vector store.")
            self.client = None
            self.persist_directory = None
            self.default_collection = collection_name or settings.chroma_collection_name
            return
        
        self.persist_directory = persist_directory or settings.chroma_persist_directory
        self.default_collection = collection_name or settings.chroma_collection_name
        
        # Initialize Chroma client with persistence
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        logger.info(f"Initialized vector store at {self.persist_directory}")

    def reset_collection(self, collection_name: Optional[str] = None) -> str:
        """Delete and recreate a collection to guarantee a clean state."""
        name = collection_name or self.default_collection

        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available - reset collection skipped")
            return name

        try:
            self.client.delete_collection(name)
            logger.info(f"Deleted existing collection: {name}")
        except Exception as e:
            logger.info(f"Collection '{name}' not deleted before reset ({type(e).__name__})")

        self.client.create_collection(
            name=name,
            metadata={"created_at": datetime.utcnow().isoformat()}
        )
        logger.info(f"Recreated collection: {name}")

        return name
    
    def _get_collection(self, collection_name: Optional[str] = None):
        """Get or create a Chroma collection.
        
        Args:
            collection_name: Collection name, uses default if None
            
        Returns:
            Chroma collection instance or None if ChromaDB unavailable
        """
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available - collection operations disabled")
            return None
        
        name = collection_name or self.default_collection
        
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name)
            logger.debug(f"Found existing collection: {name}")
            return collection
        except Exception as e:
            # Collection doesn't exist, create it
            logger.info(f"Collection '{name}' not found ({type(e).__name__}), creating new one")
            collection = self.client.create_collection(
                name=name,
                metadata={"created_at": datetime.utcnow().isoformat()}
            )
            logger.info(f"Created new collection: {name}")
            
            # Record collection creation in metrics
            try:
                from .health.metrics import get_metrics_collector
                get_metrics_collector().record_collection_created(name)
            except Exception as metrics_error:
                logger.debug(f"Failed to record collection creation metric: {metrics_error}")
            
            return collection

    def _normalize_filter_metadata(
        self,
        filter_metadata: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Convert incoming metadata filters into Chroma-compatible where clauses."""
        if not filter_metadata or not isinstance(filter_metadata, dict):
            return filter_metadata

        logical_operators = {"$and", "$or"}
        if any(key in logical_operators for key in filter_metadata):
            normalized_filters = {}
            for key, value in filter_metadata.items():
                if key in logical_operators and isinstance(value, list):
                    normalized_filters[key] = [
                        self._normalize_filter_metadata(item) if isinstance(item, dict) else item
                        for item in value
                    ]
                else:
                    normalized_filters[key] = value
            return normalized_filters

        clauses = []
        for key, value in filter_metadata.items():
            if isinstance(value, dict) and any(str(operator).startswith("$") for operator in value):
                clauses.append({key: value})
            else:
                clauses.append({key: {"$eq": value}})

        if len(clauses) == 1:
            return clauses[0]

        return {"$and": clauses}
    
    async def index_documents(
        self,
        documents: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> Dict[str, Any]:
        """Index documents into the vector store.
        
        Args:
            documents: List of documents with 'content' and optional 'metadata'
            collection_name: Target collection name
            chunk_size: Text chunk size for large documents
            chunk_overlap: Overlap between chunks
            
        Returns:
            Indexing results with document IDs and chunk information
        """
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available - using mock indexing")
            # In mock mode, just return successful result without calling embeddings
            return {
                "indexed_count": len(documents),
                "chunk_count": len(documents),
                "collection_name": collection_name or self.default_collection,
                "embedding_model": "mock-model",
                "chunk_ids": [str(uuid.uuid4()) for _ in documents]
            }
        
        collection = self._get_collection(collection_name)
        chunk_size = chunk_size or settings.max_chunk_size
        chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # Process documents into chunks
        chunks = []
        chunk_metadata = []
        chunk_ids = []
        
        for doc_idx, document in enumerate(documents):
            content = document["content"]
            metadata = document.get("metadata", {})
            
            # Add document-level metadata
            doc_metadata = {
                **metadata,
                "document_index": doc_idx,
                "indexed_at": datetime.utcnow().isoformat(),
                "original_length": len(content)
            }
            
            # Split content into chunks if necessary
            if len(content) <= chunk_size:
                # Small document, index as single chunk
                chunk_id = str(uuid.uuid4())
                chunks.append(content)
                chunk_metadata.append(doc_metadata)
                chunk_ids.append(chunk_id)
            else:
                # Large document, split into chunks
                doc_chunks = self._split_text(content, chunk_size, chunk_overlap)
                for chunk_idx, chunk_text in enumerate(doc_chunks):
                    chunk_id = str(uuid.uuid4())
                    chunk_meta = {
                        **doc_metadata,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(doc_chunks),
                        "chunk_length": len(chunk_text)
                    }
                    
                    chunks.append(chunk_text)
                    chunk_metadata.append(chunk_meta)
                    chunk_ids.append(chunk_id)
        
        # Generate embeddings for all chunks
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        embedding_result = await generate_embeddings(chunks)
        embeddings = embedding_result["embeddings"]
        
        # Store in Chroma
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=chunk_metadata,
            ids=chunk_ids
        )
        
        logger.info(f"Indexed {len(documents)} documents as {len(chunks)} chunks")
        
        return {
            "indexed_count": len(documents),
            "chunk_count": len(chunks),
            "collection_name": collection.name,
            "embedding_model": embedding_result["model"],
            "chunk_ids": chunk_ids
        }
    
    async def search(
        self,
        query: str,
        k: int = 5,
        collection_name: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Search for similar documents.
        
        Args:
            query: Search query text
            k: Number of results to return
            collection_name: Collection to search
            filter_metadata: Metadata filters
            similarity_threshold: Minimum similarity score
            
        Returns:
            Search results with documents and similarity scores
        """
        
        # Check ChromaDB availability
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available - using mock search")
            return {
                "query": query,
                "results": [
                    {
                        "id": "mock-doc-1",
                        "content": f"Mock search result for query: {query[:50]}...",
                        "metadata": {"source": "mock", "type": "example"},
                        "similarity_score": 0.85,
                        "chunk_index": None
                    }
                ],
                "total_results": 1,
                "collection_name": collection_name or self.default_collection,
                "embedding_model": "mock-model",
                "search_time_ms": 10.0
            }
        
        # Real ChromaDB search path
        logger.info("✅ Using real ChromaDB search")
        try:
            collection = self._get_collection(collection_name)
            threshold = similarity_threshold or settings.similarity_threshold
        
            # Generate query embedding
            logger.info(f"Searching for: {query[:100]}...")
            start_time = datetime.utcnow()
        
            embedding_result = await generate_embeddings([query])
            query_embedding = embedding_result["embeddings"][0]
        
            # Prepare search parameters
            search_params = {
                "query_embeddings": [query_embedding],
                "n_results": min(k, settings.max_search_results)
            }
        
            # Add metadata filter if provided
            if filter_metadata:
                search_params["where"] = self._normalize_filter_metadata(filter_metadata)
        
            # Perform search
            results = collection.query(**search_params)
        
            # Process results
            search_results = []
            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
                distances = results["distances"][0] if results["distances"] else [0.0] * len(documents)
                ids = results["ids"][0] if results["ids"] else [str(i) for i in range(len(documents))]
                
                for doc, metadata, distance, doc_id in zip(documents, metadatas, distances, ids):
                    # Convert distance to similarity score (Chroma uses L2 distance)
                    similarity_score = 1.0 / (1.0 + distance)
                    
                    # Apply similarity threshold
                    if similarity_score < threshold:
                        continue
                    
                    search_results.append({
                        "id": doc_id,
                        "content": doc,
                        "metadata": metadata or {},
                        "similarity_score": float(similarity_score),
                        "chunk_index": metadata.get("chunk_index") if metadata else None
                    })
        
            # Calculate search time
            search_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
            logger.info(f"Found {len(search_results)} results in {search_time:.1f}ms")
        
            return {
                "query": query,
                "results": search_results,
                "total_results": len(search_results),
                "collection_name": collection.name,
                "embedding_model": embedding_result["model"],
                "search_time_ms": search_time
            }

        except Exception as e:
            logger.error(f"Search error: {e}")
            # Return empty results on error
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "collection_name": collection_name or self.default_collection,
                "embedding_model": "error",
                "search_time_ms": 0.0,
                "error": str(e)
            }
    
    def _split_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """Split text into overlapping chunks.
        
        Args:
            text: Text to split
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If this is not the last chunk, try to break at word boundary
            if end < len(text):
                # Look for the last space within the chunk to avoid breaking words
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position, accounting for overlap
            start = end - chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def get_collection_info(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a collection.
        
        Args:
            collection_name: Collection name, uses default if None
            
        Returns:
            Collection information
        """
        if not CHROMADB_AVAILABLE:
            return {
                "name": collection_name or self.default_collection,
                "document_count": 0,
                "embedding_dimension": 1536,
                "metadata": {"status": "mock - ChromaDB not available"}
            }
        
        collection = self._get_collection(collection_name)
        count = collection.count()
        
        # Get embedding dimension from first document if available
        dimension = 0
        if count > 0:
            sample = collection.get(limit=1, include=["embeddings"])
            if sample["embeddings"]:
                dimension = len(sample["embeddings"][0])
        
        return {
            "name": collection.name,
            "document_count": count,
            "embedding_dimension": dimension,
            "metadata": collection.metadata or {}
        }
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections in the vector store.
        
        Returns:
            List of collection information
        """
        if not CHROMADB_AVAILABLE:
            return [{
                "name": self.default_collection,
                "document_count": 0,
                "embedding_dimension": 1536,
                "metadata": {"status": "mock - ChromaDB not available"}
            }]
        
        collections = self.client.list_collections()
        
        collection_info = []
        for collection in collections:
            try:
                info = self.get_collection_info(collection.name)
                collection_info.append(info)
            except Exception as e:
                logger.warning(f"Failed to get info for collection {collection.name}: {e}")
                collection_info.append({
                    "name": collection.name,
                    "document_count": 0,
                    "embedding_dimension": 0,
                    "metadata": {},
                    "error": str(e)
                })
        
        return collection_info
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection.
        
        Args:
            collection_name: Name of collection to delete
            
        Returns:
            True if successful
        """
        if not CHROMADB_AVAILABLE:
            logger.info(f"Mock deletion of collection: {collection_name}")
            return True
        
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False


# Global vector store instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get the global vector store instance (singleton pattern).
    
    Returns:
        Vector store instance
    """
    global _vector_store
    
    if _vector_store is None:
        _vector_store = VectorStore()
        logger.info("Initialized global vector store")
    
    return _vector_store


def reset_vector_store():
    """Reset the global vector store instance.
    
    Useful for testing or configuration changes.
    """
    global _vector_store
    _vector_store = None
    logger.info("Reset vector store instance")