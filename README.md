# Semantic Search API

A FastAPI-based semantic search service that provides document indexing and similarity search using embeddings. Built with a modular provider architecture to support multiple embedding services (OpenAI, Gemini) and Chroma vector database for efficient similarity search.

## Architecture

This project follows the same provider architecture pattern as the sibling `llm-chat-api` project:

- **Provider Interface**: Abstract base class (`BaseEmbeddingProvider`) for consistent embedding provider implementations
- **Provider Factory**: Dynamic provider selection based on configuration
- **Environment-based Configuration**: Uses Pydantic settings with `.env` file loading
- **Standardized Responses**: Consistent response format across all providers
- **Error Handling**: Comprehensive error handling with provider-specific exceptions
- **Async Support**: Fully asynchronous Python with retry logic

### System Architecture Diagram

```mermaid
graph TB
    subgraph "FastAPI Application"
        API[FastAPI Router]
        
        subgraph "Health Monitoring System"
            H1[/health - Quick Status]
            H2[/diagnostics - Comprehensive Tests]
            H3[/metrics - Performance Data]
            H4[/health/simple - Basic Check]
            H5[/dashboard - Visual Interface]
        end
        
        subgraph "Core Services"
            EMB[Embedding Service]
            VEC[Vector Store]
            IDX[Document Indexer]
            SEARCH[Search Engine]
        end
        
        subgraph "Provider Layer"
            FACT[Provider Factory]
            OAI[OpenAI Provider]
            GEM[Gemini Provider]
            MOCK[Mock Provider]
        end
        
        subgraph "External Services"
            OAPI[OpenAI API]
            GAPI[Gemini API]
            CHROMA[(ChromaDB)]
        end
    end
    
    API --> H1
    API --> H2
    API --> H3
    API --> H4
    API --> H5
    API --> EMB
    API --> IDX
    API --> SEARCH
    
    H5 --> H1
    H5 --> H2
    H2 --> FACT
    H2 --> VEC
    H2 --> EMB
    
    EMB --> FACT
    IDX --> EMB
    IDX --> VEC
    SEARCH --> EMB
    SEARCH --> VEC
    
    FACT --> OAI
    FACT --> GEM
    FACT --> MOCK
    
    OAI --> OAPI
    GEM --> GAPI
    VEC --> CHROMA
    
    classDef health fill:#e1f5fe
    classDef core fill:#f3e5f5
    classDef provider fill:#e8f5e8
    classDef external fill:#fff3e0
    
    class H1,H2,H3,H4,H5 health
    class EMB,VEC,IDX,SEARCH core
    class FACT,OAI,GEM,MOCK provider
    class OAPI,GAPI,CHROMA external
```

## Features

- 🚀 **Fast API**: Built with FastAPI for high performance and automatic OpenAPI documentation
- 🔌 **Multiple Providers**: Support for OpenAI and Gemini embedding models
- 📊 **Vector Database**: Chroma integration for efficient similarity search
- 🏗️ **Document Processing**: Automatic text chunking for large documents
- 🔍 **Semantic Search**: Similarity-based document retrieval
- ⚙️ **Flexible Configuration**: Environment-based provider and model selection
- 🛡️ **Error Handling**: Comprehensive error handling and retry logic
- 📝 **Structured Logging**: Request tracking and performance monitoring
- 🏥 **Health Monitoring**: Comprehensive health checks with diagnostics and performance benchmarking
- 📈 **Visual Dashboard**: Real-time web interface for system monitoring with auto-refresh
- 🧪 **Testing Ready**: Mock providers for development and testing

## Project Structure

```
semantic-search-api/
├── app/
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract provider interface
│   │   ├── factory.py           # Provider factory
│   │   ├── openai_provider.py   # OpenAI embedding provider
│   │   └── gemini_provider.py   # Gemini embedding provider
│   ├── health/
│   │   ├── __init__.py
│   │   ├── basic_health.py      # Fast health checks
│   │   ├── diagnostics.py       # Comprehensive diagnostics
│   │   └── metrics.py           # Performance metrics
│   ├── main.py                  # FastAPI application
│   ├── config.py                # Pydantic settings
│   ├── models.py                # Request/response models
│   ├── embeddings.py            # Embedding utilities
│   └── vectorstore.py           # Chroma vector database
├── tests/
│   └── test_basic.py            # Basic API tests
├── data/
│   └── sample_docs/             # Sample documents for testing
├── requirements.txt
├── .env.example
└── README.md
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
cd semantic-search-api

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` file:

```env
# Choose embedding provider
EMBEDDING_PROVIDER=openai

# Add your API key
OPENAI_API_KEY=your_openai_api_key_here
# OR for Gemini
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Enable mock mode for testing without API keys
USE_MOCK_EMBEDDINGS=false
```

### 3. Run the API

```bash
# Development server
uvicorn app.main:app --reload

# Production server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

## API Endpoints

### Health Monitoring

- `GET /health` - **Comprehensive health check** with startup validation, runtime checks, and configuration status
- `GET /diagnostics` - **Full system diagnostics** with real API calls, performance benchmarks, and functional tests
- `GET /metrics` - **Performance metrics** including request counts, response times, and system statistics
- `GET /health/simple` - **Basic health check** for load balancers (simple OK/ERROR response)
- `GET /dashboard` - **Visual health dashboard** with real-time monitoring interface and auto-refresh

### Core Endpoints

- `GET /` - Service information and available endpoints
- `POST /embed` - Generate embeddings for texts
- `POST /index` - Index documents into vector store
- `POST /search` - Search for similar documents
- `GET /collections` - List available collections
- `GET /models` - List available embedding models

### Health Monitoring Details

#### Quick Health Check (`/health`)
Returns comprehensive status in ~200ms:
```json
{
  "status": "healthy",
  "timestamp": "2026-04-19T12:00:00Z",
  "startup": {
    "required_vars_present": true,
    "providers_loaded": true,
    "vectorstore_initialized": true
  },
  "runtime": {
    "memory_usage_mb": 156.7,
    "uptime_seconds": 3600,
    "active_connections": 5
  },
  "configuration": {
    "embedding_provider": "openai",
    "openai_key": "sk-proj****...****Tao8A",
    "gemini_key": "AQ.Ab8R****...****8alQ",
    "vectorstore": "chroma"
  }
}
```

#### Comprehensive Diagnostics (`/diagnostics`)
Performs real API calls and benchmarks (~3-5 seconds):
```json
{
  "status": "healthy",
  "total_duration_ms": 3247,
  "api_tests": {
    "openai": {
      "status": "healthy",
      "duration_ms": 871,
      "test_embedding_dimensions": 1536,
      "models_available": ["text-embedding-3-small", "text-embedding-3-large"]
    },
    "gemini": {
      "status": "healthy", 
      "duration_ms": 541,
      "test_embedding_dimensions": 3072,
      "models_available": ["models/gemini-embedding-001"]
    }
  },
  "performance_benchmarks": {
    "single_embedding_ms": "275.8ms",
    "batch_embedding_ms": "323.9ms", 
    "embedding_throughput": "15.4 texts/sec",
    "vector_search_ms": "12.3ms",
    "status": "completed"
  },
  "functional_tests": {
    "vector_operations": {
      "status": "healthy",
      "duration_ms": 156,
      "test_steps": {
        "create_collection": "passed",
        "index_documents": "passed", 
        "search_similarity": "passed",
        "cleanup": "passed"
      }
    }
  }
}
```

#### Visual Health Dashboard (`/dashboard`)
Interactive web interface for real-time system monitoring:

**Features:**
- 🎨 **Modern UI**: Glass-morphism design with gradient background and responsive cards
- 📊 **Real-time Data**: Auto-refreshes every 30 seconds with live health status
- 🚦 **Status Indicators**: Color-coded status with emoji indicators (🟢 healthy, 🟡 degraded, 🔴 critical)
- 📱 **Responsive Design**: Optimized for both desktop and mobile viewing
- ⚡ **Fast Loading**: Fetches data from `/health` and `/diagnostics` endpoints

**Dashboard Cards:**
1. **System Health**: API keys, ChromaDB connection, provider status
2. **Runtime Metrics**: Collections count, response times, vector store status
3. **API Performance**: Live response times for OpenAI/Gemini APIs with dimensions
4. **Performance Benchmarks**: Embedding speeds, throughput, search performance

Access the dashboard at: `http://localhost:8000/dashboard`

### Example Usage

#### 1. Generate Embeddings

```bash
curl -X POST "http://localhost:8000/embed" \\
     -H "Content-Type: application/json" \\
     -d '{
       "texts": ["Hello world", "Machine learning is amazing"],
       "model": "text-embedding-3-small"
     }'
```

#### 2. Index Documents

```bash
curl -X POST "http://localhost:8000/index" \\
     -H "Content-Type: application/json" \\
     -d '{
       "documents": [
         {
           "content": "FastAPI is a modern web framework for building APIs.",
           "metadata": {"source": "documentation", "topic": "web development"}
         },
         {
           "content": "Vector databases enable semantic search capabilities.",
           "metadata": {"source": "blog", "topic": "databases"}
         }
       ]
     }'
```

#### 3. Search Documents

```bash
curl -X POST "http://localhost:8000/search" \\
     -H "Content-Type: application/json" \\
     -d '{
       "query": "Python web frameworks",
       "k": 5,
       "similarity_threshold": 0.1
     }'
```

## Configuration

### Provider Settings

Configure embedding providers in `.env`:

```env
# Provider selection
EMBEDDING_PROVIDER=openai  # or gemini

# OpenAI settings
OPENAI_API_KEY=your_openai_key_here
OPENAI_DEFAULT_MODEL=text-embedding-3-small
OPENAI_TIMEOUT=30.0

# Gemini settings  
GEMINI_API_KEY=your_gemini_key_here
GEMINI_DEFAULT_MODEL=models/gemini-embedding-001
GEMINI_TIMEOUT=30.0

# Health monitoring
USE_MOCK_EMBEDDINGS=false  # Set to true for development without API keys
```

### Available Models

#### OpenAI Models
- `text-embedding-3-small` (1536 dimensions) - Recommended for most use cases
- `text-embedding-3-large` (3072 dimensions) - Higher quality, more expensive
- `text-embedding-ada-002` (1536 dimensions) - Legacy model

#### Gemini Models  
- `models/gemini-embedding-001` (768 dimensions) - Standard embedding model
- `models/gemini-embedding-2-preview` (768 dimensions) - Preview model with improvements

**Note**: Model names must include the `models/` prefix for Gemini providers.

### Vector Database Settings

```env
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=documents
```

### Document Processing

```env
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=200
BATCH_SIZE=10
```

## Provider Architecture

### Adding New Providers

1. Create a new provider class inheriting from `BaseEmbeddingProvider`:

```python
from app.providers.base import BaseEmbeddingProvider

class CustomEmbeddingProvider(BaseEmbeddingProvider):
    async def embed(self, texts, model=None, **kwargs):
        # Implementation here
        pass
    
    def get_available_models(self):
        return ["custom-model-1", "custom-model-2"]
    
    def validate_config(self):
        # Validate configuration
        pass
```

2. Register the provider in `factory.py`:

```python
from .custom_provider import CustomEmbeddingProvider

class ProviderFactory:
    _providers = {
        "openai": OpenAIEmbeddingProvider,
        "gemini": GeminiEmbeddingProvider,
        "custom": CustomEmbeddingProvider,  # Add here
    }
```

3. Add configuration settings to `config.py` and `.env.example`.

## Development

### Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_basic.py -v
```

### Mock Mode

For development without API keys:

```env
USE_MOCK_EMBEDDINGS=true
```

This enables mock embeddings that generate consistent vectors for testing.

### Debug Mode

```env
DEBUG=true
```

Enables additional debug endpoints and detailed logging.

## Vector Database

The API uses Chroma as the vector database for storing and searching embeddings:

- **Persistent Storage**: Data persists across restarts
- **Collection Management**: Support for multiple document collections
- **Metadata Filtering**: Search with metadata constraints
- **Efficient Search**: Fast similarity search with configurable thresholds

### Data Flow

1. **Indexing**: Documents → Text Chunking → Embeddings → Chroma Storage
2. **Search**: Query → Query Embedding → Similarity Search → Ranked Results

## Performance

- **Async Processing**: Non-blocking I/O for concurrent requests
- **Batch Processing**: Efficient batch embedding generation
- **Retry Logic**: Automatic retry for transient failures
- **Connection Pooling**: Optimized API client connections
- **Caching**: Provider instances cached as singletons

## Error Handling

The API provides comprehensive error handling:

- **Provider Errors**: Specific handling for each embedding provider
- **Rate Limiting**: Graceful handling of API rate limits
- **Configuration Errors**: Clear messages for setup issues
- **Request Validation**: Input validation with helpful error messages

## Health Monitoring System

The API includes a comprehensive 4-tier health monitoring system:

### 1. Basic Health Check (`/health/simple`)
- **Purpose**: Load balancer health checks
- **Response Time**: ~5-10ms
- **Format**: Plain text "OK" or "ERROR"
- **Use Case**: Kubernetes liveness probes, AWS ALB health checks

### 2. Comprehensive Health (`/health`)  
- **Purpose**: Detailed system status
- **Response Time**: ~100-200ms
- **Features**:
  - Startup validation (API keys, providers, database)
  - Runtime metrics (memory, uptime, connections)
  - Configuration status (masked API keys)
  - Intelligent status determination
- **Use Case**: Monitoring dashboards, admin interfaces

### 3. Full Diagnostics (`/diagnostics`)
- **Purpose**: Deep system validation
- **Response Time**: ~3-5 seconds
- **Features**:
  - Real API calls to OpenAI and Gemini
  - Performance benchmarking (embedding speed, throughput)
  - Functional testing (end-to-end vector operations)
  - Embedding dimension validation
- **Use Case**: Deployment validation, troubleshooting

### 4. Performance Metrics (`/metrics`)
- **Purpose**: Operational statistics
- **Response Time**: ~50-100ms  
- **Features**:
  - Request counts by endpoint
  - Response time percentiles
  - Error rates and types
  - Resource utilization
- **Use Case**: Performance monitoring, alerting

### Health Status Logic

The system uses intelligent status determination:
- **healthy**: All components working normally
- **degraded**: Some issues but service functional (e.g., one provider down)
- **critical**: Major issues affecting service
- **error**: System unable to respond properly

### Monitoring Integration

Perfect for integration with:
- **Prometheus**: Metrics endpoint provides structured data
- **Grafana**: Create dashboards from health data
- **DataDog**: Custom metrics and health checks
- **AWS CloudWatch**: Custom metrics and alarms
- **Kubernetes**: Health checks and readiness probes

## Dashboard UI Options

### Option 1: Static HTML Dashboard (Simplest)

**Implementation Complexity**: ⭐ (1/5)
**Development Time**: ~30 minutes
**Files Needed**: 1 HTML file + 1 endpoint

```python
# Add to main.py
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Health Dashboard</title></head>
    <body>
        <div id="health-status">Loading...</div>
        <script>
        async function updateHealth() {
            const health = await fetch('/health').then(r => r.json());
            document.getElementById('health-status').innerHTML = 
                `Status: ${health.status}`;
        }
        setInterval(updateHealth, 30000); // Update every 30s
        updateHealth();
        </script>
    </body>
    </html>
    """ 
```

### Option 2: Streamlit Dashboard

**Implementation Complexity**: ⭐⭐⭐ (3/5)
**Development Time**: ~2-3 hours
**Files Needed**: 1-2 Python files + requirements update

```python
# dashboard.py
import streamlit as st
import requests
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="API Health Dashboard", layout="wide")

# Sidebar
st.sidebar.title("Health Dashboard")
refresh_rate = st.sidebar.selectbox("Refresh Rate", [30, 60, 120])

# Main dashboard
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("System Health")
    health_data = requests.get("http://localhost:8000/health").json()
    
    # Status indicator
    status_color = {"healthy": "🟢", "degraded": "🟡", "critical": "🔴"}
    st.metric("Status", f"{status_color.get(health_data['status'], '⚪')} {health_data['status']}")
    
    st.metric("Uptime", f"{health_data['runtime']['uptime_seconds']}s")
    st.metric("Memory", f"{health_data['runtime']['memory_usage_mb']:.1f} MB")

with col2:
    st.subheader("API Performance")
    diag_data = requests.get("http://localhost:8000/diagnostics").json()
    
    # API test results
    for provider, test in diag_data['api_tests'].items():
        st.metric(f"{provider.title()} API", f"{test['duration_ms']}ms", 
                 f"{test['test_embedding_dimensions']} dims")

with col3:
    st.subheader("Performance Benchmarks")
    benchmarks = diag_data['performance_benchmarks']
    
    st.metric("Single Embedding", benchmarks['single_embedding_ms'])
    st.metric("Batch Processing", benchmarks['batch_embedding_ms'])
    st.metric("Throughput", benchmarks['embedding_throughput'])

# Charts
st.subheader("Performance Trends")
# Would need historical data storage for real trends
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1,2,3], y=[100,150,120], name="Response Time"))
st.plotly_chart(fig)

# Auto-refresh
if st.button(f"Auto-refresh every {refresh_rate}s"):
    st.rerun()
```

**Setup Requirements**:
```bash
pip install streamlit plotly
```

**Run Command**:
```bash
streamlit run dashboard.py --server.port 8501
```

### Comparison Summary

| Feature | Static HTML | Streamlit |
|---------|-------------|-----------|
| Setup Complexity | ⭐ | ⭐⭐⭐ |
| Real-time Updates | Manual refresh | Auto-refresh |
| Charts/Graphs | Basic JS charts | Professional Plotly |
| Styling | Custom CSS | Built-in themes |
| Interactivity | Limited | Rich widgets |
| Historical Data | Not included | Easy to add |
| Mobile Responsive | Manual CSS | Automatic |
| Development Time | 30 minutes | 2-3 hours |

**Recommendation**: Start with Static HTML for immediate value, upgrade to Streamlit if you need rich visualizations.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-provider`
3. Make changes following the existing patterns
4. Add tests for new functionality
5. Ensure tests pass: `pytest`
6. Submit a pull request

## License

This project is licensed under the MIT License.

## Related Projects

- **llm-chat-api**: Sibling project providing chat completions with the same provider architecture