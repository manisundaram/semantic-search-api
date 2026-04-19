import requests
import time

print('🔥 FINAL TEST: HTTP API with Real ChromaDB')
print('=' * 50)

# 1. Index documents via HTTP API
print('1. Indexing via HTTP API...')
docs = [
    {'content': 'FastAPI makes building REST APIs simple and fast', 'metadata': {'framework': 'fastapi'}},
    {'content': 'ChromaDB provides vector database capabilities', 'metadata': {'framework': 'chromadb'}},
    {'content': 'OpenAI embeddings enable semantic search', 'metadata': {'framework': 'openai'}}
]

start = time.time()
index_response = requests.post('http://localhost:8000/index', json={
    'documents': docs,
    'collection_name': 'final_test'
})
index_time = time.time() - start

if index_response.status_code == 200:
    index_result = index_response.json()
    print(f'   Success! Indexed {index_result.get("indexed_count", "N/A")} docs in {index_time:.2f}s')
else:
    print(f'   Error {index_response.status_code}: {index_response.text[:100]}')

# 2. Search via HTTP API  
print()
print('2. Searching via HTTP API...')
start = time.time()
search_response = requests.post('http://localhost:8000/search', json={
    'query': 'fast API development',
    'collection_name': 'final_test',
    'limit': 3
})
search_time = time.time() - start

if search_response.status_code == 200:
    search_result = search_response.json()
    print(f'   Search completed in {search_time:.2f}s')
    print(f'   Model: {search_result.get("embedding_model", "N/A")}')
    print(f'   Results: {search_result.get("total_results", "N/A")}')
    
    if search_result.get('results'):
        print('   📊 TOP RESULT:')
        top = search_result['results'][0]
        print(f'      Content: {top["content"]}')
        print(f'      Score: {top.get("similarity_score", 0):.3f}')
        print(f'      Framework: {top.get("metadata", {}).get("framework", "N/A")}')
    
    # Final determination
    is_real = search_result.get('embedding_model') not in ['mock-model', None]
    print()
    if is_real:
        print('🎉 VICTORY! Real ChromaDB + OpenAI API working!')
        print('✅ Complete semantic search pipeline operational!')
    else:
        print('❌ Still mock mode - need more debugging')
else:
    print(f'   Error {search_response.status_code}: {search_response.text[:100]}')