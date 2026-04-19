#!/usr/bin/env python3
"""Debug Gemini API test performance."""

import asyncio
import time
from app.providers.factory import ProviderFactory
from app.config import settings

async def test_gemini_debug():
    """Debug Gemini API call step by step."""
    print("=== GEMINI API DEBUG TEST ===")
    
    print(f"Gemini API Key present: {bool(settings.gemini_api_key)}")
    print(f"Gemini API Key prefix: {settings.gemini_api_key[:10]}..." if settings.gemini_api_key else "None")
    
    try:
        # Step 1: Create provider
        print("\n1. Creating Gemini provider...")
        provider_config = {
            "api_key": settings.gemini_api_key,
            "default_model": settings.gemini_default_model,
            "timeout": settings.gemini_timeout
        } if settings.gemini_api_key else {}
        print(f"   Config: model={provider_config.get('default_model', 'default')}")
        provider = ProviderFactory.create_provider("gemini", provider_config)
        print(f"   Provider created: {type(provider).__name__}")
        
        # Step 2: Test embedding call
        print("\n2. Testing embedding call...")
        start_time = time.time()
        test_text = "This is a debug test for Gemini API performance measurement."
        
        result = await provider.embed([test_text])
        
        duration_ms = int((time.time() - start_time) * 1000)
        print(f"   Duration: {duration_ms}ms")
        print(f"   Result type: {type(result)}")
        print(f"   Result length: {len(result) if result else 0}")
        
        if result and len(result) > 0:
            embedding = result[0]
            print(f"   Embedding type: {type(embedding)}")
            print(f"   Embedding dimensions: {len(embedding) if hasattr(embedding, '__len__') else 'Unknown'}")
        else:
            print("   No embeddings returned!")
            
        # Step 3: Test models
        print("\n3. Testing available models...")
        models = provider.get_available_models()
        print(f"   Available models: {models}")
        
        return {
            "success": True,
            "duration_ms": duration_ms,
            "dimensions": len(result[0]) if result and len(result) > 0 else None,
            "models": models
        }
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    result = asyncio.run(test_gemini_debug())
    print(f"\n=== FINAL RESULT ===")
    print(result)