#!/usr/bin/env python3
"""Check available Gemini models."""

import asyncio
from app.config import settings

async def list_gemini_models():
    """List available Gemini models to find correct embedding model name."""
    try:
        import google.generativeai as genai
        
        print("=== GEMINI AVAILABLE MODELS ===")
        print(f"API Key: {settings.gemini_api_key[:10]}..." if settings.gemini_api_key else "None")
        
        # Configure Gemini
        genai.configure(api_key=settings.gemini_api_key)
        
        # List all models
        print("\nListing all available models...")
        models = genai.list_models()
        
        print("\nAll Models:")
        for model in models:
            print(f"  - {model.name}")
            if hasattr(model, 'supported_generation_methods'):
                methods = model.supported_generation_methods
                print(f"    Methods: {methods}")
                if 'embedContent' in methods:
                    print(f"    ✅ SUPPORTS EMBEDDING")
        
        # Try to find embedding-specific models
        embedding_models = []
        for model in models:
            if hasattr(model, 'supported_generation_methods') and 'embedContent' in model.supported_generation_methods:
                embedding_models.append(model.name)
        
        print(f"\n🎯 EMBEDDING MODELS ({len(embedding_models)}):")
        for model in embedding_models:
            print(f"  ✅ {model}")
            
        return embedding_models
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    models = asyncio.run(list_gemini_models())
    print(f"\n=== SUMMARY ===")
    print(f"Available embedding models: {len(models)}")
    if models:
        print(f"Recommended model: {models[0]}")