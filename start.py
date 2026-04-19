#!/usr/bin/env python3
"""Startup script for semantic search API development."""

import os
import sys
import subprocess
from pathlib import Path


def check_env_file():
    """Check if .env file exists and create from example if not."""
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if not env_path.exists() and env_example_path.exists():
        print("📝 Creating .env file from .env.example...")
        with open(env_example_path, 'r') as src:
            content = src.read()
        
        with open(env_path, 'w') as dst:
            dst.write(content)
        
        print("✅ .env file created. Please edit it to add your API keys.")
        print("💡 You can enable mock mode by setting USE_MOCK_EMBEDDINGS=true")
        return False
    
    return True


def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print(f"Error output: {e.stderr}")
        return False


def run_server():
    """Start the development server."""
    print("🚀 Starting development server...")
    print("📖 API documentation will be available at http://localhost:8000/docs")
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--reload",
            "--host", "0.0.0.0",
            "--port", "8000"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start server: {e}")


def main():
    """Main startup function."""
    print("🔍 Semantic Search API - Startup Script")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("app/main.py").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Check/create .env file
    env_ready = check_env_file()
    
    if not env_ready:
        response = input("\n🤔 Would you like to start in mock mode? (y/n): ").lower()
        if response in ['y', 'yes']:
            # Enable mock mode in .env
            with open(".env", "r") as f:
                content = f.read()
            
            content = content.replace("USE_MOCK_EMBEDDINGS=false", "USE_MOCK_EMBEDDINGS=true")
            
            with open(".env", "w") as f:
                f.write(content)
            
            print("✅ Mock mode enabled")
        else:
            print("📝 Please edit .env file and add your API keys, then run this script again")
            sys.exit(0)
    
    print("\n🎯 Starting server...")
    run_server()


if __name__ == "__main__":
    main()