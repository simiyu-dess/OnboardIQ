#!/usr/bin/env python3
"""
Test script to verify Ollama connection and model availability
"""

from langchain_ollama import OllamaLLM
import requests

def test_ollama_connection():
    """Test if Ollama is running and accessible"""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            print("✅ Ollama is running and accessible")
            models = response.json().get('models', [])
            print(f"Available models: {[model['name'] for model in models]}")
            return True
        else:
            print(f"❌ Ollama responded with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama. Make sure it's running on localhost:11434")
        return False

def test_llama_model():
    """Test if the llama3.2 model can be loaded and used"""
    try:
        llm = OllamaLLM(
            model="llama3.2",
            base_url='http://localhost:11434',
            temperature=0.1
        )
        
        # Test a simple prompt
        response = llm.invoke("Say 'Hello, Ollama is working!'")
        print(f"✅ Model test successful: {response}")
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Ollama connection and model...")
    
    if test_ollama_connection():
        test_llama_model()
    else:
        print("Please start Ollama first: ollama serve") 