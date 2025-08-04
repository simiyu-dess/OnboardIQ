#!/usr/bin/env python3
"""
Test script to verify custom LLM wrapper with CrewAI
"""

from rag_crew import CustomOllamaLLM, RAGCrew

def test_custom_llm():
    """Test the custom LLM wrapper"""
    try:
        llm = CustomOllamaLLM(model_name="llama3.2")
        response = llm.invoke("Say 'Hello, custom LLM is working!'")
        print(f"✅ Custom LLM test successful: {response}")
        return True
    except Exception as e:
        print(f"❌ Custom LLM test failed: {e}")
        return False

def test_rag_crew():
    """Test the RAG crew with custom LLM"""
    try:
        rag_crew = RAGCrew(model_name="llama3.2")
        print("✅ RAG Crew initialization successful")
        
        # Test a simple query without documents
        result = rag_crew.generate_response("What is artificial intelligence?")
        print(f"✅ RAG Crew test successful: {result}")
        return True
    except Exception as e:
        print(f"❌ RAG Crew test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing custom LLM wrapper and RAG crew...")
    
    if test_custom_llm():
        test_rag_crew()
    else:
        print("Custom LLM test failed, skipping RAG crew test") 