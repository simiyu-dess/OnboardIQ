#!/usr/bin/env python3
"""
Test script to verify CrewAI works without explicit LLM configuration
"""

from rag_crew import RAGCrew

def test_simple_crew():
    """Test the RAG crew without explicit LLM configuration"""
    try:
        rag_crew = RAGCrew(model_name="llama3.2")
        print("✅ RAG Crew initialization successful")
        
        # Test a simple query without documents
        result = rag_crew.generate_response("What is artificial intelligence?")
        print(f"✅ RAG Crew test successful: {result}")
        return True
    except Exception as e:
        print(f"❌ RAG Crew test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing simple RAG crew...")
    test_simple_crew() 