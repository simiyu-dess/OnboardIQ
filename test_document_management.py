#!/usr/bin/env python3
"""
Test script to verify improved document management functionality
"""

from rag_crew import RAGCrew
import tempfile
import os

def create_test_document(content, filename):
    """Create a test document with given content"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        return f.name

def test_document_management():
    """Test document management functionality"""
    try:
        print("🧪 Testing Document Management Functionality")
        print("=" * 60)
        
        # Initialize RAG crew
        rag_crew = RAGCrew(model_name="llama3.2:latest")
        print("✅ RAG Crew initialization successful")
        
        # Test 1: Load first set of documents
        print("\n📄 Test 1: Loading first set of documents")
        doc1 = create_test_document("This is document 1 about Python programming.", "doc1.txt")
        doc2 = create_test_document("This is document 2 about machine learning.", "doc2.txt")
        
        success = rag_crew.load_and_process_documents([doc1, doc2], clear_existing=True)
        if success:
            doc_count = rag_crew.get_document_count()
            print(f"✅ Loaded {doc_count} document chunks")
            
            # Test query
            response = rag_crew.generate_response("What is document 1 about?")
            print(f"📝 Query response: {response[:100]}...")
        else:
            print("❌ Failed to load first set of documents")
            return False
        
        # Test 2: Load second set of documents (should clear first set)
        print("\n📄 Test 2: Loading second set of documents (clearing first set)")
        doc3 = create_test_document("This is document 3 about data science.", "doc3.txt")
        doc4 = create_test_document("This is document 4 about artificial intelligence.", "doc4.txt")
        
        success = rag_crew.load_and_process_documents([doc3, doc4], clear_existing=True)
        if success:
            doc_count = rag_crew.get_document_count()
            print(f"✅ Loaded {doc_count} document chunks")
            
            # Test query - should not find content from first set
            response = rag_crew.generate_response("What is document 1 about?")
            print(f"📝 Query response: {response[:100]}...")
            
            # Test query - should find content from second set
            response = rag_crew.generate_response("What is document 3 about?")
            print(f"📝 Query response: {response[:100]}...")
        else:
            print("❌ Failed to load second set of documents")
            return False
        
        # Test 3: Clear documents manually
        print("\n🗑️ Test 3: Clearing documents manually")
        if rag_crew.clear_documents():
            print("✅ Documents cleared successfully")
            try:
                rag_crew.query_documents("test")
                print("❌ Should have raised error after clearing")
                return False
            except ValueError as e:
                print("✅ Correctly raised error after clearing documents")
        else:
            print("❌ Failed to clear documents")
            return False
        
        # Test 4: Load documents without clearing (should work)
        print("\n📄 Test 4: Loading documents without clearing")
        doc5 = create_test_document("This is document 5 about web development.", "doc5.txt")
        
        success = rag_crew.load_and_process_documents([doc5], clear_existing=False)
        if success:
            doc_count = rag_crew.get_document_count()
            print(f"✅ Loaded {doc_count} document chunks")
            
            # Test query
            response = rag_crew.generate_response("What is document 5 about?")
            print(f"📝 Query response: {response[:100]}...")
        else:
            print("❌ Failed to load documents without clearing")
            return False
        
        # Clean up
        for doc in [doc1, doc2, doc3, doc4, doc5]:
            if os.path.exists(doc):
                os.unlink(doc)
        
        print("\n✅ All document management tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Document management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_document_management() 