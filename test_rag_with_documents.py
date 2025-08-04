#!/usr/bin/env python3
"""
Test script to demonstrate RAG functionality with actual documents
"""

from rag_crew import RAGCrew
import tempfile
import os

def create_sample_document():
    """Create a sample document for testing"""
    sample_content = """
    ARTIFICIAL INTELLIGENCE RESEARCH PAPER
    
    Title: "Advances in Machine Learning for Natural Language Processing"
    
    Abstract:
    This paper presents recent developments in machine learning techniques applied to natural language processing (NLP) tasks. Our research focuses on transformer-based architectures and their applications in text generation and understanding.
    
    Introduction:
    Natural Language Processing has seen remarkable progress in recent years, primarily driven by the development of transformer architectures. The introduction of models like BERT, GPT, and T5 has revolutionized how we approach language understanding tasks.
    
    Key Findings:
    1. Transformer models achieve state-of-the-art performance on most NLP benchmarks
    2. Pre-training on large text corpora significantly improves downstream task performance
    3. Fine-tuning techniques allow adaptation to specific domains and tasks
    4. Attention mechanisms provide interpretable insights into model decisions
    
    Methodology:
    We conducted experiments using a dataset of 1 million text samples from various sources including news articles, scientific papers, and web content. Our baseline model was trained for 100 epochs with a learning rate of 1e-4.
    
    Results:
    Our experiments showed that the transformer model achieved 95.2% accuracy on the test set, outperforming previous approaches by 8.3 percentage points. The model demonstrated particularly strong performance on question-answering tasks, achieving 92.1% accuracy compared to 78.4% for traditional RNN-based models.
    
    Conclusion:
    Transformer-based models represent a significant advancement in NLP capabilities. Future work should focus on reducing computational requirements while maintaining performance gains.
    
    References:
    [1] Vaswani, A., et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).
    [2] Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." arXiv preprint arXiv:1810.04805 (2018).
    """
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_content)
        return f.name

def test_rag_with_documents():
    """Test the RAG crew with actual documents"""
    try:
        # Create sample document
        doc_path = create_sample_document()
        print(f"✅ Created sample document: {doc_path}")
        
        # Initialize RAG crew
        rag_crew = RAGCrew(model_name="llama3.2:latest")
        print("✅ RAG Crew initialization successful")
        
        # Load and process documents
        rag_crew.load_and_process_documents([doc_path])
        print("✅ Documents loaded and processed successfully")
        
        # Test document retrieval
        print("\n🔍 Testing document retrieval...")
        relevant_docs = rag_crew.query_documents("transformer models")
        print(f"✅ Retrieved {len(relevant_docs)} relevant document chunks")
        
        # Test RAG response generation
        print("\n🤖 Testing RAG response generation...")
        query = "What are the key findings about transformer models in the research?"
        result = rag_crew.generate_response(query)
        
        print(f"\n📝 Query: {query}")
        print(f"📄 Response: {result}")
        
        # Clean up
        os.unlink(doc_path)
        print(f"\n✅ Cleaned up temporary file: {doc_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing RAG functionality with actual documents...")
    test_rag_with_documents() 