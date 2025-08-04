#!/usr/bin/env python3
"""
Test script to demonstrate out-of-context question handling
"""

from rag_crew import RAGCrew
import tempfile
import os

def create_sample_documents():
    """Create sample documents for testing"""
    # Sample CV document
    cv_content = """
    JOHN DOE - SOFTWARE ENGINEER
    
    CONTACT INFORMATION:
    Email: john.doe@email.com
    Phone: (555) 123-4567
    Location: San Francisco, CA
    
    PROFESSIONAL SUMMARY:
    Experienced software engineer with 5+ years in full-stack development.
    
    WORK EXPERIENCE:
    Senior Software Engineer - TechCorp Inc. (2020-2023)
    - Developed scalable web applications using React and Node.js
    - Led a team of 3 junior developers
    - Improved application performance by 40%
    
    Software Engineer - StartupXYZ (2018-2020)
    - Built REST APIs using Python and Django
    - Implemented CI/CD pipelines
    - Reduced deployment time by 60%
    
    EDUCATION:
    Bachelor of Science in Computer Science
    University of California, Berkeley (2014-2018)
    
    SKILLS:
    - Programming Languages: Python, JavaScript, Java, C++
    - Frameworks: React, Django, Spring Boot
    - Databases: PostgreSQL, MongoDB, Redis
    - Tools: Git, Docker, AWS, Kubernetes
    """
    
    # Sample job description
    job_content = """
    SENIOR SOFTWARE ENGINEER - FULL STACK
    
    Company: InnovationTech
    Location: Remote / San Francisco
    Type: Full-time
    
    JOB DESCRIPTION:
    We are seeking a Senior Software Engineer to join our growing team.
    
    RESPONSIBILITIES:
    - Design and implement scalable web applications
    - Collaborate with cross-functional teams
    - Mentor junior developers
    - Participate in code reviews and technical discussions
    - Contribute to architectural decisions
    
    REQUIREMENTS:
    - 5+ years of software development experience
    - Proficiency in Python, JavaScript, and React
    - Experience with cloud platforms (AWS, GCP, or Azure)
    - Strong problem-solving and communication skills
    - Bachelor's degree in Computer Science or related field
    
    PREFERRED QUALIFICATIONS:
    - Experience with microservices architecture
    - Knowledge of DevOps practices
    - Experience with machine learning or data science
    - Open source contributions
    
    BENEFITS:
    - Competitive salary and equity
    - Health, dental, and vision insurance
    - Flexible work arrangements
    - Professional development opportunities
    """
    
    # Create temporary files
    cv_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    cv_file.write(cv_content)
    cv_file.close()
    
    job_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    job_file.write(job_content)
    job_file.close()
    
    return cv_file.name, job_file.name

def test_out_of_context_handling():
    """Test how the RAG system handles out-of-context questions"""
    try:
        print("🧪 Testing Out-of-Context Question Handling")
        print("=" * 60)
        
        # Create sample documents
        cv_path, job_path = create_sample_documents()
        print(f"📄 Created sample documents: CV and Job Description")
        
        # Initialize RAG crew
        rag_crew = RAGCrew(model_name="llama3.2:latest")
        print("✅ RAG Crew initialization successful")
        
        # Load documents
        success = rag_crew.load_and_process_documents([cv_path, job_path], clear_existing=True)
        if not success:
            print("❌ Failed to load documents")
            return False
        
        print(f"✅ Loaded {rag_crew.get_document_count()} document chunks")
        
        # Test out-of-context questions
        out_of_context_queries = [
            "Who is Collins?",
            "What is Sarah's experience?",
            "Tell me about Michael's education",
            "What company did Jennifer work for?",
            "What is the salary for this position?",
            "What are the working hours?",
            "Does this company offer remote work?",
            "What is the vacation policy?",
            "Who is the hiring manager?",
            "What is the company's mission statement?"
        ]
        
        print("\n🔍 Testing Out-of-Context Questions:")
        print("-" * 40)
        
        for i, query in enumerate(out_of_context_queries, 1):
            print(f"\n{i}. Query: {query}")
            print("📝 Response:")
            
            try:
                response = rag_crew.generate_response(query)
                print(response)
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print("-" * 40)
        
        # Test in-context questions for comparison
        in_context_queries = [
            "What is John Doe's experience?",
            "What skills does John have?",
            "What are the job requirements?",
            "What company did John work for?",
            "What is John's educational background?"
        ]
        
        print("\n✅ Testing In-Context Questions:")
        print("-" * 40)
        
        for i, query in enumerate(in_context_queries, 1):
            print(f"\n{i}. Query: {query}")
            print("📝 Response:")
            
            try:
                response = rag_crew.generate_response(query)
                print(response[:200] + "..." if len(response) > 200 else response)
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print("-" * 40)
        
        # Clean up
        os.unlink(cv_path)
        os.unlink(job_path)
        
        print("\n✅ Out-of-context handling test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Out-of-context test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_out_of_context_handling() 