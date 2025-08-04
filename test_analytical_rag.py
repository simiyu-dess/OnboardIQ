#!/usr/bin/env python3
"""
Test script to demonstrate enhanced analytical RAG capabilities
"""

from rag_crew import RAGCrew
import tempfile
import os

def create_sample_cv():
    """Create a sample CV for testing"""
    cv_content = """
    CURRICULUM VITAE
    
    PERSONAL INFORMATION
    Name: John Smith
    Email: john.smith@email.com
    Phone: +1-555-0123
    Location: San Francisco, CA
    
    PROFESSIONAL SUMMARY
    Experienced software engineer with 8 years of experience in full-stack development, 
    specializing in Python, JavaScript, and cloud technologies. Proven track record of 
    leading development teams and delivering scalable applications.
    
    WORK EXPERIENCE
    
    Senior Software Engineer
    TechCorp Inc., San Francisco, CA
    January 2022 - Present
    
    • Led development of microservices architecture serving 1M+ users
    • Managed team of 5 developers and implemented CI/CD pipelines
    • Technologies: Python, Django, React, AWS, Docker, Kubernetes
    • Achieved 40% improvement in application performance
    
    Software Engineer
    StartupXYZ, San Francisco, CA
    March 2020 - December 2021
    
    • Developed REST APIs and frontend applications
    • Worked with Python, Flask, JavaScript, Vue.js
    • Collaborated with product team on feature development
    • Reduced bug reports by 30% through improved testing
    
    Junior Developer
    OldTech Solutions, San Francisco, CA
    June 2018 - February 2020
    
    • Maintained legacy applications and fixed bugs
    • Technologies: Java, Spring Boot, MySQL
    • Participated in code reviews and team meetings
    
    EDUCATION
    
    Bachelor of Science in Computer Science
    University of California, Berkeley
    Graduated: 2018
    GPA: 3.8/4.0
    
    CERTIFICATIONS
    
    • AWS Certified Solutions Architect (2023)
    • Google Cloud Professional Developer (2022)
    • Certified Scrum Master (2021)
    
    TECHNICAL SKILLS
    
    Programming Languages: Python, JavaScript, Java, SQL
    Frameworks: Django, Flask, React, Vue.js, Spring Boot
    Cloud Platforms: AWS, Google Cloud Platform
    Tools: Docker, Kubernetes, Git, Jenkins
    Databases: PostgreSQL, MySQL, MongoDB
    
    LANGUAGES
    English (Native), Spanish (Conversational)
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(cv_content)
        return f.name

def create_job_description():
    """Create a sample job description for testing"""
    job_content = """
    SENIOR SOFTWARE ENGINEER - FULL STACK
    
    Company: InnovationTech Solutions
    Location: San Francisco, CA
    Type: Full-time
    
    JOB DESCRIPTION
    
    We are seeking a Senior Software Engineer to join our growing team. You will be 
    responsible for developing and maintaining scalable web applications and APIs.
    
    RESPONSIBILITIES
    
    • Design and implement scalable microservices architecture
    • Lead development team of 3-5 engineers
    • Develop REST APIs and frontend applications
    • Implement CI/CD pipelines and DevOps practices
    • Collaborate with product and design teams
    • Mentor junior developers
    • Participate in code reviews and technical discussions
    
    REQUIREMENTS
    
    • 5+ years of experience in software development
    • Strong proficiency in Python and JavaScript
    • Experience with modern frameworks (Django, React, Vue.js)
    • Knowledge of cloud platforms (AWS, GCP, Azure)
    • Experience with containerization (Docker, Kubernetes)
    • Understanding of database design and SQL
    • Experience with version control (Git)
    • Strong problem-solving and communication skills
    • Bachelor's degree in Computer Science or related field
    
    PREFERRED QUALIFICATIONS
    
    • Experience with microservices architecture
    • Knowledge of DevOps practices and CI/CD
    • Experience leading development teams
    • Cloud certifications (AWS, GCP)
    • Experience with agile methodologies
    • Knowledge of testing frameworks and practices
    
    BENEFITS
    
    • Competitive salary and equity
    • Health, dental, and vision insurance
    • 401(k) matching
    • Flexible work arrangements
    • Professional development opportunities
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(job_content)
        return f.name

def test_analytical_queries():
    """Test analytical queries with CV and job description"""
    try:
        # Create sample documents
        cv_path = create_sample_cv()
        job_path = create_job_description()
        print(f"✅ Created sample CV: {cv_path}")
        print(f"✅ Created sample job description: {job_path}")
        
        # Initialize RAG crew
        rag_crew = RAGCrew(model_name="llama3.2:latest")
        print("✅ RAG Crew initialization successful")
        
        # Load and process documents
        rag_crew.load_and_process_documents([cv_path, job_path])
        print("✅ Documents loaded and processed successfully")
        
        # Test analytical queries
        analytical_queries = [
            "Is this candidate a good fit for the Senior Software Engineer position? Provide a detailed assessment and recommendation.",
            "Has the candidate ever worked at TechCorp Inc.? If yes, what was their role and duration?",
            "Evaluate the candidate's qualifications against the job requirements. What are their strengths and weaknesses?",
            "What is your recommendation for hiring this candidate? Include specific evidence from their CV.",
            "Compare the candidate's experience with the job requirements. What gaps exist and how significant are they?"
        ]
        
        for i, query in enumerate(analytical_queries, 1):
            print(f"\n{'='*60}")
            print(f"ANALYTICAL QUERY {i}: {query}")
            print(f"{'='*60}")
            
            try:
                response = rag_crew.generate_response(query)
                print(f"\n📝 Response:")
                print(response)
                print(f"\n✅ Query {i} completed successfully")
            except Exception as e:
                print(f"❌ Error in query {i}: {e}")
        
        # Clean up
        os.unlink(cv_path)
        os.unlink(job_path)
        print(f"\n✅ Cleaned up temporary files")
        
        return True
        
    except Exception as e:
        print(f"❌ Analytical test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing enhanced analytical RAG capabilities...")
    test_analytical_queries() 