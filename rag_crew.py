# rag_crew.py
from crewai import Agent, Task, Crew, Process
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM  # Updated import for newer package
from langchain_ollama import OllamaEmbeddings  # Updated import for newer package
import os
import shutil

# Set environment variables to configure CrewAI to use Ollama
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_API_KEY"] = "ollama"  # Dummy key for Ollama
os.environ["DEFAULT_MODEL"] = "llama3.2:latest"
os.environ["LITELLM_MODEL"] = "llama3.2:latest"

class RAGCrew:
    def __init__(self, model_name="llama3.2:latest"):
        self.model_name = model_name
        self.vector_store = None
        self.retriever = None
        self.chroma_persist_directory = "./chroma_db"
        
        # Initialize Ollama LLM with LiteLLM-compatible model name
        self.llm = OllamaLLM(
            model=f"ollama/{model_name}",  # Add ollama/ prefix for LiteLLM compatibility
            base_url='http://localhost:11434',
            temperature=0.3
        )
        
        # Initialize embeddings with correct model name
        self.embeddings = OllamaEmbeddings(
            model=model_name,
            base_url='http://localhost:11434'
        )
        
        # Initialize agents with enhanced capabilities
        self.researcher = Agent(
            role="Research Analyst",
            goal="Analyze and extract relevant information from documents, identify patterns, and gather evidence for analysis",
            backstory="Expert in document analysis, pattern recognition, and evidence gathering. Skilled at identifying relevant information from various document types including CVs, job descriptions, and company records.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[]
        )
        
        self.analyst = Agent(
            role="Business Analyst",
            goal="Analyze information, evaluate fit, assess qualifications, and provide detailed reasoning for recommendations",
            backstory="Experienced business analyst with expertise in candidate evaluation, job matching, and strategic analysis. Skilled at evaluating qualifications against requirements and providing evidence-based recommendations.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[]
        )
        
        self.writer = Agent(
            role="Recommendation Specialist",
            goal="Generate clear, actionable recommendations and comprehensive responses based on analysis",
            backstory="Expert in creating clear, actionable recommendations and comprehensive responses. Skilled at presenting analysis results in a structured, professional manner with clear reasoning and actionable insights.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[]
        )
        
        self.qa_agent = Agent(
            role="Quality Assurance & Validation",
            goal="Verify accuracy, completeness, and validity of analysis and recommendations against source documents",
            backstory="Detail-oriented specialist who ensures high quality output, validates claims against source documents, and ensures recommendations are well-supported by evidence.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[]
        )
    
    def clear_documents(self):
        """Clear all existing documents and reset the vector store"""
        try:
            if os.path.exists(self.chroma_persist_directory):
                shutil.rmtree(self.chroma_persist_directory)
                print(f"✅ Cleared existing documents from {self.chroma_persist_directory}")
            self.vector_store = None
            self.retriever = None
            return True
        except Exception as e:
            print(f"❌ Error clearing documents: {e}")
            return False
    
    def load_and_process_documents(self, file_paths, clear_existing=True):
        """Load and process documents into vector embeddings"""
        try:
            # Clear existing documents if requested
            if clear_existing:
                self.clear_documents()
            
            documents = []
            
            for file_path in file_paths:
                print(f"📄 Processing: {os.path.basename(file_path)}")
                if file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path)
                documents.extend(loader.load())
            
            print(f"✅ Loaded {len(documents)} document chunks")
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            print(f"✅ Split into {len(splits)} text chunks")
            
            # Create vector store
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.chroma_persist_directory
            )
            self.retriever = self.vector_store.as_retriever()
            
            print(f"✅ Created vector store with {len(splits)} embeddings")
            return True
            
        except Exception as e:
            print(f"❌ Error processing documents: {e}")
            return False
    
    def query_documents(self, query):
        """Retrieve relevant document chunks"""
        if not self.retriever:
            raise ValueError("Documents not loaded. Call load_and_process_documents first.")
        return self.retriever.get_relevant_documents(query)
    
    def get_document_count(self):
        """Get the number of documents in the vector store"""
        if self.vector_store:
            return len(self.vector_store.get()['documents'])
        return 0

    def check_relevance(self, query, relevant_docs, threshold=0.3):
        """Check if the retrieved documents are relevant to the query"""
        if not relevant_docs:
            return False, "No documents found"
        
        # Check if any document contains the key terms from the query
        query_terms = query.lower().split()
        key_terms = [term for term in query_terms if len(term) > 3]  # Focus on meaningful terms
        
        # Check for exact matches of key terms in documents
        relevant_content = " ".join([doc.page_content.lower() for doc in relevant_docs])
        
        # Count how many key terms are found in the documents
        found_terms = sum(1 for term in key_terms if term in relevant_content)
        relevance_score = found_terms / len(key_terms) if key_terms else 0
        
        # Additional check for named entities (people, companies, etc.)
        named_entities = []
        for term in query_terms:
            if term[0].isupper() and len(term) > 2:  # Potential named entity
                named_entities.append(term)
        
        # Check if any named entities are found in documents
        entity_found = any(entity.lower() in relevant_content for entity in named_entities)
        
        is_relevant = relevance_score >= threshold or entity_found
        
        if not is_relevant:
            missing_info = f"Query terms not found in documents: {', '.join(key_terms)}"
            if named_entities:
                missing_info += f"\nNamed entities not found: {', '.join(named_entities)}"
            return False, missing_info
        
        return True, f"Relevance score: {relevance_score:.2f}"

    def generate_out_of_context_response(self, query, missing_info):
        """Generate a response for out-of-context questions"""
        out_of_context_prompt = f"""
        The user asked: "{query}"
        
        However, this information is not available in the uploaded documents. 
        Missing information: {missing_info}
        
        Please provide a helpful response that:
        1. Acknowledges that the requested information is not in the uploaded documents
        2. Explains what information is available in the documents
        3. Suggests what the user could do to get the information they need
        4. Offers to help with other questions about the available documents
        
        Be polite, helpful, and professional in your response.
        """
        
        try:
            # Use the writer agent to generate a helpful response
            response = self.llm.predict(out_of_context_prompt)
            return response
        except Exception as e:
            return f"I apologize, but I cannot find information about '{query}' in the uploaded documents. The documents I have access to don't contain this information. Please check if you have uploaded the correct documents or try asking about information that might be available in the current documents."

    def generate_response(self, query):
        """Generate response using CrewAI agents with enhanced analytical capabilities and out-of-context handling"""
        if not self.retriever:
            raise ValueError("Documents not loaded. Call load_and_process_documents first.")
        
        # Retrieve relevant documents for the query
        relevant_docs = self.query_documents(query)
        
        # Check if the retrieved documents are relevant to the query
        is_relevant, relevance_info = self.check_relevance(query, relevant_docs)
        
        if not is_relevant:
            print(f"⚠️ Out-of-context query detected: {query}")
            print(f"📊 Relevance info: {relevance_info}")
            return self.generate_out_of_context_response(query, relevance_info)
        
        # Format the retrieved documents for context
        document_context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
        
        # Determine if this is an analytical/recommendation query
        analytical_keywords = [
            'recommend', 'recommendation', 'fit', 'suitable', 'appropriate', 'evaluate', 'assessment',
            'analysis', 'compare', 'match', 'experience', 'worked', 'employment', 'qualifications',
            'skills', 'requirements', 'candidate', 'position', 'job', 'role', 'company', 'employer'
        ]
        
        is_analytical = any(keyword in query.lower() for keyword in analytical_keywords)
        
        if is_analytical:
            # Enhanced analytical workflow for recommendations and evaluations
            research_task = Task(
                description=f"""Analyze the following documents and extract ALL relevant information for: {query}

DOCUMENTS:
{document_context}

Focus on:
1. Specific qualifications, skills, and experiences
2. Dates, durations, and timelines
3. Company names, positions, and responsibilities
4. Educational background and certifications
5. Any gaps or inconsistencies in information
6. Relevant achievements and accomplishments

Extract information that could be relevant for evaluation, comparison, or recommendation purposes.""",
                agent=self.researcher,
                expected_output="A comprehensive list of relevant facts, qualifications, experiences, and evidence extracted from the documents.",
                verbose=True
            )

            analysis_task = Task(
                description=f"""Based on the research findings, conduct a thorough analysis for: {query}

Consider:
1. How well do the qualifications match the requirements?
2. What are the strengths and weaknesses?
3. Are there any red flags or concerns?
4. What is the overall assessment?
5. What specific evidence supports your analysis?

Provide detailed reasoning and evaluation criteria.""",
                agent=self.analyst,
                expected_output="A detailed analysis with evaluation criteria, strengths/weaknesses assessment, and evidence-based reasoning.",
                context=[research_task],
                verbose=True
            )

            recommendation_task = Task(
                description=f"""Based on the analysis, provide a comprehensive response with clear recommendations for: {query}

Structure your response to include:
1. **Summary of Findings**: Key points from the analysis
2. **Assessment**: Overall evaluation and fit
3. **Recommendations**: Clear, actionable recommendations
4. **Evidence**: Specific evidence from documents supporting your conclusions
5. **Next Steps**: Suggested actions or follow-up questions

Make your response actionable and well-supported by evidence.""",
                agent=self.writer,
                expected_output="A comprehensive response with clear recommendations, assessment, and actionable insights supported by evidence.",
                context=[analysis_task],
                verbose=True
            )

            qa_task = Task(
                description=f"""Verify the accuracy and completeness of the analysis and recommendations against the original documents.

ORIGINAL DOCUMENTS:
{document_context}

Ensure:
1. All claims are supported by evidence from the documents
2. No important information was overlooked
3. The analysis is fair and balanced
4. Recommendations are reasonable and actionable
5. The response addresses the original query completely""",
                agent=self.qa_agent,
                expected_output="A validated and improved version of the response with any corrections or additions, ensuring accuracy and completeness.",
                context=[recommendation_task],
                verbose=True
            )

            # Create and run crew with analytical workflow
            crew = Crew(
                agents=[self.researcher, self.analyst, self.writer, self.qa_agent],
                tasks=[research_task, analysis_task, recommendation_task, qa_task],
                process=Process.sequential,
                verbose=True
            )

        else:
            # Standard information retrieval workflow
            research_task = Task(
                description=f"""Analyze the following documents and extract relevant information about: {query}

DOCUMENTS:
{document_context}

Focus only on information that is explicitly stated in these documents. Do not use external knowledge.""",
                agent=self.researcher,
                expected_output="A comprehensive list of relevant facts and information extracted specifically from the provided documents.",
                verbose=True
            )

            writing_task = Task(
                description=f"""Based on the research findings, generate a clear and accurate response to: {query}

Use ONLY the information from the research findings. Do not add external knowledge or assumptions.""",
                agent=self.writer,
                expected_output="A well-structured answer in natural language that addresses the query completely using only the provided document information.",
                context=[research_task],
                verbose=True
            )

            qa_task = Task(
                description=f"""Verify the accuracy and completeness of the response against the original documents.

ORIGINAL DOCUMENTS:
{document_context}

Ensure the response is factually accurate and complete based on the source documents.""",
                agent=self.qa_agent,
                expected_output="An improved version of the response with any corrections or additions, ensuring accuracy against the source documents.",
                context=[writing_task],
                verbose=True
            )

            # Create and run crew with standard workflow
            crew = Crew(
                agents=[self.researcher, self.writer, self.qa_agent],
                tasks=[research_task, writing_task, qa_task],
                process=Process.sequential,
                verbose=True
            )

        result = crew.kickoff()
        return result
    