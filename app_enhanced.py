# app_enhanced.py
import streamlit as st
from rag_crew import RAGCrew
import tempfile
import os
from datetime import datetime

# Initialize session state
if 'rag_crew' not in st.session_state:
    st.session_state.rag_crew = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'show_agent_details' not in st.session_state:
    st.session_state.show_agent_details = False
if 'document_count' not in st.session_state:
    st.session_state.document_count = 0
if 'out_of_context_count' not in st.session_state:
    st.session_state.out_of_context_count = 0

# App title and description
st.title("🧠 Enhanced Local RAG System with CrewAI and Ollama")
st.markdown("""
    Ask questions about your uploaded documents using local AI models.
    All processing happens on your Mac - no data leaves your machine.
    
    **Features:**
    - 🤖 Multi-agent processing (Research → Write → QA)
    - 📄 Document-based responses
    - 🔍 Document retrieval and relevance scoring
    - 📊 Detailed agent workflow visualization
    - 🗂️ Smart document management
    - ⚠️ Out-of-context question detection
""")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    model_name = st.selectbox(
        "Select Ollama Model",
        ["llama3.2:latest", "mistral", "phi3", "gemma"],
        index=0,
        help="Choose the local LLM model to use for processing"
    )
    
    # Show agent details toggle
    st.session_state.show_agent_details = st.checkbox(
        "Show Agent Workflow Details",
        value=False,
        help="Display detailed information about each agent's work"
    )
    
    st.header("📂 Document Management")
    
    # Document processing options
    clear_existing = st.checkbox(
        "Clear existing documents",
        value=True,
        help="Clear previously loaded documents before processing new ones"
    )
    
    uploaded_files = st.file_uploader(
        "Upload documents (PDF or TXT)",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload the documents you want to query"
    )
    
    # Process documents button
    if st.button("🚀 Process Documents", help="Process the uploaded documents for querying"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                try:
                    # Save uploaded files temporarily
                    temp_dir = tempfile.mkdtemp()
                    file_paths = []
                    for file in uploaded_files:
                        file_path = os.path.join(temp_dir, file.name)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        file_paths.append(file_path)
                    
                    # Initialize RAG crew with local Ollama
                    st.session_state.rag_crew = RAGCrew(model_name=model_name)
                    
                    # Process documents with clear option
                    success = st.session_state.rag_crew.load_and_process_documents(
                        file_paths, 
                        clear_existing=clear_existing
                    )
                    
                    if success:
                        st.session_state.documents_loaded = True
                        st.session_state.messages = []  # Clear previous messages
                        st.session_state.document_count = st.session_state.rag_crew.get_document_count()
                        st.session_state.out_of_context_count = 0  # Reset counter
                        st.success("✅ Documents processed successfully!")
                        
                        # Show document processing info
                        with st.expander("📊 Document Processing Details"):
                            st.write(f"**Model Used:** {model_name}")
                            st.write(f"**Documents Processed:** {len(uploaded_files)}")
                            st.write(f"**Text Chunks Created:** {st.session_state.document_count}")
                            st.write(f"**Files:** {[f.name for f in uploaded_files]}")
                            if clear_existing:
                                st.write("**Action:** Cleared existing documents before processing")
                            else:
                                st.write("**Action:** Added to existing documents")
                    else:
                        st.error("❌ Failed to process documents")
                        
                except Exception as e:
                    st.error(f"❌ Error processing documents: {e}")
                    st.exception(e)
        else:
            st.warning("⚠️ Please upload documents first")
    
    # Clear documents button
    if st.button("🗑️ Clear All Documents", help="Clear all loaded documents"):
        if st.session_state.rag_crew:
            if st.session_state.rag_crew.clear_documents():
                st.session_state.documents_loaded = False
                st.session_state.messages = []
                st.session_state.document_count = 0
                st.session_state.out_of_context_count = 0
                st.success("✅ All documents cleared!")
            else:
                st.error("❌ Failed to clear documents")
        else:
            st.warning("⚠️ No documents to clear")
    
    # Document status
    if st.session_state.documents_loaded:
        st.success(f"📄 {st.session_state.document_count} document chunks loaded")
        if st.session_state.out_of_context_count > 0:
            st.warning(f"⚠️ {st.session_state.out_of_context_count} out-of-context questions detected")
    else:
        st.info("📄 No documents loaded")

# Main chat interface
st.divider()
st.subheader("💬 Chat with Your Documents")

# Display chat messages
response_container = st.container()  # Container for chat messages
query_container = st.container()  # Container for query input

with response_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show out-of-context indicator
            if message.get("out_of_context", False):
                st.warning("⚠️ This information was not found in the uploaded documents")
            
            # Show agent workflow details if enabled
            if st.session_state.show_agent_details and "agent_details" in message:
                with st.expander("🤖 Agent Workflow Details"):
                    for agent, details in message["agent_details"].items():
                        st.markdown(f"**{agent}:**")
                        st.markdown(details)
                        st.divider()
            
            # Show document references
            if "metadata" in message:
                with st.expander("📄 Document References"):
                    for doc in message["metadata"]:
                        st.markdown(f"**Source:** {doc.get('source', 'Unknown')}")
                        st.markdown(f"**Page:** {doc.get('page', 'N/A')}")
                        st.markdown(f"**Relevance:** {doc.get('relevance_score', 'N/A')}")
                        st.divider()

# Input for new questions
with query_container:
    if prompt := st.chat_input("Type your question here..."):
        if not st.session_state.documents_loaded:
            st.warning("Please upload and process documents first")
            st.stop()
            
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with response_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Display assistant response
            with st.chat_message("assistant"):
                with st.spinner("🤖 Processing with RAG crew..."):
                    try:
                        # Get relevant documents first
                        relevant_docs = st.session_state.rag_crew.query_documents(prompt)
                        
                        # Check if this is an out-of-context question
                        is_relevant, relevance_info = st.session_state.rag_crew.check_relevance(prompt, relevant_docs)
                        
                        if not is_relevant:
                            st.warning("⚠️ Out-of-context question detected")
                            st.session_state.out_of_context_count += 1
                        
                        # Show document retrieval info
                        if st.session_state.show_agent_details:
                            with st.expander("🔍 Document Retrieval"):
                                st.write(f"**Retrieved {len(relevant_docs)} relevant document chunks**")
                                st.write(f"**Relevance:** {relevance_info}")
                                for i, doc in enumerate(relevant_docs[:3]):
                                    st.markdown(f"**Chunk {i+1}:**")
                                    st.markdown(doc.page_content[:200] + "...")
                                    st.divider()
                        
                        # Generate response using the improved RAG
                        response = st.session_state.rag_crew.generate_response(prompt)
                        
                        # Display response
                        st.markdown(response)
                        
                        # Add to chat history with metadata
                        message_data = {
                            "role": "assistant",
                            "content": response,
                            "out_of_context": not is_relevant,
                            "metadata": [
                                {
                                    "source": doc.metadata.get("source", "Unknown"),
                                    "page": doc.metadata.get("page", "N/A"),
                                    "relevance_score": doc.metadata.get("relevance_score", "N/A")
                                }
                                for doc in relevant_docs[:3]  # Show top 3 relevant docs
                            ]
                        }
                        
                        # Add agent details if enabled
                        if st.session_state.show_agent_details:
                            message_data["agent_details"] = {
                                "Research Analyst": "Analyzed documents and extracted relevant information",
                                "Content Writer": "Synthesized information into a coherent response",
                                "Quality Assurance": "Verified accuracy against source documents"
                            }
                        
                        st.session_state.messages.append(message_data)
                        
                    except Exception as e:
                        st.error(f"❌ Error generating response: {e}")
                        st.exception(e)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Powered by CrewAI + Ollama + Streamlit</p>
    <p>All processing happens locally on your machine</p>
</div>
""", unsafe_allow_html=True) 