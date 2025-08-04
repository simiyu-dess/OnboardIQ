# 🧠 Local RAG System with CrewAI and Ollama

A powerful local Retrieval Augmented Generation (RAG) system that uses CrewAI for multi-agent processing and Ollama for local LLM inference. All processing happens on your machine - no data leaves your local environment.

## ✨ Features

- **🔒 100% Local Processing**: All AI processing happens on your machine using Ollama
- **🤖 Multi-Agent Workflow**: Uses CrewAI with specialized agents for research, analysis, writing, and quality assurance
- **📄 Document Intelligence**: Upload PDFs and text files for intelligent querying
- **💡 Analytical Capabilities**: Get recommendations, evaluations, and detailed analysis (not just information retrieval)
- **🌐 Web Interface**: Beautiful Streamlit interface for easy interaction
- **🗂️ Smart Document Management**: Clear existing documents, add new ones, and track processing status
- **📊 Detailed Insights**: View document references, agent workflow details, and processing information

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed on your system
2. **Ollama** installed and running locally
3. **A local LLM model** (e.g., llama3.2, mistral, phi3)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd OnboardIQ
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Ollama and pull a model**:
   ```bash
   # Start Ollama (if not already running)
   ollama serve
   
   # Pull a model (in another terminal)
   ollama pull llama3.2:latest
   ```

### Usage

#### Web Interface (Recommended)

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Upload documents** (PDF or TXT files)

4. **Process documents** using the sidebar

5. **Ask questions** in the chat interface

#### Enhanced Web Interface

For more detailed agent workflow information:
```bash
streamlit run app_enhanced.py
```

#### Command Line Testing

Test the RAG system with sample documents:
```bash
python test_analytical_rag.py
```

Test document management functionality:
```bash
python test_document_management.py
```

## 🏗️ Architecture

### Core Components

- **`rag_crew.py`**: Main RAG system with CrewAI agents
- **`app.py`**: Streamlit web interface
- **`app_enhanced.py`**: Enhanced interface with detailed agent workflow
- **Chroma Vector Store**: Local vector database for document embeddings
- **Ollama LLM**: Local language model for processing

### Agent Workflow

1. **Research Analyst**: Extracts relevant information from documents
2. **Business Analyst**: Analyzes information and provides evaluation (for analytical queries)
3. **Recommendation Specialist**: Generates clear, actionable recommendations
4. **Quality Assurance**: Validates accuracy against source documents

### Query Types

The system automatically detects query types:

- **Information Retrieval**: Standard questions about document content
- **Analytical Queries**: Questions requiring analysis, evaluation, or recommendations

## 📁 Project Structure

```
OnboardIQ/
├── rag_crew.py              # Main RAG system
├── app.py                   # Streamlit web interface
├── app_enhanced.py          # Enhanced web interface
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── .gitignore              # Git ignore rules
├── test_*.py               # Test scripts
├── env/                    # Virtual environment
└── chroma_db/              # Vector database (auto-generated)
```

## 🔧 Configuration

### Model Selection

Choose from available Ollama models:
- `llama3.2:latest` (default)
- `mistral`
- `phi3`
- `gemma`

### Document Processing Options

- **Clear existing documents**: Remove old documents before processing new ones
- **Add to existing**: Keep old documents and add new ones
- **Manual clearing**: Clear all documents using the sidebar button

## 🧪 Testing

### Test Scripts

- **`test_analytical_rag.py`**: Tests analytical capabilities with sample CV and job description
- **`test_document_management.py`**: Tests document loading, clearing, and management
- **`test_rag_with_documents.py`**: Tests basic RAG functionality
- **`test_ollama.py`**: Tests Ollama connectivity

### Example Queries

**Information Retrieval**:
- "What is the candidate's educational background?"
- "What are the job requirements?"

**Analytical Queries**:
- "Is this candidate a good fit for the position?"
- "Evaluate the candidate's qualifications against the job requirements"
- "Has the candidate worked at TechCorp Inc.?"

## 🛠️ Troubleshooting

### Common Issues

1. **Ollama not running**:
   ```bash
   ollama serve
   ```

2. **Model not found**:
   ```bash
   ollama pull llama3.2:latest
   ```

3. **Documents not loading**:
   - Check file format (PDF or TXT)
   - Ensure "Clear existing documents" is checked if uploading new documents
   - Check console for error messages

4. **LLM connection errors**:
   - Verify Ollama is running on `http://localhost:11434`
   - Check model name in configuration
   - Ensure virtual environment is activated

### Debug Mode

Enable detailed logging by setting `verbose=True` in the RAG crew initialization.

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test scripts for examples
3. Open an issue on GitHub

---

**Note**: This system processes all data locally. No information is sent to external servers, ensuring complete privacy and data security. 