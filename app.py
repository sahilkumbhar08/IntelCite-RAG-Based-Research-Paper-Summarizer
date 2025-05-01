import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import google.generativeai as genai
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import tempfile
import time
from pathlib import Path
import requests
import pandas as pd
import plotly.express as px
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Set API key from environment if available, otherwise use hardcoded value
API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBTOZ78XdS9w4-3LzYna1wIDxjhItGmLws")
CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "service.json")

# Configure Google API
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH
genai.configure(api_key=API_KEY)

# Cache for storing processed data
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}

# Set up Streamlit page configuration
st.set_page_config(
    page_title="IntelCite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main layout styling */
    .main {
        background-color: #f9fafb;
    }
    
    /* Header styling */
    .header {
        padding: 1.5rem 0;
        text-align: center;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3a1c71, #d76d77, #ffaf7b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 0;
    }
    
    /* Card styling */
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #e5e7eb;
    }
    
    /* Section headings */
    .section-heading {
        font-size: 1.4rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f0f0f0;
    }
    
    /* Button styling */
    .primary-button {
        background-color: #4361ee;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        border: none;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .primary-button:hover {
        background-color: #3a56e8;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Info box styling */
    .info-box {
        background-color: #e7f3fe;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 4px;
    }
    
    /* Results box styling */
    .result-box {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 16px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #e0e0e0 !important;
    }
    
    /* Metrics styling */
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4361ee;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 1.5rem 0;
        margin-top: 2rem;
        color: #6c757d;
        font-size: 0.9rem;
        border-top: 1px solid #e5e7eb;
    }
    
    /* Progress indicators */
    .progress-container {
        margin: 1rem 0;
    }
    
    /* Form elements */
    input, textarea, select {
        border-radius: 6px !important;
        border: 1px solid #ced4da !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# App header section
st.markdown('<div class="header">', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">IntelCite</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Research Assistant for PDF Analysis, Video Processing, and Academic Paper Search</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Function to get text from PDF files
def get_pdf_text(pdf_docs):
    """Extract text from multiple PDF documents."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_bytes = pdf.read()
            pdf_reader = PdfReader(BytesIO(pdf_bytes))
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            
            # Reset file pointer for future use
            pdf.seek(0)
            
        except Exception as e:
            logger.error(f"Error reading {pdf.name}: {str(e)}")
            st.error(f"Error reading {pdf.name}: {str(e)}")
    
    return text

# Function to split text into chunks
def get_text_chunks(text):
    """Split text into manageable chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

# Function to create vector store from text chunks
@st.cache_resource
def get_vector_store(text_chunks, namespace="default"):
    """Create a vector store from text chunks with caching."""
    if not text_chunks:
        logger.warning("No text chunks provided to create vector store")
        return None
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=API_KEY
        )
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        
        # Save to a namespace-specific index
        vector_store.save_local(f"faiss_index_{namespace}")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        st.error(f"Error creating vector store: {str(e)}")
        return None

# Function to load vector store
def load_vector_store(namespace="default"):
    """Load a previously saved vector store."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=API_KEY
        )
        return FAISS.load_local(f"faiss_index_{namespace}", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        return None

# Function to create the QA chain
def get_conversational_chain(model_name="gemini-2.0-flash-exp", temperature=0.3):
    """Create a question-answering chain."""
    prompt_template = """
    You are an advanced AI research assistant. Your task is to provide detailed and accurate answers based on the provided context.
    
    Context: {context}
    
    Question: {question}
    
    Instructions:
    1. Answer the question as thoroughly as possible using ONLY the information from the provided context.
    2. If the answer is not contained within the context, explicitly state: "The answer is not available in the provided context."
    3. Format your answer in a clear, structured manner using markdown formatting when appropriate.
    4. If the context contains technical information, explain it in a way that maintains accuracy while being understandable.
    5. Cite relevant parts of the context to support your answer.
    
    Your comprehensive answer:
    """
    
    try:
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        logger.error(f"Error creating conversational chain: {str(e)}")
        st.error(f"Error creating conversational chain: {str(e)}")
        return None

# Function to process user input
def process_user_query(user_question, namespace="default"):
    """Process a user question against the vector store."""
    try:
        vector_store = load_vector_store(namespace)
        if not vector_store:
            return "Error: Unable to load the knowledge base. Please process your documents first."
        
        docs = vector_store.similarity_search(user_question, k=4)
        chain = get_conversational_chain()
        
        if not chain:
            return "Error: Unable to create the AI chain. Please try again later."
        
        response = chain(
            {"input_documents": docs, "question": user_question}, 
            return_only_outputs=True
        )
        
        return response["output_text"]
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return f"An error occurred while processing your question: {str(e)}"

# Initialize the multimodal AI agent
def initialize_agent():
    """Initialize the Phi agent with Google's Gemini model."""
    try:
        return Agent(
            name="Advanced Multimodal Research Assistant",
            model=Gemini(id="gemini-2.0-flash-exp"),
            tools=[DuckDuckGo()],
            markdown=True,
        )
    except Exception as e:
        logger.error(f"Error initializing agent: {str(e)}")
        st.error(f"Error initializing agent: {str(e)}")
        return None

# Search for academic papers
def search_academic_papers(query, max_results=5):
    """Search for academic papers related to the query."""
    try:
        # Using Semantic Scholar API
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,authors,year,abstract,url,venue"
        }
        
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            results = response.json()
            return results.get('data', [])
        else:
            logger.error(f"Failed to search papers: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error searching papers: {str(e)}")
        return []

# Initialize the multimodal agent
multimodal_agent = initialize_agent()

# Sidebar for navigation and settings
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Model selection
    st.markdown("#### Model Selection")
    model_name = st.selectbox(
        "Choose AI Model",
        ["gemini-2.0-flash-exp", "gemini-2.0-pro", "gemini-2.0-vision"],
        index=0
    )
    
    # Temperature setting
    temperature = st.slider(
        "Response Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Higher values make output more creative, lower values make it more deterministic"
    )
    
    # Divider
    st.markdown("---")
    
    # About section
    st.markdown("### About")
    st.markdown("""
    **IntelCite** is an intelligent research assistant powered by advanced AI models. It helps you:
    
    - 📄 Analyze PDF documents
    - 🎥 Extract insights from videos
    - 🔍 Find relevant academic papers
    - 📊 Visualize research data
    
    *Developed by GROUP 10 CSAIML*
    """)

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📄 PDF Analysis", 
    "🎥 Video Analysis", 
    "🔍 Research Papers", 
    "📊 Dashboard"
])

# PDF Analysis Tab
with tab1:
    st.markdown('<h2 class="section-heading">PDF Document Analysis</h2>', unsafe_allow_html=True)
    
    # Upload section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Upload Documents")
        st.markdown("""
        <div class="info-box">
        <p>Upload PDF files to analyze. The system will extract text and create a searchable knowledge base.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center;">
            <span style="font-size: 2.5rem; color: #4361ee;">📄</span>
            <p style="margin-top: 10px; margin-bottom: 0; font-weight: 500;">PDF Files</p>
        </div>
        """, unsafe_allow_html=True)
    
    pdf_docs = st.file_uploader(
        "Drag and drop PDF files here",
        accept_multiple_files=True,
        type=["pdf"],
        help="Upload one or more PDF files to analyze"
    )
    
    if pdf_docs:
        file_names = [doc.name for doc in pdf_docs]
        st.markdown(f"**Uploaded {len(pdf_docs)} file(s):**")
        
        for file in file_names:
            st.markdown(f"- {file}")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            process_button = st.button("Process PDFs", key="process_pdf_button", use_container_width=True)
        
        if process_button:
            with st.spinner("Processing your documents..."):
                # Process the PDFs
                raw_text = get_pdf_text(pdf_docs)
                
                if raw_text:
                    # Count words and estimate reading time
                    word_count = len(raw_text.split())
                    reading_time = round(word_count / 200)  # Assuming 200 words per minute reading speed
                    
                    # Create text chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    # Create vector store
                    vector_store = get_vector_store(text_chunks, namespace="pdf")
                    
                    if vector_store:
                        st.session_state.processed_data['pdf'] = {
                            'word_count': word_count,
                            'reading_time': reading_time,
                            'chunk_count': len(text_chunks)
                        }
                        
                        st.success("✅ Documents processed successfully!")
                        
                        # Display statistics in a nicer format
                        st.markdown("### Document Statistics")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("""
                            <div class="metric-card">
                                <div class="metric-value">{:,}</div>
                                <div class="metric-label">Words Analyzed</div>
                            </div>
                            """.format(word_count), unsafe_allow_html=True)
                            
                        with col2:
                            st.markdown("""
                            <div class="metric-card">
                                <div class="metric-value">{} min</div>
                                <div class="metric-label">Reading Time</div>
                            </div>
                            """.format(reading_time), unsafe_allow_html=True)
                            
                        with col3:
                            st.markdown("""
                            <div class="metric-card">
                                <div class="metric-value">{}</div>
                                <div class="metric-label">Text Segments</div>
                            </div>
                            """.format(len(text_chunks)), unsafe_allow_html=True)
                else:
                    st.warning("No text content found in the uploaded PDFs.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Query section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Ask Questions About Your Documents")
    
    if 'processed_data' in st.session_state and 'pdf' in st.session_state.processed_data:
        st.markdown("""
        <div class="info-box">
        <p>Your knowledge base is ready! Ask any question about the content of your documents.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box" style="background-color: #fff3cd; border-left-color: #ffc107;">
        <p>Please upload and process PDF documents first to create a knowledge base.</p>
        </div>
        """, unsafe_allow_html=True)
    
    user_query = st.text_area(
        "Your question",
        placeholder="What are the main findings or insights from these documents?",
        help="Ask a specific question about the content of your uploaded PDFs"
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        submit_query = st.button("Ask AI", key="pdf_query_button", use_container_width=True)
    
    if submit_query and user_query:
        if 'processed_data' in st.session_state and 'pdf' in st.session_state.processed_data:
            with st.spinner("Analyzing your question..."):
                response = process_user_query(user_query, namespace="pdf")
                
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown("### AI Response:")
                st.markdown(response)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Feedback buttons
                col1, col2, col3 = st.columns([1, 1, 4])
                with col1:
                    st.button("👍 Helpful", key="helpful_btn")
                with col2:
                    st.button("👎 Not Helpful", key="not_helpful_btn")
        else:
            st.error("Please process PDFs before asking questions.")
    st.markdown('</div>', unsafe_allow_html=True)

# Video Analysis Tab
with tab2:
    st.markdown('<h2 class="section-heading">Video Content Analysis</h2>', unsafe_allow_html=True)
    
    # Upload section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Upload Video")
        st.markdown("""
        <div class="info-box">
        <p>Upload a video file to analyze its content. The AI will process visual elements, audio, and text appearing in the video.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center;">
            <span style="font-size: 2.5rem; color: #4361ee;">🎥</span>
            <p style="margin-top: 10px; margin-bottom: 0; font-weight: 500;">Video Files</p>
        </div>
        """, unsafe_allow_html=True)
    
    video_file = st.file_uploader(
        "Drag and drop a video file here",
        type=['mp4', 'mov', 'avi'],
        help="Upload a video file to analyze (5 minutes or less recommended)"
    )
    
    if video_file:
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name
        
        # Display video preview
        st.video(video_path, format="video/mp4", start_time=0)
        
        st.session_state.video_path = video_path
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Video analysis form
    if video_file:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Analysis Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            analysis_type = st.radio(
                "Choose analysis type:",
                ["Content Summary", "Detailed Analysis", "Extract Key Moments"]
            )
        
        with col2:
            st.markdown("""
            <div class="info-box">
            <p><strong>Content Summary:</strong> Brief overview of the video content<br>
            <strong>Detailed Analysis:</strong> In-depth examination of themes and content<br>
            <strong>Extract Key Moments:</strong> Identify and describe important timestamps</p>
            </div>
            """, unsafe_allow_html=True)
        
        user_query_video = st.text_area(
            "What would you like to know about this video?",
            placeholder="E.g., What are the main points discussed in this video? or Summarize the key arguments presented."
        )
        
        analyze_button = st.button("🔍 Analyze Video", key="analyze_video_button", use_container_width=False)
            
        if analyze_button and user_query_video:
            try:
                with st.spinner("Processing video... This may take a few minutes."):
                    # Process the video
                    processed_video = upload_file(video_path)
                    
                    # Create progress bar with better styling
                    st.markdown('<div class="progress-container">', unsafe_allow_html=True)
                    progress_bar = st.progress(0)
                    
                    # Wait for processing to complete with visual feedback
                    for i in range(10):
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)
                        if processed_video.state.name != "PROCESSING":
                            break
                        progress_bar.progress((i + 1) * 10)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Generate prompt based on analysis type
                    analysis_prompt = f"""
                    You are an expert video content analyzer. Analyze the uploaded video thoroughly.
                    
                    Analysis Type: {analysis_type}
                    User Query: {user_query_video}
                    
                    Provide a {analysis_type.lower()} that addresses the user's query. Format your response using markdown for better readability.
                    Include sections with headings, bullet points where appropriate, and highlight key insights.
                    """
                    
                    # Run analysis
                    response = multimodal_agent.run(analysis_prompt, videos=[processed_video])
                    
                    # Display results
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown("### Video Analysis Results")
                    st.markdown(response.content)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Feedback buttons
                    col1, col2, col3 = st.columns([1, 1, 4])
                    with col1:
                        st.button("👍 Helpful", key="video_helpful_btn")
                    with col2:
                        st.button("👎 Not Helpful", key="video_not_helpful_btn")
                    
            except Exception as error:
                st.error(f"An error occurred during analysis: {error}")
            finally:
                # Clean up temporary file
                try:
                    Path(video_path).unlink(missing_ok=True)
                except:
                    pass
        st.markdown('</div>', unsafe_allow_html=True)

# Research Papers Tab
with tab3:
    st.markdown('<h2 class="section-heading">Academic Research Papers</h2>', unsafe_allow_html=True)
    
    # Search section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Search Academic Papers")
        st.markdown("""
        <div class="info-box">
        <p>Search for academic papers related to your research topic. The system will retrieve relevant papers from reliable sources.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div style="background-color: #f8f9fa; border-radius: 8px; padding: 15px; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center;">
            <span style="font-size: 2.5rem; color: #4361ee;">🔍</span>
            <p style="margin-top: 10px; margin-bottom: 0; font-weight: 500;">Research Papers</p>
        </div>
        """, unsafe_allow_html=True)
    
    paper_search_query = st.text_input(
        "Research topic or keywords",
        placeholder="E.g., quantum computing applications in healthcare",
        help="Enter keywords to search for relevant academic papers"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        max_results = st.select_slider(
            "Number of results",
            options=[3, 5, 7, 10],
            value=5
        )
    with col2:
        search_button = st.button("Search Papers", key="search_papers_button", use_container_width=True)
    
    if search_button and paper_search_query:
        with st.spinner("Searching academic databases..."):
            papers = search_academic_papers(paper_search_query, max_results)
            
            if papers:
                st.session_state.papers = papers
                st.success(f"Found {len(papers)} relevant papers!")
            else:
                st.warning("No papers found matching your query. Try different keywords.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display search results
    if 'papers' in st.session_state and st.session_state.papers:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Search Results")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            sort_option = st.selectbox(
                "Sort by",
                ["Relevance", "Year (Newest First)", "Year (Oldest First)"],
                index=0
            )
            
        # Sort papers based on selection
        papers = st.session_state.papers
        if sort_option == "Year (Newest First)":
            papers = sorted(papers, key=lambda x: x.get('year', 0), reverse=True)
        elif sort_option == "Year (Oldest First)":
            papers = sorted(papers, key=lambda x: x.get('year', 0))
        
        # Display papers in a more structured way
        for i, paper in enumerate(papers):
            with st.expander(f"{paper.get('title', 'Untitled Paper')} ({paper.get('year', 'N/A')})"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Authors
                    authors = ", ".join([author.get('name', '') for author in paper.get('authors', [])])
                    st.markdown(f"**Authors:** {authors}")
                    
                    # Venue
                    if paper.get('venue'):
                        st.markdown(f"**Published in:** {paper.get('venue')}")
                    
                    # Abstract
                    if paper.get('abstract'):
                        st.markdown("**Abstract:**")
                        st.markdown(f"<div style='height: 150px; overflow-y: auto; padding: 10px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #e0e0e0;'>{paper.get('abstract')}</div>", unsafe_allow_html=True)
                
                with col2:
                    # URL button
                    if paper.get('url'):
                        st.markdown(f"<a href='{paper.get('url')}' target='_blank'><div style='background-color: #4361ee; color: white; padding: 8px 12px; border-radius: 6px; text-align: center; margin-bottom: 10px; font-weight: 500;'>Read Paper</div></a>", unsafe_allow_html=True)
                    
                    # Analyze button
                    st.markdown(f"<button id='analyze_paper_{i}' style='background-color: #6c757d; color: white; padding: 8px 12px; border-radius: 6px; text-align: center; width: 100%; border: none; cursor: pointer; font-weight: 500;'>Analyze Paper</button>", unsafe_allow_html=True)
                    analyze_button = st.button(f"Analyze", key=f"analyze_paper_{i}", use_container_width=True)
                
                if analyze_button:
                    with st.spinner("Analyzing paper..."):
                        # Create prompt for analysis
                        analysis_prompt = f"""
                        Analyze the following academic paper:
                        
                        Title: {paper.get('title', 'Untitled')}
                        Authors: {authors}
                        Year: {paper.get('year', 'N/A')}
                        Abstract: {paper.get('abstract', 'No abstract available')}
                        
                        Provide a comprehensive analysis including:
                        1. Main research questions and objectives
                        2. Key methodology used
                        3. Most significant findings
                        4. Limitations of the research
                        5. Potential applications and implications
                        
                        Format your analysis as a well-structured report with headings and bullet points where appropriate.
                        """
                        
                        # Use the agent to analyze
                        response = multimodal_agent.run(analysis_prompt)
                        
                        # Display analysis with better styling
                        st.markdown("""
                        <div style="background-color: #f0f7ff; border-radius: 8px; padding: 15px; margin-top: 15px; border-left: 4px solid #4361ee;">
                            <h4 style="margin-top: 0; color: #333;">AI Analysis</h4>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(response.content)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Feedback row
                        col1, col2 = st.columns([1, 9])
                        with col1:
                            st.button("👍", key=f"paper_helpful_{i}")
                            st.button("👎", key=f"paper_not_helpful_{i}")
        
        # Export options
        st.markdown("### Export Options")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                label="Export as CSV",
                data="Title,Authors,Year,Venue\n" + "\n".join([
                    f"{p.get('title', 'Untitled')},{','.join([a.get('name', '') for a in p.get('authors', [])])},{p.get('year', 'N/A')},{p.get('venue', 'N/A')}"
                    for p in papers
                ]),
                file_name="research_papers.csv",
                mime="text/csv"
            )
        with col2:
            st.download_button(
                label="Export as Bibliography",
                data="\n\n".join([
                    f"{', '.join([a.get('name', '') for a in p.get('authors', [])])} ({p.get('year', 'N/A')}). {p.get('title', 'Untitled')}. {p.get('venue', '')}."
                    for p in papers
                ]),
                file_name="bibliography.txt",
                mime="text/plain"
            )
        st.markdown('</div>', unsafe_allow_html=True)

# Dashboard Tab
with tab4:
    st.markdown('<h2 class="section-heading">Research Dashboard</h2>', unsafe_allow_html=True)
    
    # Dashboard card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    if 'processed_data' in st.session_state or 'papers' in st.session_state:
        st.markdown("### Research Activity Overview")
        
        # Create metrics with better styling
        col1, col2, col3 = st.columns(3)
        
        # PDF metrics
        pdf_count = 0
        word_count = 0
        if 'processed_data' in st.session_state and 'pdf' in st.session_state.processed_data:
            pdf_data = st.session_state.processed_data['pdf']
            pdf_count = pdf_data.get('chunk_count', 0)
            word_count = pdf_data.get('word_count', 0)
        
        # Paper metrics
        paper_count = 0
        if 'papers' in st.session_state:
            paper_count = len(st.session_state.papers)
        
        with col1:
            st.markdown("""
            <div class="metric-card" style="border-left: 4px solid #4361ee;">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Text Segments Analyzed</div>
            </div>
            """.format(pdf_count), unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-card" style="border-left: 4px solid #2e8540;">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Words Processed</div>
            </div>
            """.format(word_count), unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-card" style="border-left: 4px solid #e9c46a;">
                <div class="metric-value">{}</div>
                <div class="metric-label">Papers Retrieved</div>
            </div>
            """.format(paper_count), unsafe_allow_html=True)
        
        # Create a chart if papers are available
        if 'papers' in st.session_state and st.session_state.papers:
            st.markdown("### Research Paper Analysis")
            
            # Get years from papers
            years = [paper.get('year', None) for paper in st.session_state.papers]
            years = [y for y in years if y]  # Filter out None values
            
            if years:
                # Count papers by year
                year_counts = {}
                for year in years:
                    if year in year_counts:
                        year_counts[year] += 1
                    else:
                        year_counts[year] = 1
                
                # Create DataFrame
                df = pd.DataFrame({
                    'Year': list(year_counts.keys()),
                    'Papers': list(year_counts.values())
                })
                
                # Sort by year
                df = df.sort_values('Year')
                
                # Create chart with better styling
                fig = px.bar(
                    df,
                    x='Year',
                    y='Papers',
                    title='Papers by Publication Year',
                    labels={'Papers': 'Number of Papers', 'Year': 'Publication Year'},
                    color='Papers',
                    color_continuous_scale='Viridis',
                    template='plotly_white'
                )
                
                # Improve chart styling
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=40, b=20),
                    title_font=dict(size=18),
                    title_x=0.5
                )
                
                # Make the bars rounder
                fig.update_traces(marker_line_width=0)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add some insights about the papers
                st.markdown("### Publication Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Most common year
                    most_common_year = max(year_counts.items(), key=lambda x: x[1])[0]
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0;">
                        <h4 style="margin-top: 0;">Publication Peak</h4>
                        <p>The highest number of papers were published in <strong>{most_common_year}</strong>, suggesting this was a period of significant research activity in this area.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col2:
                    # Year range
                    year_range = max(years) - min(years) if len(years) > 1 else 0
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0;">
                        <h4 style="margin-top: 0;">Research Timeline</h4>
                        <p>The research on this topic spans <strong>{year_range} years</strong>, from {min(years)} to {max(years)}, showing the evolution of this field over time.</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        # Empty dashboard state
        st.markdown("""
        <div style="text-align: center; padding: 40px 20px; background-color: #f8f9fa; border-radius: 10px; border: 1px dashed #ced4da;">
            <span style="font-size: 3rem; color: #6c757d; display: block; margin-bottom: 20px;">📊</span>
            <h3 style="margin-bottom: 15px; color: #495057;">Your Dashboard is Empty</h3>
            <p style="color: #6c757d; margin-bottom: 20px;">Process PDFs or search for papers to populate your research dashboard with insights and visualizations.</p>
            <p style="font-size: 0.9rem; color: #6c757d;">Start by uploading documents in the PDF Analysis tab or searching for papers in the Research Papers tab.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>ResearchGPT — AI-Powered Research Assistant | Version 2.0</p>
    <p>© 2025 GROUP 10 CSAIML. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)

# Clean up any temporary files when the app closes
def cleanup():
    if 'video_path' in st.session_state:
        try:
            Path(st.session_state.video_path).unlink(missing_ok=True)
        except:
            pass

# Register the cleanup function to run when the session ends
import atexit
atexit.register(cleanup)