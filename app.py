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

# Load environment variables
load_dotenv()

# Set API key
API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBTOZ78XdS9w4-3LzYna1wIDxjhItGmLws")
CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "service.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH
genai.configure(api_key=API_KEY)

# Cache for processed data
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {}

# Set up Streamlit page configuration
st.set_page_config(
    page_title="IntelCite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Updated CSS with vibrant colors and modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    /* Global styling */
    * {
        font-family: 'Poppins', sans-serif;
        transition: all 0.2s ease;
    }

    .main {
        background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
    }

    /* Header */
    .header {
        padding: 2rem;
        text-align: center;
        background: linear-gradient(90deg, #4f46e5, #22c55e);
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }

    .subtitle {
        font-size: 1.2rem;
        color: #fef3c7;
        opacity: 0.9;
    }

    /* Card */
    .card {
        background: #ffffff;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.05);
        border: 1px solid #e5e7eb;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.1);
    }

    /* Section headings */
    .section-heading {
        font-size: 1.6rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #4f46e5;
    }

    /* Buttons */
    .primary-button {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        color: #ffffff;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .primary-button:hover {
        background: linear-gradient(90deg, #4338ca, #6d28d9);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        transform: scale(1.05);
    }

    /* Info box */
    .info-box {
        background: #fef3c7;
        border-left: 5px solid #f59e0b;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        color: #1f2937;
    }

    /* Result box */
    .result-box {
        background: #f8fafc;
        border: 1px solid #d1d5db;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #e5e7eb;
        padding: 0.5rem;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        background: #ffffff;
        border-radius: 10px;
        padding: 12px 20px;
        font-weight: 500;
        color: #4b5563;
        border: 1px solid #d1d5db;
    }

    .stTabs [aria-selected="true"] {
        background: #4f46e5 !important;
        color: #ffffff !important;
        border-color: #4f46e5;
    }

    /* Metrics */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #d1d5db;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4f46e5;
    }

    .metric-label {
        font-size: 1rem;
        color: #6b7280;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        color: #ffffff;
        font-size: 0.95rem;
        background: linear-gradient(180deg, #4f46e5, #22c55e);
    }

    /* Progress indicators */
    .progress-container {
        margin: 1.5rem 0;
    }

    /* Form elements */
    input, textarea, select {
        border-radius: 8px !important;
        border: 1px solid #d1d5db !important;
        padding: 0.75rem !important;
        background: #f9fafb !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1f2937;
        background: #f3f4f6;
        border-radius: 8px;
        padding: 0.75rem;
    }

    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #1f2937;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    /* Custom icons for tabs */
    .tab-icon {
        margin-right: 8px;
        font-size: 1.2rem;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 1.8rem;
        }
        .card {
            padding: 1rem;
        }
        .metric-card {
            margin-bottom: 1rem;
        }
    }

    /* Spinner styling */
    .stSpinner > div {
        border-color: #4f46e5 transparent transparent transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<div class="header">', unsafe_allow_html=True)
st.markdown('<h1 class="main-title"><span class="tab-icon">🔬</span> IntelCite</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Your AI-Powered Research Companion</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Function definitions (unchanged)
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_bytes = pdf.read()
            pdf_reader = PdfReader(BytesIO(pdf_bytes))
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            pdf.seek(0)
        except Exception as e:
            logger.error(f"Error reading {pdf.name}: {str(e)}")
            st.error(f"Error reading {pdf.name}: {str(e)}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

@st.cache_resource
def get_vector_store(text_chunks, namespace="default"):
    if not text_chunks:
        logger.warning("No text chunks provided")
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(f"faiss_index_{namespace}")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        st.error(f"Error creating vector store: {str(e)}")
        return None

def load_vector_store(namespace="default"):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
        return FAISS.load_local(f"faiss_index_{namespace}", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        return None

def get_conversational_chain(model_name="gemini-2.0-flash-exp", temperature=0.3):
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

def process_user_query(user_question, namespace="default"):
    try:
        vector_store = load_vector_store(namespace)
        if not vector_store:
            return "Error: Unable to load the knowledge base. Please process your documents first."
        docs = vector_store.similarity_search(user_question, k=4)
        chain = get_conversational_chain()
        if not chain:
            return "Error: Unable to create the AI chain. Please try again later."
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return f"An error occurred while processing your question: {str(e)}"

def initialize_agent():
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

def search_academic_papers(query, max_results=5):
    try:
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,authors,year,abstract,url,venue"
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            return response.json().get('data', [])
        else:
            logger.error(f"Failed to search papers: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error searching papers: {str(e)}")
        return []

# Initialize agent
multimodal_agent = initialize_agent()

# Sidebar
with st.sidebar:
    st.markdown('<div style="background: linear-gradient(180deg, #4f46e5, #22c55e); padding: 1rem; border-radius: 12px;">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Settings", unsafe_allow_html=True)
    st.markdown("#### <span class='tab-icon'>🤖</span> Model Selection", unsafe_allow_html=True)
    model_name = st.selectbox(
        "Choose AI Model",
        ["gemini-2.0-flash-exp", "gemini-2.0-pro", "gemini-2.0-vision"],
        index=0,
        help="Select the AI model for analysis"
    )
    st.markdown("#### <span class='tab-icon'>🌡️</span> Response Creativity", unsafe_allow_html=True)
    temperature = st.slider(
        "Creativity Level",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Higher values make output more creative"
    )
    st.markdown("---")
    st.markdown("### <span class='tab-icon'>ℹ️</span> About", unsafe_allow_html=True)
    st.markdown("""
    **IntelCite** is your AI-powered research assistant, helping you:
    - 📄 Analyze PDFs
    - 🎥 Process videos
    - 🔍 Find papers
    - 📊 Visualize data
    *Developed by GROUP 10 CSAIML*
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "<span class='tab-icon'>📄</span> PDF Analysis",
    "<span class='tab-icon'>🎥</span> Video Analysis",
    "<span class='tab-icon'>🔍</span> Research Papers",
    "<span class='tab-icon'>📊</span> Dashboard"
])

# PDF Analysis Tab
with tab1:
    st.markdown('<h2 class="section-heading">PDF Document Analysis</h2>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Upload Documents")
        st.markdown("""
        <div class="info-box">
        <p><span class='tab-icon'>📤</span> Upload PDF files to create a searchable knowledge base.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4f46e5, #7c3aed); color: white; border-radius: 12px; padding: 20px; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center;">
            <span style="font-size: 3rem;">📄</span>
            <p style="margin-top: 12px; font-weight: 600;">PDF Files</p>
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
            st.markdown(f"<div style='padding: 8px; background: #f8fafc; border-radius: 6px; margin: 4px 0;'>📄 {file}</div>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            process_button = st.button(
                "<span class='tab-icon'>⚙️</span> Process PDFs",
                key="process_pdf_button",
                help="Start processing uploaded PDFs",
                use_container_width=True
            )
        if process_button:
            with st.spinner("Processing your documents..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text:
                    word_count = len(raw_text.split())
                    reading_time = round(word_count / 200)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks, namespace="pdf")
                    if vector_store:
                        st.session_state.processed_data['pdf'] = {
                            'word_count': word_count,
                            'reading_time': reading_time,
                            'chunk_count': len(text_chunks)
                        }
                        st.success("✅ Documents processed successfully!")
                        st.markdown("### Document Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown("""
                            <div class="metric-card" style="border-left: 5px solid #4f46e5;">
                                <div class="metric-value">{:,}</div>
                                <div class="metric-label">Words Analyzed</div>
                            </div>
                            """.format(word_count), unsafe_allow_html=True)
                        with col2:
                            st.markdown("""
                            <div class="metric-card" style="border-left: 5px solid #22c55e;">
                                <div class="metric-value">{} min</div>
                                <div class="metric-label">Reading Time</div>
                            </div>
                            """.format(reading_time), unsafe_allow_html=True)
                        with col3:
                            st.markdown("""
                            <div class="metric-card" style="border-left: 5px solid #f43f5e;">
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
        <p>Please upload and process PDF documents first.</p>
        </div>
        """, unsafe_allow_html=True)
    user_query = st.text_area(
        "Your question",
        placeholder="What are the main findings or insights from these documents?",
        help="Ask a specific question about the content of your uploaded PDFs"
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        submit_query = st.button(
            "<span class='tab-icon'>❓</span> Ask AI",
            key="pdf_query_button",
            use_container_width=True
        )
    if submit_query and user_query:
        if 'processed_data' in st.session_state and 'pdf' in st.session_state.processed_data:
            with st.spinner("Analyzing your question..."):
                response = process_user_query(user_query, namespace="pdf")
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown("### AI Response:")
                st.markdown(response)
                st.markdown('</div>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 1, 4])
                with col1:
                    st.button("<span class='tab-icon'>👍</span> Helpful", key="helpful_btn")
                with col2:
                    st.button("<span class='tab-icon'>👎</span> Not Helpful", key="not_helpful_btn")
        else:
            st.error("Please process PDFs before asking questions.")
    st.markdown('</div>', unsafe_allow_html=True)

# Video Analysis Tab
with tab2:
    st.markdown('<h2 class="section-heading">Video Content Analysis</h2>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Upload Video")
        st.markdown("""
        <div class="info-box">
        <p><span class='tab-icon'>🎬</span> Upload a video to analyze its content, audio, and visuals.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #22c55e, #4ade80); color: white; border-radius: 12px; padding: 20px; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center;">
            <span style="font-size: 3rem;">🎥</span>
            <p style="margin-top: 12px; font-weight: 600;">Video Files</p>
        </div>
        """, unsafe_allow_html=True)
    video_file = st.file_uploader(
        "Drag and drop a video file here",
        type=['mp4', 'mov', 'avi'],
        help="Upload a video file (5 minutes or less recommended)"
    )
    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name
        st.video(video_path, format="video/mp4", start_time=0)
        st.session_state.video_path = video_path
    st.markdown('</div>', unsafe_allow_html=True)

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
            placeholder="E.g., What are the main points discussed in this video?"
        )
        analyze_button = st.button(
            "<span class='tab-icon'>🔍</span> Analyze Video",
            key="analyze_video_button",
            use_container_width=False
        )
        if analyze_button and user_query_video:
            try:
                with st.spinner("Processing video..."):
                    processed_video = upload_file(video_path)
                    st.markdown('<div class="progress-container">', unsafe_allow_html=True)
                    progress_bar = st.progress(0)
                    for i in range(10):
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)
                        if processed_video.state.name != "PROCESSING":
                            break
                        progress_bar.progress((i + 1) * 10)
                    st.markdown('</div>', unsafe_allow_html=True)
                    analysis_prompt = f"""
                    You are an expert video content analyzer. Analyze the uploaded video thoroughly.
                    
                    Analysis Type: {analysis_type}
                    User Query: {user_query_video}
                    
                    Provide a {analysis_type.lower()} that addresses the user's query. Format your response using markdown for better readability.
                    Include sections with headings, bullet points where appropriate, and highlight key insights.
                    """
                    response = multimodal_agent.run(analysis_prompt, videos=[processed_video])
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.markdown("### Video Analysis Results")
                    st.markdown(response.content)
                    st.markdown('</div>', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns([1, 1, 4])
                    with col1:
                        st.button("<span class='tab-icon'>👍</span> Helpful", key="video_helpful_btn")
                    with col2:
                        st.button("<span class='tab-icon'>👎</span> Not Helpful", key="video_not_helpful_btn")
            except Exception as error:
                st.error(f"An error occurred during analysis: {error}")
            finally:
                try:
                    Path(video_path).unlink(missing_ok=True)
                except:
                    pass
        st.markdown('</div>', unsafe_allow_html=True)

# Research Papers Tab
with tab3:
    st.markdown('<h2 class="section-heading">Academic Research Papers</h2>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Search Academic Papers")
        st.markdown("""
        <div class="info-box">
        <p><span class='tab-icon'>🔎</span> Find relevant academic papers from trusted sources.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f43f5e, #fb7185); color: white; border-radius: 12px; padding: 20px; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center;">
            <span style="font-size: 3rem;">🔍</span>
            <p style="margin-top: 12px; font-weight: 600;">Research Papers</p>
        </div>
        """, unsafe_allow_html=True)
    paper_search_query = st.text_input(
        "Research topic or keywords",
        placeholder="E.g., quantum computing applications",
        help="Enter keywords to search for papers"
    )
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        max_results = st.select_slider(
            "Number of results",
            options=[3, 5, 7, 10],
            value=5
        )
    with col2:
        search_button = st.button(
            "<span class='tab-icon'>🔎</span> Search Papers",
            key="search_papers_button",
            use_container_width=True
        )
    if search_button and paper_search_query:
        with st.spinner("Searching academic databases..."):
            papers = search_academic_papers(paper_search_query, max_results)
            if papers:
                st.session_state.papers = papers
                st.success(f"Found {len(papers)} relevant papers!")
            else:
                st.warning("No papers found. Try different keywords.")
    st.markdown('</div>', unsafe_allow_html=True)

    if 'papers' in st.session_state and st.session_state.papers:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Search Results")
        col1, col2 = st.columns(2)
        with col1:
            sort_option = st.selectbox(
                "Sort by",
                ["Relevance", "Year (Newest First)", "Year (Oldest First)"],
                index=0
            )
        papers = st.session_state.papers
        if sort_option == "Year (Newest First)":
            papers = sorted(papers, key=lambda x: x.get('year', 0), reverse=True)
        elif sort_option == "Year (Oldest First)":
            papers = sorted(papers, key=lambda x: x.get('accept_multiple_filesyear', 0))
        for i, paper in enumerate(papers):
            with st.expander(f"{paper.get('title', 'Untitled Paper')} ({paper.get('year', 'N/A')})"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    authors = ", ".join([author.get('name', '') for author in paper.get('authors', [])])
                    st.markdown(f"**Authors:** {authors}")
                    if paper.get('venue'):
                        st.markdown(f"**Published in:** {paper.get('venue')}")
                    if paper.get('abstract'):
                        st.markdown("**Abstract:**")
                        st.markdown(f"<div style='height: 150px; overflow-y: auto; padding: 10px; background-color: #f8f9fa; border-radius: 5px; border: 1px solid #e0e0e0;'>{paper.get('abstract')}</div>", unsafe_allow_html=True)
                with col2:
                    if paper.get('url'):
                        st.markdown(f"<a href='{paper.get('url')}' target='_blank'><div style='background-color: #4f46e5; color: white; padding: 8px 12px; border-radius: 6px; text-align: center; margin-bottom: 10px; font-weight: 500;'>Read Paper</div></a>", unsafe_allow_html=True)
                    analyze_button = st.button(
                        "<span class='tab-icon'>🔍</span> Analyze",
                        key=f"analyze_paper_{i}",
                        use_container_width=True
                    )
                if analyze_button:
                    with st.spinner("Analyzing paper..."):
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
                        response = multimodal_agent.run(analysis_prompt)
                        st.markdown("""
                        <div style="background-color: #f0f7ff; border-radius: 8px; padding: 15px; margin-top: 15px; border-left: 4px solid #4f46e5;">
                            <h4 style="margin-top: 0; color: #333;">AI Analysis</h4>
                        """, unsafe_allow_html=True)
                        st.markdown(response.content)
                        st.markdown("</div>", unsafe_allow_html=True)
                        col1, col2 = st.columns([1, 9])
                        with col1:
                            st.button("<span class='tab-icon'>👍</span>", key=f"paper_helpful_{i}")
                            st.button("<span class='tab-icon'>👎</span>", key=f"paper_not_helpful_{i}")
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
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if 'processed_data' in st.session_state or 'papers' in st.session_state:
        st.markdown("### Research Activity Overview")
        col1, col2, col3 = st.columns(3)
        pdf_count = st.session_state.processed_data.get('pdf', {}).get('chunk_count', 0)
        word_count = st.session_state.processed_data.get('pdf', {}).get('word_count', 0)
        paper_count = len(st.session_state.get('papers', []))
        with col1:
            st.markdown("""
            <div class="metric-card" style="border-left: 5px solid #4f46e5;">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Text Segments</div>
            </div>
            """.format(pdf_count), unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card" style="border-left: 5px solid #22c55e;">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Words Processed</div>
            </div>
            """.format(word_count), unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card" style="border-left: 5px solid #f43f5e;">
                <div class="metric-value">{}</div>
                <div class="metric-label">Papers Retrieved</div>
            </div>
            """.format(paper_count), unsafe_allow_html=True)
        if 'papers' in st.session_state and st.session_state.papers:
            st.markdown("### Research Paper Analysis")
            years = [paper.get('year', None) for paper in st.session_state.papers]
            years = [y for y in years if y]
            if years:
                year_counts = {year: years.count(year) for year in set(years)}
                df = pd.DataFrame({
                    'Year': list(year_counts.keys()),
                    'Papers': list(year_counts.values())
                }).sort_values('Year')
                fig = px.bar(
                    df,
                    x='Year',
                    y='Papers',
                    title='Papers by Publication Year',
                    labels={'Papers': 'Number of Papers', 'Year': 'Publication Year'},
                    color='Papers',
                    color_continuous_scale='Plasma',
                    template='plotly_white'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=40, b=20),
                    title_font=dict(size=18, family='Poppins'),
                    title_x=0.5
                )
                fig.update_traces(marker_line_width=0)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("### Publication Insights")
                col1, col2 = st.columns(2)
                with col1:
                    most_common_year = max(year_counts.items(), key=lambda x: x[1])[0]
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0;">
                        <h4 style="margin-top: 0;">Publication Peak</h4>
                        <p>The highest number of papers were published in <strong>{most_common_year}</strong>.</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    year_range = max(years) - min(years) if len(years) > 1 else 0
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0;">
                        <h4 style="margin-top: 0;">Research Timeline</h4>
                        <p>The research spans <strong>{year_range} years</strong>, from {min(years)} to {max(years)}.</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #f3f4f6, #e5e7eb); border-radius: 12px; border: 1px dashed #9ca3af;">
            <span style="font-size: 3.5rem; color: #6b7280; display: block; margin-bottom: 20px;">📊</span>
            <h3 style="margin-bottom: 15px; color: #1f2937;">Your Dashboard Awaits</h3>
            <p style="color: #6b7280; margin-bottom: 20px;">Analyze PDFs or search papers to unlock insights.</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>IntelCite — AI-Powered Research Assistant | Version 2.0</p>
    <p>© 2025 GROUP 10 CSAIML. All rights reserved.</p>
    <p>
        <a href="#" style="color: #fef3c7; margin: 0 10px;"><span class='tab-icon'>🌐</span> Website</a> |
        <a href="#" style="color: #fef3c7; margin: 0 10px;"><span class='tab-icon'>📧</span> Contact</a>
    </p>
</div>
""", unsafe_allow_html=True)

# Cleanup
def cleanup():
    if 'video_path' in st.session_state:
        try:
            Path(st.session_state.video_path).unlink(missing_ok=True)
        except:
            pass

import atexit
atexit.register(cleanup)
