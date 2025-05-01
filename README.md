
# IntelCite: RAG Based Research Paper Summarizer

## 📄 Overview
**IntelCite** is an advanced document analysis system specifically designed for academic research papers. Leveraging **Retrieval-Augmented Generation (RAG)**, it extracts, processes, and summarizes content from PDFs, enabling users to query complex scholarly documents and receive accurate, contextualized responses.

---

## 🚀 Features

- **Intelligent PDF Processing**  
  Extracts and preserves structure including text, tables, and figures.

- **Advanced RAG Pipeline**  
  Combines retrieval-based and generative AI for precise, context-aware answers.

- **Research-Focused Summarization**  
  Creates concise summaries tailored to academic writing.

- **Interactive Query System**  
  Ask domain-specific questions about any paper content.

- **Citation-Aware Responses**  
  Maintains traceability to source sections in every answer.

---

## 🧠 Technical Architecture

### 📥 Document Ingestion & Processing
- Loads PDFs using `PyPDFLoader` with formatting preservation.
- Segments content with `RecursiveCharacterTextSplitter` using research-optimized settings.
- Maintains hierarchical document structure across sections.

### 🔍 Embedding & Vectorization
- Generates vector embeddings via `GoogleGenerativeAIEmbeddings`.
- Stores vectors in a `Chroma` database for similarity-based search.
- Uses sliding-window context to preserve relevance during retrieval.

### 🤖 Query Processing & Generation
- Retrieves semantically similar document segments.
- Generates answers via `ChatGoogleGenerativeAI`.
- Uses academic prompting techniques to retain scholarly tone and accuracy.

---

## 📦 Dependencies

This project uses the following Python libraries:

- `langchain_community` – for LLM workflow orchestration  
- `pypdf` – for PDF content extraction  
- `langchain_google_genai` – Google Generative AI model integration  
- `langchain_chroma` – for building and querying the vector store

---

## 🔧 Installation

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install langchain-community pypdf langchain-google-genai langchain-chroma
```

---

## 📝 Usage Guide

### 🔹 Basic Usage

```python
from pdf_summarizer_rag import PDFSummarizer

# Initialize the summarizer
summarizer = PDFSummarizer("path/to/research_paper.pdf")

# Generate summary
summary = summarizer.generate_summary()
print(summary)

# Ask a specific question
answer = summarizer.query("What methodology was used in this research?")
print(answer)
```

---

### 🔹 Advanced Configuration

```python
from pdf_summarizer_rag import PDFSummarizer, SummarizerConfig

config = SummarizerConfig(
    chunk_size=1000,
    chunk_overlap=200,
    embedding_model="models/embedding-001",
    llm_model="gemini-pro",
    temperature=0.3
)

summarizer = PDFSummarizer("path/to/research_paper.pdf", config=config)

methodology = summarizer.extract_section("methodology")
results = summarizer.extract_section("results")
```

---

## ⚙️ Performance Optimization

- **Semantic Chunking**: Splits content using document structure
- **Context Preservation**: Maintains section hierarchy
- **Embedding Caching**: Avoids recomputation
- **Query Preprocessing**: Reformulates prompts for optimal retrieval

---

## 🌱 Future Development

- Integration with academic databases (e.g., arXiv, PubMed)
- Multi-document summarization & cross-referencing
- Support for equations and LaTeX math handling
- Export annotated insights and summaries

---

## 📄 License

MIT License

Copyright (c) 2025 **Sahil Kumbhar**

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions...

(You may add the full license text if needed.)
