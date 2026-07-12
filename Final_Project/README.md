# 🚗 DriveWise – Metadata-Aware Automotive RAG Assistant

DriveWise is a Retrieval-Augmented Generation (RAG) based automotive assistant that answers user questions using information retrieved from official car brochures.

The system uses metadata-aware and section-aware retrieval to provide relevant, brochure-grounded answers with source page references.

## 🚀 Live Demo

Try the deployed DriveWise application:

**[Launch DriveWise](https://uditguptajecrcfoundationcei-er68zm6amacnqs8e6z7bzn.streamlit.app/)**

## ✨ Features

- Official brochure-based question answering
- Brand and car model metadata filtering
- Section-aware document retrieval
- Semantic search using vector embeddings
- FAISS vector database
- CrossEncoder re-ranking
- Context window control
- Gemini-based answer generation
- Brochure and page-level source attribution
- Query logging
- Response time tracking
- Interactive Streamlit web interface

## 🚘 Supported Cars

### Hyundai
- Venue
- Creta

### Tata
- Nexon
- Punch

## 🧠 RAG Architecture

Official Car Brochures  
↓  
PDF Text Extraction  
↓  
Document Creation  
↓  
Text Chunking  
↓  
Metadata Enrichment  
↓  
Sentence Transformer Embeddings  
↓  
FAISS Vector Database  
↓  
Brand + Model + Section Filtering  
↓  
Semantic Retrieval  
↓  
CrossEncoder Re-ranking  
↓  
Context Selection  
↓  
Gemini LLM  
↓  
Answer + Source Attribution

## 🛠️ Technologies Used

- Python
- LangChain
- FAISS
- Sentence Transformers
- CrossEncoder
- Gemini API
- Streamlit
- Pandas

## 🔍 How It Works

1. Official car brochures are processed and converted into text documents.
2. The extracted documents are divided into smaller text chunks.
3. Metadata such as brand, model, page number, source, and section is attached to each chunk.
4. Sentence Transformer embeddings are generated for the document chunks.
5. The embeddings are stored in a FAISS vector database.
6. The user's question is classified into a relevant section such as safety, ADAS, performance, or interior.
7. Metadata filtering restricts retrieval to the selected brand, model, and section.
8. Relevant chunks are retrieved using semantic similarity search.
9. A CrossEncoder re-ranks the retrieved chunks.
10. The highest-quality context is provided to Gemini.
11. Gemini generates a concise answer using only the brochure context.
12. The application displays the answer with brochure and page-level sources.

## 📁 Project Structure

    Final_Project/
    ├── app.py
    ├── DriveWise_RAG_Final.ipynb
    ├── README.md
    ├── requirements.txt
    └── drivewise_faiss/
        ├── index.faiss
        └── index.pkl

## ▶️ Run Locally

Install the required dependencies:

    pip install -r requirements.txt

Create a Streamlit secrets file:

    .streamlit/secrets.toml

Add your Gemini API key:

    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

Run the application:

    streamlit run app.py

## 🔐 Security

API keys are managed using Streamlit Secrets and are not stored directly in the source code or GitHub repository.

## 📊 Example Questions

- What are the safety features of Nexon?
- What ADAS features are available in Venue?
- What engine options are available in Creta?
- What infotainment features are available in Punch?
- Does Creta have a panoramic sunroof?

## 🎯 Future Improvements

- Add brochures from more automotive brands
- Support car-to-car comparison
- Add hybrid search using BM25 and vector retrieval
- Implement conversational memory
- Add multilingual question answering
- Improve query classification using an ML-based router

## 👨‍💻 Author

**Udit Gupta**

B.Tech – Artificial Intelligence & Data Science  
JECRC Foundation
