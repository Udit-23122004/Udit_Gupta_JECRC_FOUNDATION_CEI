# 🚗 DriveWise – Metadata-Aware Automotive RAG Assistant

DriveWise is a Retrieval-Augmented Generation (RAG) based automotive assistant that answers user questions using official car brochures.

## Features

- Brochure-based question answering
- Metadata filtering by brand and car model
- Section-aware retrieval
- FAISS vector database
- CrossEncoder re-ranking
- Context window control
- Source attribution with brochure page numbers
- Query logging and response time tracking
- Streamlit web interface

## Technologies Used

- Python
- LangChain
- FAISS
- Sentence Transformers
- CrossEncoder
- Gemini API
- Streamlit
- Pandas

## Supported Cars

- Hyundai Venue
- Hyundai Creta
- Tata Nexon
- Tata Punch

## RAG Workflow

PDF Brochures → Text Extraction → Chunking → Embeddings → FAISS → Metadata Filtering → Re-ranking → Context Control → Gemini → Answer + Sources

## Author

Udit Gupta