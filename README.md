# RAG_IMPLEMENTATION

# 📘 PDF Q&A App

This Streamlit application allows you to upload PDF documents and ask natural language questions about their contents. It uses **Google Gemini (Generative AI)** for embedding and answering queries and **FAISS** for vector-based document search.

---

## 🚀 Features

- Upload multiple PDFs
- Extract and index text using FAISS
- Ask questions in natural language
- Retrieve answers from document context using Google Gemini
- Download full Q&A history as a `.txt` file
- Clean and modern UI with Streamlit

---

## 🛠️ Tech Stack

- **Streamlit**: Interactive UI
- **LangChain**: Prompt engineering & document processing
- **FAISS**: Vector store for semantic similarity search
- **Google Generative AI (Gemini)**: Embedding & response generation
- **PyPDF2**: PDF text extraction
