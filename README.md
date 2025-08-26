# Abstract Comparison Tool

This Streamlit-based web app allows users to compare two research abstracts by extracting and analyzing key elements such as technologies used, architecture style, development tools, and more. The tool uses NLP techniques and cosine similarity to quantify how similar two abstracts are, based on user-selected parameters.

## Features
- Supports both file upload (PDF, DOCX) and text input
- Parameter-based comparison (e.g., Technology, Architecture Style, etc.)
- Uses Sentence Transformers and cosine similarity for comparing abstracts
- Keyword extraction powered by Spacy & WordNet
- Outputs similarity score and extracted elements per abstract

## Tech Stack
- **Streamlit** — Web interface  
- **Sentence Transformers** — Semantic text embeddings  
- **Scikit-learn** — Cosine similarity computation  
- **spaCy** — NLP pipeline  
- **WordNet via NLTK** — Synonym enhancement for keyword matching  
- **PyMuPDF (fitz)** — PDF text extraction  
- **python-docx** — DOCX text extraction  

## Parameters for Comparison
Users can choose from the following abstract parameters:
- Technology Used  
- Development Tools Used  
- Project Objective  
- Architecture Style  
- System Functionalities  
- System Quality Attributes  
- Algorithms Used  
- Protocols Used  
- Processes Used  
