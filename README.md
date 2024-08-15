# ChatPDF: A Streamlit-based RAG Application

## Overview

ChatPDF is a Streamlit application that allows users to upload PDF documents and chat with an AI assistant about the content of these documents. It uses Ollama's llama3 model for language processing, LangChain for Retrieval-Augmented Generation (RAG), and Chroma as a vector database.

## Features

- Upload multiple PDF documents
- Chat interface to ask questions about the uploaded documents
- Uses RAG to provide context-aware answers
- Leverages llama3 model through Ollama for natural language processing
- Utilizes Chroma as an efficient vector database for document storage and retrieval

## Prerequisites

- Python 3.8+
- Ollama with llama3 model installed

## Installation

1. Clone this repository:

2. Install the required packages:

pip install -r requirements.txt

3. Ensure Ollama is installed and the llama3 model is available.

## Usage

1. Run the Streamlit app:

2. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Upload one or more PDF documents using the file uploader.

4. Once the documents are ingested, you can start asking questions in the chat interface.

## How it Works

1. **Document Ingestion**: When you upload PDF files, the application uses PyPDF to read the content, splits it into chunks, and stores these chunks in a Chroma vector database using FastEmbed embeddings.

2. **Question Answering**: When you ask a question, the application:
- Retrieves relevant document chunks from the vector database
- Uses these chunks as context for the llama3 model
- Generates an answer based on the context and your question

3. **User Interface**: The Streamlit interface provides an intuitive way to upload documents and interact with the AI assistant.

## Files

- `main.py`: The Streamlit application script
- `rag.py`: Contains the RAG implementation using LangChain and Chroma