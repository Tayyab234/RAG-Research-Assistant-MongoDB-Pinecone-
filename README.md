# Hybrid RAG Research Assistant

🚧 Project Status: Work in Progress 🚧

## Overview
This project implements a Hybrid Retrieval-Augmented Generation (RAG) system that enables users to upload documents and perform intelligent question answering, research, and content generation over them using large language models.

The system follows a hybrid architecture using:
- **Vector Database (Pinecone)** for semantic search
- **Document Database (MongoDB)** for storing actual text chunks

## Features
- Document upload (PDF, TXT, DOCX)
- Automatic chunking and embedding generation
- Hybrid storage (Pinecone + MongoDB)
- Multiple retrieval modes:
  - QA mode
  - Generation mode
  - Research mode
- Scalable architecture for future improvements

## Architecture
User Query → Embedding → Pinecone Search → Retrieve IDs → Fetch Text from MongoDB → LLM Response

## Tech Stack
- Python
- FastAPI
- MongoDB
- Pinecone
- LangChain (partial usage)
- LLM integration

## Project Status
This project is actively under development. Current focus is on improving retrieval quality, optimizing chunking strategies, and refining multi-mode retrieval.

## Future Plans
- Agent integration
- Advanced reranking
- Better compression strategies
- Caching and performance optimization
