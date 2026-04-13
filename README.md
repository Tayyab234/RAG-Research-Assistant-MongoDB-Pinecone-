📚 DocuMind RAG Engine

DocuMind is a multi-mode Retrieval-Augmented Generation (RAG) system designed for intelligent document understanding. It enables accurate question answering, structured content generation, and deep research analysis — all while enforcing strict context grounding and minimizing hallucinations.

🚀 Features
🧠 Multi-Mode Intelligence

Supports three advanced processing modes:

-  QA Mode → Precise, context-grounded answers
-  Generation Mode → MCQs, summaries, and structured outputs
-  Research Mode → Deep analytical reasoning across documents

📄 Document-Centric RAG

-  Fully grounded in uploaded documents
-  MongoDB-based chunk storage
-  Pinecone-powered vector similarity search

🔍 Smart Retrieval System

-  Dynamic k selection based on query type
-  Supports MMR and similarity-based retrieval
-  File-aware chunk filtering for precise context

⚖️ Strict / Flexible Reasoning Control

-  Strict Mode → Answers only from retrieved documents
-  Non-Strict Mode → Controlled fallback when context is insufficient
-  Built-in hallucination prevention layer

🧩 Map-Reduce Processing

  Efficient handling of large documents:
  
  -  MAP → Chunk-level reasoning
  -  REDUCE → Final synthesis
  -  Token-aware batching for optimized performance

📊 Adaptive Query Analysis

  Automatically determines:

  -  Query type (QA / Generation / Research)
  -  Scope (full document vs partial retrieval)
  -  Query optimization strategy

🏗️ Architecture

    User Query
        ↓
    Query Analyzer (mode + scope detection)
        ↓
    Retriever (Pinecone + filters)
        ↓
    Context Builder (MongoDB chunks)
        ↓
    Processing Engine:
        ├── QA Pipeline
        ├── Generation Pipeline
        └── Research Pipeline
        ↓
    Map-Reduce Engine (if needed)
        ↓
    LLM Response Engine
        ↓
    Final Answer

⚙️ Tech Stack

-  FastAPI – Backend API framework
-  LangChain – LLM orchestration
-  Pinecone – Vector database
-  MongoDB – Document chunk storage
-  OpenAI / LLMs – Reasoning engine
-  tiktoken – Token management

📦 Core Components

  1. Query Analyzer

    - Detects user intent
    - Optimizes queries
    - Determines processing mode and scope

  2. Retrieval Layer

    - Embedding-based vector search
    - File-level filtering
    - Active document selection

  3. Map-Reduce Engine

    - Splits large documents into batches
    - Performs chunk-level reasoning
    - Aggregates final output

  4. Strict Context Guard

    - Prevents hallucination
    - Ensures document grounding
    - Enforces retrieval dependency

🔐 Safety Design

  ❌ No unrestricted hallucination in strict mode
  ❌ No external knowledge injection (strict mode)
  ✅ Fully document-grounded responses
  ✅ Controlled fallback in non-strict mode

⚡ Performance Optimization

  Chunk size: 500 tokens
  Overlap: 100 tokens
  Map batch size: 3–5 chunks
  Token-aware processing
  Async retrieval pipeline

🧠 Key Innovations

  DocuMind is more than a standard RAG pipeline:

  - Multi-mode reasoning engine
  - Strict hallucination control layer
  - Map-reduce document intelligence
  - Scope-aware query analysis
  - Hybrid retrieval architecture


📌 Example Usage

  🧠 QA Mode (Precise Document Grounded Answers)

   - According to the uploaded climate change report, what are the main human activities   responsible for greenhouse gas emissions, and how do they individually contribute to global  warming?

  🧩 Generation Mode (Structured Output Creation)

    - Generate 10 MCQs from the climate change document focusing on causes, effects, and mitigation strategies. Each question must include 4 options and exactly one correct answer with explanation.
    
    - From the document, create:
      - 5 MCQs
      - 5 True/False statements
      - 3 short-answer questions

      Ensure all questions are strictly based on the provided context.

  🔬 Research Mode (Deep Analytical Reasoning)

    - Analyze the climate change document and evaluate the effectiveness of the mitigation strategies mentioned. Compare their potential impact on developing vs developed countries based on the document context.