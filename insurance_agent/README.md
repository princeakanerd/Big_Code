# Agentic RAG Framework for Insurance Claim Adjudication

This project implements an autonomous, multi-agent Retrieval-Augmented Generation (RAG) system designed to navigate hierarchical health insurance documents and adjudicate complex medical claims. Moving beyond standard AI wrappers, this pipeline leverages layout-aware document indexing, a collaborative state machine orchestration, and integrated hallucination verification.

## Core Features

*   **Hierarchical Document Parsing**: Utilizes **Docling** for advanced layout analysis of complex PDF policies. Instead of indiscriminately chunking text, it intelligently identifies structure (paragraphs, lists) and maintains spatial metadata (exact page index tracking) to guarantee accurate downstream citations.
*   **Vector Search & Semantic Embeddings**: Employs **ChromaDB** coupled with Google GenAI semantic embedding models to reason against complex policy limits, coverage guidelines, and CPT-coded protocols.
*   **Multi-Agent LangGraph Orchestration**:
    *   **Information Extraction Agent**: Normalizes raw hospital bills into standardized, strict Pydantic models mapping ICD/CPT codes and procedures.
    *   **Planner Retrieval Agent**: Formulates multi-hop semantic queries to extract relevant clauses from the vector database.
    *   **Adjudication Agent**: Drafts coverage alignment decisions constrained tightly by the specific policy rules while embedding inline spatial citations.
    *   **Citation Verifier Agent (Judge)**: Acts as an independent error-checking node. Using cyclic feedback loops, it strictly audits draft decisions to flag and correct LLM hallucinations prior to outputting a response.
*   **Automated Evaluation Suite**: The integrated `evaluate.py` suite serves as an autonomous benchmark measuring exact-match classification capability and tracking the framework's Empirical Hallucination Catch Rate.

## Setup & Installation

1. Activate your virtual environment and install the required dependencies:
```bash
pip install langgraph langchain langchain-core langchain-chroma langchain-google-genai docling pydantic python-dotenv fpdf
```

2. Establish your Google Gemini API Key within a `.env` file or export it globally:
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

## Quick Start Guide

### 1. Build the Vector Database
Place your policy document directly into the `data/` directory named as `sample_policy.pdf`. Running the database architect will index the document and persist the embedded chunks into a local `chroma_db` folder.
```bash
python database.py
```

### 2. Execute a Standalone Workflow
Run the LangGraph node network against a single hospital bill. Watch the terminal for individual routing and verification updates as the state transitions.
```bash
python main.py
```

### 3. Run Benchmark Evacuation
Process multiple synthetic medical claims against your ground truths to prove 'Demonstrable Reliability' across the pipeline.
```bash
python evaluate.py
```
