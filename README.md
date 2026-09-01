# Deterministic Support Triage Agent

🚀 **Live Demo:** [Open NEXA Support Triage Dashboard](https://support-triage-agent-havm8qsnc7y5ma8kktsupu.streamlit.app/)

A deterministic AI support triage system using TF-IDF RAG, cosine similarity, regex-based risk detection, and optional local Llama 3 generation through Ollama.

## Overview

The Deterministic Support Triage Agent processes technical support queries and decides whether they should:

- 🤖 Receive an automatic AI-generated response
- 📚 Retrieve relevant information from a verified knowledge base
- ⚠️ Be escalated to human review due to risky content
- 🔍 Be escalated because of low retrieval confidence

The project demonstrates a practical AI pipeline combining Retrieval-Augmented Generation (RAG), deterministic routing, risk detection, and local LLM integration.

## Architecture

```text
User Query
    │
    ▼
Risk Detection Engine
    │
    ├── Risky Query ──────────────► HUMAN TRIAGE
    │
    ▼
TF-IDF RAG Engine
    │
    ▼
Cosine Similarity Retrieval
    │
    ├── Low Similarity ───────────► HUMAN TRIAGE
    │
    ▼
Strict Local LLM Generator
    │
    ▼
AUTO RESPONSE
Features
🔎 TF-IDF RAG Retrieval

The system converts knowledge-base documents into TF-IDF vectors and retrieves the most relevant document using cosine similarity.

User Query
    ↓
TF-IDF Vectorization
    ↓
Cosine Similarity
    ↓
Most Relevant Knowledge Section
⚠️ Risk Detection Engine

Regex-based detection identifies potentially unreliable or high-liability concepts.

Examples include:

Overunity
Free energy
Perpetual motion
Anti-gravity blueprints
UFO propulsion
Warp drive generators
Zero-point power

Risky queries are automatically routed to:

HUMAN_TRIAGE
📊 Confidence-Based Routing

After retrieval, the similarity score is compared against a configurable threshold.

Similarity Score >= Threshold
        ↓
AUTO_RESPOND

Similarity Score < Threshold
        ↓
HUMAN_TRIAGE

This prevents the AI model from responding when the knowledge base does not confidently support the query.

🤖 Local LLM Generation

The project integrates with a locally running Llama model through Ollama.

The LLM receives:

The user's query
The retrieved knowledge-base context
A strict grounding instruction

The model is instructed to answer only using the provided context, reducing unsupported responses.

Technology Stack
Python
Scikit-learn
TF-IDF Vectorization
Cosine Similarity
Regular Expressions
Retrieval-Augmented Generation (RAG)
Ollama
Llama 3
Requests
GitHub Actions
Unit Testing
Project Structure
support-triage-agent/
│
├── triage_agent.py
├── tests/
│   └── test_triage_agent.py
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
Installation

Clone the repository:

git clone https://github.com/adityapandereio6-sketch/support-triage-agent.git

Navigate to the project directory:

cd support-triage-agent

Install dependencies:

pip install -r requirements.txt
Ollama Setup

Install Ollama and download the Llama model:

ollama pull llama3

Start the Ollama server:

ollama serve

The application connects to:

http://localhost:11434
Running the Project

Run:

python triage_agent.py

The verification runner tests:

A valid technical query
A risky query detected by the risk engine
Example Workflow
Valid Technical Query
        ↓
Risk Check Passed
        ↓
TF-IDF Retrieval
        ↓
Similarity Above Threshold
        ↓
Local Llama Generation
        ↓
AUTO_RESPOND
Testing

Run the unit tests locally:

python -m unittest discover -s tests

The test suite verifies:

TF-IDF retrieval functionality
Risk keyword detection
Human escalation for risky queries
Low-confidence routing
Automatic response routing

The project also uses GitHub Actions CI to automatically run tests whenever code is pushed to the repository.

Future Improvements
Streamlit web interface
Larger document knowledge bases
Multi-document retrieval
Persistent vector storage
Conversation history
REST API integration
Docker deployment
Advanced semantic embeddings
Author

Aditya Pandere

GitHub: https://github.com/adityapandereio6-sketch

License

This project is licensed under the MIT License.
