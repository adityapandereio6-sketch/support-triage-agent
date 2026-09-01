# 🤖 Support Triage Agent

An AI-powered customer support triage system that analyzes incoming support queries, retrieves relevant knowledge-base information, detects potential risk, and routes queries based on confidence.

## Overview

Support Triage Agent is designed to simulate an intelligent first-line support system.

The application combines:

- TF-IDF-based information retrieval
- Cosine similarity for query matching
- Confidence-based routing
- Regex-based risk detection
- Local LLM response generation using Ollama and Llama 3
- Interactive Streamlit interface

The system helps determine whether a customer query can be answered automatically or should be escalated for additional support.

---

## ✨ Features

### 🔍 Knowledge Retrieval

The system converts support documents and user queries into TF-IDF vectors and uses cosine similarity to find the most relevant knowledge-base entry.

### 📊 Confidence-Based Routing

The similarity score is used to estimate retrieval confidence.

- High-confidence queries can be answered using retrieved knowledge.
- Low-confidence queries can be flagged for escalation.

### ⚠️ Risk Detection

Regex-based pattern matching is used to detect potentially sensitive or high-risk queries.

Examples may include:

- Security-related issues
- Unauthorized access
- Account compromise
- Urgent escalation scenarios

### 🧠 Local LLM Integration

The project integrates with Ollama to generate responses using a locally running Llama model.

This allows the system to combine deterministic retrieval and routing with natural-language response generation.

### 🌐 Streamlit Dashboard

The project includes an interactive Streamlit application for submitting support queries and viewing the triage results.

---

## 🏗️ Architecture

```text
User Support Query
        │
        ▼
Streamlit Interface (app.py)
        │
        ▼
Support Triage Engine (triage_agent.py)
        │
        ├── TF-IDF Vectorization
        │
        ├── Cosine Similarity Retrieval
        │
        ├── Confidence Evaluation
        │
        ├── Regex Risk Detection
        │
        └── Routing Decision
                │
                ▼
        ┌───────────────────┐
        │                   │
        ▼                   ▼
 Automated Response     Escalation
        │                   │
        ▼                   ▼
 Local Ollama LLM    Human Support
```

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- TF-IDF
- Cosine Similarity
- Regular Expressions
- Ollama
- Llama 3

---

## 📁 Project Structure

```text
support-triage-agent/
│
├── app.py
├── triage_agent.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/adityapandereio6-sketch/support-triage-agent.git
```

Navigate to the project directory:

```bash
cd support-triage-agent
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧠 Ollama Setup

Install Ollama and make sure the Ollama service is running locally.

Pull the required Llama model:

```bash
ollama pull llama3
```

Then verify that Ollama is running before starting the application.

---

## 🌐 Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the local URL displayed in your terminal.

---

## 🔄 Example Workflow

```text
User enters support query
        │
        ▼
Query converted to TF-IDF vector
        │
        ▼
Cosine similarity search
        │
        ▼
Confidence score calculated
        │
        ├── High Confidence
        │       │
        │       ▼
        │   Retrieve knowledge
        │       │
        │       ▼
        │   Generate response
        │
        └── Low Confidence
                │
                ▼
             Escalate
```

---

## 🚀 Future Improvements

Potential improvements include:

- Persistent knowledge-base storage
- Vector database integration
- Semantic embeddings
- Multi-document retrieval
- Conversation memory
- User authentication
- Cloud-based LLM support
- Support analytics dashboard
- REST API integration

---

## 👨‍💻 Author

**Aditya Pandere**

GitHub: https://github.com/adityapandereio6-sketch

---

## 📄 License

This project is intended for educational and portfolio purposes.
