"""
=============================================================================
Deterministic Support Triage Agent
=============================================================================
Architecture:
  - Knowledge Base  : Anti-gravity physics text, chunked into topical sections
  - RAG Engine      : TF-IDF vectorization + cosine similarity retrieval (scikit-learn)
  - Risk Engine     : Regex-based high-liability term detection
  - LLM Gen Layer   : Strict local Llama 3 model integration via Ollama
  - Router          : Decides between auto-answer, human escalation, and RAG retrieval
=============================================================================
Dependencies: 
  - pip install scikit-learn requests
  - Ensure Ollama is running locally with: `ollama run llama3`
=============================================================================
"""

import re
import json
import requests
from typing import Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: KNOWLEDGE BASE
# topical anti-gravity RAG knowledge base.
# ─────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE = [
    # [KB-00] Negative Mass — Core Definition
    """
    Negative Mass and Exotic Matter: Core Definition.
    In Newtonian gravity, mass plays three distinct roles: inertial mass (resistance
    to acceleration, F=ma), passive gravitational mass (response to a gravitational
    field), and active gravitational mass (source of a gravitational field).
    The Equivalence Principle — a cornerstone of General Relativity — demands that
    all three are equal. A hypothetical material with negative values for any of
    these roles is termed exotic matter. No Standard Model particle has been
    observed with negative inertial mass.
    """,

    # [KB-01] Negative Mass — Physical Consequences
    """
    Negative Mass and Exotic Matter: Solutions and Wormholes.
    Exotic matter with negative energy density is mathematically permitted in General Relativity. 
    It is required to support the throat of traversable wormholes (Morris-Thorne wormholes) and to
    construct the metric for the Alcubierre Warp Drive. In quantum field theory, local negative energy 
    densities are demonstrated by the Casimir effect, though scaling this to macroscopic exotic matter 
    remains speculative.
    """,

    # [KB-02] The Bondi Negative Mass Paradox
    """
    The Bondi Negative Mass Paradox (Runaway Motion).
    Hermann Bondi (1957) analyzed the interaction of positive mass (+M) and negative mass (-m). 
    The positive mass attracts the negative mass, while the negative mass repels the positive mass. 
    This leads to 'runaway motion' where both masses accelerate indefinitely in the same direction 
    from rest. Although total momentum and total kinetic energy remain zero (since negative mass carries 
    negative kinetic energy), this perpetual self-acceleration presents thermodynamic and stability paradoxes.
    """,

    # [KB-03] Cosmological Constant and Dark Energy
    """
    Cosmological Constant (Lambda) and Dark Energy.
    The cosmological constant (Lambda) represents a positive vacuum energy density with an equation of 
    state w = p / (rho * c^2) of exactly -1. This massive negative pressure opposes standard gravitational 
    collapse and drives the accelerating expansion of the universe. This acts as a form of repulsive 
    gravity on cosmological scales, though it is far too dilute to be utilized locally.
    """,

    # [KB-04] Energy Conditions in General Relativity
    """
    Energy Conditions in General Relativity.
    General Relativity uses energy conditions to rule out unphysical or pathological spacetimes. 
    The Null Energy Condition (NEC) is violated by wormhole throats. The Weak Energy Condition (WEC) 
    is violated by negative mass. The Strong Energy Condition (SEC) ensures gravity is always attractive; 
    it is violated by dark energy and Lambda, as proven by cosmic acceleration. The Dominant Energy Condition 
    (DEC) governs causality and subluminal energy flux.
    """,

    # [KB-05] Experimental Constraints (ALPHA-g and CERN)
    """
    Experimental Constraints and Progress.
    The 2023 ALPHA-g experiment at CERN observed the effect of gravity on antihydrogen (antimatter) 
    and confirmed that antimatter falls downward, ruling out basic antimatter anti-gravity hypotheses. 
    Additionally, the Schoen-Yau Positive Mass Theorem proves that total ADM mass must be non-negative 
    for any stable, physically reasonable spacetime satisfying the Dominant Energy Condition.
    """
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: TF-IDF RAG GROUNDING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class SklearnRAGEngine:
    def __init__(self, knowledge_base: list[str]):
        self.kb = knowledge_base
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.kb)

    def retrieve(self, query: str) -> Tuple[int, str, float]:
        """Retrieves the most relevant section from the KB based on cosine similarity."""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        best_idx = similarities.argmax()
        best_sim = similarities[best_idx]
        return int(best_idx), self.kb[best_idx], float(best_sim)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: GENERATION LAYER (OLLAMA INTEGRATION)
# ─────────────────────────────────────────────────────────────────────────────

class LocalLlamaGenerator:
    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3"):
        self.host = host
        self.model = model
        self.generate_url = f"{self.host}/api/generate"

    def generate_response(self, query: str, context: str) -> str:
        """
        Queries the local Llama 3 model with a strict system prompt to ground
        its generation strictly in the retrieved context section.
        """
        system_prompt = (
            "You are a strict technical support assistant for a theoretical physics research team.\n"
            "Your task is to answer the user's query using ONLY the provided context block below.\n"
            "Follow these strict guidelines:\n"
            "1. Answer the query accurately, professionally, and technically using ONLY facts in the context.\n"
            "2. Do not extrapolate, speculate, or introduce any outside physics knowledge.\n"
            "3. If the context does not contain the answer to the query, respond exactly with: "
            "'I am sorry, but the provided database does not contain sufficient technical details to answer this query.'\n\n"
            f"--- START CONTEXT BLOCK ---\n{context.strip()}\n--- END CONTEXT BLOCK ---"
        )

        payload = {
            "model": self.model,
            "prompt": query,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,      # Maximum determinism
                "top_p": 0.1,
                "num_predict": 256       # Prevent verbose rambling
            }
        }

        try:
            response = requests.post(self.generate_url, json=payload, timeout=20)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                return f"[Error: Local Llama server returned status code {response.status_code}]"
        except requests.exceptions.RequestException as e:
            # Elegant fallback description if local Ollama is not active
            return (
                f"[GENERATION OFFLINE] Could not connect to local Ollama server at {self.host}.\n"
                f"Reason: {e}\n"
                f"Ensure 'ollama serve' is running and model '{self.model}' is installed.\n\n"
                f"[DRY-RUN FALLBACK] If generation was online, Llama 3 would have answered the query "
                f"strictly grounded in the retrieved context section below:\n\n{context.strip()}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: RISK ENGINE AND ROUTER
# ─────────────────────────────────────────────────────────────────────────────

class TriageAgent:
    def __init__(self, knowledge_base: list[str], sim_threshold: float = 0.15):
        self.rag_engine = SklearnRAGEngine(knowledge_base)
        self.generator = LocalLlamaGenerator()
        self.sim_threshold = sim_threshold
        
        # Risk keywords for pseudoscience / high liability
        self.risk_patterns = [
            r"\boverunity\b",
            r"\bfree\s+energy\b",
            r"\bperpetual\s+motion\b",
            r"\banti[- ]gravity\s+blueprint(s)?\b",
            r"\bufo\s+propulsion\b",
            r"\bwarp\s+drive\s+generator\b",
            r"\bzero[- ]point\s+power\b"
        ]
        self.risk_regex = re.compile("|".join(self.risk_patterns), re.IGNORECASE)

    def route_and_process(self, query: str) -> Dict:
        # Step 1: Risk Engine Scan
        risk_match = self.risk_regex.search(query)
        if risk_match:
            return {
                "status": "HUMAN_TRIAGE",
                "reason": "HIGH_LIABILITY_PSEUDOSCIENCE",
                "flagged_term": risk_match.group(0),
                "response": (
                    "This ticket has been routed to human review. The request contains unscientific "
                    "or speculative high-liability engineering concepts (e.g., perpetual motion, overunity). "
                    "We do not generate answers or support files for these concepts."
                )
            }

        # Step 2: RAG Grounding Retrieval
        doc_idx, context, similarity = self.rag_engine.retrieve(query)

        # Step 3: Threshold Routing
        if similarity < self.sim_threshold:
            return {
                "status": "HUMAN_TRIAGE",
                "reason": f"LOW_GROUNDING_CONFIDENCE (Similarity: {similarity:.4f} < Threshold: {self.sim_threshold})",
                "response": (
                    "This ticket has been routed to human review. The query could not be confidently "
                    "matched to the verified research database."
                )
            }

        # Step 4: Strict LLM Response Generation
        llm_response = self.generator.generate_response(query, context)

        return {
            "status": "AUTO_RESPOND",
            "retrieved_section": f"KB-{doc_idx:02d}",
            "similarity_score": round(similarity, 4),
            "response": llm_response
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: TEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_verification():
    agent = TriageAgent(KNOWLEDGE_BASE)

    print("=" * 80)
    print("DETERMINISTIC SUPPORT TRIAGE AGENT VERIFICATION")
    print("=" * 80)

    # Test Case 1: Valid technical query inside the knowledge base
    valid_query = "What did the CERN ALPHA-g experiment show about antimatter falling?"
    print(f"\n[TEST 1] VALID TECHNICAL QUERY: '{valid_query}'")
    result_1 = agent.route_and_process(valid_query)
    print(f"  Status    : {result_1['status']}")
    if result_1['status'] == "AUTO_RESPOND":
        print(f"  Source    : {result_1['retrieved_section']} (Similarity: {result_1['similarity_score']})")
    print(f"  Response  :\n{result_1['response']}")
    print("-" * 80)

    # Test Case 2: High-liability/sci-fi query caught by risk engine
    risky_query = "Can you supply a schematic for an overunity free energy warp drive generator?"
    print(f"\n[TEST 2] HIGH-LIABILITY QUERY: '{risky_query}'")
    result_2 = agent.route_and_process(risky_query)
    print(f"  Status    : {result_2['status']}")
    print(f"  Reason    : {result_2.get('reason')}")
    print(f"  Flagged   : '{result_2.get('flagged_term')}'")
    print(f"  Response  :\n{result_2['response']}")
    print("=" * 80)


if __name__ == "__main__":
    run_agent_verification()
