"""
=============================================================================
Deterministic Support Triage Agent
=============================================================================

Architecture:
- Knowledge Base : Physics research text divided into topical sections
- RAG Engine     : TF-IDF vectorization + cosine similarity retrieval
- Risk Engine    : Regex-based risk term detection
- LLM Layer      : Optional local Llama 3 integration through Ollama
- Router         : Routes queries to auto-response or human triage

Dependencies:
    pip install scikit-learn requests

Optional Ollama setup:
    ollama pull llama3
    ollama serve
=============================================================================
"""

import re
import requests

from typing import Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# SECTION 1: KNOWLEDGE BASE
# =============================================================================

KNOWLEDGE_BASE = [

    # [KB-00] Negative Mass — Core Definition
    """
    Negative Mass and Exotic Matter: Core Definition.

    In Newtonian gravity, mass plays three distinct roles:
    inertial mass (resistance to acceleration, F=ma),
    passive gravitational mass (response to a gravitational field),
    and active gravitational mass (source of a gravitational field).

    The Equivalence Principle, a cornerstone of General Relativity,
    demands that all three are equal.

    A hypothetical material with negative values for any of these
    roles is termed exotic matter.

    No Standard Model particle has been observed with negative
    inertial mass.
    """,

    # [KB-01] Negative Mass — Physical Consequences
    """
    Negative Mass and Exotic Matter: Solutions and Wormholes.

    Exotic matter with negative energy density is mathematically
    permitted in General Relativity.

    It is required to support the throat of traversable wormholes
    (Morris-Thorne wormholes) and to construct the metric for the
    Alcubierre Warp Drive.

    In quantum field theory, local negative energy densities are
    demonstrated by the Casimir effect, though scaling this to
    macroscopic exotic matter remains speculative.
    """,

    # [KB-02] The Bondi Negative Mass Paradox
    """
    The Bondi Negative Mass Paradox (Runaway Motion).

    Hermann Bondi (1957) analyzed the interaction of positive mass
    (+M) and negative mass (-m).

    The positive mass attracts the negative mass, while the negative
    mass repels the positive mass.

    This leads to runaway motion where both masses accelerate
    indefinitely in the same direction from rest.

    Although total momentum and total kinetic energy remain zero
    since negative mass carries negative kinetic energy, this
    perpetual self-acceleration presents thermodynamic and stability
    paradoxes.
    """,

    # [KB-03] Cosmological Constant and Dark Energy
    """
    Cosmological Constant (Lambda) and Dark Energy.

    The cosmological constant (Lambda) represents a positive vacuum
    energy density with an equation of state:

    w = p / (rho * c^2)

    of exactly -1.

    This negative pressure opposes standard gravitational collapse
    and drives the accelerating expansion of the universe.

    This acts as a form of repulsive gravity on cosmological scales,
    though it is far too dilute to be utilized locally.
    """,

    # [KB-04] Energy Conditions in General Relativity
    """
    Energy Conditions in General Relativity.

    General Relativity uses energy conditions to rule out unphysical
    or pathological spacetimes.

    The Null Energy Condition (NEC) is violated by wormhole throats.

    The Weak Energy Condition (WEC) is violated by negative mass.

    The Strong Energy Condition (SEC) ensures gravity is always
    attractive and is violated by dark energy and Lambda.

    The Dominant Energy Condition (DEC) governs causality and
    subluminal energy flux.
    """,

    # [KB-05] Experimental Constraints
    """
    Experimental Constraints and Progress.

    The 2023 ALPHA-g experiment at CERN observed the effect of
    gravity on antihydrogen (antimatter) and confirmed that
    antimatter falls downward.

    This ruled out basic antimatter anti-gravity hypotheses.

    Additionally, the Schoen-Yau Positive Mass Theorem proves that
    total ADM mass must be non-negative for any stable,
    physically reasonable spacetime satisfying the Dominant Energy
    Condition.
    """
]


# =============================================================================
# SECTION 2: TF-IDF RAG ENGINE
# =============================================================================

class SklearnRAGEngine:

    def __init__(self, knowledge_base: list[str]):

        self.kb = knowledge_base

        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.kb
        )


    def retrieve(
        self,
        query: str
    ) -> Tuple[int, str, float]:

        """
        Retrieves the most relevant knowledge-base section
        using cosine similarity.
        """

        query_vector = self.vectorizer.transform(
            [query]
        )

        similarities = cosine_similarity(
            query_vector,
            self.tfidf_matrix
        ).flatten()

        best_idx = similarities.argmax()

        best_similarity = similarities[best_idx]

        return (
            int(best_idx),
            self.kb[best_idx],
            float(best_similarity)
        )


# =============================================================================
# SECTION 3: LOCAL LLM GENERATION LAYER
# =============================================================================

class LocalLlamaGenerator:

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3"
    ):

        self.host = host
        self.model = model

        self.generate_url = (
            f"{self.host}/api/generate"
        )


    def generate_response(
        self,
        query: str,
        context: str
    ) -> str:

        """
        Sends the query and retrieved context to a locally
        running Ollama Llama model.
        """

        system_prompt = (
            "You are a strict technical support assistant "
            "for a theoretical physics research team.\n\n"

            "Your task is to answer the user's query using "
            "ONLY the provided context block.\n\n"

            "Follow these strict rules:\n"

            "1. Answer accurately and professionally using "
            "ONLY facts in the context.\n"

            "2. Do not speculate or introduce outside "
            "knowledge.\n"

            "3. If the context does not contain sufficient "
            "information, respond exactly with:\n\n"

            "'I am sorry, but the provided database does not "
            "contain sufficient technical details to answer "
            "this query.'\n\n"

            "--- START CONTEXT BLOCK ---\n"

            f"{context.strip()}\n"

            "--- END CONTEXT BLOCK ---"
        )

        payload = {

            "model": self.model,

            "prompt": query,

            "system": system_prompt,

            "stream": False,

            "options": {

                "temperature": 0.0,

                "top_p": 0.1,

                "num_predict": 256

            }
        }


        try:

            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=20
            )

            if response.status_code == 200:

                return response.json().get(
                    "response",
                    ""
                ).strip()

            return (
                "[ERROR] Local Llama server returned "
                f"status code {response.status_code}"
            )


        except requests.exceptions.RequestException:

            return (
                "[GENERATION OFFLINE]\n\n"

                "Could not connect to the local Ollama server.\n\n"

                f"Expected server: {self.host}\n"
                f"Expected model: {self.model}\n\n"

                "To enable local generation, install Ollama "
                "and run:\n\n"

                "ollama pull llama3\n"
                "ollama serve\n\n"

                "[RETRIEVED CONTEXT]\n\n"

                f"{context.strip()}"
            )


# =============================================================================
# SECTION 4: TRIAGE AGENT
# =============================================================================

class TriageAgent:

    def __init__(
        self,
        knowledge_base: list[str],
        sim_threshold: float = 0.15
    ):

        self.rag_engine = SklearnRAGEngine(
            knowledge_base
        )

        self.generator = LocalLlamaGenerator()

        self.sim_threshold = sim_threshold


        # Risk patterns

        self.risk_patterns = [

            r"\boverunity\b",

            r"\bfree\s+energy\b",

            r"\bperpetual\s+motion\b",

            r"\banti[- ]gravity\s+blueprint(s)?\b",

            r"\bufo\s+propulsion\b",

            r"\bwarp\s+drive\s+generator\b",

            r"\bzero[- ]point\s+power\b"

        ]


        self.risk_regex = re.compile(

            "|".join(self.risk_patterns),

            re.IGNORECASE

        )


    def route_and_process(
        self,
        query: str
    ) -> Dict:


        # ---------------------------------------------------------------------
        # STEP 1: RISK DETECTION
        # ---------------------------------------------------------------------

        risk_match = self.risk_regex.search(
            query
        )


        if risk_match:

            return {

                "status": "HUMAN_TRIAGE",

                "reason":
                    "HIGH_LIABILITY_PSEUDOSCIENCE",

                "flagged_term":
                    risk_match.group(0),

                "response": (

                    "This ticket has been routed to human "
                    "review. The request contains potentially "
                    "unreliable or high-liability concepts."

                )
            }


        # ---------------------------------------------------------------------
        # STEP 2: RAG RETRIEVAL
        # ---------------------------------------------------------------------

        doc_idx, context, similarity = (
            self.rag_engine.retrieve(query)
        )


        # ---------------------------------------------------------------------
        # STEP 3: CONFIDENCE ROUTING
        # ---------------------------------------------------------------------

        if similarity < self.sim_threshold:

            return {

                "status": "HUMAN_TRIAGE",

                "reason": (

                    "LOW_GROUNDING_CONFIDENCE "
                    f"(Similarity: {similarity:.4f} < "
                    f"Threshold: {self.sim_threshold})"

                ),

                "response": (

                    "This ticket has been routed to human "
                    "review because the query could not be "
                    "confidently matched to the verified "
                    "knowledge base."

                )
            }


        # ---------------------------------------------------------------------
        # STEP 4: LOCAL LLM GENERATION
        # ---------------------------------------------------------------------

        llm_response = (
            self.generator.generate_response(
                query,
                context
            )
        )


        return {

            "status": "AUTO_RESPOND",

            "retrieved_section":
                f"KB-{doc_idx:02d}",

            "similarity_score":
                round(similarity, 4),

            "response":
                llm_response

        }


# =============================================================================
# SECTION 5: VERIFICATION RUNNER
# =============================================================================

def run_agent_verification():

    agent = TriageAgent(
        KNOWLEDGE_BASE
    )


    print("=" * 80)

    print(
        "DETERMINISTIC SUPPORT TRIAGE AGENT VERIFICATION"
    )

    print("=" * 80)


    # -------------------------------------------------------------------------
    # TEST 1: VALID TECHNICAL QUERY
    # -------------------------------------------------------------------------

    valid_query = (

        "What did the CERN ALPHA-g experiment "
        "show about antimatter falling?"

    )


    print(
        f"\n[TEST 1] VALID TECHNICAL QUERY:\n"
        f"{valid_query}"
    )


    result_1 = agent.route_and_process(
        valid_query
    )


    print(
        f"\nStatus: {result_1['status']}"
    )


    if result_1["status"] == "AUTO_RESPOND":

        print(
            "Retrieved Section: "
            f"{result_1['retrieved_section']}"
        )

        print(
            "Similarity Score: "
            f"{result_1['similarity_score']}"
        )


    print(
        "\nResponse:\n"
        f"{result_1['response']}"
    )


    print("-" * 80)


    # -------------------------------------------------------------------------
    # TEST 2: RISKY QUERY
    # -------------------------------------------------------------------------

    risky_query = (

        "Can you supply a schematic for an "
        "overunity free energy warp drive generator?"

    )


    print(
        f"\n[TEST 2] HIGH-LIABILITY QUERY:\n"
        f"{risky_query}"
    )


    result_2 = agent.route_and_process(
        risky_query
    )


    print(
        f"\nStatus: {result_2['status']}"
    )


    print(
        f"Reason: {result_2.get('reason')}"
    )


    print(
        f"Flagged Term: "
        f"{result_2.get('flagged_term')}"
    )


    print(
        "\nResponse:\n"
        f"{result_2['response']}"
    )


    print("=" * 80)


# =============================================================================
# PROGRAM ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    run_agent_verification()
