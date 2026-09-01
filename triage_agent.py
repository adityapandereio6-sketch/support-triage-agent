"""
=============================================================================
Deterministic Support Triage Agent
=============================================================================

Architecture:
- Knowledge Base  : Physics research content split into topical sections
- RAG Engine      : TF-IDF + cosine similarity retrieval
- Risk Engine     : Regex-based risk detection
- LLM Layer       : Optional local Llama 3 generation through Ollama
- Router          : Auto-response or human-triage decision

The application supports two modes:

1. LOCAL MODE
   If Ollama is running, responses are generated using Llama 3.

2. DEPLOYMENT MODE
   If Ollama is unavailable, the system returns a clean response based on
   the retrieved knowledge-base context.

Dependencies:
    pip install scikit-learn requests
=============================================================================
"""

import re
import requests

from typing import Dict, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# SECTION 1 — KNOWLEDGE BASE
# =============================================================================

KNOWLEDGE_BASE = [

    # -------------------------------------------------------------------------
    # KB-00
    # -------------------------------------------------------------------------

    """
    Negative Mass and Exotic Matter: Core Definition.

    In Newtonian gravity, mass plays three distinct roles:

    - Inertial mass: resistance to acceleration.
    - Passive gravitational mass: response to a gravitational field.
    - Active gravitational mass: source of a gravitational field.

    The Equivalence Principle, a cornerstone of General Relativity,
    requires these forms of mass to be equivalent.

    A hypothetical material with negative values for these roles is
    commonly referred to as exotic matter.

    No Standard Model particle has been observed with negative inertial mass.
    """,


    # -------------------------------------------------------------------------
    # KB-01
    # -------------------------------------------------------------------------

    """
    Negative Mass and Exotic Matter: Wormholes and Theoretical Applications.

    Exotic matter with negative energy density is mathematically discussed
    within General Relativity.

    Such exotic matter is associated with theoretical models including
    traversable wormholes and the Alcubierre warp-drive metric.

    Quantum field theory can demonstrate localized negative energy density
    under specific conditions, such as the Casimir effect.

    However, scaling these effects into macroscopic quantities of exotic
    matter remains speculative.
    """,


    # -------------------------------------------------------------------------
    # KB-02
    # -------------------------------------------------------------------------

    """
    The Bondi Negative Mass Paradox: Runaway Motion.

    Hermann Bondi analyzed interactions between positive and hypothetical
    negative mass.

    In a simplified theoretical scenario, positive mass attracts negative
    mass while negative mass can produce repulsive gravitational behavior.

    This can result in theoretical runaway motion where both objects
    accelerate continuously in the same direction.

    Such behavior raises questions about stability and physical realism.
    """,


    # -------------------------------------------------------------------------
    # KB-03
    # -------------------------------------------------------------------------

    """
    Cosmological Constant and Dark Energy.

    The cosmological constant, commonly represented by Lambda, represents
    vacuum energy associated with the accelerating expansion of the universe.

    Dark energy is associated with negative pressure on cosmological scales.

    This negative pressure can oppose gravitational collapse and contribute
    to the observed accelerated expansion of the universe.

    These effects occur at cosmological scales and cannot currently be used
    as a practical local propulsion system.
    """,


    # -------------------------------------------------------------------------
    # KB-04
    # -------------------------------------------------------------------------

    """
    Energy Conditions in General Relativity.

    General Relativity uses several energy conditions to describe physically
    reasonable distributions of matter and energy.

    These include:

    - Null Energy Condition (NEC)
    - Weak Energy Condition (WEC)
    - Strong Energy Condition (SEC)
    - Dominant Energy Condition (DEC)

    Certain theoretical spacetime geometries, including traversable
    wormholes, may require violations of specific energy conditions.

    Dark energy is associated with violations of the Strong Energy Condition
    in cosmological contexts.
    """,


    # -------------------------------------------------------------------------
    # KB-05
    # -------------------------------------------------------------------------

    """
    Experimental Constraints and Progress.

    The 2023 ALPHA-g experiment at CERN observed the effect of gravity on
    antihydrogen, which is antimatter.

    The experiment confirmed that antihydrogen falls downward under Earth's
    gravity.

    This result ruled out basic hypotheses proposing that antimatter exhibits
    ordinary anti-gravity behavior.

    Positive mass theorems in General Relativity also provide important
    constraints on physically reasonable spacetime configurations.
    """
]


# =============================================================================
# SECTION 2 — TF-IDF RAG ENGINE
# =============================================================================

class SklearnRAGEngine:
    """
    Retrieves the most relevant knowledge-base section using
    TF-IDF vectorization and cosine similarity.
    """

    def __init__(self, knowledge_base: list[str]):

        self.knowledge_base = knowledge_base

        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.knowledge_base
        )


    def retrieve(
        self,
        query: str
    ) -> Tuple[int, str, float]:

        query_vector = self.vectorizer.transform(
            [query]
        )

        similarities = cosine_similarity(
            query_vector,
            self.tfidf_matrix
        ).flatten()

        best_index = similarities.argmax()

        best_similarity = similarities[
            best_index
        ]

        return (
            int(best_index),
            self.knowledge_base[best_index],
            float(best_similarity)
        )


# =============================================================================
# SECTION 3 — LOCAL OLLAMA GENERATION
# =============================================================================

class LocalLlamaGenerator:
    """
    Optional local Llama generation through Ollama.

    If Ollama is unavailable, the system returns None and the
    TriageAgent uses a clean RAG fallback response.
    """

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
    ) -> str | None:

        system_prompt = (
            "You are a strict technical support assistant.\n\n"

            "Answer the user's query using ONLY the provided "
            "knowledge-base context.\n\n"

            "Rules:\n"

            "1. Do not introduce outside information.\n"
            "2. Do not speculate.\n"
            "3. Keep the answer concise and professional.\n"
            "4. If the context does not contain enough information, "
            "say so clearly.\n\n"

            "--- KNOWLEDGE BASE CONTEXT ---\n"
            f"{context.strip()}\n"
            "--- END CONTEXT ---"
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
                timeout=10
            )


            if response.status_code == 200:

                generated_text = response.json().get(
                    "response",
                    ""
                ).strip()


                if generated_text:

                    return generated_text


            return None


        except requests.exceptions.RequestException:

            return None


# =============================================================================
# SECTION 4 — TRIAGE AGENT
# =============================================================================

class TriageAgent:
    """
    Main deterministic routing engine.

    Pipeline:

        User Query
            ↓
        Risk Detection
            ↓
        TF-IDF Retrieval
            ↓
        Similarity Validation
            ↓
        Ollama Generation OR RAG Fallback
    """


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


        # ---------------------------------------------------------------------
        # Risk Detection Patterns
        # ---------------------------------------------------------------------

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


    # =========================================================================
    # CLEAN RAG FALLBACK
    # =========================================================================

    def create_rag_fallback(
        self,
        context: str
    ) -> str:

        """
        Creates a clean response when Ollama is unavailable.

        This makes the deployed Streamlit application fully usable
        without requiring a local Ollama server.
        """

        clean_context = context.strip()

        return (
            "Based on the verified knowledge base:\n\n"
            f"{clean_context}"
        )


    # =========================================================================
    # MAIN ROUTING FUNCTION
    # =========================================================================

    def route_and_process(
        self,
        query: str
    ) -> Dict:


        # ---------------------------------------------------------------------
        # STEP 1 — RISK DETECTION
        # ---------------------------------------------------------------------

        risk_match = self.risk_regex.search(
            query
        )


        if risk_match:

            return {

                "status": "HUMAN_TRIAGE",

                "reason": (
                    "HIGH_LIABILITY_PSEUDOSCIENCE"
                ),

                "flagged_term": (
                    risk_match.group(0)
                ),

                "response": (
                    "This ticket has been routed to human review. "
                    "The request contains a restricted speculative "
                    "or high-liability concept detected by the risk "
                    "engine."
                )

            }


        # ---------------------------------------------------------------------
        # STEP 2 — RAG RETRIEVAL
        # ---------------------------------------------------------------------

        document_index, context, similarity = (
            self.rag_engine.retrieve(
                query
            )
        )


        # ---------------------------------------------------------------------
        # STEP 3 — CONFIDENCE CHECK
        # ---------------------------------------------------------------------

        if similarity < self.sim_threshold:

            return {

                "status": "HUMAN_TRIAGE",

                "reason": (
                    "LOW_GROUNDING_CONFIDENCE "
                    f"({similarity:.4f} < "
                    f"{self.sim_threshold:.4f})"
                ),

                "response": (
                    "This ticket has been routed to human review "
                    "because the query could not be confidently "
                    "matched to the verified knowledge base."
                )

            }


        # ---------------------------------------------------------------------
        # STEP 4 — OPTIONAL LLM GENERATION
        # ---------------------------------------------------------------------

        llm_response = (
            self.generator.generate_response(
                query,
                context
            )
        )


        # ---------------------------------------------------------------------
        # STEP 5 — DEPLOYMENT FALLBACK
        # ---------------------------------------------------------------------

        if llm_response:

            final_response = llm_response

            response_mode = "LOCAL_LLM"

        else:

            final_response = (
                self.create_rag_fallback(
                    context
                )
            )

            response_mode = "RAG_FALLBACK"


        # ---------------------------------------------------------------------
        # FINAL RESULT
        # ---------------------------------------------------------------------

        return {

            "status": "AUTO_RESPOND",

            "retrieved_section": (
                f"KB-{document_index:02d}"
            ),

            "similarity_score": (
                round(similarity, 4)
            ),

            "response_mode": (
                response_mode
            ),

            "response": (
                final_response
            )

        }


# =============================================================================
# SECTION 5 — LOCAL VERIFICATION RUNNER
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
    # TEST 1 — VALID QUERY
    # -------------------------------------------------------------------------

    valid_query = (
        "What did the CERN ALPHA-g experiment "
        "show about antimatter falling?"
    )


    print(
        "\n[TEST 1] VALID TECHNICAL QUERY"
    )

    print(
        f"Query: {valid_query}"
    )


    result_1 = agent.route_and_process(
        valid_query
    )


    print(
        f"\nStatus: {result_1['status']}"
    )

    print(
        f"Source: "
        f"{result_1.get('retrieved_section')}"
    )

    print(
        f"Similarity: "
        f"{result_1.get('similarity_score')}"
    )

    print(
        f"Mode: "
        f"{result_1.get('response_mode')}"
    )

    print(
        "\nResponse:"
    )

    print(
        result_1["response"]
    )


    print(
        "\n" + "-" * 80
    )


    # -------------------------------------------------------------------------
    # TEST 2 — RISKY QUERY
    # -------------------------------------------------------------------------

    risky_query = (
        "Can you provide an overunity "
        "free energy generator design?"
    )


    print(
        "\n[TEST 2] HIGH-LIABILITY QUERY"
    )

    print(
        f"Query: {risky_query}"
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
        "\nResponse:"
    )

    print(
        result_2["response"]
    )


    print(
        "\n" + "=" * 80
    )


# =============================================================================
# PROGRAM ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    run_agent_verification()
