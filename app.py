import html
import streamlit as st

from triage_agent import TriageAgent, KNOWLEDGE_BASE


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="NEXA | Triage Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CUSTOM STYLING
# =============================================================================

custom_style = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background: radial-gradient(
        circle at 50% 10%,
        #1e1b4b 0%,
        #111827 45%,
        #0f172a 100%
    );
}

/* Header */

.header-container {
    background: rgba(30, 41, 59, 0.45);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(165, 180, 252, 0.15);
    border-radius: 20px;
    padding: 30px 32px;
    margin-bottom: 30px;
    text-align: center;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.25);
}

.gradient-title {
    background: linear-gradient(
        135deg,
        #c7d2fe 0%,
        #818cf8 45%,
        #6366f1 100%
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 8px;
}

.subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
}

/* Cards */

.card {
    background: rgba(30, 41, 59, 0.50);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 18px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

/* Status badges */

.badge-auto {
    background: linear-gradient(
        135deg,
        #059669,
        #10b981
    );

    color: white;
    font-weight: 700;
    padding: 8px 16px;
    border-radius: 999px;
    display: inline-block;
    font-size: 0.85rem;
    letter-spacing: 0.04em;
}

.badge-human {
    background: linear-gradient(
        135deg,
        #dc2626,
        #ef4444
    );

    color: white;
    font-weight: 700;
    padding: 8px 16px;
    border-radius: 999px;
    display: inline-block;
    font-size: 0.85rem;
    letter-spacing: 0.04em;
}

/* Metrics */

.metric-label {
    font-size: 0.80rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 700;
}

.metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    color: #e2e8f0;
    font-weight: 700;
    word-break: break-word;
}

/* Output */

.output-box {
    background: rgba(15, 23, 42, 0.80);
    border: 1px solid rgba(129, 140, 248, 0.15);
    border-radius: 12px;
    padding: 20px;
    line-height: 1.7;
    margin-top: 10px;
    color: #e2e8f0;
}

/* Text area */

textarea {
    border-radius: 12px !important;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: rgba(17, 24, 39, 0.95);
}

</style>
"""

st.markdown(
    custom_style,
    unsafe_allow_html=True,
)


# =============================================================================
# INITIALIZE TRIAGE AGENT
# =============================================================================

if "agent" not in st.session_state:
    st.session_state.agent = TriageAgent(KNOWLEDGE_BASE)

agent = st.session_state.agent


# =============================================================================
# QUICK TEST FUNCTIONS
# =============================================================================

def set_antimatter_query():
    st.session_state.ticket_input = (
        "What did the 2023 CERN ALPHA-g experiment "
        "show about antimatter falling?"
    )


def set_overunity_query():
    st.session_state.ticket_input = (
        "Can you provide a free energy "
        "overunity generator design?"
    )


def set_low_confidence_query():
    st.session_state.ticket_input = (
        "How do I bake a chocolate cake "
        "in outer space?"
    )


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.title("🛡️ Control Panel")

    st.write(
        "Configure the similarity threshold and review "
        "the risk detection patterns."
    )

    st.divider()

    # -------------------------------------------------------------------------
    # CONFIDENCE THRESHOLD
    # -------------------------------------------------------------------------

    st.subheader("⚙️ Confidence Threshold")

    threshold = st.slider(
        "Minimum Cosine Similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.05,
    )

    agent.sim_threshold = threshold

    st.caption(
        f"Current Threshold: {threshold:.2f}"
    )

    st.divider()

    # -------------------------------------------------------------------------
    # RISK PATTERNS
    # -------------------------------------------------------------------------

    st.subheader("🚫 Risk Detection Patterns")

    st.caption(
        "The following concepts trigger automatic "
        "human triage."
    )

    risk_terms = [
        "Overunity",
        "Free Energy",
        "Perpetual Motion",
        "Anti-Gravity Blueprint",
        "UFO Propulsion",
        "Warp Drive Generator",
        "Zero-Point Power",
    ]

    for term in risk_terms:
        st.markdown(f"🔴 `{term}`")


# =============================================================================
# APPLICATION HEADER
# =============================================================================

st.markdown(
    '<div class="header-container">'
    '<div class="gradient-title">NEXA TRIAGE OPERATIONAL CONTROL</div>'
    '<div class="subtitle">Deterministic RAG Routing • Local Llama 3 Grounding • Regex Risk Engine</div>'
    '</div>',
    unsafe_allow_html=True
)


# =============================================================================
# MAIN DASHBOARD
# =============================================================================

col_left, col_right = st.columns([1, 1.2])


# =============================================================================
# LEFT COLUMN — INPUT
# =============================================================================

with col_left:

    st.subheader("📥 Incoming Customer Ticket")

    ticket_query = st.text_area(
        "Enter support ticket or query:",
        placeholder=(
            "Ask a question about antimatter, "
            "negative mass, wormholes, or related "
            "physics topics..."
        ),
        height=180,
        key="ticket_input",
    )

    st.subheader("⚡ Quick Test Scenarios")

    c1, c2, c3 = st.columns(3)

    c1.button(
        "🧪 Antimatter",
        on_click=set_antimatter_query,
    )

    c2.button(
        "⚠️ Overunity",
        on_click=set_overunity_query,
    )

    c3.button(
        "❓ Low Confidence",
        on_click=set_low_confidence_query,
    )

    st.divider()

    st.subheader("⚙️ Current Configuration")

    st.write(
        f"Similarity Threshold: **{threshold:.2f}**"
    )


# =============================================================================
# RIGHT COLUMN — RESULTS
# =============================================================================

with col_right:

    st.subheader("🛡️ Operational Routing State")

    if ticket_query.strip():

        with st.spinner(
            "Processing through RAG and safety router..."
        ):
            result = agent.route_and_process(
                ticket_query
            )

        # ---------------------------------------------------------------------
        # STATUS
        # ---------------------------------------------------------------------

        status = result["status"]

        if status == "AUTO_RESPOND":

            badge_html = (
                '<span class="badge-auto">'
                '🤖 AUTO RESPOND'
                '</span>'
            )

        else:

            badge_html = (
                '<span class="badge-human">'
                '🧑‍💻 HUMAN TRIAGE REQUIRED'
                '</span>'
            )

        st.markdown(
            f"""
            <div class="card">

                <span class="metric-label">
                    System Route Decision
                </span>

                <br><br>

                {badge_html}

            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---------------------------------------------------------------------
        # AUTO RESPONSE METRICS
        # ---------------------------------------------------------------------

        if status == "AUTO_RESPOND":

            retrieved_section = result.get(
                "retrieved_section",
                "N/A",
            )

            similarity_score = result.get(
                "similarity_score",
                0.0,
            )

            st.markdown(
                f"""
                <div class="card">

                    <span class="metric-label">
                        Retrieved Knowledge Block
                    </span>

                    <br>

                    <span class="metric-val">
                        {html.escape(str(retrieved_section))}
                    </span>

                    <br><br>

                    <span class="metric-label">
                        Cosine Similarity Score
                    </span>

                    <br>

                    <span class="metric-val">
                        {similarity_score:.4f}
                    </span>

                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---------------------------------------------------------------------
        # HUMAN TRIAGE METRICS
        # ---------------------------------------------------------------------

        else:

            reason = result.get(
                "reason",
                "Unknown",
            )

            flagged_term = result.get(
                "flagged_term",
                "N/A",
            )

            st.markdown(
                f"""
                <div class="card">

                    <span class="metric-label">
                        Escalation Reason
                    </span>

                    <br>

                    <span class="metric-val">
                        {html.escape(str(reason))}
                    </span>

                    <br><br>

                    <span class="metric-label">
                        Flagged Term
                    </span>

                    <br>

                    <span class="metric-val">
                        {html.escape(str(flagged_term))}
                    </span>

                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---------------------------------------------------------------------
        # RESPONSE OUTPUT
        # ---------------------------------------------------------------------

        st.subheader("💬 Generated Output")

        response_text = html.escape(
            str(result["response"])
        ).replace(
            "\n",
            "<br>"
        )

        st.markdown(
            f"""
            <div class="output-box">
                {response_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # EMPTY STATE
    # -------------------------------------------------------------------------

    else:

        st.info(
            "📥 Enter a support ticket or select one "
            "of the Quick Test Scenarios to begin."
        )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "NEXA Support Triage Agent • "
    "TF-IDF RAG • Cosine Similarity • "
    "Risk Detection • Ollama"
)
