"""
=============================================================================
Streamlit Frontend Dashboard for Support Triage Agent
=============================================================================
Architecture:
  - Framework: Streamlit
  - Backend  : Imports TriageAgent and KNOWLEDGE_BASE from triage_agent.py
  - Styling  : High-end custom dark/glassmorphic theme with animated elements
=============================================================================
Dependencies: pip install streamlit scikit-learn requests
Run Command : streamlit run app.py
=============================================================================
"""

import streamlit as st
from triage_agent import TriageAgent, KNOWLEDGE_BASE

# ─────────────────────────────────────────────────────────────────────────────
# 1. UI CONFIGURATION & THEME STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEXA | Triage Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS injecting Google Fonts, glassmorphic containers, and custom badges
custom_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');

/* Typography & Background */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    color: #E2E8F0;
}

/* App Background */
.stApp {
    background: radial-gradient(circle at 50% 10%, #1e1b4b 0%, #0f172a 100%);
}

/* Glassmorphism Title Container */
.header-container {
    background: rgba(30, 41, 59, 0.4);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    text-align: center;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.gradient-title {
    background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #4f46e5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 6px;
    letter-spacing: -0.02em;
}

.subtitle {
    color: #94A3B8;
    font-size: 1.1rem;
    font-weight: 300;
}

/* Action Cards */
.card {
    background: rgba(30, 41, 59, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.2);
}

/* Glowing Badges */
.badge-auto {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    color: #FFFFFF;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 9999px;
    display: inline-block;
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
    font-size: 0.85rem;
    text-transform: uppercase;
}

.badge-human {
    background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
    color: #FFFFFF;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 9999px;
    display: inline-block;
    box-shadow: 0 0 15px rgba(239, 68, 68, 0.4);
    font-size: 0.85rem;
    text-transform: uppercase;
}

.metric-label {
    font-size: 0.85rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}

.metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.2rem;
    color: #E2E8F0;
    font-weight: 700;
}

/* Output styling */
.output-box {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    padding: 16px;
    font-family: 'Outfit', sans-serif;
    line-height: 1.6;
    margin-top: 8px;
}
</style>
"""
st.markdown(custom_style, unsafe_allow_html=True)

# Initialize Triage Agent in Streamlit session state so it preserves memory
if "agent" not in st.session_state:
    st.session_state.agent = TriageAgent(KNOWLEDGE_BASE)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR CONTROL PANEL & SAFETY OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Control Panel")
    st.markdown("Configure operational parameters and review threat definitions in real-time.")
    st.markdown("---")
    
    st.markdown("### ⚙️ Confidence Threshold")
    # Dynamically bind TF-IDF Similarity Threshold slider
    st.session_state.agent.sim_threshold = st.slider(
        "Min Cosine Similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.05,
        key="sim_threshold_slider"
    )
    st.markdown(f"Current Limit: `{st.session_state.agent.sim_threshold:.2f}`")
    
    st.markdown("---")
    st.markdown("### 🚫 Blacklisted Risk Patterns")
    st.markdown("The regex safety net automatically intercepts and redirects these high-liability triggers:")
    
    # Dynamically extract and display formatted blacklist terms directly from the model
    for raw_pattern in st.session_state.agent.risk_patterns:
        cleaned_term = raw_pattern.replace(r"\b", "").replace(r"\s+", " ").replace(r"(s)?", "s")
        st.markdown(f"🔴 `{cleaned_term}`")

# ─────────────────────────────────────────────────────────────────────────────
# 2. HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="header-container">
        <div class="gradient-title">NEXA TRIAGE OPERATIONAL CONTROL</div>
        <div class="subtitle">Deterministic RAG Routing • Local Llama 3 Grounding • Regex Risk Engine</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. INTERACTIVE CONTROL DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.markdown("### 📥 INCOMING CUSTOMER TICKET")
    
    # Input Area
    ticket_query = st.text_area(
        "Enter support ticket / query:",
        placeholder="Type physics questions, technical specs, or requests here...",
        height=160,
        key="ticket_input_area"
    )
    
    # Quick Test Queries
    st.markdown("#### ⚡ Quick Test Scenarios")
    c1, c2, c3 = st.columns(3)
    if c1.button("📋 Anti-Matter Gravity", key="btn_antimatter"):
        st.session_state.ticket_input_area = "What did the 2023 CERN ALPHA-g experiment show about antimatter falling?"
        st.rerun()
    if c2.button("⚠️ Speculative Overunity", key="btn_overunity"):
        st.session_state.ticket_input_area = "Where can I download working blueprints for a free energy generator?"
        st.rerun()
    if c3.button("❓ Low Confidence Query", key="btn_lowconf"):
        st.session_state.ticket_input_area = "Explain how to bake a chocolate cake in outer space."
        st.rerun()

    # Configured State info display
    st.markdown("#### ⚙️ Configured Settings")
    st.markdown(f"Current Similarity Threshold: `{st.session_state.agent.sim_threshold:.2f}` (Adjust in sidebar)")

with col_right:
    st.markdown("### 🛡️ OPERATIONAL ROUTING STATE")

    if ticket_query.strip():
        # Process through backend triage agent
        with st.spinner("Processing through deterministic RAG and Safety Router..."):
            result = st.session_state.agent.route_and_process(ticket_query)

        # 1. Triage Status Badge
        status = result["status"]
        if status == "AUTO_RESPOND":
            badge_html = '<span class="badge-auto">🤖 Auto-Respond</span>'
        else:
            badge_html = '<span class="badge-human">🧑‍💻 Human Triage Required</span>'
            
        st.markdown(
            f"""
            <div class="card">
                <span class="metric-label">System Route Decision</span><br/>
                <div style="margin-top: 8px; margin-bottom: 8px;">{badge_html}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )

        # 2. Metric Details (Similarity or Flagged terms)
        if status == "AUTO_RESPOND":
            st.markdown(
                f"""
                <div class="card">
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <span class="metric-label">Retrieved Grounding Block</span><br/>
                            <span class="metric-val">{result.get('retrieved_section', 'N/A')}</span>
                        </div>
                        <div>
                            <span class="metric-label">Cosine Similarity Score</span><br/>
                            <span class="metric-val">{result.get('similarity_score', 0.0):.4f}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            reason = result.get("reason", "Unknown")
            flagged = result.get("flagged_term", "N/A")
            st.markdown(
                f"""
                <div class="card">
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <span class="metric-label">Escalation Reason</span><br/>
                            <span class="metric-val" style="color: #F87171;">{reason}</span>
                        </div>
                        <div>
                            <span class="metric-label">Flagged Term</span><br/>
                            <span class="metric-val" style="color: #F87171;">{flagged}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # 3. Output Response Panel
        st.markdown("#### 💬 GENERATED OUTPUT / SYSTEM STATEMENT")
        st.markdown(
            f"""
            <div class="output-box">
                {result['response'].replace('\n', '<br/>')}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            """
            <div style="text-align: center; margin-top: 40px; padding: 40px; border: 1px dashed rgba(255,255,255,0.1); border-radius: 12px; background: rgba(30, 41, 59, 0.2);">
                <span style="font-size: 3rem;">📥</span>
                <h4 style="margin-top: 10px; color: #94A3B8;">Awaiting Customer Ticket Input</h4>
                <p style="color: #64748B; font-size: 0.9rem;">Submit a ticket query or click a quick-test scenario to view routing logic live.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
