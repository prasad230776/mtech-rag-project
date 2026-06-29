import streamlit as st
import requests
import json
import sys
import base64
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))
from src.utilities.evaluator import get_comparison_summary, run_evaluation_for_version


# Helper to read local image and convert to base64
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return f"data:image/png;base64,{base64.b64encode(img_file.read()).decode()}"
    except Exception:
        return ""


logo_base64 = get_base64_image("assets/images/logosiddrtaha.png")

# Setup page config
st.set_page_config(
    page_title="Multi-Stage Hallucination Mitigation RAG", page_icon="🛡️", layout="wide"
)

import os

# API endpoint configuration
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")

# Create a top section layout for header and horizontal theme selector
# Custom Premium Styling (Dark/Light Mode Compatible Glassmorphism)
st.markdown(
    """
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    /* Base Reset & Font Family */
    * {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Clean Top Header Spacing */
    [data-testid="stDecoration"] {
        display: none !important;
        height: 0px !important;
    }
    [data-testid="stHeader"] {
        background-color: transparent !important;
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }
    
    /* Hide Deploy button specifically */
    div[data-testid="stHeaderActionElements"] button:nth-child(1),
    div[data-testid="stHeaderActionElements"] button:first-of-type,
    .stDeployButton,
    .stAppDeployButton,
    iframe[title="Deploy"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Add theme indicator label before the three dots options button */
    [data-testid="stHeader"]::after {
        content: "Change Theme ➜";
        position: absolute;
        right: 50px;
        top: 21px;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.8rem;
        font-weight: 700;
        color: #6366f1;
        opacity: 1.0;
        letter-spacing: 0.5px;
        pointer-events: none;
        z-index: 999999;
        text-transform: uppercase;
    }
    
    div[data-testid="stAppViewContainer"] > section {
        padding-top: 0px !important;
    }
    .main .block-container,
    div[data-testid="stAppViewBlockContainer"] {
        padding-top: 0 !important;
        padding-bottom: 1rem !important;
        margin-top: 0 !important;
    }

    /* Make both columns start at same height */
    div[data-testid="stHorizontalBlock"] {
        align-items: flex-start !important;
    }
    div[data-testid="column"] {
        padding-top: 0px !important;
        margin-top: 0px !important;
    }
    div[data-testid="column"] > div {
        padding-top: 0px !important;
        margin-top: 0px !important;
    }

    /* Premium Glassmorphic Layout Card Styling */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(17, 25, 40, 0.6) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.3) !important;
        margin-top: 0 !important;
        padding: 14px 18px 18px 18px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(99, 102, 241, 0.35) !important;
        box-shadow: 0 12px 35px -10px rgba(99, 102, 241, 0.15) !important;
    }

    /* Custom Title Typography */
    .gradient-text {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* High-contrast Badges */
    .badge {
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        margin-right: 6px;
        letter-spacing: 0.2px;
        font-family: 'Space Grotesk', sans-serif;
    }
    .badge-accept {
        background-color: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
    .badge-escalate {
        background-color: rgba(239, 68, 68, 0.12);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .badge-version {
        background-color: rgba(99, 102, 241, 0.12);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.25);
    }

    /* Custom Scrollbar for grounding sources */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.3);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 102, 241, 0.5);
    }

    /* Header Responsive Layout */
    .header-container {
        display: flex;
        align-items: flex-end;
        gap: 20px;
        margin-bottom: 5px;
    }
    .header-logo {
        height: 115px;
        width: auto;
        display: block;
        margin-bottom: 0px;
        filter: drop-shadow(0 0 8px rgba(99, 102, 241, 0.15));
    }
    .header-text-container {
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        padding-bottom: 12px;
    }
    
    @media (max-width: 768px) {
        .header-container {
            flex-direction: column !important;
            align-items: center !important;
            text-align: center !important;
            gap: 15px !important;
            margin-bottom: 15px !important;
        }
        .header-logo {
            height: 90px !important;
            margin-bottom: 5px !important;
        }
        .header-text-container {
            align-items: center !important;
            padding-bottom: 0px !important;
        }
        /* 1. Move whole page/header up on mobile but keep arrows/menus visible */
        [data-testid="stHeader"] {
            background-color: rgba(0, 0, 0, 0) !important;
            background: transparent !important;
        }
        .main .block-container,
        div[data-testid="stAppViewBlockContainer"] {
            padding-top: 35px !important;
            margin-top: -35px !important;
        }
        /* 2. Decrease space between header and chat assistant */
        .responsive-hr {
            margin-top: 5px !important;
            margin-bottom: 10px !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            padding: 8px 12px 10px 12px !important;
        }
        /* 3. Decrease height of the chat display container to fit viewport */
        div[data-testid="stVerticalBlock"][style*="height"] {
            height: 140px !important;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

# Application Header
st.markdown(
    f"""
<div class="header-container">
    <img src="{logo_base64}" class="header-logo">
    <div class="header-text-container">
        <h1 class="gradient-text" style="font-size: 2.3rem; margin: 0; padding: 0; line-height: 1.0;">SEAT CHAT BOT</h1>
        <h2 style="font-size: 1.15rem; color: #f8fafc !important; font-weight: 600; margin: 6px 0 3px 0; padding: 0; font-family: 'Space Grotesk'; letter-spacing: 0.2px; line-height: 1.1;">Siddartha Educational Academy Group of Institutions</h2>
        <p style="color: #94a3b8 !important; font-size: 0.90rem; margin: 2px 0 0 0; padding: 0; font-weight: 400; line-height: 1.1;">Multi-Stage Hallucination Mitigation RAG (V0 - V5)</p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<hr class='responsive-hr' style='border: 0; height: 1px; background: linear-gradient(to right, rgba(99,102,241,0), rgba(99,102,241,0.5), rgba(99,102,241,0)); margin-bottom: 25px;'>",
    unsafe_allow_html=True,
)

# Sidebar Configuration panel
st.sidebar.markdown(
    "<h2 style=\"font-family: 'Space Grotesk'; font-weight: 600; color: #f1f5f9 !important;\">🛡️ Navigation</h2>",
    unsafe_allow_html=True,
)
menu = st.sidebar.radio(
    "Go To",
    ["💬 Chatbot Assistant", "📈 Performance Graphs", "📤 Ingest Knowledge Base"],
)

st.sidebar.markdown(
    "<hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True
)

if menu == "💬 Chatbot Assistant":
    st.sidebar.markdown(
        "<h3 style=\"font-family: 'Space Grotesk'; font-weight: 600; color: #f1f5f9 !important; font-size: 1.1rem; margin-top:0px;\">⚙️ Pipeline Settings</h3>",
        unsafe_allow_html=True,
    )
    pipeline_version = st.sidebar.selectbox(
        "Select Framework Version",
        [
            "V0 (Basic RAG)",
            "V1 (Web Fallback)",
            "V2 (Reranking)",
            "V3 (Claim Verification)",
            "V4 (Confidence Scoring)",
            "V5 (Query Intelligence)",
        ],
        index=0,
    )

    # Extract version string
    version_code = pipeline_version.split(" ")[0].lower()

    # Show confidence threshold slider only for V4 and V5 pipelines
    if version_code in ["v4", "v5"]:
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.70, step=0.05
        )
    else:
        confidence_threshold = 0.70  # Default fallback value


# Initialize Chat Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if menu == "💬 Chatbot Assistant":
    # Chat Layout Columns
    main_chat, doc_viewer = st.columns([0.65, 0.35])

    with main_chat:
        with st.container(border=True):
            st.markdown(
                "<h3 style=\"font-family: 'Space Grotesk'; font-weight: 600; margin-top:0px; color: #f8fafc !important;\">💬 Chat Assistant</h3>",
                unsafe_allow_html=True,
            )

            # Display Chat History
            chat_container = st.container(height=310)
            with chat_container:
                for chat in st.session_state.chat_history:
                    if chat["role"] == "user":
                        with st.chat_message("user", avatar="👤"):
                            st.markdown(f"**{chat['content']}**")
                    else:
                        with st.chat_message("assistant", avatar="🛡️"):
                            # Check decision to color answer
                            if chat.get("decision") == "ESCALATE":
                                st.markdown(
                                    f"<span style='color: #fca5a5;'>⚠️ {chat['content']}</span>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(chat["content"])

                            # Mini metric badges below assistant responses
                            if "confidence_score" in chat:
                                status_class = (
                                    "badge-accept"
                                    if chat["decision"] == "ACCEPT"
                                    else "badge-escalate"
                                )
                                badges_html = f'<span class="badge badge-version">{chat["version"]}</span>'
                                badges_html += f'<span class="badge {status_class}">Confidence: {chat["confidence_score"]:.2f}</span>'
                                if chat["decision"] == "ESCALATE":
                                    badges_html += f'<span class="badge {status_class}">ESCALATED</span>'
                                st.markdown(
                                    f'<div style="margin-top: 8px;">{badges_html}</div>',
                                    unsafe_allow_html=True,
                                )

        # Chat Input Box
        user_input = st.chat_input(
            "Ask a question about academics, placements, admissions, hostels..."
        )

        if user_input:
            # Append user message
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )

            # Display instantly
            with chat_container:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(f"**{user_input}**")

            # Call API backend
            with st.spinner("Generating reliable response..."):
                try:
                    payload = {
                        "question": user_input,
                        "version": version_code,
                        "escalation_threshold": confidence_threshold,
                    }
                    response = requests.post(f"{API_URL}/ask", json=payload)

                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": data["answer"],
                                "decision": data["decision"],
                                "confidence_score": data["confidence_score"],
                                "version": data["version"].upper(),
                                "sources": data["sources"],
                                "metrics": data["metrics"],
                                "raw_answer": data["raw_answer"],
                                "intent_type": data.get("intent_type", "SINGLE"),
                            }
                        )
                        st.rerun()
                    else:
                        st.error(
                            f"Backend API error: {response.json().get('detail', 'Unknown error')}"
                        )
                except Exception as e:
                    st.error(
                        f"Failed to connect to FastAPI backend: {str(e)}. Make sure `uvicorn main:app` is running."
                    )

    # Doc Viewer Sidebar column (Right-hand metrics and sources panels)
    with doc_viewer:
        with st.container(border=True):
            st.markdown(
                '<h3 style="font-family: \'Space Grotesk\'; font-weight: 600; margin-top:0px; color: #f8fafc !important; font-size: 1.25rem; display: flex; align-items: center; gap: 8px;"><span style="color: #6366f1;">📊</span> Pipeline Analytics</h3>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p style="color: #94a3b8 !important; font-size: 0.8rem; margin-bottom: 20px; line-height: 1.3;">Real-time checks of relevance, faithfulness, and grounding sources.</p>',
                unsafe_allow_html=True,
            )

            if st.session_state.chat_history:
                # Get last assistant result
                last_response = [
                    c for c in st.session_state.chat_history if c["role"] == "assistant"
                ]
                if last_response:
                    latest = last_response[-1]

                    # Display decision banner
                    if latest["decision"] == "ACCEPT":
                        st.markdown(
                            """
                        <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.05)); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 10px; padding: 12px 14px; margin-bottom: 8px; display: flex; align-items: center; gap: 10px;">
                            <span style="background: #10b981; color: white !important; border-radius: 50%; width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: bold; box-shadow: 0 0 8px rgba(16, 185, 129, 0.4); line-height: 1;">✓</span>
                            <div>
                                <div style="color: #f8fafc !important; font-weight: 600; font-size: 0.85rem;">Response Approved</div>
                                <div style="color: #34d399 !important; font-size: 0.7rem; font-weight: 500;">Pipeline factual validation checks passed</div>
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            """
                        <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.05)); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 10px; padding: 12px 14px; margin-bottom: 8px; display: flex; align-items: center; gap: 10px;">
                            <span style="background: #ef4444; color: white !important; border-radius: 50%; width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: bold; box-shadow: 0 0 8px rgba(239, 68, 68, 0.4); line-height: 1;">!</span>
                            <div>
                                <div style="color: #f8fafc !important; font-weight: 600; font-size: 0.85rem;">Escalation Triggered</div>
                                <div style="color: #f87171 !important; font-size: 0.7rem; font-weight: 500;">Verification scores below confidence threshold</div>
                            </div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04); border-radius: 8px; padding: 6px 12px; margin-bottom: 5px;">
                        <span style="color: #94a3b8 !important; font-size: 0.8rem;">Query Classification</span>
                        <span style="color: #818cf8 !important; background: rgba(129, 140, 248, 0.1); border: 1px solid rgba(129, 140, 248, 0.2); padding: 1px 6px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; font-family: monospace;">{latest.get('intent_type', 'SINGLE')}</span>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Metric Columns (Custom Styled)
                    metrics = latest.get("metrics", {})
                    cr_val = metrics.get("retrieval_relevance", 0.0)
                    f_val = metrics.get("faithfulness", 0.0)

                    st.markdown(
                        f"""
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 5px;">
                        <div style="background: rgba(99, 102, 241, 0.02); border: 1px solid rgba(99, 102, 241, 0.06); border-radius: 10px; padding: 10px 6px; text-align: center;">
                            <div style="color: #94a3b8 !important; font-size: 0.75rem; margin-bottom: 2px;">Context Relevance</div>
                            <div style="color: #818cf8 !important; font-size: 1.25rem; font-weight: 700; font-family: 'Space Grotesk', sans-serif;">{cr_val:.1%}</div>
                        </div>
                        <div style="background: rgba(52, 211, 153, 0.02); border: 1px solid rgba(52, 211, 153, 0.06); border-radius: 10px; padding: 10px 6px; text-align: center;">
                            <div style="color: #94a3b8 !important; font-size: 0.75rem; margin-bottom: 2px;">Faithfulness Score</div>
                            <div style="color: #34d399 !important; font-size: 1.25rem; font-weight: 700; font-family: 'Space Grotesk', sans-serif;">{f_val:.1%}</div>
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Confidence Progress bar
                    conf_val = latest.get("confidence_score", 0.0)
                    conf_color = (
                        "#34d399"
                        if conf_val >= 0.70
                        else "#fbbf24" if conf_val >= 0.50 else "#f87171"
                    )
                    st.markdown(
                        f"""
                    <div style="background: rgba(255, 255, 255, 0.01); border: 1px solid rgba(255, 255, 255, 0.04); border-radius: 10px; padding: 10px 12px; margin-bottom: 5px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                            <span style="color: #cbd5e1 !important; font-size: 0.8rem; font-weight: 600;">Overall Confidence Score</span>
                            <span style="color: {conf_color} !important; font-size: 0.9rem; font-weight: 700; font-family: 'Space Grotesk', sans-serif;">{conf_val:.2f}</span>
                        </div>
                        <div style="width: 100%; background-color: rgba(255,255,255,0.05); border-radius: 100px; height: 6px; overflow: hidden;">
                            <div style="width: {min(100, max(0, conf_val * 100)):.1f}%; background: linear-gradient(to right, #6366f1, {conf_color}); height: 100%; border-radius: 100px;"></div>
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Detailed Raw Output Collapsible
                    if latest.get("raw_answer") != latest.get("answer"):
                        with st.expander("🔍 View Raw Generated Response"):
                            st.caption(
                                "This was the initial answer before claim verification filtered unsupported claims:"
                            )
                            st.write(latest["raw_answer"])

                    # Source Documents display
                    st.markdown(
                        "<h4 style=\"font-family: 'Space Grotesk'; margin-top: 10px; color: #e2e8f0 !important; font-size: 1.0rem; font-weight: 600; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;\">📄 Grounding Sources</h4>",
                        unsafe_allow_html=True,
                    )
                    sources = latest.get("sources", [])
                    if sources:
                        import html

                        sources_html = '<div style="max-height: 220px; overflow-y: auto; padding-right: 4px; display: flex; flex-direction: column; gap: 6px;">'
                        for idx, src in enumerate(sources):
                            escaped_src = html.escape(src)
                            sources_html += f'<div style="background: rgba(99, 102, 241, 0.03); border: 1px solid rgba(99, 102, 241, 0.08); border-radius: 8px; padding: 8px 12px; display: flex; align-items: center;"><span style="color: #818cf8 !important; font-size: 1rem; margin-right: 10px; flex-shrink: 0;">📄</span><div style="color: #cbd5e1 !important; font-size: 0.8rem; font-family: monospace; word-break: break-all; line-height: 1.2;">{escaped_src}</div></div>'
                        sources_html += "</div>"
                        st.markdown(sources_html, unsafe_allow_html=True)
                    else:
                        st.markdown(
                            """
                        <div style="background: rgba(255,255,255,0.01); border: 1px dashed rgba(255,255,255,0.08); border-radius: 8px; padding: 12px; text-align: center; color: #94a3b8 !important; font-size: 0.8rem;">
                            No sources retrieved for this query.
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        """
                    <div style="text-align: center; padding: 40px 10px; color: #94a3b8 !important; border: 1px dashed rgba(255,255,255,0.08); border-radius: 10px; background: rgba(255,255,255,0.01); margin-top: 15px;">
                        <div style="font-size: 2.2rem; margin-bottom: 12px;">📊</div>
                        <div style="font-weight: 600; color: #f8fafc !important; margin-bottom: 6px; font-family: 'Space Grotesk', sans-serif; font-size: 0.95rem;">No Analytics Loaded</div>
                        <div style="font-size: 0.8rem; line-height: 1.4; max-width: 220px; margin: 0 auto; color: #64748b !important;">Ask a question in the chatbot interface to view real-time confidence scores, relevance metrics, and grounding evidence.</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    """
                <div style="text-align: center; padding: 40px 10px; color: #94a3b8 !important; border: 1px dashed rgba(255,255,255,0.08); border-radius: 10px; background: rgba(255,255,255,0.01); margin-top: 15px;">
                    <div style="font-size: 2.2rem; margin-bottom: 12px;">📊</div>
                    <div style="font-weight: 600; color: #f8fafc !important; margin-bottom: 6px; font-family: 'Space Grotesk', sans-serif; font-size: 0.95rem;">No Analytics Loaded</div>
                    <div style="font-size: 0.8rem; line-height: 1.4; max-width: 220px; margin: 0 auto; color: #64748b !important;">Ask a question in the chatbot interface to view real-time confidence scores, relevance metrics, and grounding evidence.</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )


elif menu == "📈 Performance Graphs":
    # Performance Graphs Tab
    with st.container(border=True):
        st.markdown(
            "<h3 style=\"font-family: 'Space Grotesk'; font-weight: 600; margin-top:0px; color: #f8fafc !important;\">📊 Framework Version Evaluation</h3>",
            unsafe_allow_html=True,
        )

        st.write(
            "This dashboard displays the comparative evaluation metrics (Retrieval Relevance, Answer Relevance, and Faithfulness) across all six incremental versions of the framework (V0 to V5), as established in the M.Tech project research methodology."
        )

        col_btn, col_info = st.columns([0.4, 0.6])
        with col_btn:
            if st.button("🔄 Run Live Golden Batch Evaluation"):
                with st.spinner(
                    "Executing golden dataset queries across V0-V5... (Approx 45s)"
                ):
                    try:
                        for v in ["v0", "v1", "v2", "v3", "v4", "v5"]:
                            run_evaluation_for_version(v)
                        st.success(
                            "Batch evaluation completed and CSV metrics updated successfully!"
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Batch evaluation failed: {str(e)}")
        with col_info:
            st.caption(
                "Running the batch evaluation invokes the golden dataset queries (across placements, hostels, admissions, etc.) sequentially through all 6 framework versions and logs the metrics to a CSV file."
            )

        summary = get_comparison_summary()

        # Metrics Averages Table
        st.write("#### 📈 Average Precision Metrics")
        st.dataframe(
            summary.style.format(
                {
                    "Retrieval_Relevance": "{:.2%}",
                    "Answer_Relevance": "{:.2%}",
                    "Faithfulness": "{:.2%}",
                }
            ),
            use_container_width=True,
        )

        # Plot Comparison Line Graphs (Matplotlib style matching the paper)
        st.write("#### 📉 Performance Comparison Graphs")

        graph_option = st.selectbox(
            "Choose Figure to View:",
            [
                "Combined Metrics Comparison",
                "Figure 3: Retrieval Relevance (RR) Across Framework Versions",
                "Figure 4: Answer Relevance (AR) Across Framework Versions",
                "Figure 5: Faithfulness (F) Across Framework Versions",
            ],
        )

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 5.5))

            if graph_option == "Combined Metrics Comparison":
                ax.plot(
                    summary["Version"],
                    summary["Retrieval_Relevance"],
                    marker="o",
                    label="Retrieval Relevance (RR)",
                    color="#818cf8",
                    linewidth=2.5,
                )
                ax.plot(
                    summary["Version"],
                    summary["Answer_Relevance"],
                    marker="s",
                    label="Answer Relevance (AR)",
                    color="#34d399",
                    linewidth=2.5,
                )
                ax.plot(
                    summary["Version"],
                    summary["Faithfulness"],
                    marker="^",
                    label="Faithfulness (F)",
                    color="#f87171",
                    linewidth=2.5,
                )
                title = "Performance Metrics Trends Across Framework Versions (V0 - V5)"
            elif (
                graph_option
                == "Figure 3: Retrieval Relevance (RR) Across Framework Versions"
            ):
                ax.plot(
                    summary["Version"],
                    summary["Retrieval_Relevance"],
                    marker="o",
                    label="Retrieval Relevance (RR)",
                    color="#818cf8",
                    linewidth=2.5,
                )
                title = "Figure 3: Retrieval Relevance Across Framework Versions"
            elif (
                graph_option
                == "Figure 4: Answer Relevance (AR) Across Framework Versions"
            ):
                ax.plot(
                    summary["Version"],
                    summary["Answer_Relevance"],
                    marker="s",
                    label="Answer Relevance (AR)",
                    color="#34d399",
                    linewidth=2.5,
                )
                title = "Figure 4: Answer Relevance Across Framework Versions"
            else:
                ax.plot(
                    summary["Version"],
                    summary["Faithfulness"],
                    marker="^",
                    label="Faithfulness (F)",
                    color="#f87171",
                    linewidth=2.5,
                )
                title = "Figure 5: Faithfulness Across Framework Versions"

            # Style chart labels
            ax.set_title(title, fontsize=13, pad=15, color="#f8fafc", weight="bold")
            ax.set_xlabel("Framework Version", fontsize=11, labelpad=8, color="#cbd5e1")
            ax.set_ylabel("Metric Score", fontsize=11, labelpad=8, color="#cbd5e1")
            ax.set_ylim(0.4, 1.05)
            ax.grid(True, linestyle="--", alpha=0.15)
            ax.legend(
                facecolor="#1e293b", edgecolor="none", labelcolor="#cbd5e1", fontsize=10
            )
            ax.tick_params(colors="#cbd5e1", labelsize=10)

            # Style background
            fig.patch.set_facecolor("#111827")
            ax.set_facecolor("#1f2937")
            ax.spines["bottom"].set_color("#374151")
            ax.spines["top"].set_color("#374151")
            ax.spines["right"].set_color("#374151")
            ax.spines["left"].set_color("#374151")

            st.pyplot(fig)
        except Exception as e:
            st.error(f"Plotting failed: {str(e)}")

else:
    # Ingest Knowledge Base Page
    with st.container(border=True):
        st.markdown(
            "<h3 style=\"font-family: 'Space Grotesk'; font-weight: 600; margin-top:0px; color: #f8fafc !important;\">📤 Ingest Knowledge Base</h3>",
            unsafe_allow_html=True,
        )
        st.write(
            "Upload institutional documents and FAQ datasets to build or update the RAG system's knowledge base. New uploads are processed, chunked, embedded using local Hugging Face embedding models, and persisted to the local Chroma Vector Database."
        )

        col_pdf, col_csv = st.columns(2)
        with col_pdf:
            st.markdown(
                "<h4 style=\"font-family: 'Space Grotesk'; color: #818cf8 !important; font-weight: 600;\">📂 Institutional PDFs</h4>",
                unsafe_allow_html=True,
            )
            uploaded_pdfs = st.file_uploader(
                "Upload PDFs (e.g. seat_handbook.pdf)",
                type=["pdf"],
                accept_multiple_files=True,
                key="ingest_pdfs",
            )
        with col_csv:
            st.markdown(
                "<h4 style=\"font-family: 'Space Grotesk'; color: #34d399 !important; font-weight: 600;\">📂 FAQ Datasets</h4>",
                unsafe_allow_html=True,
            )
            uploaded_csvs = st.file_uploader(
                "Upload CSVs (e.g. seat_faq_dataset.csv)",
                type=["csv"],
                accept_multiple_files=True,
                key="ingest_csvs",
            )

        st.markdown(
            "<hr style='border-color: rgba(255,255,255,0.05); margin-top: 20px; margin-bottom: 20px;'>",
            unsafe_allow_html=True,
        )

        if st.button("🚀 Process & Load Knowledge Base", use_container_width=True):
            if not uploaded_pdfs and not uploaded_csvs:
                st.error("Please upload at least one PDF or CSV file first.")
            else:
                with st.spinner(
                    "Processing uploads & building database... (This involves semantic chunking, local embeddings, and vector DB indexing)"
                ):
                    try:
                        req_files = []
                        if uploaded_pdfs:
                            for pdf in uploaded_pdfs:
                                req_files.append(
                                    (
                                        "pdfs",
                                        (pdf.name, pdf.getvalue(), "application/pdf"),
                                    )
                                )
                        if uploaded_csvs:
                            for csv in uploaded_csvs:
                                req_files.append(
                                    ("csvs", (csv.name, csv.getvalue(), "text/csv"))
                                )

                        res = requests.post(f"{API_URL}/ingest", files=req_files)
                        if res.status_code == 200:
                            st.success(
                                "🎉 Ingestion successful! Knowledge base reloaded and Chroma DB indexed successfully."
                            )
                        else:
                            st.error(
                                f"Ingestion failed: {res.json().get('detail', 'Unknown error')}"
                            )
                    except Exception as e:
                        st.error(f"Error connecting to backend: {str(e)}")
