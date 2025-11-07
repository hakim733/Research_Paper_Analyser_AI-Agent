import streamlit as st
import json
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image

from modules import LLMBackend, RAGResearchProcessorLLM, LLMUCREvaluator, CIREstimator


st.set_page_config(page_title="📘 Research Paper Analyzer", layout="wide", initial_sidebar_state="collapsed")

# --- Custom CSS for futuristic theme ---
st.markdown("""
    <style>
        /* Main app background - dark futuristic theme */
        .stApp {
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
            background-attachment: fixed;
            color: #f1f5f9 !important;
            font-family: 'Segoe UI', 'Inter', -apple-system, sans-serif;
        }
        
        /* Make all text more visible */
        .stMarkdown, .stMarkdown p, .stMarkdown li {
            color: #f1f5f9 !important;
        }
        
        /* Animated background particles effect */
        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                radial-gradient(circle at 20% 50%, rgba(99, 102, 241, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(139, 92, 246, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 40% 20%, rgba(59, 130, 246, 0.1) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }

        /* Header styling with logo */
        .header-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1.5rem;
            padding: 2rem 0;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
            border-bottom: 2px solid rgba(99, 102, 241, 0.3);
            border-radius: 0 0 20px 20px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(99, 102, 241, 0.2);
            backdrop-filter: blur(10px);
        }
        
        .header-logo {
            height: 80px;
            width: auto;
            filter: drop-shadow(0 0 10px rgba(99, 102, 241, 0.5));
            animation: pulse-glow 3s ease-in-out infinite;
        }
        
        @keyframes pulse-glow {
            0%, 100% { filter: drop-shadow(0 0 10px rgba(99, 102, 241, 0.5)); }
            50% { filter: drop-shadow(0 0 20px rgba(139, 92, 246, 0.8)); }
        }
        
        .header-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #e0e7ff !important;
            text-shadow: 0 0 20px rgba(99, 102, 241, 0.8), 0 0 40px rgba(139, 92, 246, 0.5);
            letter-spacing: 1px;
        }

        /* Text input styling */
        .stTextInput > div > div > input {
            background-color: rgba(30, 41, 59, 0.9) !important;
            color: #f1f5f9 !important;
            border: 1px solid rgba(99, 102, 241, 0.5) !important;
            border-radius: 10px !important;
            padding: 0.75rem !important;
            transition: all 0.3s ease !important;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #6366f1 !important;
            box-shadow: 0 0 15px rgba(99, 102, 241, 0.4) !important;
            outline: none !important;
        }
        
        .stTextInput label {
            color: #e0e7ff !important;
            font-weight: 500 !important;
        }

        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.75rem 2rem !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 25px rgba(99, 102, 241, 0.6) !important;
            background: linear-gradient(135deg, #818cf8 0%, #a78bfa 100%) !important;
        }

        /* Metric cards */
        [data-testid="stMetricValue"] {
            font-size: 2.5rem !important;
            font-weight: 700 !important;
            color: #e0e7ff !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #c7d2fe !important;
            font-size: 0.9rem !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
        }

        /* Subheaders */
        h2, h3 {
            color: #e0e7ff !important;
            font-weight: 600 !important;
            margin-top: 2rem !important;
            margin-bottom: 1rem !important;
            text-transform: uppercase !important;
            letter-spacing: 2px !important;
            border-left: 4px solid #6366f1 !important;
            padding-left: 1rem !important;
        }
        
        h1 {
            color: #e0e7ff !important;
        }

        /* Info boxes */
        .stInfo {
            background-color: rgba(99, 102, 241, 0.15) !important;
            border-left: 4px solid #6366f1 !important;
            border-radius: 8px !important;
            padding: 1rem !important;
            color: #e0e7ff !important;
        }
        
        .stInfo > div {
            color: #e0e7ff !important;
        }

        /* Success messages */
        .stSuccess {
            background-color: rgba(34, 197, 94, 0.15) !important;
            border-left: 4px solid #22c55e !important;
            border-radius: 8px !important;
            color: #e0e7ff !important;
        }
        
        .stSuccess > div {
            color: #e0e7ff !important;
        }

        /* Spinner */
        .stSpinner > div {
            border-top-color: #6366f1 !important;
        }
        
        .stSpinner label {
            color: #e0e7ff !important;
        }

        /* Chat bubbles */
        .chat-bubble-user {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.3) 0%, rgba(139, 92, 246, 0.3) 100%);
            padding: 1.25rem;
            border-radius: 15px;
            color: #f1f5f9 !important;
            margin-bottom: 0.75rem;
            border: 1px solid rgba(99, 102, 241, 0.3);
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.2);
        }
        
        .chat-bubble-ai {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(51, 65, 85, 0.9) 100%);
            padding: 1.25rem;
            border-radius: 15px;
            color: #e0e7ff !important;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(99, 102, 241, 0.2);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }

        /* Divider */
        hr {
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent 0%, rgba(99, 102, 241, 0.5) 50%, transparent 100%);
            margin: 2.5rem 0;
        }

        /* JSON viewer */
        .stJson {
            background-color: rgba(15, 23, 42, 0.8) !important;
            border: 1px solid rgba(99, 102, 241, 0.3) !important;
            border-radius: 10px !important;
            padding: 1rem !important;
        }

        /* Plotly charts - dark theme */
        .js-plotly-plot {
            background-color: rgba(15, 23, 42, 0.5) !important;
            border-radius: 10px !important;
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(15, 23, 42, 0.5);
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #818cf8 0%, #a78bfa 100%);
        }
    </style>
""", unsafe_allow_html=True)

# --- Header with Logo ---
st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(139, 92, 246, 0.25) 100%);
        border-bottom: 3px solid rgba(99, 102, 241, 0.4);
        border-radius: 0 0 25px 25px;
        margin: -1rem -1rem 2rem -1rem;
        padding: 2rem 3rem;
        box-shadow: 0 10px 40px rgba(99, 102, 241, 0.3);
    ">
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.2, 2.8])
with col1:
    try:
        logo = Image.open("assets/mau_logo.png")
        st.image(logo, width=180, use_container_width=False)
    except Exception as e:
        st.write("Logo")

with col2:
    st.markdown("""
        <div style="padding-top: 2rem;">
            <h1 style="
                font-size: 3.5rem;
                font-weight: 700;
                color: #e0e7ff !important;
                text-shadow: 0 0 25px rgba(99, 102, 241, 0.9), 0 0 50px rgba(139, 92, 246, 0.6);
                letter-spacing: 3px;
                margin: 0;
                line-height: 1.1;
            ">Research Paper Analyzer</h1>
            <p style="
                color: #c7d2fe !important;
                font-size: 1.4rem;
                margin-top: 0.8rem;
                margin-bottom: 0;
                font-weight: 500;
                letter-spacing: 1px;
            ">Malmö Universitet</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; color: #e0e7ff !important; font-size: 1.1rem; margin-bottom: 2rem; padding: 0 2rem;">
        Analyze any research paper for structure, citation impact, and claim reliability — 
        powered by AI-driven insights and interactive visualizations.
    </div>
""", unsafe_allow_html=True)

# Get API key from environment variable or user input
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    api_key = st.text_input("🔑 Enter your GROQ API Key:", type="password")
else:
    st.info("✅ Using API key from environment variable")

url = st.text_input("📎 Enter PDF URL (e.g. arXiv):")

if api_key and url and st.button("🚀 Run Analysis"):
    llm = LLMBackend(api_key=api_key)
    proc = RAGResearchProcessorLLM(llm)
    cir = CIREstimator(llm)
    eval = LLMUCREvaluator(llm)

    with st.spinner("🔍 Processing paper..."):
        text, pages = proc.extract_document_text(url)
        analysis = proc.analyze_research_paper(text)
        chunks = proc.create_rag_chunks(pages)
        cir_res = cir.compute_cir(analysis.get("title", ""), analysis.get("abstract", ""))
        claims = """The Transformer architecture introduced self-attention for sequence modeling.
        It eliminated recurrence and convolution, achieving state-of-the-art results in translation tasks."""
        ucr = eval.analyze_claim_support(claims, chunks)

    st.success("✅ Analysis Complete!")

    # ---- Summary Cards ----
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Citations", cir_res["citations"])
    col2.metric("Novelty", f"{cir_res['novelty']*100:.0f}%")
    col3.metric("CIR Score", f"{cir_res['CIR']*100:.0f}%")

    # ---- Radar Chart (CIR components) ----
    st.subheader("📈 CIR Component Radar Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[cir_res["novelty"], cir_res["CIR"], cir_res["citations"]/100],
        theta=["Novelty", "CIR", "Citations"],
        fill='toself',
        name="CIR Profile",
        line_color='#6366f1'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1], gridcolor='rgba(99, 102, 241, 0.3)'),
            bgcolor='rgba(15, 23, 42, 0.5)'
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#c7d2fe')
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- UCR Bar Chart ----
    st.subheader("⚖️ Claim Support Evaluation (UCR)")
    labels = ["Supported", "Unsupported"]
    values = [ucr["supported"], ucr["unsupported"]]
    fig2 = go.Figure([go.Bar(
        x=labels, 
        y=values, 
        marker_color=["#22c55e", "#ef4444"],
        marker_line_color='#6366f1',
        marker_line_width=2
    )])
    fig2.update_layout(
        yaxis_title="Count", 
        xaxis_title="Claim Type",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.5)',
        font=dict(color='#c7d2fe')
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ---- Word Cloud of Key Concepts ----
    if analysis.get("key_concepts"):
        st.subheader("🌐 Key Concept Cloud")
        text_for_wc = " ".join(analysis["key_concepts"])
        wc = WordCloud(
            width=800, 
            height=400, 
            background_color="#0a0e27",
            colormap='viridis',
            relative_scaling=0.5
        ).generate(text_for_wc)
        plt.figure(figsize=(12, 6), facecolor='#0a0e27')
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(plt)

    # ---- Full Paper Summary ----
    st.subheader("📄 Paper Analysis Summary")
    st.json(analysis)

    # Store chunks and processor in session state for chatbot
    st.session_state["rag_chunks"] = chunks
    st.session_state["proc"] = proc
    st.session_state["llm"] = llm

# ---- Chatbot: Ask the Paper ----
# Only show chatbot if analysis has been run
if "rag_chunks" in st.session_state and st.session_state["rag_chunks"]:
    st.markdown("---")
    st.subheader("Chat with the Paper")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_query = st.text_input("Ask a question about the paper:")

    if user_query:
        with st.spinner("Thinking..."):
            # Retrieve relevant chunks for context
            retrieved = st.session_state["proc"].retrieve_relevant_chunks(
                user_query, 
                st.session_state["rag_chunks"], 
                top_k=3
            )

            # Build a chat prompt using the retrieved context
            context = "\n\n".join([r["text"] for r in retrieved])
            prompt = f"""You are a research assistant. Use the following context from the paper to answer the question.

Context:
{context}
Question: {user_query}
Answer: """

            answer = st.session_state["llm"].chat(prompt)

            # Update chat history
            st.session_state["chat_history"].append({"user": user_query, "assistant": answer})

    # Display chat history
    for chat in st.session_state["chat_history"]:
        st.markdown(f"<div class='chat-bubble-user'>🧑 <b>You:</b> {chat['user']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble-ai'>🤖 <b>AI:</b> {chat['assistant']}</div>", unsafe_allow_html=True)
        st.markdown("---")