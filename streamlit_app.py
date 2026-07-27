"""
streamlit_app.py
==================
MarketMind AI -- main Streamlit entrypoint.

Orchestrates the full RAG pipeline (PDF upload -> extraction -> cleaning ->
chunking -> embedding -> ChromaDB -> retrieval -> grounded generation) behind
a premium, ChatGPT/Perplexity-style dashboard, plus a standalone Campaign
Angle & Audience Extractor marketing-insights tool.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import time

import streamlit as st

import config
from utils import format_bytes, load_module, logger, truncate

# --------------------------------------------------------------------------- #
# Dynamically load the numerically-prefixed pipeline modules
# --------------------------------------------------------------------------- #
documents_mod = load_module("mm_documents", "01_documents.py")
preprocessing_mod = load_module("mm_preprocessing", "02_preprocessing.py")
chunking_mod = load_module("mm_chunking", "03_chunking.py")
vectors_mod = load_module("mm_vectors", "04_vector_representation.py")
store_mod = load_module("mm_store", "05_create_chroma_store.py")
retrieve_mod = load_module("mm_retrieve", "06_retrieve_context.py")
prompting_mod = load_module("mm_prompting", "07_prompting.py")

# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title=f"{config.APP_NAME} | {config.APP_SUBTITLE}",
    page_icon=str(config.FAVICON_PATH) if config.FAVICON_PATH.exists() else "📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Custom CSS -- Deep Navy / Slate Blue / White / Soft Gray design system
# --------------------------------------------------------------------------- #
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --navy: #0B1120;
    --navy-light: #16213E;
    --slate: #475569;
    --slate-light: #64748B;
    --accent: #3B82F6;
    --accent-soft: #93C5FD;
    --white: #FFFFFF;
    --soft-gray: #F1F5F9;
    --border: #E2E8F0;
    --text-dark: #0F172A;
    --text-muted: #64748B;
    --success: #16A34A;
    --warning: #D97706;
    --danger: #DC2626;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Sora', sans-serif !important; color: var(--text-dark); }

.stApp { background: var(--soft-gray); }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--navy) 0%, var(--navy-light) 100%);
}
section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12); }

.sidebar-brand {
    display: flex; align-items: center; gap: 12px;
    padding: 4px 0 18px 0;
}
.sidebar-brand img { border-radius: 10px; }
.sidebar-brand .brand-name { font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.15rem; color: #fff; margin: 0; }
.sidebar-brand .brand-sub { font-size: 0.72rem; color: #93A3C4; margin: 0; }

.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 999px;
    font-size: 0.75rem; font-weight: 600;
    background: rgba(255,255,255,0.08);
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.dot-ready { background: #22C55E; box-shadow: 0 0 6px #22C55E; }
.dot-empty { background: #94A3B8; }

.kb-meta-row {
    display: flex; justify-content: space-between;
    font-size: 0.8rem; padding: 5px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.kb-meta-label { color: #93A3C4 !important; }
.kb-meta-value { color: #fff !important; font-weight: 600; text-align: right; }

/* ---------- Hero ---------- */
.hero-wrap {
    background: linear-gradient(120deg, var(--navy) 0%, #1E293B 55%, var(--navy-light) 100%);
    border-radius: 20px;
    padding: 40px 44px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 12px 32px rgba(15, 23, 42, 0.25);
}
.hero-wrap::after {
    content: "";
    position: absolute; top: -60px; right: -60px;
    width: 260px; height: 260px; border-radius: 50%;
    background: radial-gradient(circle, rgba(59,130,246,0.35) 0%, rgba(59,130,246,0) 70%);
}
.hero-eyebrow {
    color: var(--accent-soft); font-size: 0.78rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 10px;
}
.hero-title {
    color: #fff; font-family: 'Sora', sans-serif; font-weight: 800;
    font-size: 2.15rem; margin: 0 0 10px 0; line-height: 1.15;
}
.hero-subtitle { color: #B9C4DE; font-size: 1.02rem; max-width: 640px; margin: 0; }

/* ---------- Cards ---------- */
.mm-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px 24px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    margin-bottom: 18px;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.mm-card:hover { box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08); }

.answer-card {
    background: var(--white);
    border-left: 4px solid var(--accent);
    border-radius: 12px;
    padding: 22px 26px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
    font-size: 1rem; line-height: 1.65; color: var(--text-dark);
    animation: fadeIn 0.4s ease;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(6px);} to { opacity: 1; transform: translateY(0);} }

.source-card {
    background: var(--soft-gray);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 0.85rem;
}
.source-header {
    display: flex; justify-content: space-between; align-items: center;
    font-weight: 700; color: var(--navy); margin-bottom: 4px;
}
.similarity-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; font-weight: 600;
    padding: 2px 8px; border-radius: 999px;
    background: var(--accent); color: #fff;
}
.source-meta { color: var(--text-muted); font-size: 0.76rem; margin-bottom: 6px; }
.source-text { color: var(--slate); font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; line-height: 1.5; }

/* ---------- Buttons ---------- */
.stButton > button {
    background: var(--navy); color: #fff; border: none;
    border-radius: 10px; font-weight: 600; padding: 0.55em 1.2em;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover { background: var(--accent); transform: translateY(-1px); box-shadow: 0 6px 16px rgba(59,130,246,0.35); }

/* ---------- Misc ---------- */
.section-label {
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--slate-light); margin-bottom: 6px;
}
.mm-divider { border: none; border-top: 1px solid var(--border); margin: 22px 0; }
.pill { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.72rem; font-weight: 700; }
.pill-navy { background: var(--navy); color: #fff; }
.pill-soft { background: #DBEAFE; color: #1D4ED8; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Session state initialization
# --------------------------------------------------------------------------- #
def _init_state() -> None:
    defaults = {
        "kb_ready": store_mod.collection_exists(),
        "current_pdf_name": None,
        "chat_history": [],
        "marketing_insights": None,
        "processing_log": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


# --------------------------------------------------------------------------- #
# Pipeline orchestration
# --------------------------------------------------------------------------- #
def process_uploaded_pdf(uploaded_file) -> bool:
    """Run the full ingestion pipeline on a newly-uploaded PDF.

    Returns True on success, False on failure (with a user-facing error
    already displayed).
    """
    progress = st.progress(0, text="Starting...")
    status = st.empty()

    try:
        # --- Step 1: Extraction ---------------------------------------- #
        status.info("📄 Extracting text from PDF...")
        progress.progress(10, text="Extracting PDF text...")
        file_bytes = uploaded_file.getvalue()
        if len(file_bytes) == 0:
            st.error("The uploaded file is empty. Please choose a valid PDF.")
            return False

        pages = documents_mod.extract_pdf_text(file_bytes, uploaded_file.name)

        # --- Step 2: Cleaning -------------------------------------------#
        status.info("🧹 Cleaning and normalizing text...")
        progress.progress(30, text="Cleaning text...")
        cleaned_pages = preprocessing_mod.preprocess_page_documents(pages)
        if not cleaned_pages:
            st.error(
                "No usable text remained after cleaning. The PDF may be "
                "mostly images or non-text content."
            )
            return False

        # --- Step 3: Chunking -------------------------------------------#
        status.info("✂️ Splitting into adaptive chunks...")
        progress.progress(45, text="Chunking document...")
        chunks = chunking_mod.chunk_page_documents(cleaned_pages)

        # --- Step 4: Embeddings ------------------------------------------#
        status.info("🧠 Generating embeddings (this can take a moment)...")
        progress.progress(65, text="Generating embeddings...")
        texts = [c.text for c in chunks]
        embeddings = vectors_mod.embed_texts(texts)

        # --- Step 5: Vector store -----------------------------------------#
        status.info("🗄️ Building the vector database...")
        progress.progress(85, text="Building ChromaDB collection...")
        store_mod.build_vector_store(chunks, embeddings, reset_existing=True)

        progress.progress(100, text="Done!")
        status.success(
            f"✅ Knowledge base ready — {len(chunks)} chunks indexed from "
            f"{len(cleaned_pages)} page(s) of **{uploaded_file.name}**."
        )
        st.session_state.kb_ready = True
        st.session_state.current_pdf_name = uploaded_file.name
        st.session_state.marketing_insights = None
        time.sleep(0.6)
        return True

    except documents_mod.DocumentExtractionError as exc:
        st.error(f"📄 Document error: {exc}")
    except vectors_mod.EmbeddingError as exc:
        st.error(f"🧠 Embedding error: {exc}")
    except store_mod.VectorStoreError as exc:
        st.error(f"🗄️ Vector store error: {exc}")
    except Exception as exc:  # noqa: BLE001 - final safety net, never crash the app
        logger.exception("Unexpected error while processing PDF")
        st.error(f"⚠️ An unexpected error occurred while processing the file: {exc}")
    return False


def answer_question(query: str):
    """Retrieve context and generate a grounded answer; returns (answer, chunks) or (None, None)."""
    if not config.is_llm_configured():
        st.error(
            "⚠️ OPENROUTER_API_KEY is not configured. Add it to "
            "`.streamlit/secrets.toml` (see `secrets.toml.example`) or as an "
            "environment variable, then reload the app."
        )
        return None, None

    try:
        collection = store_mod.get_or_create_collection()
        retrieved = retrieve_mod.retrieve_top_k(
            query, collection, vectors_mod.embed_query, top_k=config.TOP_K
        )
    except retrieve_mod.RetrievalError as exc:
        st.error(f"🔎 Retrieval error: {exc}")
        return None, None
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected retrieval error")
        st.error(f"⚠️ Could not search the knowledge base: {exc}")
        return None, None

    try:
        answer = prompting_mod.generate_answer(query, retrieved)
    except RuntimeError as exc:
        st.error(f"⚠️ {exc}")
        return None, None
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected LLM generation error")
        st.error(
            "🌐 The AI model could not be reached. This is usually a network "
            f"or OpenRouter issue. Details: {exc}"
        )
        return None, None

    return answer, retrieved


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    if config.LOGO_PATH.exists():
        import base64

        logo_b64 = base64.b64encode(config.LOGO_PATH.read_bytes()).decode()
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <img src="data:image/png;base64,{logo_b64}" width="42" height="42" />
                <div>
                    <p class="brand-name">{config.APP_NAME}</p>
                    <p class="brand-sub">{config.APP_SUBTITLE}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"### {config.APP_NAME}")
        st.caption(config.APP_SUBTITLE)

    stats = store_mod.get_collection_stats()
    is_ready = stats["exists"] and stats["count"] > 0
    st.session_state.kb_ready = is_ready

    dot_class = "dot-ready" if is_ready else "dot-empty"
    status_text = "Knowledge base ready" if is_ready else "No knowledge base yet"
    st.markdown(
        f'<span class="status-pill"><span class="status-dot {dot_class}"></span>{status_text}</span>',
        unsafe_allow_html=True,
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Knowledge Base</div>', unsafe_allow_html=True)

    meta_rows = [
        ("Embedding model", config.EMBEDDING_MODEL_NAME.split("/")[-1]),
        ("LLM (OpenRouter)", config.OPENROUTER_MODEL),
        ("Collection", config.COLLECTION_NAME),
        ("Chunks indexed", str(stats["count"])),
        ("Top-K retrieval", str(config.TOP_K)),
    ]
    for label, value in meta_rows:
        st.markdown(
            f'<div class="kb-meta-row"><span class="kb-meta-label">{label}</span>'
            f'<span class="kb-meta-value">{value}</span></div>',
            unsafe_allow_html=True,
        )

    if stats["sources"]:
        st.markdown('<div class="section-label" style="margin-top:14px;">Uploaded Document(s)</div>', unsafe_allow_html=True)
        for src in stats["sources"]:
            st.markdown(f"📄 {truncate(src, 40)}")

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Upload &amp; Build</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a PDF", type=["pdf"], label_visibility="collapsed",
        help="Market reports, research papers, business reports, and more.",
    )

    build_clicked = st.button("⚙️ Build Vector Store", use_container_width=True)
    reset_clicked = st.button("🗑️ Reset Knowledge Base", use_container_width=True)

    if reset_clicked:
        try:
            store_mod.reset_vector_store()
            st.session_state.kb_ready = False
            st.session_state.current_pdf_name = None
            st.session_state.chat_history = []
            st.session_state.marketing_insights = None
            st.success("Knowledge base reset.")
            st.rerun()
        except store_mod.VectorStoreError as exc:
            st.error(str(exc))

    if build_clicked:
        if uploaded_file is None:
            st.warning("Please choose a PDF file to upload first.")
        else:
            with st.container():
                success = process_uploaded_pdf(uploaded_file)
            if success:
                st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)
    with st.expander("ℹ️ About this project"):
        st.markdown(
            f"""
            **{config.APP_NAME}** v{config.APP_VERSION}

            A retrieval-augmented generation platform that turns any uploaded
            PDF (market reports, research papers, competitor analyses) into
            a queryable knowledge base with grounded, cited answers — plus a
            one-click marketing intelligence extractor.

            **Pipeline:** Upload → Extract → Clean → Chunk → Embed → Index →
            Retrieve → Generate → Cite

            Built with Streamlit, ChromaDB, LlamaIndex, Sentence-Transformers,
            and OpenRouter.
            """
        )

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
    <div class="hero-wrap">
        <div class="hero-eyebrow">AI-Powered Market Intelligence</div>
        <div class="hero-title">Ask your documents anything.</div>
        <p class="hero-subtitle">
            Upload a market report, research paper, or competitor analysis and
            get grounded, cited answers instantly — plus an automatic
            marketing strategy brief extracted from the same document.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Main tabs
# --------------------------------------------------------------------------- #
tab_ask, tab_marketing = st.tabs(["💬 Ask Questions", "📈 Campaign & Audience Extractor"])

# ============================== TAB 1: ASK ================================ #
with tab_ask:
    if not st.session_state.kb_ready:
        st.markdown(
            """
            <div class="mm-card">
                <b>No knowledge base yet.</b><br/>
                Upload a PDF in the sidebar and click <b>Build Vector Store</b> to get started.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        col_q, col_btn = st.columns([5, 1])
        with col_q:
            query = st.text_input(
                "Ask a question about your document",
                placeholder='e.g. "Summarize the key findings" or "What are the main trends?"',
                label_visibility="collapsed",
                key="query_input",
            )
        with col_btn:
            ask_clicked = st.button("Ask →", use_container_width=True)

        if ask_clicked and query.strip():
            with st.spinner("Searching the knowledge base and generating a grounded answer..."):
                answer, retrieved = answer_question(query.strip())

            if answer is not None:
                st.session_state.chat_history.insert(0, {
                    "query": query.strip(),
                    "answer": answer,
                    "sources": retrieved,
                })
        elif ask_clicked:
            st.warning("Please type a question first.")

        for turn in st.session_state.chat_history:
            st.markdown('<div class="section-label">Question</div>', unsafe_allow_html=True)
            st.markdown(f"**{turn['query']}**")
            st.markdown('<div class="section-label" style="margin-top:14px;">Answer</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-card">{turn["answer"]}</div>', unsafe_allow_html=True)

            if turn["sources"]:
                with st.expander(f"📚 Sources ({len(turn['sources'])})"):
                    for src in turn["sources"]:
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <div class="source-header">
                                    <span>Source {src.rank}</span>
                                    <span class="similarity-badge">{src.similarity:.0%} match</span>
                                </div>
                                <div class="source-meta">
                                    📄 {src.source} &nbsp;·&nbsp; Page {src.page_number} &nbsp;·&nbsp; Chunk #{src.chunk_index}
                                </div>
                                <div class="source-text">{truncate(src.text, 280)}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            st.markdown('<hr class="mm-divider"/>', unsafe_allow_html=True)

# ========================= TAB 2: MARKETING INSIGHTS ======================= #
with tab_marketing:
    st.markdown(
        """
        <div class="mm-card">
            <b>Campaign Angle & Audience Extractor</b><br/>
            One click to analyze your uploaded document and extract target
            audience, positioning, campaign ideas, SEO keywords, buyer
            journey, and a full SWOT — grounded strictly in your document.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.kb_ready:
        st.info("Upload and build a knowledge base first (see sidebar).")
    else:
        analyze_clicked = st.button("✨ Analyze Document for Marketing Insights")

        if analyze_clicked:
            if not config.is_llm_configured():
                st.error(
                    "⚠️ OPENROUTER_API_KEY is not configured. Add it to "
                    "`.streamlit/secrets.toml` or as an environment variable."
                )
            else:
                with st.spinner("Analyzing document and extracting marketing insights..."):
                    try:
                        collection = store_mod.get_or_create_collection()
                        # Pull a broad sample of chunks for a holistic brief.
                        broad_context = retrieve_mod.retrieve_top_k(
                            "overview summary key information about the product, market, "
                            "audience, and business",
                            collection,
                            vectors_mod.embed_query,
                            top_k=min(12, config.TOP_K * 3),
                        )
                        raw_json = prompting_mod.generate_marketing_insights(broad_context)
                        insights = json.loads(raw_json)
                        st.session_state.marketing_insights = insights
                    except retrieve_mod.RetrievalError as exc:
                        st.error(f"🔎 Retrieval error: {exc}")
                    except json.JSONDecodeError:
                        st.error(
                            "⚠️ The AI model returned an unexpected format. Please try again."
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("Marketing insight generation failed")
                        st.error(f"⚠️ Could not generate marketing insights: {exc}")

        insights = st.session_state.marketing_insights
        if insights:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="mm-card">', unsafe_allow_html=True)
                st.markdown("#### 🎯 Target Audience")
                st.write(insights.get("target_audience", "—"))
                st.markdown("#### 👤 Customer Persona")
                st.write(insights.get("customer_persona", "—"))
                st.markdown("#### 💡 Unique Selling Proposition")
                st.write(insights.get("usp", "—"))
                st.markdown("#### 📍 Product Positioning")
                st.write(insights.get("product_positioning", "—"))
                st.markdown("#### 💎 Value Proposition")
                st.write(insights.get("value_proposition", "—"))
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="mm-card">', unsafe_allow_html=True)
                st.markdown("#### 😖 Pain Points")
                for item in insights.get("pain_points", []):
                    st.markdown(f"- {item}")
                st.markdown("#### ✅ Customer Needs")
                for item in insights.get("customer_needs", []):
                    st.markdown(f"- {item}")
                st.markdown("</div>", unsafe_allow_html=True)

            with c2:
                st.markdown('<div class="mm-card">', unsafe_allow_html=True)
                st.markdown("#### 📣 Marketing Angle")
                st.write(insights.get("marketing_angle", "—"))
                st.markdown("#### 🚀 Campaign Ideas")
                for item in insights.get("campaign_ideas", []):
                    st.markdown(f"- {item}")
                st.markdown("#### ✍️ Ad Headlines")
                for item in insights.get("ad_headlines", []):
                    st.markdown(f'<span class="pill pill-soft">{item}</span>', unsafe_allow_html=True)
                st.markdown("#### 👉 CTA Suggestions")
                for item in insights.get("cta_suggestions", []):
                    st.markdown(f"- {item}")
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="mm-card">', unsafe_allow_html=True)
                st.markdown("#### 🔑 SEO Keywords")
                keywords = insights.get("seo_keywords", [])
                st.markdown(
                    " ".join(f'<span class="pill pill-navy">{k}</span>' for k in keywords),
                    unsafe_allow_html=True,
                )
                st.markdown("#### 🛤️ Buyer Journey")
                for i, stage in enumerate(insights.get("buyer_journey", []), start=1):
                    st.markdown(f"**{i}.** {stage}")
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="mm-card">', unsafe_allow_html=True)
            st.markdown("#### 🧭 Marketing Opportunities")
            for item in insights.get("marketing_opportunities", []):
                st.markdown(f"- {item}")
            st.markdown("</div>", unsafe_allow_html=True)

            swot = insights.get("swot", {})
            if swot:
                st.markdown('<div class="mm-card">', unsafe_allow_html=True)
                st.markdown("#### 🧩 SWOT Analysis")
                s1, s2, s3, s4 = st.columns(4)
                for col, key, label, emoji in [
                    (s1, "strengths", "Strengths", "💪"),
                    (s2, "weaknesses", "Weaknesses", "⚠️"),
                    (s3, "opportunities", "Opportunities", "🌱"),
                    (s4, "threats", "Threats", "🛑"),
                ]:
                    with col:
                        st.markdown(f"**{emoji} {label}**")
                        for item in swot.get(key, []):
                            st.markdown(f"- {item}")
                st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
    <div style="text-align:center; color:#94A3B8; font-size:0.78rem; padding: 24px 0 8px 0;">
        {config.APP_NAME} v{config.APP_VERSION} · Answers are grounded strictly in your uploaded document(s)
    </div>
    """,
    unsafe_allow_html=True,
)
