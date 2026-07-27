# MarketMind AI

**AI-Powered Marketing & Market Research Assistant**

MarketMind AI is a production-style Retrieval-Augmented Generation (RAG) platform that turns any uploaded PDF — market reports, research papers, competitor analyses, white papers, or academic papers — into a queryable, cited knowledge base, plus a one-click marketing intelligence extractor (audience, positioning, campaigns, SEO, SWOT, and more).

Answers are **strictly grounded** in the uploaded document. If the document doesn't contain the answer, the assistant says so instead of guessing.

---

## Overview

| | |
|---|---|
| **Frontend** | Streamlit (custom navy/slate dashboard) |
| **Vector store** | ChromaDB (persistent, on-disk) |
| **Chunking** | LlamaIndex `SentenceSplitter` (adaptive) |
| **Embeddings** | `BAAI/bge-small-en-v1.5` (Sentence-Transformers) |
| **LLM** | Any OpenRouter model (via OpenAI-compatible SDK) |
| **PDF parsing** | `pypdf` |

## Architecture

```
User uploads PDF
      │
      ▼
01_documents.py        → extract raw text per page (pypdf)
      │
      ▼
02_preprocessing.py    → clean text, strip headers/footers, normalize whitespace
      │
      ▼
03_chunking.py         → adaptive sentence-aware chunking (LlamaIndex)
      │
      ▼
04_vector_representation.py → generate embeddings (BAAI/bge-small-en-v1.5)
      │
      ▼
05_create_chroma_store.py   → persist chunks + embeddings in ChromaDB
      │
      ▼
06_retrieve_context.py      → embed query, retrieve Top-K similar chunks
      │
      ▼
07_prompting.py             → build grounded RAG prompt, call LLM via OpenRouter
      │
      ▼
streamlit_app.py            → display cited, grounded answer
```

Every step logs its duration and status; every step fails gracefully (never crashes the app) and surfaces a clear error message in the UI instead.

## Folder Structure

```
MarketMind_AI/
│
├── assets/
│   ├── logo.png
│   └── favicon.png
│
├── data/                       # scratch space (gitignored contents)
├── chroma_db/                  # persistent vector store (gitignored contents)
├── .streamlit/
│   └── secrets.toml.example    # copy to secrets.toml and fill in
│
├── config.py                   # central configuration (reads secrets/env)
├── utils.py                    # logging, dynamic module loader, timing, formatting
├── streamlit_app.py            # main dashboard application
│
├── 01_documents.py             # PDF text extraction
├── 02_preprocessing.py         # text cleaning
├── 03_chunking.py              # adaptive chunking
├── 04_vector_representation.py # embedding generation
├── 05_create_chroma_store.py   # ChromaDB build/reset/stats
├── 06_retrieve_context.py      # Top-K retrieval
├── 07_prompting.py             # RAG + marketing-insight prompts, LLM calls
│
├── requirements.txt
├── README.md
└── .gitignore
```

> **Note on file names:** steps `01`–`07` start with digits, which are not valid Python import identifiers. `utils.load_module()` loads them dynamically via `importlib`, so the required file names are preserved exactly while still being fully importable.

## Installation

```bash
git clone <your-repo-url>
cd MarketMind_AI
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

1. Copy the secrets template:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
2. Open `.streamlit/secrets.toml` and add your [OpenRouter](https://openrouter.ai/keys) API key:
   ```toml
   OPENROUTER_API_KEY = "sk-or-v1-..."
   OPENROUTER_MODEL = "openai/gpt-4o-mini"
   TOP_K = 5
   ```
3. Alternatively, set the same values as environment variables (a `.env` file also works, via `python-dotenv`):
   ```bash
   export OPENROUTER_API_KEY="sk-or-v1-..."
   export OPENROUTER_MODEL="openai/gpt-4o-mini"
   ```

`config.py` checks `st.secrets` first, then falls back to environment variables — no API key is ever hard-coded.

## How to Run

```bash
streamlit run streamlit_app.py
```

Then, in the app:
1. Upload a PDF in the sidebar.
2. Click **Build Vector Store**.
3. Ask questions in the **Ask Questions** tab, or run the **Campaign & Audience Extractor** tab for a full marketing brief.

## Example Questions

- "Summarize the key findings of this report."
- "What are the main trends discussed in this document?"
- "What marketing opportunities are mentioned?"
- "What benchmarks or metrics does this report reference?"
- "Who is the target audience for this product?"

## Screenshots

> _Add screenshots here after your first run:_
> - `docs/screenshot-dashboard.png` — main dashboard with sidebar
> - `docs/screenshot-answer.png` — a grounded answer with sources expanded
> - `docs/screenshot-marketing.png` — the Campaign & Audience Extractor tab

## GitHub Instructions

```bash
git init
git add .
git commit -m "Initial commit: MarketMind AI"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

`.gitignore` already excludes `secrets.toml`, the local vector store contents, logs, and virtual environments.

## Streamlit Cloud Deployment

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and create a new app pointing at `streamlit_app.py`.
3. In **App settings → Secrets**, paste the contents of your filled-in `secrets.toml`:
   ```toml
   OPENROUTER_API_KEY = "sk-or-v1-..."
   OPENROUTER_MODEL = "openai/gpt-4o-mini"
   TOP_K = 5
   ```
4. Deploy. The app will build its ChromaDB index automatically the first time a PDF is uploaded.

> Streamlit Cloud's file system resets on redeploy — the vector store is rebuilt from the uploaded PDF each session, so no manual setup is needed.

## Error Handling

The app degrades gracefully instead of crashing for:

- Missing/invalid API key
- No PDF uploaded / empty PDF
- Corrupted, encrypted, or image-only (non-text) PDFs
- Embedding generation failures
- OpenRouter/network failures
- ChromaDB read/write failures
- Malformed LLM output (e.g. non-JSON marketing insights)

## Future Improvements

- Multi-PDF knowledge bases with per-document filtering
- Conversation memory (multi-turn follow-up questions)
- Export answers and marketing briefs to PDF/DOCX
- Support for additional embedding models and local LLMs
- User authentication for multi-tenant deployments

## License

MIT License — free to use and modify for academic and portfolio purposes.
