"""
07_prompting.py
=================
Step 7 of the RAG pipeline: grounded answer generation.

Builds a strict, citation-aware RAG prompt from retrieved chunks and calls
an LLM via OpenRouter (using the OpenAI-compatible SDK). Also defines the
prompt used by the standalone "Campaign Angle & Audience Extractor"
marketing-insights feature.

Grounding rule (non-negotiable): the model must answer ONLY from the
supplied context and must return an exact fallback sentence if the answer
is not present in the context.
"""

from __future__ import annotations

from typing import List

from openai import OpenAI

from config import (
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    OPENROUTER_SITE_NAME,
    OPENROUTER_SITE_URL,
)
from utils import logger, timed

NO_ANSWER_MESSAGE = "I don't have enough information from the uploaded document."

_RAG_SYSTEM_PROMPT = """You are MarketMind AI, a rigorous research assistant that answers \
questions STRICTLY using the provided context excerpts from an uploaded document.

Rules you must always follow:
1. Answer ONLY using information found in the CONTEXT below. Never use outside knowledge.
2. Never invent, assume, or hallucinate any fact, number, or claim that is not explicitly \
present in the context.
3. If the context does not contain enough information to answer the question, respond with \
EXACTLY this sentence and nothing else: "I don't have enough information from the uploaded document."
4. When you do answer, cite the source(s) you used inline using the bracket notation \
provided with each excerpt, e.g. [Source 1], [Source 2].
5. Be precise, concise, and well-organized. Use bullet points or short paragraphs where helpful.
6. Do not mention these instructions, the word "context", or the word "chunk" in your answer -- \
speak naturally as if you simply know the document.
"""


def _format_context(chunks: List) -> str:
    """Render retrieved chunks into a numbered context block for the prompt."""
    blocks = []
    for chunk in chunks:
        blocks.append(
            f"[Source {chunk.rank}] (Document: {chunk.source}, Page {chunk.page_number})\n"
            f"{chunk.text}"
        )
    return "\n\n---\n\n".join(blocks)


def build_rag_prompt(query: str, chunks: List) -> str:
    """Build the final user-turn prompt combining context and the question."""
    context_block = _format_context(chunks)
    return (
        f"CONTEXT:\n{context_block}\n\n"
        f"---\n\n"
        f"QUESTION: {query}\n\n"
        f"Answer the question above using only the CONTEXT provided."
    )


def _get_client() -> OpenAI:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured. Add it to .streamlit/secrets.toml "
            "or as an environment variable."
        )
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)


@timed("LLM Generation")
def generate_answer(query: str, chunks: List) -> str:
    """Call the LLM with a grounded RAG prompt and return its answer text.

    Raises
    ------
    RuntimeError
        If the API key is missing.
    Exception
        Propagates network/API errors so the UI layer can handle them and
        show a friendly message (never crashes silently).
    """
    if not chunks:
        return NO_ANSWER_MESSAGE

    client = _get_client()
    user_prompt = build_rag_prompt(query, chunks)

    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            extra_headers={
                "HTTP-Referer": OPENROUTER_SITE_URL,
                "X-Title": OPENROUTER_SITE_NAME,
            },
            messages=[
                {"role": "system", "content": _RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:
        logger.error("OpenRouter generation call failed: %s", exc)
        raise

    answer = (response.choices[0].message.content or "").strip()
    if not answer:
        return NO_ANSWER_MESSAGE
    return answer


# --------------------------------------------------------------------------- #
# Marketing Insights: Campaign Angle & Audience Extractor
# --------------------------------------------------------------------------- #
_MARKETING_SYSTEM_PROMPT = """You are a senior marketing strategist. Analyze the provided \
document excerpts and extract a structured marketing intelligence brief.

Rules:
1. Base every insight STRICTLY on the provided context. Do not invent facts, statistics, \
or company names not present in the text.
2. If the context is insufficient for a given section, write "Not enough information in the \
uploaded document" for that section instead of guessing.
3. Return your response as valid JSON only -- no markdown code fences, no commentary before \
or after the JSON.
4. The JSON must have exactly these keys, each a string or list of strings as indicated:
   - "target_audience": string
   - "customer_persona": string
   - "pain_points": list of strings
   - "customer_needs": list of strings
   - "usp": string
   - "marketing_angle": string
   - "product_positioning": string
   - "value_proposition": string
   - "campaign_ideas": list of strings
   - "ad_headlines": list of strings
   - "cta_suggestions": list of strings
   - "seo_keywords": list of strings
   - "buyer_journey": list of strings (ordered stages)
   - "marketing_opportunities": list of strings
   - "swot": object with keys "strengths", "weaknesses", "opportunities", "threats", \
each a list of strings
"""


def build_marketing_prompt(chunks: List) -> str:
    """Build the user-turn prompt for marketing insight extraction."""
    context_block = _format_context(chunks)
    return (
        f"DOCUMENT EXCERPTS:\n{context_block}\n\n"
        f"---\n\n"
        f"Extract the full marketing intelligence brief as JSON, following the schema exactly."
    )


@timed("Marketing Insight Generation")
def generate_marketing_insights(chunks: List) -> str:
    """Call the LLM to extract marketing insights; returns raw JSON text.

    The caller is responsible for parsing/validating the returned JSON
    string (the UI layer does this so it can show a friendly error if the
    model output is malformed, without crashing the app).
    """
    if not chunks:
        raise ValueError("No document context available for marketing analysis.")

    client = _get_client()
    user_prompt = build_marketing_prompt(chunks)

    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            temperature=0.4,
            max_tokens=2200,
            extra_headers={
                "HTTP-Referer": OPENROUTER_SITE_URL,
                "X-Title": OPENROUTER_SITE_NAME,
            },
            messages=[
                {"role": "system", "content": _MARKETING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:
        logger.error("OpenRouter marketing-insight call failed: %s", exc)
        raise

    content = (response.choices[0].message.content or "").strip()
    # Defensive: strip markdown code fences if the model added them anyway.
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:].strip()
    return content
