"""
Configuration settings for the Scaler Learner Support RAG pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "corpus"
DATA_DIR = BASE_DIR / "data"
TRACES_DB_PATH = DATA_DIR / "traces.db"
TRACES_JSONL_PATH = DATA_DIR / "traces.jsonl"
INDEX_STORE_PATH = DATA_DIR / "vector_index.json"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1200"))       # in characters (~300-400 tokens)
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))   # in characters (~40-50 tokens)

# Retrieval Configuration
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.25"))
HYBRID_ALPHA = float(os.getenv("RAG_HYBRID_ALPHA", "0.5"))  # 0.5 dense + 0.5 sparse

# LLM & Embedding Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY", "")
GROK_API_KEY = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY", "")
LLM_PROVIDER = os.getenv("RAG_LLM_PROVIDER", "auto")

# Default model selection based on configured keys
if (os.getenv("GROQ_API_KEY", "").startswith("gsk_") or os.getenv("GROK_API_KEY", "").startswith("gsk_")):
    default_model = "openai/gpt-oss-120b"
elif (os.getenv("GROK_API_KEY", "").startswith("xai-") or os.getenv("XAI_API_KEY", "").startswith("xai-")):
    default_model = "grok-2-latest"
else:
    default_model = "gpt-4o-mini"

LLM_MODEL = os.getenv("RAG_LLM_MODEL", default_model)

# System Prompt
SYSTEM_PROMPT = """You are Scaler's official Learner Support AI Assistant. Your role is to answer learner questions accurately, concisely, and strictly based on the provided reference context.

CRITICAL INSTRUCTIONS:
1. Grounding: Answer ONLY using the facts directly mentioned in the Context. Do NOT use prior knowledge or extrapolate.
2. Citations: Every factual claim must be attributed to its source chunk using brackets, e.g. [Chunk 1] or [Chunk 2].
3. No Relevant Context: If the answer cannot be found in the provided context, state clearly and politely:
   "I apologize, but I could not find information regarding this in the official Scaler documentation. Please contact the learner support team at support@scaler.com for further assistance."
4. Tone & Style: Be professional, empathetic, clear, and structured (use bullet points where appropriate). Never speculate on fees, dates, or guarantees not explicitly stated."""
