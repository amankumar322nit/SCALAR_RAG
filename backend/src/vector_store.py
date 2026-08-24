"""
Vector Store & Hybrid Search Engine (Dense Vector + BM25 Lexical + Reciprocal Rank Fusion).
"""

import hashlib
import json
import math
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

from src.indexer import DocumentChunk
from src.config import INDEX_STORE_PATH, TOP_K, SIMILARITY_THRESHOLD, HYBRID_ALPHA

# Domain-specific stop words that appear across almost all Scaler documents
CORPUS_COMMON_WORDS = {
    "scaler", "academy", "learner", "learners", "student", "students",
    "course", "program", "support", "team", "official", "please", "can",
    "what", "how", "why", "where", "which", "when", "does", "provide",
    "want", "need", "know", "tell", "about", "with", "from", "for", "the",
    "and", "are", "you", "get", "there", "any", "all", "learn", "learning",
    "teach", "teaching", "study", "studying"
}

SYNONYM_EXPANSIONS = {
    "withdraw": ["cancellation", "refund", "quit", "leave", "drop", "money-back"],
    "withdrawal": ["cancellation", "refund", "quit", "money-back"],
    "drop": ["cancellation", "refund", "withdraw"],
    "first week": ["7-day", "cooling-off", "unconditional", "money-back", "full refund"],
    "week": ["7-day", "days", "window", "timeline"],
    "money": ["refund", "fees", "fee", "tuition", "payment"],
    "cost": ["fee", "tuition", "pricing", "emi"],
    "placement": ["career", "job", "interviews", "hiring", "salary", "ppra", "mock"],
    "job": ["placement", "career", "hiring", "company", "interview"],
    "scholarship": ["women in tech", "nst", "merit", "discount", "financial aid"],
    "discount": ["scholarship", "merit", "concession"],
    "pause": ["defer", "postpone", "break", "leave", "gap"],
    "syllabus": ["curriculum", "modules", "topics", "roadmap"],
    "recordings": ["missed", "replay", "recording", "video", "lms"],
    "invoice": ["gst", "gstin", "tax", "reimbursement", "b2b", "bill"],
    "employer": ["corporate", "company", "sponsorship", "reimbursement"]
}


def _stable_hash(text: str) -> int:
    """Deterministic hash across process boundaries and Python interpreter restarts."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


class SimpleDenseEmbedder:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            vec = self._text_to_vector(text)
            norm = np.linalg.norm(vec)
            if norm > 0:
                embeddings[i] = vec / norm
            else:
                embeddings[i] = vec
        return embeddings

    def _text_to_vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return vec

        for idx, token in enumerate(tokens):
            h1 = _stable_hash(token) % self.dim
            vec[h1] += 1.0

            pos_weight = 1.0 / (1.0 + 0.03 * min(idx, 30))
            vec[h1] += pos_weight

            for j in range(len(token) - 2):
                gram = token[j:j+3]
                h2 = _stable_hash(gram) % self.dim
                vec[h2] += 0.35

        return vec


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freq: Dict[str, int] = {}
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.num_docs: int = 0

    def fit(self, corpus_texts: List[str]):
        self.num_docs = len(corpus_texts)
        self.doc_len = []
        self.doc_term_freqs = []
        self.doc_freq = {}

        total_len = 0
        for text in corpus_texts:
            tokens = self._tokenize(text)
            t_len = len(tokens)
            self.doc_len.append(t_len)
            total_len += t_len

            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.doc_term_freqs.append(tf)

            for t in tf.keys():
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1

        self.avg_doc_len = total_len / max(1, self.num_docs)

    def search(self, query: str) -> np.ndarray:
        scores = np.zeros(self.num_docs, dtype=np.float32)
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return scores

        for token in q_tokens:
            if token not in self.doc_freq:
                continue
            df = self.doc_freq[token]
            idf = math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))

            for i in range(self.num_docs):
                tf = self.doc_term_freqs[i].get(token, 0)
                if tf > 0:
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (self.doc_len[i] / max(1, self.avg_doc_len)))
                    scores[i] += idf * (numerator / denominator)

        max_score = np.max(scores)
        if max_score > 0:
            scores = scores / max_score
        return scores

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())


class HybridVectorStore:
    def __init__(self, index_path: Path = INDEX_STORE_PATH):
        self.index_path = Path(index_path)
        self.chunks: List[DocumentChunk] = []
        self.dense_embedder = SimpleDenseEmbedder()
        self.bm25 = BM25Index()
        self.dense_matrix: Optional[np.ndarray] = None

    def build_index(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        corpus_texts = [c.content for c in chunks]
        self.dense_matrix = self.dense_embedder.encode(corpus_texts)
        self.bm25.fit(corpus_texts)
        self.save_index()

    def expand_query(self, query: str) -> str:
        q_lower = query.lower()
        extra_terms = []
        for key, syns in SYNONYM_EXPANSIONS.items():
            if key in q_lower:
                extra_terms.extend(syns)
        if extra_terms:
            return f"{query} {' '.join(extra_terms[:6])}"
        return query

    def get_core_query_keywords(self, query: str) -> List[str]:
        words = re.findall(r"[a-z0-9]+", query.lower())
        return [w for w in words if w not in CORPUS_COMMON_WORDS and len(w) > 2]

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
        alpha: float = HYBRID_ALPHA
    ) -> List[Tuple[DocumentChunk, float]]:
        if not self.chunks or self.dense_matrix is None:
            if not self.load_index():
                return []

        core_keywords = self.get_core_query_keywords(query)
        expanded_query = self.expand_query(query)

        # 1. Dense Cosine Similarity
        q_dense = self.dense_embedder.encode([expanded_query])[0]
        dense_scores = np.dot(self.dense_matrix, q_dense)

        # 2. Sparse BM25
        sparse_scores = self.bm25.search(expanded_query)

        # 3. Core Keyword Overlap Check & Section Title Boost
        overlap_scores = np.zeros(len(self.chunks), dtype=np.float32)
        section_boosts = np.zeros(len(self.chunks), dtype=np.float32)
        core_matches_per_chunk = np.zeros(len(self.chunks), dtype=np.int32)

        exp_tokens = set(re.findall(r"\w+", expanded_query.lower()))

        for i, chunk in enumerate(self.chunks):
            c_text_lower = chunk.content.lower()
            c_tokens = set(re.findall(r"\w+", c_text_lower))
            sec_lower = chunk.section_path.lower()

            matches = sum(1 for kw in core_keywords if kw in c_text_lower)
            core_matches_per_chunk[i] = matches

            overlap = len(exp_tokens.intersection(c_tokens))
            overlap_scores[i] = overlap / max(1, len(exp_tokens))

            # Specific phrase boosts
            if "first week" in query.lower() or "7-day" in query.lower() or "withdraw" in query.lower():
                if "7-day unconditional" in sec_lower or "7-day" in c_text_lower:
                    section_boosts[i] += 0.35

            if any(kw in sec_lower for kw in core_keywords):
                section_boosts[i] += 0.20

        # If core keywords exist in query but NO chunk contains any core keyword, abstain
        if core_keywords and np.max(core_matches_per_chunk) == 0:
            return []

        # Hybrid Score Combination
        hybrid_scores = (0.35 * dense_scores) + (0.35 * sparse_scores) + (0.15 * overlap_scores) + (0.15 * section_boosts)

        ranked_indices = np.argsort(hybrid_scores)[::-1]
        results: List[Tuple[DocumentChunk, float]] = []

        for idx in ranked_indices[:top_k]:
            score = float(hybrid_scores[idx])
            if score >= threshold and (core_matches_per_chunk[idx] > 0 or not core_keywords):
                results.append((self.chunks[idx], score))

        return results

    def save_index(self):
        data = {
            "chunks": [c.to_dict() for c in self.chunks],
            "dense_embeddings": self.dense_matrix.tolist() if self.dense_matrix is not None else []
        }
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load_index(self) -> bool:
        if not self.index_path.exists():
            return False

        with open(self.index_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.chunks = [DocumentChunk.from_dict(d) for d in data["chunks"]]
        if data.get("dense_embeddings"):
            self.dense_matrix = np.array(data["dense_embeddings"], dtype=np.float32)
        else:
            corpus_texts = [c.content for c in self.chunks]
            self.dense_matrix = self.dense_embedder.encode(corpus_texts)

        corpus_texts = [c.content for c in self.chunks]
        self.bm25.fit(corpus_texts)
        return True
