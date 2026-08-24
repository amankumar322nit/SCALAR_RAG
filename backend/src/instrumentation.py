"""
Structured Instrumentation, Observability Trace Engine, and Online Evaluation System.
Supports stdout, JSONL, SQLite persistence, and live Online Evaluation metrics.
"""

import json
import sqlite3
import time
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.config import TRACES_DB_PATH, TRACES_JSONL_PATH


COMMON_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were",
    "will", "with", "this", "these", "those", "you", "your", "they", "their",
    "can", "could", "should", "would", "may", "might", "must", "scaler",
    "please", "official", "support", "according", "policy", "policies", "stated",
    "documentation", "team", "section", "doc", "chunk", "information", "regarding"
}


def compute_grounding_score(answer: str, context_chunks: List[Dict[str, Any]]) -> float:
    """
    Computes claim-level lexical entailment and grounding score (0.0 to 1.0)
    between generated answer assertions and retrieved context chunks.
    """
    if not context_chunks:
        return 0.0
    context_text = " ".join([
        c.get("content", "") or c.get("snippet", "") or c.get("raw_content", "") or c.get("content_preview", "")
        for c in context_chunks
    ]).lower()
    context_words = set(re.findall(r"\b[a-z0-9]+\b", context_text))
    if not context_words:
        return 0.0

    # Strip citation tags like [DOC:01-SEC:02]
    clean_answer = re.sub(r"\[[^\]]+\]", "", answer)
    sentences = [s.strip() for s in re.split(r"[.\n;]", clean_answer) if len(s.strip()) > 10]
    if not sentences:
        return 1.0

    sentence_grounding_scores = []
    for s in sentences:
        words = [w for w in re.findall(r"\b[a-z0-9]+\b", s.lower()) if w not in COMMON_STOPWORDS and len(w) > 2]
        if not words:
            continue
        grounded_count = sum(1 for w in words if w in context_words)
        sentence_grounding_scores.append(grounded_count / len(words))

    if not sentence_grounding_scores:
        return 1.0

    mean_grounding = sum(sentence_grounding_scores) / len(sentence_grounding_scores)
    return round(float(mean_grounding), 2)


class StructuredTracer:
    def __init__(self, db_path: Path = TRACES_DB_PATH, jsonl_path: Path = TRACES_JSONL_PATH):
        self.db_path = Path(db_path)
        self.jsonl_path = Path(jsonl_path)
        self._init_sqlite()

    def _init_sqlite(self):
        """Create SQLite schema for query observability traces and online evals."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_traces (
                    trace_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    top_k_chunks TEXT NOT NULL,
                    similarity_scores TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    prompt_tokens INTEGER,
                    answer TEXT NOT NULL,
                    citations TEXT,
                    retrieval_latency_ms REAL,
                    generation_latency_ms REAL,
                    total_latency_ms REAL NOT NULL,
                    provider_used TEXT,
                    status TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS online_evals (
                    trace_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    faithfulness_score REAL NOT NULL,
                    context_relevance REAL NOT NULL,
                    citation_density REAL NOT NULL,
                    is_out_of_scope INTEGER NOT NULL,
                    user_rating INTEGER DEFAULT 0,
                    user_feedback TEXT,
                    flagged_for_review INTEGER DEFAULT 0,
                    review_reason TEXT,
                    FOREIGN KEY(trace_id) REFERENCES query_traces(trace_id)
                )
            """)
            conn.commit()

    def record_trace(self, trace: Dict[str, Any], emit_stdout: bool = True) -> Dict[str, Any]:
        """Save trace to SQLite, append to JSONL, run online evaluation, and emit log."""
        if "timestamp" not in trace:
            trace["timestamp"] = datetime.now(timezone.utc).isoformat()

        # 1. Append to JSONL
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

        # 2. Save to SQLite query_traces
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO query_traces (
                        trace_id, timestamp, query, top_k_chunks, similarity_scores,
                        prompt, prompt_tokens, answer, citations, retrieval_latency_ms,
                        generation_latency_ms, total_latency_ms, provider_used, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trace.get("trace_id", ""),
                    trace.get("timestamp", ""),
                    trace.get("query", ""),
                    json.dumps(trace.get("retrieved_chunks", []), ensure_ascii=False),
                    json.dumps([c.get("similarity_score", 0.0) for c in trace.get("retrieved_chunks", [])]),
                    trace.get("prompt", ""),
                    trace.get("prompt_tokens", 0),
                    trace.get("answer", ""),
                    json.dumps(trace.get("citations", []), ensure_ascii=False),
                    trace.get("retrieval_latency_ms", 0.0),
                    trace.get("generation_latency_ms", 0.0),
                    trace.get("total_latency_ms", 0.0),
                    trace.get("provider_used", ""),
                    trace.get("status", "SUCCESS")
                ))
                conn.commit()
        except Exception:
            pass

        # 3. Execute Real-Time Online Evaluation
        online_eval = self._evaluate_online(trace)
        trace["online_eval"] = online_eval

        # 4. Emit structured trace to stdout
        if emit_stdout:
            self._emit_stdout(trace)

        return trace

    def _evaluate_online(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously / in-line evaluate live production queries:
        - Context Relevance: Mean similarity score of retrieved context
        - Citation Density: Citations per key claim
        - Faithfulness Grounding: Checks if answer terms align with context
        - Anomaly / Drift Flagging: Identifies unanswered or low-confidence queries
        """
        answer = trace.get("answer", "")
        retrieved_chunks = trace.get("retrieved_chunks", [])
        citations = trace.get("citations", [])
        scores = [c.get("similarity_score", 0.0) for c in retrieved_chunks]
        is_out_of_scope = 1 if ("could not find information" in answer.lower() or not retrieved_chunks) else 0

        # Context Relevance
        avg_score = round(sum(scores) / max(1, len(scores)), 3) if scores else 0.0

        # Citation Density
        sentences = [s.strip() for s in re.split(r"[.!?\n]", answer) if len(s.strip()) > 10]
        citation_density = round(min(1.0, len(citations) / max(1, len(sentences))), 2)

        # Real-time Faithfulness Estimation via Claim-Level Context Grounding
        if is_out_of_scope:
            faithfulness = 1.0
        else:
            faithfulness = compute_grounding_score(answer, retrieved_chunks)

        # Flagging Criteria for Human-in-the-loop review
        flagged = 0
        review_reasons = []
        if avg_score < 0.28 and not is_out_of_scope:
            flagged = 1
            review_reasons.append("Low Context Relevance (<0.28)")
        if faithfulness < 0.70 and not is_out_of_scope:
            flagged = 1
            review_reasons.append(f"Low Faithfulness Grounding ({faithfulness:.2f})")
        if trace.get("total_latency_ms", 0) > 4000:
            flagged = 1
            review_reasons.append("High Latency (>4000ms)")

        review_reason_str = ", ".join(review_reasons) if review_reasons else "Normal"

        # Persist Online Eval to SQLite
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO online_evals (
                        trace_id, timestamp, query, faithfulness_score, context_relevance,
                        citation_density, is_out_of_scope, user_rating, user_feedback,
                        flagged_for_review, review_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trace.get("trace_id", ""),
                    trace.get("timestamp", ""),
                    trace.get("query", ""),
                    faithfulness,
                    avg_score,
                    citation_density,
                    is_out_of_scope,
                    0,
                    "",
                    flagged,
                    review_reason_str
                ))
                conn.commit()
        except Exception:
            pass

        return {
            "faithfulness_score": faithfulness,
            "context_relevance": avg_score,
            "citation_density": citation_density,
            "is_out_of_scope": bool(is_out_of_scope),
            "flagged_for_review": bool(flagged),
            "review_reason": review_reason_str
        }

    def record_user_feedback(self, trace_id: str, rating: int, feedback: str = "") -> bool:
        """Record learner thumbs up (+1) or thumbs down (-1) feedback."""
        try:
            flagged = 1 if rating == -1 else 0
            reason_add = "User Thumbs Down Feedback" if rating == -1 else ""
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE online_evals 
                    SET user_rating = ?, 
                        user_feedback = ?,
                        flagged_for_review = CASE WHEN ? == 1 THEN 1 ELSE flagged_for_review END,
                        review_reason = CASE WHEN ? != '' THEN review_reason || ' | ' || ? ELSE review_reason END
                    WHERE trace_id = ?
                """, (rating, feedback, flagged, reason_add, reason_add, trace_id))
                conn.commit()
                return True
        except Exception:
            return False

    def get_online_eval_summary(self) -> Dict[str, Any]:
        """Aggregate real-time metrics across all live production traffic."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_queries,
                        AVG(faithfulness_score) as avg_faithfulness,
                        AVG(context_relevance) as avg_context_relevance,
                        AVG(citation_density) as avg_citation_density,
                        SUM(CASE WHEN is_out_of_scope = 1 THEN 1 ELSE 0 END) as total_abstentions,
                        SUM(CASE WHEN user_rating = 1 THEN 1 ELSE 0 END) as thumbs_up,
                        SUM(CASE WHEN user_rating = -1 THEN 1 ELSE 0 END) as thumbs_down,
                        SUM(CASE WHEN flagged_for_review = 1 THEN 1 ELSE 0 END) as flagged_queries
                    FROM online_evals
                """)
                row = cursor.fetchone()
                total = row["total_queries"] or 0
                thumbs_up = row["thumbs_up"] or 0
                thumbs_down = row["thumbs_down"] or 0
                rated_total = thumbs_up + thumbs_down
                satisfaction_pct = round((thumbs_up / rated_total) * 100, 1) if rated_total > 0 else 100.0

                # Latency metrics from query_traces
                cursor.execute("SELECT total_latency_ms FROM query_traces ORDER BY total_latency_ms ASC")
                latencies = [r[0] for r in cursor.fetchall()]
                p50 = latencies[len(latencies) // 2] if latencies else 0.0
                p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0

                # Recent flagged queries
                cursor.execute("""
                    SELECT q.trace_id, q.timestamp, q.query, q.answer, o.review_reason, o.user_rating
                    FROM online_evals o
                    JOIN query_traces q ON o.trace_id = q.trace_id
                    WHERE o.flagged_for_review = 1
                    ORDER BY q.timestamp DESC LIMIT 10
                """)
                flagged_rows = [dict(r) for r in cursor.fetchall()]

                return {
                    "total_live_queries": total,
                    "avg_faithfulness": round((row["avg_faithfulness"] or 0.0) * 100, 1),
                    "avg_context_relevance": round((row["avg_context_relevance"] or 0.0), 3),
                    "avg_citation_density": round((row["avg_citation_density"] or 0.0) * 100, 1),
                    "abstention_rate_pct": round(((row["total_abstentions"] or 0) / max(1, total)) * 100, 1),
                    "user_satisfaction_pct": satisfaction_pct,
                    "total_thumbs_up": thumbs_up,
                    "total_thumbs_down": thumbs_down,
                    "flagged_for_review_count": row["flagged_queries"] or 0,
                    "latency_p50_ms": round(p50, 1),
                    "latency_p95_ms": round(p95, 1),
                    "recent_flagged_queries": flagged_rows
                }
        except Exception as e:
            return {"error": str(e), "total_live_queries": 0}

    def _emit_stdout(self, trace: Dict[str, Any]):
        """Print clean, structured JSON trace to stdout with online eval metrics."""
        online_eval = trace.get("online_eval", {})
        print("\n" + "=" * 80)
        print(f"📊 [STRUCTURED RAG TRACE] - ID: {trace.get('trace_id')}")
        print("=" * 80)
        print(json.dumps({
            "timestamp": trace.get("timestamp"),
            "query": trace.get("query"),
            "total_latency_ms": trace.get("total_latency_ms"),
            "retrieval_latency_ms": trace.get("retrieval_latency_ms"),
            "generation_latency_ms": trace.get("generation_latency_ms"),
            "status": trace.get("status"),
            "num_retrieved": len(trace.get("retrieved_chunks", [])),
            "retrieved_chunks": [
                {
                    "chunk_id": c.get("chunk_id"),
                    "doc_path": c.get("doc_path"),
                    "section_path": c.get("section_path"),
                    "score": round(c.get("similarity_score", 0.0), 3)
                } for c in trace.get("retrieved_chunks", [])
            ],
            "citations": trace.get("citations", []),
            "online_eval": {
                "faithfulness": f"{int(online_eval.get('faithfulness_score', 1.0) * 100)}%",
                "context_relevance": online_eval.get("context_relevance", 0.0),
                "citation_density": f"{int(online_eval.get('citation_density', 0.0) * 100)}%",
                "flagged": online_eval.get("flagged_for_review", False),
                "review_reason": online_eval.get("review_reason", "Normal")
            },
            "answer_preview": trace.get("answer", "")[:180] + ("..." if len(trace.get("answer", "")) > 180 else "")
        }, indent=2))
        print("=" * 80 + "\n")

    def get_recent_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent query traces from SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT q.*, o.faithfulness_score, o.context_relevance, o.citation_density, o.user_rating, o.flagged_for_review
                    FROM query_traces q
                    LEFT JOIN online_evals o ON q.trace_id = o.trace_id
                    ORDER BY q.timestamp DESC LIMIT ?
                """, (limit,))
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    item = dict(row)
                    item["retrieved_chunks"] = json.loads(item["top_k_chunks"])
                    item["citations"] = json.loads(item["citations"]) if item["citations"] else []
                    results.append(item)
                return results
        except Exception:
            return []
