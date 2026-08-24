"""
End-to-End RAG Pipeline Orchestrator for Scaler Learner Support.
"""

import time
import uuid
from typing import Dict, Any, Optional, List
from src.indexer import HierarchicalIndexer, DocumentChunk
from src.vector_store import HybridVectorStore
from src.llm import LLMClient
from src.instrumentation import StructuredTracer
from src.config import TOP_K, SIMILARITY_THRESHOLD, HYBRID_ALPHA


class RAGPipeline:
    def __init__(
        self,
        indexer: Optional[HierarchicalIndexer] = None,
        vector_store: Optional[HybridVectorStore] = None,
        llm_client: Optional[LLMClient] = None,
        tracer: Optional[StructuredTracer] = None
    ):
        self.indexer = indexer or HierarchicalIndexer()
        self.vector_store = vector_store or HybridVectorStore()
        self.llm = llm_client or LLMClient()
        self.tracer = tracer or StructuredTracer()

        # Ensure index exists or build it
        self._ensure_index()

    def _ensure_index(self):
        """Load index from storage or build automatically if not present."""
        if not self.vector_store.load_index():
            print("🚀 Building initial vector and lexical index from corpus...")
            self.ingest()

    def ingest(self) -> int:
        """Scan corpus directory, chunk documents, and index in vector store."""
        chunks = self.indexer.load_and_chunk_all()
        self.vector_store.build_index(chunks)
        print(f"✅ Successfully indexed {len(chunks)} chunks across corpus.")
        return len(chunks)

    def query(
        self,
        question: str,
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
        alpha: float = HYBRID_ALPHA,
        emit_stdout: bool = True
    ) -> Dict[str, Any]:
        """
        Execute full RAG pipeline:
        1. Hybrid Dense-Sparse retrieval
        2. Relevance filtering
        3. Prompt assembly & LLM generation
        4. Structured trace recording
        """
        total_start = time.time()
        trace_id = str(uuid.uuid4())

        # 1. Retrieval Stage
        retrieval_start = time.time()
        retrieved_results = self.vector_store.search(
            query=question,
            top_k=top_k,
            threshold=threshold,
            alpha=alpha
        )
        retrieval_latency_ms = round((time.time() - retrieval_start) * 1000, 2)

        # 2. Check if context found
        if not retrieved_results:
            total_latency_ms = round((time.time() - total_start) * 1000, 2)
            out_of_scope_answer = (
                "I apologize, but I could not find information regarding this in the official Scaler "
                "documentation. Please contact the learner support team at support@scaler.com for further assistance."
            )
            trace = {
                "trace_id": trace_id,
                "query": question,
                "retrieved_chunks": [],
                "prompt": f"Question: {question}\nContext: None",
                "prompt_tokens": len(question.split()),
                "answer": out_of_scope_answer,
                "citations": [],
                "retrieval_latency_ms": retrieval_latency_ms,
                "generation_latency_ms": 0.0,
                "total_latency_ms": total_latency_ms,
                "provider_used": "direct_abstention",
                "status": "NO_RELEVANT_CONTEXT"
            }
            self.tracer.record_trace(trace, emit_stdout=emit_stdout)
            return {
                "answer": out_of_scope_answer,
                "sources": [],
                "citations": [],
                "trace": trace
            }

        # 3. Generation Stage
        gen_start = time.time()
        llm_response = self.llm.generate(question, retrieved_results)
        generation_latency_ms = round((time.time() - gen_start) * 1000, 2)
        total_latency_ms = round((time.time() - total_start) * 1000, 2)

        # 4. Format Sources
        sources = []
        for i, (chunk, score) in enumerate(retrieved_results, start=1):
            sources.append({
                "chunk_tag": f"[Chunk {i}]",
                "chunk_id": chunk.chunk_id,
                "doc_path": chunk.doc_path,
                "doc_title": chunk.doc_title,
                "section_path": chunk.section_path,
                "similarity_score": round(score, 4),
                "snippet": chunk.raw_content.strip()
            })

        # 5. Record Trace
        trace = {
            "trace_id": trace_id,
            "query": question,
            "retrieved_chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_path": chunk.doc_path,
                    "section_path": chunk.section_path,
                    "similarity_score": round(score, 4),
                    "content_preview": chunk.raw_content[:200]
                } for chunk, score in retrieved_results
            ],
            "prompt": llm_response.get("prompt", ""),
            "prompt_tokens": llm_response.get("prompt_tokens", 0),
            "answer": llm_response.get("answer", ""),
            "citations": llm_response.get("citations", []),
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "total_latency_ms": total_latency_ms,
            "provider_used": llm_response.get("provider_used", "local"),
            "status": "SUCCESS"
        }

        self.tracer.record_trace(trace, emit_stdout=emit_stdout)

        return {
            "answer": llm_response.get("answer", ""),
            "sources": sources,
            "citations": llm_response.get("citations", []),
            "latency_ms": total_latency_ms,
            "trace": trace
        }
