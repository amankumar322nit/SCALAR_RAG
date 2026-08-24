"""
LLM Generation Engine and Provider Interfaces (Groq, Grok/xAI, OpenAI, Gemini, Local Grounded Generator).
"""

import os
import re
import json
import time
from typing import List, Dict, Any, Tuple
from src.indexer import DocumentChunk
from src.config import SYSTEM_PROMPT, OPENAI_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, GROK_API_KEY, LLM_PROVIDER, LLM_MODEL


class LLMClient:
    def __init__(
        self,
        provider: str = LLM_PROVIDER,
        model: str = LLM_MODEL,
        groq_key: str = GROQ_API_KEY,
        grok_key: str = GROK_API_KEY,
        openai_key: str = OPENAI_API_KEY,
        gemini_key: str = GEMINI_API_KEY
    ):
        self.provider = provider
        self.model = model
        self.groq_key = groq_key or os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY") or ""
        self.grok_key = grok_key or os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY") or ""
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY", "")
        self.gemini_key = gemini_key or os.getenv("GEMINI_API_KEY", "")

    def format_prompt(self, query: str, chunks: List[Tuple[DocumentChunk, float]]) -> str:
        """Construct prompt with structured context blocks and explicit citation identifiers."""
        if not chunks:
            return f"Question: {query}\n\nContext: None provided."

        context_blocks = []
        for i, (chunk, score) in enumerate(chunks, start=1):
            block = (
                f"[Chunk {i}] (Source: {chunk.doc_path} | Section: {chunk.section_path} | Score: {score:.3f})\n"
                f"{chunk.raw_content.strip()}"
            )
            context_blocks.append(block)

        formatted_context = "\n\n".join(context_blocks)
        prompt = (
            f"Context Information:\n"
            f"---------------------\n"
            f"{formatted_context}\n"
            f"---------------------\n\n"
            f"Question: {query}\n\n"
            f"Answer strictly based on the context above. Include source citations [Chunk X] for all factual statements:"
        )
        return prompt

    def generate(self, query: str, chunks: List[Tuple[DocumentChunk, float]]) -> Dict[str, Any]:
        prompt = self.format_prompt(query, chunks)
        start_time = time.time()

        if not chunks:
            answer = (
                "I apologize, but I could not find information regarding this in the official Scaler "
                "documentation. Please contact the learner support team at support@scaler.com for further assistance."
            )
            return {
                "answer": answer,
                "citations": [],
                "prompt": prompt,
                "prompt_tokens": len(prompt.split()),
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "provider_used": "direct_refusal"
            }

        # 1. Groq API (when key starts with gsk_)
        active_key = self.groq_key or self.grok_key
        if active_key and (self.provider in ("auto", "groq", "grok", "xai")):
            try:
                if active_key.startswith("gsk_"):
                    res = self._generate_groq(prompt, active_key)
                else:
                    res = self._generate_grok(prompt, active_key)
                res["latency_ms"] = round((time.time() - start_time) * 1000, 2)
                res["prompt"] = prompt
                res["prompt_tokens"] = len(prompt.split())
                return res
            except Exception as e:
                print(f"⚠️ Groq/Grok API error: {e}. Falling back.")

        # 2. OpenAI API
        if self.openai_key and self.provider in ("auto", "openai"):
            try:
                res = self._generate_openai(prompt)
                res["latency_ms"] = round((time.time() - start_time) * 1000, 2)
                res["prompt"] = prompt
                res["prompt_tokens"] = len(prompt.split())
                return res
            except Exception as e:
                print(f"⚠️ OpenAI API call error: {e}.")

        # 3. Gemini API
        if self.gemini_key and self.provider in ("auto", "gemini"):
            try:
                res = self._generate_gemini(prompt)
                res["latency_ms"] = round((time.time() - start_time) * 1000, 2)
                res["prompt"] = prompt
                res["prompt_tokens"] = len(prompt.split())
                return res
            except Exception as e:
                print(f"⚠️ Gemini API call error: {e}.")

        # 4. Offline Grounded Synthesis Engine
        res = self._generate_local_grounded(query, chunks, prompt)
        res["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        res["prompt"] = prompt
        res["prompt_tokens"] = len(prompt.split())
        return res

    def _generate_groq(self, prompt: str, key: str) -> Dict[str, Any]:
        """Generate response via Groq API endpoint."""
        import requests
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        # Choose valid Groq model
        model_name = self.model if ("gpt-oss" in self.model or "qwen" in self.model or "compound" in self.model) else "openai/gpt-oss-120b"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
        }
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"].strip()
        citations = self._extract_citations(raw_text)
        return {
            "answer": raw_text,
            "citations": citations,
            "provider_used": f"groq ({model_name})"
        }

    def _generate_grok(self, prompt: str, key: str) -> Dict[str, Any]:
        """Generate response via xAI / Grok API endpoint."""
        import requests
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        model_name = self.model if "grok" in self.model.lower() else "grok-2-latest"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
        }
        resp = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"].strip()
        citations = self._extract_citations(raw_text)
        return {
            "answer": raw_text,
            "citations": citations,
            "provider_used": f"xai ({model_name})"
        }

    def _generate_openai(self, prompt: str) -> Dict[str, Any]:
        import requests
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model if "gpt" in self.model else "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
        }
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"].strip()
        citations = self._extract_citations(raw_text)
        return {
            "answer": raw_text,
            "citations": citations,
            "provider_used": "openai"
        }

    def _generate_gemini(self, prompt: str) -> Dict[str, Any]:
        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": f"{SYSTEM_PROMPT}\n\n{prompt}"}]
            }],
            "generationConfig": {"temperature": 0.1}
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        citations = self._extract_citations(raw_text)
        return {
            "answer": raw_text,
            "citations": citations,
            "provider_used": "gemini"
        }

    def _generate_local_grounded(
        self,
        query: str,
        chunks: List[Tuple[DocumentChunk, float]],
        prompt: str
    ) -> Dict[str, Any]:
        top_chunk, top_score = chunks[0]
        q_lower = query.lower()
        stopwords = {"what", "is", "the", "how", "can", "i", "do", "a", "an", "for", "to", "in", "of", "and", "if", "want", "get", "are", "there", "any"}
        q_tokens = set(re.findall(r"\w+", q_lower)) - stopwords

        scored_lines = []

        for i, (chunk, chunk_score) in enumerate(chunks, start=1):
            chunk_tag = f"[Chunk {i}]"
            lines = [l.strip() for l in chunk.raw_content.splitlines() if l.strip()]

            for line in lines:
                l_lower = line.lower()
                l_tokens = set(re.findall(r"\w+", l_lower))
                overlap_count = len(q_tokens.intersection(l_tokens))
                
                sec_tokens = set(re.findall(r"\w+", chunk.section_path.lower()))
                sec_overlap = len(q_tokens.intersection(sec_tokens))

                line_score = (overlap_count * 2.5) + (sec_overlap * 1.5) + (chunk_score * 2.0)
                if line.startswith("-") or line.startswith("•") or line.startswith("**"):
                    line_score += 0.5

                clean_line = re.sub(r"^[-*•]\s*", "", line)
                if clean_line and len(clean_line) > 15:
                    scored_lines.append((line_score, clean_line, chunk_tag, chunk))

        scored_lines.sort(key=lambda x: x[0], reverse=True)

        selected_lines = []
        seen_texts = set()
        citations = []

        for score, line_text, chunk_tag, chunk in scored_lines:
            if line_text not in seen_texts and len(selected_lines) < 5:
                seen_texts.add(line_text)
                selected_lines.append(f"• {line_text} {chunk_tag}")
                if chunk_tag not in citations:
                    citations.append(chunk_tag)

        if not selected_lines:
            for l in [l.strip() for l in top_chunk.raw_content.splitlines() if l.strip()][:4]:
                clean_l = re.sub(r"^[-*•]\s*", "", l)
                selected_lines.append(f"• {clean_l} [Chunk 1]")
            citations = ["[Chunk 1]"]

        summary_points = "\n".join(selected_lines)
        answer = (
            f"Based on Scaler's official documentation regarding {top_chunk.doc_title} ({top_chunk.section_path}):\n\n"
            f"{summary_points}\n\n"
            f"If you have additional questions, you can reach learner support at support@scaler.com."
        )

        return {
            "answer": answer,
            "citations": citations,
            "provider_used": "local_grounded_engine"
        }

    def _extract_citations(self, text: str) -> List[str]:
        # Match [Chunk 1], [Chunk  1], [Chunk 1], [Source: Chunk 1] etc.
        raw_matches = re.findall(r"\[(?:Source:?\s*)?Chunk[\s\u202f\u00a0]*(\d+)\]", text, re.IGNORECASE)
        if not raw_matches:
            # Check for alternative citation patterns like (Chunk 1) or Chunk 1:
            raw_matches = re.findall(r"(?:\[|\()?(?:Source:?\s*)?Chunk[\s\u202f\u00a0]*(\d+)(?:\]|\))?", text, re.IGNORECASE)
        citations = [f"[Chunk {m}]" for m in sorted(list(set(raw_matches)))]
        return citations
