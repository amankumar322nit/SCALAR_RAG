"""
Document Parser and Hierarchical Chunker for the RAG pipeline.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any
from src.config import CORPUS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


class DocumentChunk:
    def __init__(
        self,
        chunk_id: str,
        doc_path: str,
        doc_title: str,
        section_path: str,
        content: str,
        raw_content: str,
        char_count: int,
        metadata: Dict[str, Any] = None
    ):
        self.chunk_id = chunk_id
        self.doc_path = doc_path
        self.doc_title = doc_title
        self.section_path = section_path
        self.content = content          # With breadcrumb prefix
        self.raw_content = raw_content  # Body text only
        self.char_count = char_count
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_path": self.doc_path,
            "doc_title": self.doc_title,
            "section_path": self.section_path,
            "content": self.content,
            "raw_content": self.raw_content,
            "char_count": self.char_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        return cls(
            chunk_id=data["chunk_id"],
            doc_path=data["doc_path"],
            doc_title=data["doc_title"],
            section_path=data["section_path"],
            content=data["content"],
            raw_content=data.get("raw_content", data["content"]),
            char_count=data["char_count"],
            metadata=data.get("metadata", {}),
        )


class HierarchicalIndexer:
    def __init__(self, corpus_dir: Path = CORPUS_DIR, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.corpus_dir = Path(corpus_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_and_chunk_all(self) -> List[DocumentChunk]:
        """Discover and hierarchically chunk all documents in the corpus directory."""
        if not self.corpus_dir.exists():
            raise FileNotFoundError(f"Corpus directory not found at: {self.corpus_dir}")

        chunks: List[DocumentChunk] = []
        doc_files = list(self.corpus_dir.rglob("*.md")) + list(self.corpus_dir.rglob("*.txt"))
        doc_files.sort()

        chunk_counter = 1
        for file_path in doc_files:
            relative_path = file_path.relative_to(self.corpus_dir).as_posix()
            doc_chunks = self.process_document(file_path, relative_path, start_id=chunk_counter)
            chunk_counter += len(doc_chunks)
            chunks.extend(doc_chunks)

        return chunks

    def process_document(self, file_path: Path, relative_path: str, start_id: int = 1) -> List[DocumentChunk]:
        """Parse a single markdown/text file into hierarchical chunks."""
        text = file_path.read_text(encoding="utf-8")
        lines = text.splitlines()

        doc_title = file_path.stem.replace("_", " ").title()
        sections: List[Dict[str, Any]] = []

        current_h1 = doc_title
        current_h2 = ""
        current_h3 = ""
        current_lines = []

        def save_current_section():
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    path_parts = [p for p in [current_h1, current_h2, current_h3] if p]
                    section_path = " > ".join(path_parts)
                    sections.append({
                        "doc_title": current_h1,
                        "section_path": section_path,
                        "text": body
                    })
            current_lines.clear()

        for line in lines:
            h1_match = re.match(r"^#\s+(.+)$", line)
            h2_match = re.match(r"^##\s+(.+)$", line)
            h3_match = re.match(r"^###\s+(.+)$", line)

            if h1_match:
                save_current_section()
                current_h1 = h1_match.group(1).strip()
                current_h2 = ""
                current_h3 = ""
            elif h2_match:
                save_current_section()
                current_h2 = h2_match.group(1).strip()
                current_h3 = ""
            elif h3_match:
                save_current_section()
                current_h3 = h3_match.group(1).strip()
            else:
                current_lines.append(line)

        save_current_section()

        # If no markdown headings were detected, treat the entire file as one section
        if not sections and text.strip():
            sections.append({
                "doc_title": doc_title,
                "section_path": doc_title,
                "text": text.strip()
            })

        # Now chunk each section if it exceeds chunk_size
        chunks: List[DocumentChunk] = []
        for sec in sections:
            sec_text = sec["text"]
            sec_chunks = self._split_text_with_overlap(sec_text, self.chunk_size, self.overlap)

            for i, chunk_text in enumerate(sec_chunks):
                breadcrumb = f"[Document: {relative_path} | Section: {sec['section_path']}]\n"
                full_content = breadcrumb + chunk_text
                chunk_id = f"chunk_{start_id + len(chunks):04d}"

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        doc_path=relative_path,
                        doc_title=sec["doc_title"],
                        section_path=sec["section_path"],
                        content=full_content,
                        raw_content=chunk_text,
                        char_count=len(full_content),
                        metadata={
                            "section_index": len(chunks),
                            "sub_chunk_index": i,
                            "filename": file_path.name
                        }
                    )
                )

        return chunks

    def _split_text_with_overlap(self, text: str, max_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks along paragraph/sentence boundaries."""
        if len(text) <= max_size:
            return [text]

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_len = 0

        for p in paragraphs:
            p_len = len(p)
            if current_len + p_len + 2 <= max_size:
                current_chunk.append(p)
                current_len += p_len + 2
            else:
                if current_chunk:
                    chunk_str = "\n\n".join(current_chunk)
                    chunks.append(chunk_str)
                    # Retain last paragraph for overlap if within overlap budget
                    if len(current_chunk[-1]) <= overlap:
                        current_chunk = [current_chunk[-1], p]
                        current_len = len(current_chunk[0]) + p_len + 2
                    else:
                        current_chunk = [p]
                        current_len = p_len
                else:
                    # Single paragraph exceeds max_size, split by sentences/lines
                    lines = p.split("\n")
                    sub_buf = []
                    sub_len = 0
                    for line in lines:
                        if sub_len + len(line) + 1 <= max_size:
                            sub_buf.append(line)
                            sub_len += len(line) + 1
                        else:
                            if sub_buf:
                                chunks.append("\n".join(sub_buf))
                            sub_buf = [line]
                            sub_len = len(line)
                    if sub_buf:
                        current_chunk = ["\n".join(sub_buf)]
                        current_len = sub_len

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks if chunks else [text]
