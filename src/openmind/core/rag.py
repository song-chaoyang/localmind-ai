"""
Retrieval-Augmented Generation (RAG) pipeline for OpenMind.

Provides a lightweight, dependency-minimal RAG implementation with:
- Recursive character-based text splitting
- Multi-format document loading (txt, md, csv, json, py, js, etc.)
- Hash-based fallback embeddings (deterministic, no external deps)
- Optional sentence-transformers embeddings when available
- In-memory vector store with cosine similarity search
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Text Splitter
# ---------------------------------------------------------------------------

class TextSplitter:
    """Recursive character-based text splitter.

    Splits text into chunks of approximately *chunk_size* characters,
    respecting *chunk_overlap* overlap between consecutive chunks and
    attempting to split on paragraph, sentence, or word boundaries.

    Args:
        chunk_size: Target size for each chunk in characters.
        chunk_overlap: Number of overlapping characters between chunks.
        separators: Ordered list of separator strings tried during recursive
            splitting.  The splitter tries the first separator; if the
            resulting piece is still too long it moves to the next, and so on.

    Example::

        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_text("Long document text ...")
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or list(self.DEFAULT_SEPARATORS)

    def split_text(self, text: str) -> List[str]:
        """Split *text* into chunks.

        Args:
            text: The input text to split.

        Returns:
            A list of text chunks.
        """
        if not text:
            return []
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the given separators."""
        if len(text) <= self.chunk_size:
            return [text]

        # Find the best separator
        for sep in separators:
            if sep == "":
                # Last resort: hard split
                chunks = []
                for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                    chunk = text[i : i + self.chunk_size]
                    if chunk.strip():
                        chunks.append(chunk)
                return chunks

            if sep in text:
                parts = text.split(sep)
                chunks: List[str] = []
                current = ""
                for part in parts:
                    candidate = current + sep + part if current else part
                    if len(candidate) <= self.chunk_size:
                        current = candidate
                    else:
                        if current:
                            chunks.append(current)
                        # If a single part is too long, recurse
                        if len(part) > self.chunk_size:
                            sub_chunks = self._recursive_split(part, separators[separators.index(sep) + 1 :])
                            chunks.extend(sub_chunks)
                            current = ""
                        else:
                            current = part
                if current:
                    chunks.append(current)

                # Merge small trailing chunks with overlap
                merged = self._merge_with_overlap(chunks)
                return merged

        # No separator found -- hard split
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i : i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def _merge_with_overlap(self, chunks: List[str]) -> List[str]:
        """Ensure consecutive chunks share *chunk_overlap* characters."""
        if len(chunks) <= 1:
            return chunks
        result: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = result[-1]
            overlap_text = prev[-self.chunk_overlap :] if len(prev) > self.chunk_overlap else prev
            # Only prepend overlap if it doesn't duplicate too much
            if chunks[i].startswith(overlap_text):
                result.append(chunks[i])
            else:
                merged = overlap_text + chunks[i]
                if len(merged) > self.chunk_size:
                    merged = merged[: self.chunk_size]
                result.append(merged)
        return result


# ---------------------------------------------------------------------------
# Document Loader
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A loaded document with metadata.

    Attributes:
        content: The extracted text content.
        source: Original file path or identifier.
        metadata: Arbitrary metadata attached to the document.
    """

    content: str
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentLoader:
    """Multi-format document loader.

    Supports plain text, markdown, CSV, JSON, and source code files
    (Python, JavaScript, TypeScript, HTML, CSS, etc.).

    Example::

        loader = DocumentLoader()
        doc = loader.load_file("notes.md")
        docs = loader.load_directory("./docs/")
    """

    # File extensions mapped to their expected MIME-like category
    TEXT_EXTENSIONS: Dict[str, str] = {
        ".txt": "text",
        ".md": "markdown",
        ".markdown": "markdown",
        ".csv": "csv",
        ".json": "json",
        ".py": "code",
        ".js": "code",
        ".ts": "code",
        ".tsx": "code",
        ".jsx": "code",
        ".html": "code",
        ".htm": "code",
        ".css": "code",
        ".scss": "code",
        ".yaml": "code",
        ".yml": "code",
        ".xml": "code",
        ".toml": "code",
        ".ini": "code",
        ".cfg": "code",
        ".sh": "code",
        ".bash": "code",
        ".zsh": "code",
        ".rs": "code",
        ".go": "code",
        ".java": "code",
        ".c": "code",
        ".cpp": "code",
        ".h": "code",
        ".rb": "code",
        ".php": "code",
        ".sql": "code",
        ".r": "code",
        ".R": "code",
    }

    def load_file(self, file_path: str | Path) -> Document:
        """Load a single file into a :class:`Document`.

        Args:
            file_path: Path to the file.

        Returns:
            A :class:`Document` with the extracted content.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not supported.
        """
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        category = self.TEXT_EXTENSIONS.get(ext)

        if category is None:
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                f"Supported: {', '.join(sorted(self.TEXT_EXTENSIONS.keys()))}"
            )

        raw = path.read_text(encoding="utf-8", errors="replace")

        if category == "csv":
            content = self._parse_csv(raw)
        elif category == "json":
            content = self._parse_json(raw)
        else:
            content = raw

        return Document(
            content=content,
            source=str(path),
            metadata={"extension": ext, "category": category, "filename": path.name},
        )

    def load_directory(self, dir_path: str | Path, recursive: bool = True) -> List[Document]:
        """Load all supported files from a directory.

        Args:
            dir_path: Path to the directory.
            recursive: Whether to traverse subdirectories.

        Returns:
            A list of loaded :class:`Document` objects.
        """
        directory = Path(dir_path).expanduser().resolve()
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        pattern = "**/*" if recursive else "*"
        documents: List[Document] = []
        for file_path in sorted(directory.glob(pattern)):
            if file_path.is_file() and file_path.suffix.lower() in self.TEXT_EXTENSIONS:
                try:
                    documents.append(self.load_file(file_path))
                except Exception as exc:
                    # Skip files that fail to load rather than aborting
                    continue
        return documents

    @staticmethod
    def _parse_csv(raw: str) -> str:
        """Convert CSV text into a readable paragraph format."""
        reader = csv.reader(raw.strip().splitlines())
        rows = list(reader)
        if not rows:
            return ""
        headers = rows[0]
        lines: List[str] = []
        for row in rows[1:]:
            pairs = [f"{h}: {v}" for h, v in zip(headers, row) if v.strip()]
            if pairs:
                lines.append("; ".join(pairs))
        return "\n".join(lines)

    @staticmethod
    def _parse_json(raw: str) -> str:
        """Flatten a JSON document into readable text."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        return json.dumps(data, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class SimpleEmbedder:
    """Text embedder with a deterministic hash-based fallback.

    If the ``sentence_transformers`` package is available, real dense
    embeddings are computed using the specified model.  Otherwise a
    deterministic hash-based projection is used as a fallback -- this
    still allows cosine-similarity search to function, though with
    lower quality semantic matching.

    Args:
        model_name: Name of the sentence-transformers model, or
            ``"hash"`` to force the hash-based fallback.

    Example::

        embedder = SimpleEmbedder(model_name="hash")
        vector = embedder.embed("Hello, world!")
    """

    _DIMENSION = 256  # dimensionality of the hash-based vectors

    def __init__(self, model_name: str = "hash") -> None:
        self.model_name = model_name
        self._model: Any = None
        self._use_hash = model_name == "hash"

        if not self._use_hash:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

                self._model = SentenceTransformer(model_name)
                self._use_hash = False
            except ImportError:
                import warnings

                warnings.warn(
                    "sentence-transformers not installed; falling back to hash-based embeddings. "
                    "Install with: pip install sentence-transformers",
                    UserWarning,
                    stacklevel=2,
                )
                self._use_hash = True

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        if self._use_hash:
            return self._DIMENSION
        # Infer from the model
        sample = self._model.encode(["dim"])  # type: ignore[union-attr]
        return len(sample[0])  # type: ignore[index]

    def embed(self, text: str) -> List[float]:
        """Produce an embedding vector for *text*.

        Args:
            text: Input text.

        Returns:
            A list of floats representing the embedding.
        """
        if self._use_hash:
            return self._hash_embed(text)
        vec = self._model.encode([text])[0]  # type: ignore[union-attr]
        return vec.tolist()  # type: ignore[union-attr]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts at once.

        Args:
            texts: List of input strings.

        Returns:
            A list of embedding vectors.
        """
        if self._use_hash:
            return [self._hash_embed(t) for t in texts]
        vecs = self._model.encode(texts)  # type: ignore[union-attr]
        return vecs.tolist()  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Hash-based fallback
    # ------------------------------------------------------------------

    def _hash_embed(self, text: str) -> List[float]:
        """Deterministic hash-based embedding (fallback).

        Uses multiple hash rounds with different seeds to produce a
        fixed-size float vector.
        """
        import hashlib

        dim = self._DIMENSION
        vector: List[float] = []
        for i in range(dim):
            h = hashlib.sha256(f"{text}__dim_{i}".encode("utf-8")).hexdigest()
            # Map first 8 hex chars to a float in [-1, 1]
            val = int(h[:8], 16) / 0xFFFFFFFF
            vector.append(val * 2.0 - 1.0)
        return vector


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single result from a vector similarity search.

    Attributes:
        text: The matching text chunk.
        score: Cosine similarity score (0.0 to 1.0, higher is better).
        source: Source document identifier.
        metadata: Arbitrary metadata.
    """

    text: str
    score: float
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """In-memory vector store with cosine similarity search.

    Stores embedding vectors alongside their source text and metadata,
    and retrieves the most similar documents for a given query vector.

    Args:
        dimension: Expected dimensionality of embedding vectors.

    Example::

        store = VectorStore(dimension=256)
        store.add("Hello world", [0.1, 0.2, ...], source="doc1.txt")
        results = store.search([0.1, 0.2, ...], top_k=3)
    """

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension
        self._texts: List[str] = []
        self._vectors: List[List[float]] = []
        self._sources: List[str] = []
        self._metadata: List[Dict[str, Any]] = []

    @property
    def size(self) -> int:
        """Number of stored vectors."""
        return len(self._texts)

    def add(
        self,
        text: str,
        vector: List[float],
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a single document to the store.

        Args:
            text: Document text.
            vector: Embedding vector.
            source: Source identifier.
            metadata: Optional metadata.
        """
        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector dimension {len(vector)} does not match store dimension {self.dimension}"
            )
        self._texts.append(text)
        self._vectors.append(vector)
        self._sources.append(source)
        self._metadata.append(metadata or {})

    def add_batch(
        self,
        texts: List[str],
        vectors: List[List[float]],
        sources: Optional[List[str]] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add multiple documents at once.

        Args:
            texts: List of document texts.
            vectors: List of embedding vectors.
            sources: Optional list of source identifiers.
            metadata_list: Optional list of metadata dicts.
        """
        for i, (text, vector) in enumerate(zip(texts, vectors)):
            source = sources[i] if sources and i < len(sources) else ""
            meta = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            self.add(text=text, vector=vector, source=source, metadata=meta)

    def search(self, query_vector: List[float], top_k: int = 5) -> List[SearchResult]:
        """Find the most similar documents to *query_vector*.

        Args:
            query_vector: The query embedding.
            top_k: Number of results to return.

        Returns:
            A list of :class:`SearchResult` objects sorted by descending
            similarity score.
        """
        if not self._vectors:
            return []

        scores: List[tuple[int, float]] = []
        for idx, vec in enumerate(self._vectors):
            sim = self._cosine_similarity(query_vector, vec)
            scores.append((idx, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        results: List[SearchResult] = []
        for idx, score in scores[:top_k]:
            results.append(
                SearchResult(
                    text=self._texts[idx],
                    score=score,
                    source=self._sources[idx],
                    metadata=self._metadata[idx],
                )
            )
        return results

    def clear(self) -> None:
        """Remove all stored documents."""
        self._texts.clear()
        self._vectors.clear()
        self._sources.clear()
        self._metadata.clear()

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """End-to-end RAG pipeline that ties together splitting, embedding,
    and retrieval.

    Args:
        chunk_size: Characters per text chunk.
        chunk_overlap: Overlap between consecutive chunks.
        embedding_model: Name of the embedding model, or ``"hash"`` for
            the deterministic fallback.
        max_chunks: Default number of chunks to retrieve per query.

    Example::

        rag = RAGPipeline(chunk_size=500, chunk_overlap=100)
        rag.ingest_file("knowledge_base.md")
        context = rag.build_context("How does the system work?")
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "hash",
        max_chunks: int = 5,
    ) -> None:
        self.splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = SimpleEmbedder(model_name=embedding_model)
        self.store = VectorStore(dimension=self.embedder.dimension)
        self.loader = DocumentLoader()
        self.max_chunks = max_chunks
        self._doc_count = 0

    def ingest_file(self, file_path: str | Path) -> int:
        """Load and ingest a single file into the vector store.

        Args:
            file_path: Path to the file.

        Returns:
            The number of chunks created and stored.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is unsupported.
        """
        doc = self.loader.load_file(file_path)
        return self._ingest_document(doc)

    def ingest_text(
        self,
        text: str,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Ingest raw text directly into the vector store.

        Args:
            text: The text to ingest.
            source: An optional source identifier.
            metadata: Optional metadata.

        Returns:
            The number of chunks created and stored.
        """
        doc = Document(content=text, source=source, metadata=metadata or {})
        return self._ingest_document(doc)

    def _ingest_document(self, doc: Document) -> int:
        """Split a document, embed the chunks, and store them."""
        chunks = self.splitter.split_text(doc.content)
        if not chunks:
            return 0

        vectors = self.embedder.embed_batch(chunks)
        sources = [doc.source] * len(chunks)
        metadatas = [
            {**doc.metadata, "chunk_index": i, "total_chunks": len(chunks)}
            for i in range(len(chunks))
        ]

        self.store.add_batch(
            texts=chunks,
            vectors=vectors,
            sources=sources,
            metadata_list=metadatas,
        )
        self._doc_count += 1
        return len(chunks)

    def query(self, query_text: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """Retrieve the most relevant chunks for a query.

        Args:
            query_text: The query string.
            top_k: Number of results. Defaults to :attr:`max_chunks`.

        Returns:
            A list of :class:`SearchResult` objects.
        """
        k = top_k if top_k is not None else self.max_chunks
        query_vector = self.embedder.embed(query_text)
        return self.store.search(query_vector, top_k=k)

    def build_context(self, query_text: str, top_k: Optional[int] = None) -> str:
        """Build a RAG context string suitable for injecting into a prompt.

        Args:
            query_text: The user query.
            top_k: Number of chunks to include.

        Returns:
            A formatted string containing the retrieved context, or an
            empty string if no documents have been ingested.
        """
        results = self.query(query_text, top_k=top_k)
        if not results:
            return ""

        parts: List[str] = ["--- Retrieved Context ---"]
        for i, result in enumerate(results, 1):
            source_label = f" [Source: {result.source}]" if result.source else ""
            parts.append(f"\n[Chunk {i}{source_label} (score: {result.score:.3f})]\n{result.text}")
        parts.append("--- End Context ---\n")
        return "\n".join(parts)

    @property
    def stats(self) -> Dict[str, Any]:
        """Return summary statistics about the RAG pipeline."""
        return {
            "documents_ingested": self._doc_count,
            "chunks_stored": self.store.size,
            "embedding_model": self.embedder.model_name,
            "embedding_dimension": self.embedder.dimension,
            "chunk_size": self.splitter.chunk_size,
            "chunk_overlap": self.splitter.chunk_overlap,
        }
