"""RAG (Retrieval-Augmented Generation) Pipeline for NexusMind.

Provides document loading, text splitting, embedding, vector storage,
and retrieval for augmenting LLM responses with external knowledge.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RAGError(Exception):
    """Base exception for RAG operations."""


class DocumentLoadError(RAGError):
    """Raised when a document cannot be loaded."""


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """A document or document chunk with metadata."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = ""
    embedding: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = hashlib.sha256(self.content.encode()).hexdigest()[:16]


@dataclass
class SearchResult:
    """A search result from the vector store."""

    document: Document
    score: float
    chunk_index: int = 0


# ---------------------------------------------------------------------------
# Document Loader
# ---------------------------------------------------------------------------


class DocumentLoader:
    """Load documents from various file formats.

    Supported formats: .txt, .md, .py, .js, .ts, .json, .yaml, .yml,
    .toml, .cfg, .ini, .html, .css, .sh, .bash, .sql, .csv
    """

    _SUPPORTED_EXTENSIONS: set[str] = {
        ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx",
        ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
        ".html", ".htm", ".css", ".sh", ".bash", ".sql", ".csv",
        ".rs", ".go", ".java", ".c", ".cpp", ".h", ".hpp",
        ".rb", ".php", ".r", ".jl", ".scala", ".kt",
    }

    @classmethod
    def load(cls, path: str | Path) -> list[Document]:
        """Load a document from a file path.

        Args:
            path: Path to the file.

        Returns:
            List of Document objects (usually one).

        Raises:
            DocumentLoadError: If the file cannot be loaded.
        """
        path = Path(path)
        if not path.exists():
            raise DocumentLoadError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in cls._SUPPORTED_EXTENSIONS:
            raise DocumentLoadError(
                f"Unsupported file format: {ext}. "
                f"Supported: {', '.join(sorted(cls._SUPPORTED_EXTENSIONS))}"
            )

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = path.read_text(encoding="latin-1")
            except Exception as e:
                raise DocumentLoadError(f"Cannot read file {path}: {e}") from e

        doc = Document(
            content=content,
            metadata={
                "source": str(path),
                "filename": path.name,
                "extension": ext,
                "size_bytes": path.stat().st_size,
            },
        )
        return [doc]

    @classmethod
    def load_many(cls, paths: list[str | Path]) -> list[Document]:
        """Load multiple documents.

        Args:
            paths: List of file paths.

        Returns:
            List of Document objects.

        Raises:
            DocumentLoadError: If any file cannot be loaded.
        """
        documents: list[Document] = []
        for path in paths:
            documents.extend(cls.load(path))
        return documents

    @classmethod
    def load_directory(
        cls,
        directory: str | Path,
        recursive: bool = True,
        extensions: set[str] | None = None,
    ) -> list[Document]:
        """Load all supported documents from a directory.

        Args:
            directory: Directory path.
            recursive: Whether to search subdirectories.
            extensions: Optional set of extensions to filter.

        Returns:
            List of Document objects.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise DocumentLoadError(f"Not a directory: {directory}")

        target_extensions = extensions or cls._SUPPORTED_EXTENSIONS
        pattern = "**/*" if recursive else "*"
        documents: list[Document] = []

        for file_path in sorted(directory.glob(pattern)):
            if file_path.is_file() and file_path.suffix.lower() in target_extensions:
                try:
                    documents.extend(cls.load(file_path))
                except DocumentLoadError as e:
                    logger.warning("Skipping %s: %s", file_path, e)

        return documents


# ---------------------------------------------------------------------------
# Text Splitter
# ---------------------------------------------------------------------------


class TextSplitter:
    """Split documents into chunks with overlap for RAG processing.

    Supports multiple splitting strategies and configurable chunk sizes.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        """Initialize the text splitter.

        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Number of overlapping characters between chunks.
            separators: List of separator strings in priority order.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks.

        Args:
            text: The text to split.

        Returns:
            List of text chunks.
        """
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        remaining = text

        while len(remaining) > self.chunk_size:
            # Find the best split point
            split_pos = self._find_split_point(
                remaining[: self.chunk_size + self.chunk_overlap]
            )
            chunk = remaining[:split_pos].strip()
            if chunk:
                chunks.append(chunk)
            remaining = remaining[split_pos - self.chunk_overlap:]

        if remaining.strip():
            chunks.append(remaining.strip())

        return chunks

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks.

        Args:
            documents: List of Document objects.

        Returns:
            List of chunked Document objects.
        """
        chunked: list[Document] = []
        for i, doc in enumerate(documents):
            chunks = self.split_text(doc.content)
            for j, chunk in enumerate(chunks):
                chunked_doc = Document(
                    content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": j,
                        "total_chunks": len(chunks),
                        "parent_id": doc.id,
                        "document_index": i,
                    },
                )
                chunked.append(chunked_doc)
        return chunked

    def _find_split_point(self, text: str) -> int:
        """Find the best position to split text.

        Tries separators in priority order, falling back to chunk_size.

        Args:
            text: Text to find split point in.

        Returns:
            Character position for the split.
        """
        for sep in self.separators:
            if not sep:
                continue
            # Search from the end of the chunk_size portion
            search_start = max(0, self.chunk_size - len(sep) * 2)
            search_end = min(len(text), self.chunk_size + self.chunk_overlap)
            pos = text.rfind(sep, search_start, search_end)
            if pos > 0:
                return pos + len(sep)

        return self.chunk_size


# ---------------------------------------------------------------------------
# Simple Embedder
# ---------------------------------------------------------------------------


class SimpleEmbedder:
    """Hash-based embedding fallback with optional sentence-transformers.

    Provides basic text embeddings using character frequency analysis.
    When sentence-transformers is available, uses it for higher quality embeddings.
    """

    _DIMENSION = 128
    _model = None

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize the embedder.

        Args:
            model_name: Optional sentence-transformers model name.
        """
        self.model_name = model_name
        self._use_transformers = False
        self._try_load_transformers()

    def _try_load_transformers(self) -> None:
        """Try to load sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            model = self.model_name or "all-MiniLM-L6-v2"
            self._model = SentenceTransformer(model)
            self._use_transformers = True
            self._DIMENSION = self._model.get_sentence_embedding_dimension()
            logger.info("Loaded sentence-transformers model: %s", model)
        except ImportError:
            logger.info("sentence-transformers not available, using hash-based embeddings")
        except Exception as e:
            logger.warning("Failed to load sentence-transformers: %s", e)

    @property
    def dimension(self) -> int:
        """Get the embedding dimension.

        Returns:
            Number of dimensions in the embedding vectors.
        """
        return self._DIMENSION

    def embed(self, text: str) -> list[float]:
        """Generate an embedding for the given text.

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        if self._use_transformers and self._model is not None:
            embedding = self._model.encode(text).tolist()
            return embedding
        return self._hash_embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if self._use_transformers and self._model is not None:
            embeddings = self._model.encode(texts).tolist()
            return embeddings
        return [self._hash_embed(text) for text in texts]

    def _hash_embed(self, text: str) -> list[float]:
        """Generate a hash-based embedding vector.

        Uses character frequency analysis to create a fixed-dimension vector.

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the embedding.
        """
        text = text.lower().strip()
        embedding = [0.0] * self._DIMENSION

        # Character frequency-based embedding
        for i, char in enumerate(text):
            idx = ord(char) % self._DIMENSION
            embedding[idx] += 1.0 / (i + 1)

        # Bigram-based embedding
        for i in range(len(text) - 1):
            bigram = text[i : i + 2]
            idx = hash(bigram) % self._DIMENSION
            embedding[idx] += 0.5

        # Normalize
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------


class VectorStore:
    """In-memory vector store with cosine similarity search.

    Stores document embeddings and provides efficient similarity search
    for RAG retrieval.
    """

    def __init__(self, dimension: int = 128) -> None:
        """Initialize the vector store.

        Args:
            dimension: Dimension of embedding vectors.
        """
        self.dimension = dimension
        self._documents: dict[str, Document] = {}
        self._embeddings: dict[str, list[float]] = {}

    def add(self, document: Document) -> None:
        """Add a document with its embedding to the store.

        Args:
            document: Document with embedding set.
        """
        if document.embedding:
            if len(document.embedding) != self.dimension:
                # Resize embedding
                document.embedding = self._resize_embedding(document.embedding)
        self._documents[document.id] = document
        if document.embedding:
            self._embeddings[document.id] = document.embedding

    def add_many(self, documents: list[Document]) -> None:
        """Add multiple documents to the store.

        Args:
            documents: List of Documents with embeddings.
        """
        for doc in documents:
            self.add(doc)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Search for similar documents.

        Args:
            query_embedding: The query embedding vector.
            top_k: Maximum number of results.
            threshold: Minimum similarity score.

        Returns:
            List of SearchResult objects sorted by similarity.
        """
        if not self._embeddings:
            return []

        query = self._resize_embedding(query_embedding)
        results: list[SearchResult] = []

        for doc_id, embedding in self._embeddings.items():
            score = self._cosine_similarity(query, embedding)
            if score >= threshold:
                doc = self._documents.get(doc_id)
                if doc:
                    results.append(SearchResult(document=doc, score=score))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def delete(self, document_id: str) -> bool:
        """Delete a document from the store.

        Args:
            document_id: Document ID to delete.

        Returns:
            True if the document was found and deleted.
        """
        if document_id in self._documents:
            del self._documents[document_id]
            self._embeddings.pop(document_id, None)
            return True
        return False

    @property
    def size(self) -> int:
        """Number of documents in the store."""
        return len(self._documents)

    def clear(self) -> None:
        """Remove all documents from the store."""
        self._documents.clear()
        self._embeddings.clear()

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity score between -1 and 1.
        """
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def _resize_embedding(self, embedding: list[float]) -> list[float]:
        """Resize an embedding to match the store dimension.

        Args:
            embedding: Original embedding vector.

        Returns:
            Resized embedding vector.
        """
        if len(embedding) == self.dimension:
            return embedding
        if len(embedding) > self.dimension:
            # Downsample by averaging
            result = [0.0] * self.dimension
            for i in range(self.dimension):
                start = i * len(embedding) // self.dimension
                end = (i + 1) * len(embedding) // self.dimension
                result[i] = sum(embedding[start:end]) / max(1, end - start)
            return result
        # Upsample by interpolation
        result = list(embedding) + [0.0] * (self.dimension - len(embedding))
        return result


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    """End-to-end RAG pipeline for document ingestion and retrieval.

    Combines document loading, text splitting, embedding, and vector
    storage into a unified pipeline.

    Example::

        rag = RAGPipeline()
        rag.ingest("/path/to/documents/")
        results = rag.query("How does authentication work?")
        context = rag.build_context("How does authentication work?")
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedder_model: str | None = None,
    ) -> None:
        """Initialize the RAG pipeline.

        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap between chunks.
            embedder_model: Optional sentence-transformers model name.
        """
        self.splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = SimpleEmbedder(model_name=embedder_model)
        self.store = VectorStore(dimension=self.embedder.dimension)
        self._stats = {
            "documents_ingested": 0,
            "chunks_created": 0,
            "queries_performed": 0,
        }

    def ingest(self, paths: list[str | Path]) -> int:
        """Ingest documents into the RAG pipeline.

        Args:
            paths: List of file paths or directories to ingest.

        Returns:
            Number of chunks created.
        """
        all_documents: list[Document] = []

        for path in paths:
            path = Path(path)
            if path.is_dir():
                all_documents.extend(DocumentLoader.load_directory(path))
            elif path.is_file():
                all_documents.extend(DocumentLoader.load(path))
            else:
                logger.warning("Path not found: %s", path)

        if not all_documents:
            return 0

        # Split documents into chunks
        chunks = self.splitter.split_documents(all_documents)

        # Embed and store
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed_batch(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            self.store.add(chunk)

        self._stats["documents_ingested"] += len(all_documents)
        self._stats["chunks_created"] += len(chunks)

        logger.info(
            "Ingested %d documents -> %d chunks",
            len(all_documents), len(chunks),
        )
        return len(chunks)

    def ingest_text(self, text: str, metadata: dict[str, Any] | None = None) -> int:
        """Ingest raw text into the RAG pipeline.

        Args:
            text: The text to ingest.
            metadata: Optional metadata for the document.

        Returns:
            Number of chunks created.
        """
        doc = Document(content=text, metadata=metadata or {})
        chunks = self.splitter.split_documents([doc])
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed_batch(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            self.store.add(chunk)

        self._stats["chunks_created"] += len(chunks)
        return len(chunks)

    def query(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Query the RAG pipeline for relevant documents.

        Args:
            query: The query string.
            top_k: Maximum number of results.
            threshold: Minimum similarity score.

        Returns:
            List of SearchResult objects.
        """
        query_embedding = self.embedder.embed(query)
        results = self.store.search(query_embedding, top_k=top_k, threshold=threshold)
        self._stats["queries_performed"] += 1
        return results

    def build_context(self, query: str, max_tokens: int = 2000) -> str:
        """Build a context string from relevant documents.

        Args:
            query: The query string.
            max_tokens: Approximate maximum token budget.

        Returns:
            Formatted context string for LLM prompts.
        """
        results = self.query(query, top_k=5)
        if not results:
            return ""

        parts: list[str] = []
        total_chars = max_tokens * 4  # rough estimate

        for i, result in enumerate(results):
            source = result.document.metadata.get("source", "unknown")
            chunk = f"[{i+1}] (Source: {source})\n{result.document.content}"
            if sum(len(p) for p in parts) + len(chunk) > total_chars:
                break
            parts.append(chunk)

        return "\n\n---\n\n".join(parts)

    def get_stats(self) -> dict[str, Any]:
        """Get RAG pipeline statistics.

        Returns:
            Dictionary with pipeline statistics.
        """
        return {
            **self._stats,
            "store_size": self.store.size,
            "embedding_dimension": self.embedder.dimension,
            "using_transformers": self.embedder._use_transformers,
        }
