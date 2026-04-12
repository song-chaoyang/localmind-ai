"""
RAG (Retrieval-Augmented Generation) Pipeline for LocalMind.

Provides document ingestion, chunking, embedding, and retrieval
capabilities for augmenting model responses with external knowledge.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A document or document chunk."""

    id: str = ""
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    score: float = 0.0

    def __post_init__(self):
        if not self.id:
            self.id = hashlib.sha256(
                self.content.encode()[:1024]
            ).hexdigest()[:16]


@dataclass
class ChunkConfig:
    """Configuration for text chunking."""

    chunk_size: int = 512
    chunk_overlap: int = 50
    separators: List[str] = field(default_factory=lambda: [
        "\n\n", "\n", ". ", " ", ""
    ])


class TextSplitter:
    """
    Text splitting utility for breaking documents into chunks.

    Supports multiple splitting strategies and respects
    document structure (paragraphs, sentences, etc.).
    """

    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks of approximately config.chunk_size.

        Uses recursive character splitting with the configured separators.
        """
        if not text:
            return []

        chunks = []
        remaining = text

        while remaining:
            if len(remaining) <= self.config.chunk_size:
                chunks.append(remaining.strip())
                break

            # Find the best split point
            split_point = self._find_split_point(remaining)
            chunk = remaining[:split_point].strip()
            if chunk:
                chunks.append(chunk)

            # Move past the split point, with overlap
            overlap_start = max(0, split_point - self.config.chunk_overlap)
            remaining = remaining[overlap_start:]

        return [c for c in chunks if c.strip()]

    def _find_split_point(self, text: str) -> int:
        """Find the best point to split text."""
        target_len = self.config.chunk_size

        # Try each separator in order of preference
        for separator in self.config.separators:
            if not separator:
                continue

            # Look for separator near the target length
            search_start = max(0, target_len - 100)
            search_end = min(len(text), target_len + 100)
            search_region = text[search_start:search_end]

            # Find the last occurrence of the separator
            last_sep = search_region.rfind(separator)
            if last_sep != -1:
                return search_start + last_sep + len(separator)

        # Fallback: split at target length
        return target_len

    def split_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """Split a list of documents into chunks."""
        chunks = []
        for doc in documents:
            text_chunks = self.split_text(doc.content)
            for i, chunk_text in enumerate(text_chunks):
                chunk = Document(
                    content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(text_chunks),
                    },
                )
                chunks.append(chunk)
        return chunks


class DocumentLoader:
    """
    Utility for loading documents from various file formats.
    """

    SUPPORTED_EXTENSIONS = {
        ".txt", ".md", ".markdown",
        ".json", ".csv",
        ".py", ".js", ".ts", ".java", ".go", ".rs",
        ".html", ".xml", ".yaml", ".yml", ".toml",
    }

    @classmethod
    def load_file(cls, file_path: Path) -> Document:
        """Load a document from a file."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = file_path.suffix.lower()
        if extension not in cls.SUPPORTED_EXTENSIONS:
            logger.warning(
                f"File extension '{extension}' may not be fully supported. "
                f"Attempting to load as text."
            )

        content = file_path.read_text(encoding="utf-8", errors="replace")

        return Document(
            content=content,
            metadata={
                "source": str(file_path),
                "filename": file_path.name,
                "extension": extension,
                "size": len(content),
            },
        )

    @classmethod
    def load_directory(
        cls,
        directory: Path,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
    ) -> List[Document]:
        """Load all supported documents from a directory."""
        extensions = extensions or list(cls.SUPPORTED_EXTENSIONS)
        documents = []

        pattern = "**/*" if recursive else "*"
        for file_path in sorted(directory.glob(pattern)):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    doc = cls.load_file(file_path)
                    documents.append(doc)
                    logger.debug(f"Loaded document: {file_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")

        return documents


class SimpleEmbedder:
    """
    Simple embedding generator using TF-IDF-like approach.

    For production use, replace with proper sentence-transformers
    or OpenAI embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._use_sentence_transformers = False

    def _load_model(self) -> None:
        """Lazily load the embedding model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._use_sentence_transformers = True
            logger.info(f"Loaded embedding model: {self.model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Using simple hash-based embeddings. "
                "For better results: pip install sentence-transformers"
            )
            self._use_sentence_transformers = False

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        self._load_model()

        if self._use_sentence_transformers and self._model:
            embeddings = self._model.encode(texts, show_progress_bar=False)
            return [e.tolist() for e in embeddings]

        # Fallback: simple hash-based embeddings
        return [self._simple_hash_embedding(text) for text in texts]

    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed([text])[0]

    def _simple_hash_embedding(self, text: str, dim: int = 384) -> List[float]:
        """Generate a simple hash-based embedding (fallback)."""
        words = re.findall(r'\w+', text.lower())
        embedding = [0.0] * dim

        for i, word in enumerate(words):
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = hash_val % dim
            embedding[idx] += 1.0 / (i + 1)

        # Normalize
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding


class VectorStore:
    """
    Simple in-memory vector store with cosine similarity search.

    For production, replace with ChromaDB, FAISS, or Qdrant.
    """

    def __init__(self):
        self._documents: List[Document] = []
        self._embedder = SimpleEmbedder()

    def add(self, documents: List[Document]) -> None:
        """Add documents to the store."""
        texts = [doc.content for doc in documents]
        embeddings = self._embedder.embed(texts)

        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
            self._documents.append(doc)

        logger.debug(f"Added {len(documents)} documents to vector store")

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[Document]:
        """Search for similar documents."""
        if not self._documents:
            return []

        query_embedding = self._embedder.embed_single(query)

        # Calculate cosine similarity
        scored = []
        for doc in self._documents:
            if doc.embedding:
                score = self._cosine_similarity(query_embedding, doc.embedding)
                if score >= min_score:
                    doc_copy = Document(
                        id=doc.id,
                        content=doc.content,
                        metadata=doc.metadata,
                        score=score,
                    )
                    scored.append(doc_copy)

        # Sort by score descending
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = sum(x * x for x in a) ** 0.5
        magnitude_b = sum(x * x for x in b) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    def count(self) -> int:
        """Return the number of documents in the store."""
        return len(self._documents)

    def clear(self) -> None:
        """Clear all documents."""
        self._documents.clear()


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.

    Orchestrates document loading, chunking, embedding, and retrieval
    to augment model responses with relevant context.
    """

    def __init__(self, config=None):
        self.config = config
        self.splitter = TextSplitter()
        self.loader = DocumentLoader()
        self.vector_store = VectorStore()
        self._persist_dir = Path.home() / ".localmind" / "documents" / "vectors"

    def ingest_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Ingest a file into the RAG pipeline.

        Args:
            file_path: Path to the file
            metadata: Optional metadata

        Returns:
            Number of chunks created
        """
        file_path = Path(file_path)
        doc = self.loader.load_file(file_path)

        if metadata:
            doc.metadata.update(metadata)

        # Split into chunks
        chunks = self.splitter.split_documents([doc])

        # Add to vector store
        self.vector_store.add(chunks)

        logger.info(f"Ingested '{file_path.name}' into {len(chunks)} chunks")
        return len(chunks)

    def ingest_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
    ) -> int:
        """
        Ingest all supported files from a directory.

        Args:
            directory: Path to the directory
            recursive: Whether to search subdirectories

        Returns:
            Total number of chunks created
        """
        directory = Path(directory)
        docs = self.loader.load_directory(directory, recursive=recursive)
        chunks = self.splitter.split_documents(docs)
        self.vector_store.add(chunks)

        logger.info(
            f"Ingested {len(docs)} documents from '{directory}' "
            f"into {len(chunks)} chunks"
        )
        return len(chunks)

    def ingest_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Ingest raw text into the RAG pipeline.

        Args:
            text: The text to ingest
            metadata: Optional metadata

        Returns:
            Number of chunks created
        """
        doc = Document(content=text, metadata=metadata or {})
        chunks = self.splitter.split_documents([doc])
        self.vector_store.add(chunks)
        return len(chunks)

    async def query(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query the RAG pipeline for relevant documents.

        Args:
            query: The search query
            top_k: Number of results to return

        Returns:
            List of relevant documents with scores
        """
        k = top_k or 5
        results = self.vector_store.search(query, top_k=k)

        return [
            {
                "content": doc.content,
                "score": doc.score,
                "metadata": doc.metadata,
                "id": doc.id,
            }
            for doc in results
        ]

    def build_context(self, query: str, top_k: int = 3) -> str:
        """
        Build a context string from relevant documents for prompt injection.

        Args:
            query: The search query
            top_k: Number of documents to include

        Returns:
            Formatted context string
        """
        results = self.vector_store.search(query, top_k=top_k)

        if not results:
            return ""

        context_parts = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            context_parts.append(
                f"[Document {i}] (Source: {source}, Score: {doc.score:.3f})\n"
                f"{doc.content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG pipeline statistics."""
        return {
            "total_documents": self.vector_store.count(),
            "splitter_config": {
                "chunk_size": self.splitter.config.chunk_size,
                "chunk_overlap": self.splitter.config.chunk_overlap,
            },
        }
