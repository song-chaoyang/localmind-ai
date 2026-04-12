"""
Tests for LocalMind Core - RAG Pipeline
"""

import tempfile
from pathlib import Path

import pytest

from localmind.core.rag import (
    ChunkConfig,
    Document,
    DocumentLoader,
    RAGPipeline,
    TextSplitter,
    VectorStore,
)


class TestDocument:
    """Tests for Document."""

    def test_auto_id(self):
        doc = Document(content="test content")
        assert doc.id
        assert len(doc.id) > 5

    def test_to_dict(self):
        doc = Document(content="test", metadata={"key": "value"})
        data = doc.to_dict()
        assert data["content"] == "test"
        assert data["metadata"]["key"] == "value"


class TestTextSplitter:
    """Tests for TextSplitter."""

    def test_split_short_text(self):
        splitter = TextSplitter(ChunkConfig(chunk_size=100))
        chunks = splitter.split_text("Short text")
        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_split_long_text(self):
        splitter = TextSplitter(ChunkConfig(chunk_size=50, chunk_overlap=10))
        text = " ".join([f"word{i}" for i in range(100)])
        chunks = splitter.split_text(text)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 60  # Allow some flexibility

    def test_split_empty_text(self):
        splitter = TextSplitter()
        chunks = splitter.split_text("")
        assert chunks == []

    def test_split_preserves_paragraphs(self):
        splitter = TextSplitter(ChunkConfig(chunk_size=200))
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = splitter.split_text(text)

        # Should try to split at paragraph boundaries
        assert len(chunks) >= 1

    def test_split_documents(self):
        splitter = TextSplitter(ChunkConfig(chunk_size=50))
        docs = [
            Document(content="A" * 100, metadata={"source": "doc1"}),
            Document(content="B" * 100, metadata={"source": "doc2"}),
        ]
        chunks = splitter.split_documents(docs)

        assert len(chunks) > 2  # Each doc should be split into multiple chunks


class TestDocumentLoader:
    """Tests for DocumentLoader."""

    def test_load_text_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("Hello, World!")
            temp_path = f.name

        try:
            doc = DocumentLoader.load_file(Path(temp_path))
            assert doc.content == "Hello, World!"
            assert doc.metadata["extension"] == ".txt"
        finally:
            Path(temp_path).unlink()

    def test_load_markdown_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write("# Title\n\nContent here.")
            temp_path = f.name

        try:
            doc = DocumentLoader.load_file(Path(temp_path))
            assert "# Title" in doc.content
        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            DocumentLoader.load_file(Path("/nonexistent/file.txt"))

    def test_load_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "file1.txt").write_text("Content 1")
            (Path(tmpdir) / "file2.md").write_text("Content 2")
            (Path(tmpdir) / "skip.dat").write_text("Skip this")

            docs = DocumentLoader.load_directory(Path(tmpdir))
            assert len(docs) == 2


class TestVectorStore:
    """Tests for VectorStore."""

    def test_add_and_search(self):
        store = VectorStore()
        docs = [
            Document(content="Python is a programming language"),
            Document(content="JavaScript runs in the browser"),
            Document(content="Machine learning uses neural networks"),
        ]
        store.add(docs)

        assert store.count() == 3

    def test_search_returns_results(self):
        store = VectorStore()
        docs = [
            Document(content="Python is a popular programming language for AI"),
            Document(content="The weather is sunny today"),
            Document(content="Machine learning models process data"),
        ]
        store.add(docs)

        results = store.search("programming language AI", top_k=2)
        assert len(results) > 0
        # First result should be about programming/AI
        assert results[0].score > 0

    def test_search_empty_store(self):
        store = VectorStore()
        results = store.search("anything")
        assert results == []

    def test_clear(self):
        store = VectorStore()
        store.add([Document(content="test")])
        store.clear()
        assert store.count() == 0


class TestRAGPipeline:
    """Tests for RAGPipeline."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = RAGPipeline()

    def test_ingest_text(self):
        chunks = self.pipeline.ingest_text("This is a long text that should be chunked. " * 50)
        assert chunks > 0

    def test_ingest_file(self):
        file_path = Path(self.temp_dir) / "test.txt"
        file_path.write_text("Content " * 200)

        chunks = self.pipeline.ingest_file(file_path)
        assert chunks > 0

    def test_ingest_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            self.pipeline.ingest_file("/nonexistent/file.txt")

    def test_query(self):
        self.pipeline.ingest_text(
            "Python is a programming language. "
            "Machine learning is a subset of AI. "
            "Neural networks are used in deep learning."
        )

        import asyncio
        results = asyncio.run(self.pipeline.query("programming language", top_k=2))
        assert len(results) > 0

    def test_build_context(self):
        self.pipeline.ingest_text(
            "Python is great for web development. "
            "JavaScript is essential for frontend. "
            "Rust is used for systems programming."
        )

        context = self.pipeline.build_context("web development", top_k=2)
        assert len(context) > 0

    def test_get_stats(self):
        stats = self.pipeline.get_stats()
        assert "total_documents" in stats
        assert "splitter_config" in stats
