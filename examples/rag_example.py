"""
RAG (Retrieval-Augmented Generation) Example — LocalMind

This example demonstrates how to ingest documents and use
RAG to augment AI responses with external knowledge.
"""

import asyncio
import tempfile
from pathlib import Path

from localmind import LocalMind
from localmind.core.rag import RAGPipeline


def main():
    """Run the RAG example."""

    print("🧠 LocalMind RAG Example\n")

    # ─── 1. Create RAG Pipeline ────────────────────────────────
    print("📋 Creating RAG pipeline...")
    rag = RAGPipeline()

    # ─── 2. Ingest Raw Text ────────────────────────────────────
    print("\n📄 Ingesting raw text...")
    text = """
    LocalMind is an open-source AI operating system that runs entirely on your
    local machine. It provides a unified platform for model management,
    intelligent agents, plugin ecosystem, and workflow orchestration.

    Key features of LocalMind:
    1. Privacy-first: All data stays on your machine
    2. Multi-model support: Works with Ollama, llama.cpp, and more
    3. Agent system: Built-in agents for research, coding, and analysis
    4. Plugin ecosystem: Extend functionality with community plugins
    5. RAG pipeline: Augment responses with your own documents
    6. Workflow engine: Orchestrate complex multi-step tasks

    LocalMind is designed to be the "Linux of AI" — giving users full control
    over their artificial intelligence experience. Unlike cloud-based solutions,
    LocalMind ensures that your data never leaves your machine.

    The project is licensed under MIT and welcomes contributions from the
    open-source community. Whether you're a developer, researcher, or AI
    enthusiast, LocalMind provides the tools you need to harness the power
    of AI while maintaining complete privacy and control.
    """.strip()

    chunks = rag.ingest_text(text, metadata={"source": "about_localmind"})
    print(f"   Ingested into {chunks} chunks")

    # ─── 3. Ingest a File ──────────────────────────────────────
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False
    ) as f:
        f.write("""
        # Machine Learning Basics

        Machine learning is a subset of artificial intelligence that enables
        systems to learn and improve from experience without being explicitly
        programmed.

        ## Types of Machine Learning

        ### Supervised Learning
        The algorithm learns from labeled training data. Examples include:
        - Image classification
        - Spam detection
        - Price prediction

        ### Unsupervised Learning
        The algorithm finds patterns in unlabeled data. Examples include:
        - Customer segmentation
        - Anomaly detection
        - Dimensionality reduction

        ### Reinforcement Learning
        The algorithm learns through trial and error. Examples include:
        - Game playing (AlphaGo)
        - Robotics
        - Autonomous driving

        ## Popular Frameworks
        - PyTorch
        - TensorFlow
        - scikit-learn
        - JAX
        """.strip())
        temp_path = f.name

    try:
        print(f"\n📄 Ingesting file: {temp_path}")
        file_chunks = rag.ingest_file(temp_path)
        print(f"   Ingested into {file_chunks} chunks")
    finally:
        Path(temp_path).unlink()

    # ─── 4. Query the RAG System ───────────────────────────────
    print("\n🔍 Querying RAG system...")

    queries = [
        "What is LocalMind?",
        "What are the types of machine learning?",
        "How does LocalMind protect privacy?",
    ]

    for query in queries:
        print(f"\n   Query: '{query}'")
        context = rag.build_context(query, top_k=2)
        if context:
            print(f"   Found relevant context ({len(context)} chars)")
            # Show first 200 chars of context
            print(f"   Preview: {context[:200]}...")
        else:
            print("   No relevant context found")

    # ─── 5. Async Query ────────────────────────────────────────
    async def async_query():
        results = await rag.query("popular ML frameworks", top_k=3)
        print(f"\n📊 Async query results: {len(results)} found")
        for r in results:
            print(f"   Score: {r['score']:.3f} | Source: {r['metadata'].get('source', 'unknown')}")

    asyncio.run(async_query())

    # ─── 6. Pipeline Stats ─────────────────────────────────────
    stats = rag.get_stats()
    print(f"\n📈 Pipeline Stats:")
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Chunk size: {stats['splitter_config']['chunk_size']}")
    print(f"   Chunk overlap: {stats['splitter_config']['chunk_overlap']}")

    print("\n✅ RAG example complete! 👋")


if __name__ == "__main__":
    main()
