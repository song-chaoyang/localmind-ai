"""
Quick Start Example — LocalMind

This example demonstrates the basic usage of LocalMind:
loading a model, chatting, and using agents.
"""

import asyncio
from localmind import LocalMind
from localmind.agents import ResearchAgent, CodeAgent, WriterAgent


async def main():
    """Run the quick start example."""

    # ─── 1. Initialize LocalMind ───────────────────────────────
    print("🧠 Initializing LocalMind...")
    mind = LocalMind()

    # ─── 2. Load a Model ───────────────────────────────────────
    print("\n📦 Loading model...")
    try:
        mind.load_model("llama3")
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️  Could not load model: {e}")
        print("   Make sure Ollama is running: https://ollama.ai")
        print("   Then run: ollama pull llama3")
        return

    # ─── 3. Simple Chat ────────────────────────────────────────
    print("\n💬 Chatting with LocalMind...")
    response = await mind.chat(
        "Explain quantum computing in one simple paragraph.",
        temperature=0.7,
    )
    print(f"\n🤖 AI: {response}\n")

    # ─── 4. Streaming Chat ─────────────────────────────────────
    print("💬 Streaming response...")
    print("🤖 AI: ", end="", flush=True)
    async for chunk in mind.chat_stream(
        "What are the top 3 programming languages in 2025?"
    ):
        print(chunk, end="", flush=True)
    print("\n")

    # ─── 5. Use Agents ─────────────────────────────────────────
    print("\n🤖 Using Agents...")

    # Register agents
    researcher = ResearchAgent(engine=mind)
    coder = CodeAgent(engine=mind)
    writer = WriterAgent(engine=mind)

    mind.register_agent("researcher", researcher)
    mind.register_agent("coder", coder)
    mind.register_agent("writer", writer)

    # Run a research task
    result = await mind.execute_agent(
        "researcher",
        "What are the latest trends in open-source AI?"
    )
    print(f"\n📋 Research Result:\n{result}\n")

    # ─── 6. Multi-Agent Collaboration ──────────────────────────
    print("🤝 Multi-Agent Collaboration...")
    result = await mind.collaborate(
        agents=[researcher, writer],
        task="Research the benefits of local AI and write a short blog post about it",
    )
    print(f"\n📋 Collaboration Result:\n{result}\n")

    # ─── 7. System Stats ───────────────────────────────────────
    stats = mind.get_stats()
    print(f"\n📊 System Stats:")
    print(f"   Model: {stats['model_name']}")
    print(f"   Agents: {stats['agents_registered']}")
    print(f"   Messages: {stats['memory']['short_term_messages']}")

    # ─── 8. Cleanup ────────────────────────────────────────────
    mind.shutdown()
    print("\n✅ LocalMind shut down. Goodbye! 👋")


if __name__ == "__main__":
    asyncio.run(main())
