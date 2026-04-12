"""
Workflow Example — LocalMind

This example demonstrates how to create and run workflows
for multi-step AI tasks.
"""

import asyncio
from localmind import LocalMind
from localmind.core import Workflow


async def main():
    """Run the workflow example."""

    print("🧠 Initializing LocalMind...")
    mind = LocalMind()

    # ─── Example 1: Simple Sequential Workflow ─────────────────
    print("\n📋 Example 1: Simple Sequential Workflow")

    workflow = Workflow("Content Pipeline")

    workflow.add_step(
        "research",
        "Research Topic",
        function=lambda x: "AI agents are autonomous software entities...",
    )
    workflow.add_step(
        "outline",
        "Create Outline",
        function=lambda x: "1. Introduction\n2. Key Concepts\n3. Applications",
        depends_on=["research"],
    )
    workflow.add_step(
        "draft",
        "Write Draft",
        function=lambda x: "Draft content based on outline...",
        depends_on=["outline"],
    )
    workflow.add_step(
        "review",
        "Review & Edit",
        function=lambda x: "Reviewed and polished content.",
        depends_on=["draft"],
    )

    print(workflow.visualize())

    results = await workflow.run()
    for step_id, result in results.items():
        status_icon = "✅" if result.status.value == "completed" else "❌"
        print(f"  {status_icon} {step_id}: {result.status.value} ({result.duration_ms:.0f}ms)")

    # ─── Example 2: Conditional Workflow ───────────────────────
    print("\n📋 Example 2: Conditional Workflow")

    workflow2 = Workflow("Smart Router")

    workflow2.add_step(
        "classify",
        "Classify Input",
        function=lambda x: "technical",
    )
    workflow2.add_step(
        "tech_handler",
        "Handle Technical Query",
        function=lambda x: "Technical response generated.",
        depends_on=["classify"],
        condition=lambda r: r["classify"].output == "technical",
    )
    workflow2.add_step(
        "general_handler",
        "Handle General Query",
        function=lambda x: "General response generated.",
        depends_on=["classify"],
        condition=lambda r: r["classify"].output != "technical",
    )

    results2 = await workflow2.run()
    for step_id, result in results2.items():
        print(f"  {step_id}: {result.status.value}")

    # ─── Example 3: Workflow with Agents ───────────────────────
    print("\n📋 Example 3: Workflow with Agents")

    try:
        mind.load_model("llama3")

        from localmind.agents import ResearchAgent, WriterAgent

        researcher = ResearchAgent(engine=mind)
        writer = WriterAgent(engine=mind)

        workflow3 = Workflow("AI-Assisted Report")

        workflow3.add_step(
            "research",
            "Research Phase",
            agent="researcher",
            function=lambda x: "Research findings about local AI trends...",
        )
        workflow3.add_step(
            "write",
            "Writing Phase",
            agent="writer",
            depends_on=["research"],
            function=lambda x: "A comprehensive report on local AI trends...",
        )

        results3 = await workflow3.run(mind)
        for step_id, result in results3.items():
            print(f"  {step_id}: {result.status.value}")

    except Exception as e:
        print(f"  ⚠️  Skipped (model not available): {e}")

    # ─── Example 4: Parallel-like Workflow ─────────────────────
    print("\n📋 Example 4: Independent Steps (Parallelizable)")

    workflow4 = Workflow("Data Processing")

    # These steps have no dependencies, so they could run in parallel
    workflow4.add_step("step_a", "Process A", function=lambda x: "Result A")
    workflow4.add_step("step_b", "Process B", function=lambda x: "Result B")
    workflow4.add_step("step_c", "Process C", function=lambda x: "Result C")

    # Final step depends on all three
    workflow4.add_step(
        "combine",
        "Combine Results",
        function=lambda x: "Combined: A + B + C",
        depends_on=["step_a", "step_b", "step_c"],
    )

    results4 = await workflow4.run()
    print(f"  Total steps: {len(results4)}")
    print(f"  Workflow status: {workflow4.status.value}")

    mind.shutdown()
    print("\n✅ Done! 👋")


if __name__ == "__main__":
    asyncio.run(main())
