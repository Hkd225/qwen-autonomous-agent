"""
main.py - Entry Point & Demo
==============================
Demonstrasi penggunaan Autonomous Agent dengan berbagai task.

Jalankan dengan:
  python main.py                   # Demo default
  python main.py --task "your task" # Custom task
  python main.py --mode interactive  # Mode interaktif
  python main.py --mode goals       # Mode goal pursuit
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time

# Pastikan module path benar
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import AutonomousAgent, AgentConfig
from memory.goal_memory import GoalPriority


# ─────────────────────────────────────────────
# Demo Tasks
# ─────────────────────────────────────────────

DEMO_TASKS = [
    {
        "name": "Riset Sederhana",
        "task": "Cari informasi tentang model language Qwen 2.5 dan jelaskan keunggulannya",
        "type": "search",
    },
    {
        "name": "Kalkulasi Matematika",
        "task": (
            "Hitung berapa lama waktu yang dibutuhkan untuk membayar hutang Rp 50 juta "
            "dengan bunga 12% per tahun, jika membayar Rp 1 juta per bulan"
        ),
        "type": "calculation",
    },
    {
        "name": "Analisis & Reasoning",
        "task": (
            "Bandingkan pendekatan Retrieval-Augmented Generation (RAG) dengan "
            "fine-tuning dalam konteks pengembangan chatbot untuk perusahaan. "
            "Kapan menggunakan masing-masing?"
        ),
        "type": "analysis",
    },
    {
        "name": "Coding Task",
        "task": (
            "Tulis fungsi Python yang menghitung bilangan Fibonacci ke-N menggunakan "
            "memoization, dan hitung Fibonacci ke-35. Tunjukkan hasilnya."
        ),
        "type": "coding",
    },
    {
        "name": "Goal-Based Task",
        "task": (
            "Buat panduan 5 langkah untuk memulai belajar machine learning dari nol, "
            "dengan resource yang konkrit untuk setiap langkah"
        ),
        "type": "planning",
    },
]


# ─────────────────────────────────────────────
# Demo Modes
# ─────────────────────────────────────────────

async def demo_single_task(agent: AutonomousAgent, task: str):
    """Jalankan satu task dan tampilkan hasilnya."""
    print(f"\n{'='*60}")
    print(f"TASK: {task}")
    print(f"{'='*60}\n")

    trace = await agent.run(task)

    print(f"\n{'='*60}")
    print("HASIL:")
    print(f"{'='*60}")
    print(trace.final_answer)
    print(f"\n{'─'*40}")
    print(f"Status: {'✓ SUKSES' if trace.success else '✗ GAGAL'}")
    print(f"Steps: {len(trace.steps)}")
    print(f"Durasi: {trace.duration_ms/1000:.1f}s")
    print(f"Tokens: {trace.total_tokens}")

    return trace


async def demo_multiple_tasks(agent: AutonomousAgent):
    """Jalankan semua demo tasks."""
    print("\n" + "="*60)
    print("DEMO: Multiple Tasks")
    print("="*60)

    results = []
    for i, task_info in enumerate(DEMO_TASKS, 1):
        print(f"\n[{i}/{len(DEMO_TASKS)}] {task_info['name']}")
        print("-" * 40)

        trace = await demo_single_task(agent, task_info["task"])
        results.append({
            "name": task_info["name"],
            "success": trace.success,
            "steps": len(trace.steps),
            "duration_s": round(trace.duration_ms / 1000, 1),
        })

        # Pause singkat antara tasks
        await asyncio.sleep(1)

    # Ringkasan
    print(f"\n{'='*60}")
    print("RINGKASAN SEMUA TASKS:")
    print(f"{'='*60}")
    success_count = sum(1 for r in results if r["success"])
    print(f"Success rate: {success_count}/{len(results)} tasks")
    print()
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['name']}: {r['steps']} steps, {r['duration_s']}s")


async def demo_interactive(agent: AutonomousAgent):
    """Mode interaktif: user input tasks secara real-time."""
    print("\n" + "="*60)
    print("MODE INTERAKTIF")
    print("Ketik task kamu, atau 'quit' untuk keluar")
    print("Ketik 'status' untuk melihat status agent")
    print("Ketik 'skills' untuk melihat skill library")
    print("="*60)

    while True:
        try:
            print("\n")
            task = input("Task: ").strip()

            if not task:
                continue
            elif task.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break
            elif task.lower() == "status":
                status = agent.get_status()
                print("\nAgent Status:")
                print(json.dumps(status, indent=2, ensure_ascii=False))
            elif task.lower() == "skills":
                print("\n" + agent.self_improvement.skill_library.summary())
            elif task.lower() == "goals":
                print("\nGoal Tree:")
                print(agent.goal_memory.format_tree())
            else:
                await demo_single_task(agent, task)

        except KeyboardInterrupt:
            print("\n\nInterrupted!")
            break
        except EOFError:
            break


async def demo_goal_pursuit(agent: AutonomousAgent):
    """Demo goal hierarchy dan autonomous pursuit."""
    print("\n" + "="*60)
    print("DEMO: Goal Hierarchy & Autonomous Pursuit")
    print("="*60)

    # Set ultimate goal
    goal = await agent.set_goal(
        title="Buat Panduan Machine Learning",
        description=(
            "Buat panduan komprehensif untuk pemula yang ingin belajar "
            "machine learning, termasuk roadmap, resources, dan project ideas"
        ),
        priority=GoalPriority.HIGH,
    )

    print(f"\nGoal dibuat: {goal.title}")

    # Decompose ke sub-goals
    print("Decomposing goal ke sub-goals...")
    subgoals = await agent.decompose_goal(goal, n_subgoals=3)

    print(f"\n{len(subgoals)} sub-goals dibuat:")
    for sg in subgoals:
        print(f"  • {sg.title}")

    # Tampilkan goal tree
    print("\nGoal Tree:")
    print(agent.goal_memory.format_tree())

    # Pursuit goals
    print("\nMengejar goals secara otomatis...")
    traces = await agent.pursue_goals(max_goals=3)

    # Summary
    print(f"\n{'='*60}")
    print(f"SELESAI: {sum(1 for t in traces if t.success)}/{len(traces)} goals berhasil")


# ─────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Autonomous Agent berbasis Qwen 2.5 7B"
    )
    parser.add_argument(
        "--task", type=str, default="",
        help="Task yang akan diselesaikan"
    )
    parser.add_argument(
        "--mode", type=str, default="single",
        choices=["single", "demo", "interactive", "goals"],
        help="Mode operasi: single/demo/interactive/goals"
    )
    parser.add_argument(
        "--model", type=str, default="qwen2.5:7b",
        help="Nama model Ollama"
    )
    parser.add_argument(
        "--ollama-url", type=str, default="http://localhost:11434",
        help="URL Ollama server"
    )
    parser.add_argument(
        "--max-steps", type=int, default=15,
        help="Maksimal langkah per task"
    )
    parser.add_argument(
        "--storage-dir", type=str, default="./agent_storage",
        help="Direktori penyimpanan"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose logging"
    )
    parser.add_argument(
        "--no-plan", action="store_true",
        help="Disable planning system"
    )
    parser.add_argument(
        "--no-reflect", action="store_true",
        help="Disable Reflexion"
    )

    args = parser.parse_args()

    # Setup config
    config = AgentConfig(
        model_name=args.model,
        ollama_url=args.ollama_url,
        max_iterations=args.max_steps,
        storage_dir=args.storage_dir,
        verbose=args.verbose,
        log_level="DEBUG" if args.verbose else "INFO",
        use_planning=not args.no_plan,
        use_reflection=not args.no_reflect,
    )

    # Initialize agent
    print("\n🤖 Initializing Autonomous Agent...")
    print(f"   Model: {config.model_name}")
    print(f"   Ollama: {config.ollama_url}")
    print(f"   Storage: {config.storage_dir}")

    agent = AutonomousAgent(config)

    try:
        if args.mode == "single":
            task = args.task or DEMO_TASKS[0]["task"]
            await demo_single_task(agent, task)

        elif args.mode == "demo":
            await demo_multiple_tasks(agent)

        elif args.mode == "interactive":
            await demo_interactive(agent)

        elif args.mode == "goals":
            await demo_goal_pursuit(agent)

        # Tampilkan performance summary di akhir
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY:")
        summary = agent.self_improvement.get_performance_summary()
        for key, val in summary.items():
            print(f"  {key}: {val}")

        suggestions = agent.self_improvement.get_improvement_suggestions()
        if suggestions:
            print("\nSaran Improvement:")
            for s in suggestions:
                print(f"  • {s}")

    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
