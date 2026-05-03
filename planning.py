"""
planning.py - Hierarchical Planning System
===========================================
Sistem perencanaan untuk agent menggunakan pendekatan Tree of Thoughts
dan Hierarchical Task Network (HTN) inspired planning.

Referensi Paper:
- Tree of Thoughts (Yao et al., 2023): "Tree of Thoughts: Deliberate Problem Solving with LLMs"
  → Explore multiple reasoning paths secara parallel
  → Evaluate dan select path terbaik
  → Backtrack jika path gagal
- HTN Planning: Hierarchical Task Network
  → Decompose high-level task ke primitive tasks
  → Tasks punya preconditions dan effects
- AutoGPT / BabyAGI: Task management dan self-directed planning
  → Agent membuat dan memodifikasi planning-nya sendiri
- ReAct (Yao et al., 2022): Interleave reasoning dan acting
  → Plan diupdate setiap step berdasarkan observasi baru
- Voyager (Wang et al., 2023): Curriculum & plan management
  → Plans disimpan sebagai reusable templates

Komponen Utama:
- PlanNode: Satu langkah dalam plan
- Plan: Kumpulan PlanNodes yang terorganisir
- PlanGenerator: Generate plan baru dari task description
- PlanValidator: Validasi feasibility plan
- PlanExecutionTracker: Track progress plan
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PlanNodeStatus(Enum):
    """Status satu langkah dalam plan."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanNode:
    """
    Satu node/langkah dalam execution plan.

    Bisa merepresentasikan:
    - Action: Eksekusi tool spesifik
    - Decision: Putuskan path berdasarkan kondisi
    - Checkpoint: Verifikasi progress
    - Sub-plan: Nested plan untuk task kompleks
    """
    description: str                        # Deskripsi langkah
    action: str = ""                        # Tool/action yang akan dieksekusi
    action_input: Dict[str, Any] = field(default_factory=dict)
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: PlanNodeStatus = PlanNodeStatus.PENDING
    depends_on: List[str] = field(default_factory=list)
    expected_output: str = ""               # Apa yang diharapkan dari langkah ini
    actual_output: str = ""                 # Hasil aktual
    priority: int = 1                       # Prioritas eksekusi
    is_optional: bool = False               # Apakah langkah ini opsional?
    retry_count: int = 0
    max_retries: int = 2
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mark_completed(self, output: str = ""):
        self.status = PlanNodeStatus.COMPLETED
        self.actual_output = output

    def mark_failed(self, reason: str = ""):
        self.status = PlanNodeStatus.FAILED
        self.metadata["failure_reason"] = reason
        self.retry_count += 1

    @property
    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries

    @property
    def is_ready(self) -> bool:
        """Apakah node siap dieksekusi (semua deps sudah selesai)?"""
        return self.status == PlanNodeStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "description": self.description,
            "action": self.action,
            "action_input": self.action_input,
            "status": self.status.value,
            "depends_on": self.depends_on,
            "expected_output": self.expected_output,
            "priority": self.priority,
            "is_optional": self.is_optional,
        }


@dataclass
class Plan:
    """
    Plan eksekusi lengkap untuk sebuah task.

    Berisi urutan PlanNodes yang harus dieksekusi,
    beserta metadata dan tracking progress.
    """
    task: str                               # Task yang direncanakan
    nodes: List[PlanNode] = field(default_factory=list)
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: float = field(default_factory=time.time)
    feasibility_score: float = 0.7          # Estimasi kemungkinan berhasil (0-1)
    estimated_steps: int = 0
    rationale: str = ""                     # Alasan mengapa plan ini dibuat
    alternative_plans: List[str] = field(default_factory=list)  # ID alternatif

    @property
    def current_node(self) -> Optional[PlanNode]:
        """Node yang sedang dikerjakan atau berikutnya."""
        for node in self.nodes:
            if node.status == PlanNodeStatus.IN_PROGRESS:
                return node
        for node in self.nodes:
            if node.status == PlanNodeStatus.PENDING:
                return node
        return None

    @property
    def completed_nodes(self) -> List[PlanNode]:
        return [n for n in self.nodes if n.status == PlanNodeStatus.COMPLETED]

    @property
    def failed_nodes(self) -> List[PlanNode]:
        return [n for n in self.nodes if n.status == PlanNodeStatus.FAILED]

    @property
    def progress(self) -> float:
        """Progress plan (0.0 - 1.0)."""
        if not self.nodes:
            return 0.0
        mandatory = [n for n in self.nodes if not n.is_optional]
        if not mandatory:
            return 1.0
        done = sum(1 for n in mandatory if n.status == PlanNodeStatus.COMPLETED)
        return done / len(mandatory)

    @property
    def is_complete(self) -> bool:
        mandatory = [n for n in self.nodes if not n.is_optional]
        return all(n.status == PlanNodeStatus.COMPLETED for n in mandatory)

    @property
    def has_failures(self) -> bool:
        return any(n.status == PlanNodeStatus.FAILED for n in self.nodes)

    def format(self) -> str:
        """Format plan sebagai teks yang mudah dibaca."""
        lines = [
            f"Plan: {self.task}",
            f"ID: {self.plan_id}",
            f"Feasibility: {self.feasibility_score:.0%}",
            f"Progress: {self.progress:.0%}",
            f"Steps ({len(self.nodes)}):",
        ]
        for i, node in enumerate(self.nodes, 1):
            status_symbols = {
                PlanNodeStatus.PENDING: "○",
                PlanNodeStatus.IN_PROGRESS: "◉",
                PlanNodeStatus.COMPLETED: "✓",
                PlanNodeStatus.FAILED: "✗",
                PlanNodeStatus.SKIPPED: "→",
            }
            symbol = status_symbols.get(node.status, "?")
            optional = " [opsional]" if node.is_optional else ""
            lines.append(
                f"  {i}. {symbol} {node.description}{optional}"
                f"{f' → {node.action}' if node.action else ''}"
            )
        return "\n".join(lines)


class PlanGenerator:
    """
    Generate plans menggunakan LLM dengan Tree of Thoughts approach.

    Proses:
    1. Analisis task dan buat beberapa candidate plans (breadth)
    2. Evaluasi setiap plan berdasarkan feasibility dan completeness
    3. Pilih plan terbaik atau gabungkan elemen terbaik dari beberapa plan

    Terinspirasi dari Tree of Thoughts: explore multiple reasoning paths
    sebelum commit ke satu path.
    """

    def __init__(self, llm_interface, tool_registry=None):
        self.llm = llm_interface
        self.tools = tool_registry

    async def generate(
        self,
        task: str,
        context: str = "",
        constraints: Optional[List[str]] = None,
        n_candidates: int = 3,
    ) -> Plan:
        """
        Generate plan terbaik untuk task.

        Args:
            task: Deskripsi task
            context: Informasi tambahan (state saat ini, history, dll)
            constraints: Batasan-batasan yang harus dipatuhi
            n_candidates: Jumlah candidate plans yang digenerate

        Returns:
            Plan terbaik
        """
        logger.info(f"[PlanGenerator] Membuat plan untuk: {task[:100]}")

        # Dapatkan daftar tools yang tersedia
        tools_desc = ""
        if self.tools:
            tool_names = self.tools.tool_names
            tools_desc = f"Tools tersedia: {', '.join(tool_names)}"

        constraints_text = ""
        if constraints:
            constraints_text = "\nBatasan:\n" + "\n".join(f"- {c}" for c in constraints)

        # Generate plan dengan LLM
        plan_prompt = f"""Buat rencana eksekusi step-by-step untuk menyelesaikan task berikut.

Task: {task}
Konteks: {context[:500] if context else "Tidak ada konteks tambahan"}
{tools_desc}
{constraints_text}

Buat plan yang:
1. Praktis dan bisa dieksekusi
2. Urut secara logis (prerequisites sebelum dependents)
3. Setiap langkah jelas dan actionable
4. Estimasi feasibility (0-1)

Respond dengan JSON:
{{
    "plan_rationale": "alasan mengapa plan ini dipilih",
    "feasibility_score": 0.0-1.0,
    "steps": [
        {{
            "step_num": 1,
            "description": "deskripsi langkah",
            "action": "nama tool (kosong jika reasoning/thinking)",
            "action_input": {{}},
            "expected_output": "apa yang diharapkan dari langkah ini",
            "is_optional": false,
            "depends_on_steps": []
        }}
    ]
}}"""

        try:
            response = await self.llm.generate_json(
                [{"role": "user", "content": plan_prompt}],
                temperature=0.7,
            )

            plan = self._parse_plan_response(task, response)
            logger.info(
                f"[PlanGenerator] Plan dibuat: {len(plan.nodes)} langkah, "
                f"feasibility={plan.feasibility_score:.2f}"
            )
            return plan

        except Exception as e:
            logger.error(f"Gagal generate plan dengan LLM: {e}")
            return self._create_fallback_plan(task)

    def _parse_plan_response(self, task: str, response: Dict[str, Any]) -> Plan:
        """Parse response LLM menjadi Plan object."""
        nodes = []
        for step_data in response.get("steps", []):
            node = PlanNode(
                description=step_data.get("description", ""),
                action=step_data.get("action", ""),
                action_input=step_data.get("action_input", {}),
                expected_output=step_data.get("expected_output", ""),
                is_optional=step_data.get("is_optional", False),
                priority=len(nodes) + 1,
            )
            # Map depends_on_steps (nomor step) ke node IDs
            # Kita akan resolve ini setelah semua nodes dibuat
            node.metadata["depends_on_steps"] = step_data.get("depends_on_steps", [])
            nodes.append(node)

        # Resolve dependencies
        for i, node in enumerate(nodes):
            dep_steps = node.metadata.get("depends_on_steps", [])
            for step_num in dep_steps:
                idx = step_num - 1
                if 0 <= idx < len(nodes) and idx != i:
                    node.depends_on.append(nodes[idx].node_id)

        return Plan(
            task=task,
            nodes=nodes,
            feasibility_score=float(response.get("feasibility_score", 0.7)),
            estimated_steps=len(nodes),
            rationale=response.get("plan_rationale", ""),
        )

    def _create_fallback_plan(self, task: str) -> Plan:
        """Buat plan minimal jika LLM gagal."""
        return Plan(
            task=task,
            nodes=[
                PlanNode(
                    description=f"Analisis task: {task[:100]}",
                    action="",
                    expected_output="Pemahaman task",
                ),
                PlanNode(
                    description="Eksekusi task berdasarkan analisis",
                    action="",
                    expected_output="Hasil task",
                ),
            ],
            feasibility_score=0.5,
            rationale="Fallback plan karena LLM tidak tersedia",
        )


class PlanValidator:
    """
    Validasi kelayakan sebuah plan sebelum dieksekusi.

    Cek:
    - Apakah semua tools yang dibutuhkan tersedia?
    - Apakah urutan langkah logis?
    - Apakah ada circular dependencies?
    - Apakah plan terlalu kompleks/sederhana?
    """

    def __init__(self, tool_registry=None):
        self.tools = tool_registry

    def validate(self, plan: Plan) -> Tuple[bool, List[str]]:
        """
        Validasi plan.

        Returns:
            (is_valid: bool, issues: List[str])
        """
        issues = []

        # Cek plan tidak kosong
        if not plan.nodes:
            issues.append("Plan kosong - tidak ada langkah")
            return False, issues

        # Cek tools tersedia
        if self.tools:
            for node in plan.nodes:
                if node.action and node.action not in self.tools.tool_names:
                    if not node.is_optional:
                        issues.append(
                            f"Tool '{node.action}' tidak tersedia "
                            f"(dibutuhkan oleh: '{node.description}')"
                        )

        # Cek circular dependencies
        circular = self._check_circular_deps(plan)
        if circular:
            issues.append(f"Circular dependency terdeteksi: {circular}")

        # Cek plan terlalu panjang
        if len(plan.nodes) > 20:
            issues.append(f"Plan terlalu panjang ({len(plan.nodes)} langkah), "
                         "pertimbangkan untuk memecah task")

        # Cek feasibility score
        if plan.feasibility_score < 0.3:
            issues.append(
                f"Feasibility score rendah ({plan.feasibility_score:.2f}), "
                "task mungkin tidak bisa diselesaikan"
            )

        is_valid = len([i for i in issues if "tidak tersedia" in i]) == 0
        return is_valid, issues

    def _check_circular_deps(self, plan: Plan) -> Optional[str]:
        """Deteksi circular dependency menggunakan DFS."""
        id_to_idx = {n.node_id: i for i, n in enumerate(plan.nodes)}
        visited = set()
        in_stack = set()

        def dfs(node_id: str) -> Optional[str]:
            if node_id in in_stack:
                return node_id
            if node_id in visited:
                return None

            visited.add(node_id)
            in_stack.add(node_id)

            idx = id_to_idx.get(node_id)
            if idx is not None:
                for dep_id in plan.nodes[idx].depends_on:
                    result = dfs(dep_id)
                    if result:
                        return result

            in_stack.discard(node_id)
            return None

        for node in plan.nodes:
            result = dfs(node.node_id)
            if result:
                return result
        return None


class ReplanningStrategy(Enum):
    """Strategi re-planning ketika eksekusi gagal."""
    RETRY = "retry"             # Coba langkah yang sama lagi
    SKIP = "skip"               # Skip langkah yang gagal
    REPLAN = "replan"           # Generate plan baru
    FALLBACK = "fallback"       # Gunakan plan alternatif
    ABORT = "abort"             # Hentikan eksekusi


class PlanningSystem:
    """
    Main Planning System yang mengintegrasikan semua komponen.

    Mengelola full planning lifecycle:
    1. Generate plan dari task
    2. Validate plan
    3. Track execution
    4. Handle failures dengan re-planning
    """

    def __init__(self, llm_interface, tool_registry=None):
        self.llm = llm_interface
        self.generator = PlanGenerator(llm_interface, tool_registry)
        self.validator = PlanValidator(tool_registry)
        self.current_plan: Optional[Plan] = None
        self.plan_history: List[Plan] = []

    async def create_plan(
        self,
        task: str,
        context: str = "",
        constraints: Optional[List[str]] = None,
    ) -> Plan:
        """
        Buat dan validasi plan untuk task.

        Flow:
        1. Generate plan
        2. Validate
        3. Perbaiki jika ada issues kecil
        4. Simpan sebagai current plan
        """
        plan = await self.generator.generate(task, context, constraints)

        is_valid, issues = self.validator.validate(plan)
        if issues:
            logger.warning(f"[Planning] Issues ditemukan: {issues}")

        self.current_plan = plan
        self.plan_history.append(plan)

        logger.info(
            f"[Planning] Plan siap: {len(plan.nodes)} langkah, "
            f"valid={is_valid}, "
            f"feasibility={plan.feasibility_score:.2f}"
        )
        return plan

    async def handle_failure(
        self,
        failed_node: PlanNode,
        failure_reason: str,
        task: str,
        context: str = "",
    ) -> Tuple[ReplanningStrategy, Optional[Plan]]:
        """
        Handle kegagalan satu langkah dalam plan.

        Putuskan strategi re-planning berdasarkan:
        - Apakah bisa di-retry?
        - Seberapa kritis langkah ini?
        - Apakah ada alternatif?

        Returns:
            (strategy, new_plan_if_replanning)
        """
        # Jika bisa retry, coba lagi
        if failed_node.can_retry and not failed_node.is_optional:
            logger.info(
                f"[Planning] Retry langkah: {failed_node.description} "
                f"(attempt {failed_node.retry_count + 1}/{failed_node.max_retries})"
            )
            return ReplanningStrategy.RETRY, None

        # Jika opsional, skip
        if failed_node.is_optional:
            logger.info(f"[Planning] Skip langkah opsional: {failed_node.description}")
            return ReplanningStrategy.SKIP, None

        # Jika critical dan tidak bisa retry, replan
        logger.info(
            f"[Planning] Replan diperlukan karena kegagalan: {failure_reason}"
        )
        new_context = (
            f"{context}\n\n"
            f"CATATAN: Langkah '{failed_node.description}' gagal karena: {failure_reason}. "
            "Hindari pendekatan yang sama dalam plan baru."
        )
        new_plan = await self.create_plan(task, new_context)
        return ReplanningStrategy.REPLAN, new_plan

    def get_next_executable_node(self) -> Optional[PlanNode]:
        """Dapatkan node berikutnya yang siap dieksekusi."""
        if not self.current_plan:
            return None

        completed_ids = {
            n.node_id for n in self.current_plan.nodes
            if n.status == PlanNodeStatus.COMPLETED
        }

        for node in self.current_plan.nodes:
            if node.status != PlanNodeStatus.PENDING:
                continue
            # Cek semua dependencies sudah selesai
            if all(dep_id in completed_ids for dep_id in node.depends_on):
                return node

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Statistik planning system."""
        return {
            "plans_created": len(self.plan_history),
            "current_plan": {
                "task": self.current_plan.task[:50] if self.current_plan else None,
                "progress": self.current_plan.progress if self.current_plan else 0,
                "nodes": len(self.current_plan.nodes) if self.current_plan else 0,
            } if self.current_plan else None,
        }
