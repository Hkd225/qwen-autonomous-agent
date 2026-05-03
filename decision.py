"""
decision.py - Decision Making System
======================================
Sistem pengambilan keputusan agent menggunakan pola ReAct
dengan self-reflection (Reflexion) setelah setiap action.

Referensi Paper:
- ReAct (Yao et al., 2022): "ReAct: Synergizing Reasoning and Acting in Language Models"
  → Interleave Thought → Action → Observation loop
  → Thought menganalisis situasi, Action memilih tool, Observation memproses hasil
- Reflexion (Shinn et al., 2023): "Reflexion: Language Agents with Verbal Reinforcement Learning"
  → Setelah setiap action, agent merefleksikan apakah ini langkah yang benar
  → Reflection disimpan ke memory untuk mencegah kesalahan yang sama
  → "Verbal reinforcement": kata-kata sebagai gradient signal
- Chain-of-Thought (Wei et al., 2022): Explicit reasoning steps
  → Setiap Thought menggunakan CoT untuk reasoning yang lebih baik

Loop Utama:
  Perceive → Think (CoT) → Decide Action → Execute → Observe → Reflect → Update Memory
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Tipe keputusan yang bisa dibuat agent."""
    EXECUTE_TOOL = "execute_tool"       # Jalankan tool
    THINK = "think"                     # Reasoning/thinking saja
    CLARIFY = "clarify"                 # Minta klarifikasi
    FINISH = "finish"                   # Task selesai
    REPLAN = "replan"                   # Ubah plan
    WAIT = "wait"                       # Tunggu kondisi tertentu


@dataclass
class Thought:
    """
    Satu thought dalam reasoning chain.

    Terinspirasi dari ReAct's "Thought" component:
    "Thought: I need to search for information about..."
    """
    content: str                        # Isi pemikiran
    timestamp: float = field(default_factory=time.time)
    reasoning_type: str = "analysis"    # "analysis", "planning", "reflection", "verification"
    confidence: float = 0.7             # Seberapa yakin dengan thought ini
    leads_to: Optional[str] = None      # Action apa yang dihasilkan thought ini


@dataclass
class ActionDecision:
    """
    Keputusan untuk mengambil satu action.

    Berisi semua informasi untuk eksekusi action satu langkah.
    """
    decision_type: DecisionType
    thought: Thought                    # Reasoning yang mengarah ke keputusan ini
    action_name: str = ""               # Nama tool (jika EXECUTE_TOOL)
    action_input: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = ""          # Apa yang diharapkan terjadi
    confidence: float = 0.7            # Confidence dalam keputusan ini
    alternatives: List[str] = field(default_factory=list)  # Alternative yang dipertimbangkan
    urgency: float = 0.5               # 0=tidak urgent, 1=sangat urgent

    @property
    def is_final(self) -> bool:
        return self.decision_type == DecisionType.FINISH

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.decision_type.value,
            "action": self.action_name,
            "input": self.action_input,
            "thought": self.thought.content[:200],
            "confidence": round(self.confidence, 2),
        }


@dataclass
class Reflection:
    """
    Refleksi setelah satu action.

    Terinspirasi dari Reflexion paper: verbal self-evaluation
    yang digunakan sebagai "learning signal" untuk iterasi berikutnya.
    """
    action_taken: str                   # Action yang baru diambil
    outcome: str                        # Hasil aktual
    expected_outcome: str               # Hasil yang diharapkan
    was_successful: bool                # Berhasil atau tidak?
    lessons: List[str] = field(default_factory=list)  # Pelajaran yang dipetik
    what_went_well: List[str] = field(default_factory=list)
    what_went_wrong: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)  # Rekomendasi langkah berikutnya
    confidence_score: float = 0.7       # Confidence setelah refleksi
    timestamp: float = field(default_factory=time.time)

    def to_memory_entry(self) -> str:
        """Format refleksi untuk disimpan ke memory."""
        lines = [f"Refleksi - Action: {self.action_taken}"]
        lines.append(f"Sukses: {'Ya' if self.was_successful else 'Tidak'}")
        if self.lessons:
            lines.append(f"Pelajaran: {'; '.join(self.lessons)}")
        if self.what_went_wrong:
            lines.append(f"Masalah: {'; '.join(self.what_went_wrong)}")
        return "\n".join(lines)


class ReActDecisionMaker:
    """
    Decision maker menggunakan ReAct (Reasoning + Acting) pattern.

    Implementasi loop:
    1. OBSERVE: Kumpulkan konteks saat ini (task, memory, history)
    2. THINK: Generate reasoning chain (CoT)
    3. DECIDE: Pilih action berdasarkan reasoning
    4. [Execute action - dilakukan oleh agent loop]
    5. REFLECT: Evaluasi hasil action (Reflexion)
    6. UPDATE: Simpan learning ke memory

    Perbedaan dengan Reflexion pure:
    - ReAct fokus pada think-act cycle
    - Reflexion menambahkan verbal reinforcement setelah failure
    - Kita kombinasikan keduanya: think sebelum act, reflect setelah act
    """

    def __init__(self, llm_interface, tool_registry=None, memory_system=None):
        self.llm = llm_interface
        self.tools = tool_registry
        self.memory = memory_system

        # History untuk in-context learning
        self.thought_history: List[Thought] = []
        self.reflection_history: List[Reflection] = []
        self.decision_history: List[ActionDecision] = []

        logger.info("ReActDecisionMaker diinisialisasi")

    # ─────────────────────────────
    # Core ReAct Loop Components
    # ─────────────────────────────

    async def think(
        self,
        task: str,
        current_context: str,
        observations: List[str],
        step_count: int = 0,
    ) -> Thought:
        """
        Generate thought menggunakan Chain-of-Thought reasoning.

        Komponen think dalam ReAct:
        "Thought: I need to first search for X, then calculate Y based on the results."
        """
        # Bangun konteks untuk thinking
        obs_text = "\n".join([f"- {obs}" for obs in observations[-5:]])

        # Tambahkan refleksi terbaru sebagai context
        recent_reflections = ""
        if self.reflection_history:
            latest = self.reflection_history[-2:]
            ref_parts = []
            for r in latest:
                if r.lessons:
                    ref_parts.append(f"Pelajaran sebelumnya: {r.lessons[0]}")
            recent_reflections = "\n".join(ref_parts)

        # Tools context
        tools_text = ""
        if self.tools:
            tools_text = f"Tools tersedia: {', '.join(self.tools.tool_names)}"

        think_prompt = f"""Kamu adalah agen AI yang menggunakan Chain-of-Thought reasoning.
Analisis situasi saat ini dan putuskan langkah terbaik berikutnya.

Task: {task}

Observasi terbaru:
{obs_text if obs_text else "Belum ada observasi"}

{tools_text}

{f"Catatan dari refleksi sebelumnya:{chr(10)}{recent_reflections}" if recent_reflections else ""}

Konteks tambahan:
{current_context[:500] if current_context else "Tidak ada konteks tambahan"}

Langkah ke-{step_count + 1}

Lakukan reasoning dengan format:
{{
    "analysis": "analisis situasi saat ini",
    "key_information_known": "informasi penting yang sudah kita ketahui",
    "information_gaps": "apa yang masih perlu kita ketahui",
    "reasoning": "mengapa langkah berikutnya ini yang terbaik",
    "next_step": "apa yang harus dilakukan selanjutnya",
    "confidence": 0.0-1.0
}}"""

        try:
            response = await self.llm.generate_json(
                [{"role": "user", "content": think_prompt}],
                temperature=0.5,
            )

            thought_content = (
                f"Analisis: {response.get('analysis', '')}\n"
                f"Langkah selanjutnya: {response.get('next_step', '')}"
            )

            thought = Thought(
                content=thought_content,
                reasoning_type="analysis",
                confidence=float(response.get("confidence", 0.7)),
                leads_to=response.get("next_step", ""),
            )

            self.thought_history.append(thought)
            logger.debug(
                f"[Think] Thought generated (confidence={thought.confidence:.2f})"
            )
            return thought

        except Exception as e:
            logger.error(f"Think gagal: {e}")
            thought = Thought(
                content=f"Perlu menganalisis task: {task[:100]}",
                confidence=0.5,
            )
            self.thought_history.append(thought)
            return thought

    async def decide(
        self,
        task: str,
        thought: Thought,
        current_context: str = "",
        force_finish: bool = False,
    ) -> ActionDecision:
        """
        Buat keputusan action berdasarkan thought.

        Pilihan action:
        - Jalankan tool
        - Berikan final answer
        - Minta klarifikasi
        - Re-plan
        """
        if force_finish:
            return ActionDecision(
                decision_type=DecisionType.FINISH,
                thought=thought,
                action_name="",
                expected_outcome="Task selesai",
                confidence=1.0,
            )

        # Tentukan tools yang tersedia
        available_tools = self.tools.tool_names if self.tools else []
        tools_json = json.dumps(available_tools)

        decide_prompt = f"""Berdasarkan reasoning ini, tentukan action berikutnya.

Reasoning:
{thought.content}

Task: {task}

Tools tersedia: {tools_json}

Putuskan action dengan format JSON:
{{
    "decision_type": "execute_tool" | "think" | "clarify" | "finish",
    "action_name": "nama_tool (kosong jika bukan execute_tool)",
    "action_input": {{}},
    "expected_outcome": "apa yang akan terjadi",
    "reasoning": "mengapa action ini dipilih",
    "confidence": 0.0-1.0,
    "is_final_answer": false,
    "final_answer": "jawaban final jika decision_type=finish"
}}

PENTING: Jika task sudah selesai atau kamu sudah punya jawaban lengkap, gunakan decision_type="finish"."""

        try:
            response = await self.llm.generate_json(
                [{"role": "user", "content": decide_prompt}],
                temperature=0.3,
            )

            # Parse decision type
            decision_type_str = response.get("decision_type", "think")
            try:
                decision_type = DecisionType(decision_type_str)
            except ValueError:
                decision_type = DecisionType.THINK

            # Override ke FINISH jika indicated
            if response.get("is_final_answer", False):
                decision_type = DecisionType.FINISH

            decision = ActionDecision(
                decision_type=decision_type,
                thought=thought,
                action_name=response.get("action_name", ""),
                action_input=response.get("action_input", {}),
                expected_outcome=response.get("expected_outcome", ""),
                confidence=float(response.get("confidence", 0.6)),
            )

            # Jika FINISH, simpan final answer di action_input
            if decision_type == DecisionType.FINISH:
                decision.action_input["final_answer"] = response.get("final_answer", "")

            self.decision_history.append(decision)
            logger.info(
                f"[Decide] {decision_type.value} → {decision.action_name or 'N/A'} "
                f"(confidence={decision.confidence:.2f})"
            )
            return decision

        except Exception as e:
            logger.error(f"Decide gagal: {e}")
            return ActionDecision(
                decision_type=DecisionType.THINK,
                thought=thought,
                confidence=0.4,
            )

    async def reflect(
        self,
        action_taken: str,
        action_input: Dict[str, Any],
        actual_outcome: str,
        expected_outcome: str,
        was_successful: bool,
        task: str = "",
    ) -> Reflection:
        """
        Buat refleksi setelah action dieksekusi.

        Terinspirasi dari Reflexion paper: verbal self-evaluation
        yang digunakan untuk belajar dari pengalaman.

        Args:
            action_taken: Nama action/tool yang dieksekusi
            action_input: Input yang diberikan
            actual_outcome: Hasil aktual
            expected_outcome: Hasil yang diharapkan
            was_successful: Apakah berhasil?
            task: Context task keseluruhan

        Returns:
            Reflection dengan lessons learned
        """
        reflect_prompt = f"""Evaluasi action yang baru saja dilakukan.

Action: {action_taken}
Input: {json.dumps(action_input, ensure_ascii=False)[:200]}
Expected outcome: {expected_outcome[:200]}
Actual outcome: {actual_outcome[:300]}
Berhasil: {'Ya' if was_successful else 'Tidak'}
Task konteks: {task[:200] if task else 'N/A'}

Lakukan refleksi dalam JSON:
{{
    "what_went_well": ["hal-hal yang berjalan baik"],
    "what_went_wrong": ["hal-hal yang tidak berjalan sesuai harapan"],
    "lessons_learned": ["pelajaran konkrit untuk tidak mengulangi kesalahan"],
    "next_steps": ["rekomendasi langkah selanjutnya"],
    "confidence_score": 0.0-1.0,
    "should_replan": false
}}"""

        try:
            response = await self.llm.generate_json(
                [{"role": "user", "content": reflect_prompt}],
                temperature=0.4,
            )

            reflection = Reflection(
                action_taken=action_taken,
                outcome=actual_outcome[:500],
                expected_outcome=expected_outcome,
                was_successful=was_successful,
                lessons=response.get("lessons_learned", []),
                what_went_well=response.get("what_went_well", []),
                what_went_wrong=response.get("what_went_wrong", []),
                next_steps=response.get("next_steps", []),
                confidence_score=float(response.get("confidence_score", 0.5)),
            )

            self.reflection_history.append(reflection)

            # Simpan ke memory jika ada
            if self.memory and not was_successful and reflection.lessons:
                try:
                    from memory.episodic_memory import Episode
                    episode = Episode(
                        content=reflection.to_memory_entry(),
                        episode_type="reflection",
                        importance=7.0,  # Failures penting untuk diingat
                        task_context=task[:200],
                        outcome="failure",
                    )
                    await self.memory.store(episode)
                except Exception as e:
                    logger.debug(f"Gagal simpan reflection ke memory: {e}")

            logger.info(
                f"[Reflect] {'✓' if was_successful else '✗'} "
                f"action={action_taken}, "
                f"lessons={len(reflection.lessons)}"
            )
            return reflection

        except Exception as e:
            logger.error(f"Reflect gagal: {e}")
            return Reflection(
                action_taken=action_taken,
                outcome=actual_outcome[:300],
                expected_outcome=expected_outcome,
                was_successful=was_successful,
            )

    # ─────────────────────────────
    # Utility Methods
    # ─────────────────────────────

    def parse_react_output(self, raw_output: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse output LLM dalam format ReAct.

        Format:
        Thought: ...
        Action: tool_name
        Action Input: {...}

        Returns:
            (thought, action, action_input)
        """
        thought = None
        action = None
        action_input = None

        # Extract Thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)", raw_output, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract Action
        action_match = re.search(r"Action:\s*(.+?)(?=Action Input:|$)", raw_output, re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()

        # Extract Action Input
        input_match = re.search(r"Action Input:\s*(.+?)(?=Observation:|$)", raw_output, re.DOTALL)
        if input_match:
            input_str = input_match.group(1).strip()
            try:
                action_input = json.loads(input_str)
            except json.JSONDecodeError:
                action_input = {"query": input_str}

        # Final Answer
        final_match = re.search(r"Final Answer:\s*(.+)", raw_output, re.DOTALL)
        if final_match:
            action = "FINISH"
            action_input = {"answer": final_match.group(1).strip()}

        return thought, action, action_input

    def should_finish(
        self,
        task: str,
        observations: List[str],
        step_count: int,
        max_steps: int = 15,
    ) -> Tuple[bool, str]:
        """
        Evaluasi apakah task sudah selesai.

        Returns:
            (should_finish: bool, reason: str)
        """
        # Batas iterasi
        if step_count >= max_steps:
            return True, f"Batas maksimal {max_steps} langkah tercapai"

        # Cek apakah ada decision FINISH dalam history
        if self.decision_history:
            last = self.decision_history[-1]
            if last.decision_type == DecisionType.FINISH:
                return True, "Decision FINISH diberikan"

        return False, ""

    def get_stats(self) -> Dict[str, Any]:
        """Statistik decision making."""
        success_reflections = sum(1 for r in self.reflection_history if r.was_successful)
        total_reflections = len(self.reflection_history)

        decision_types: Dict[str, int] = {}
        for d in self.decision_history:
            key = d.decision_type.value
            decision_types[key] = decision_types.get(key, 0) + 1

        return {
            "total_thoughts": len(self.thought_history),
            "total_decisions": len(self.decision_history),
            "total_reflections": total_reflections,
            "success_rate": (
                round(success_reflections / max(1, total_reflections) * 100, 1)
            ),
            "decision_breakdown": decision_types,
            "avg_confidence": (
                round(sum(d.confidence for d in self.decision_history) /
                      max(1, len(self.decision_history)), 3)
            ),
        }
