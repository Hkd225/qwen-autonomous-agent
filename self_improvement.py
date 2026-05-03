"""
self_improvement.py - Self-Improvement Module
==============================================
Modul untuk agent meningkatkan kemampuannya sendiri dari pengalaman.
Menyimpan skill sukses, menganalisis kegagalan, dan mengoptimasi prompts.

Referensi Paper:
- Voyager (Wang et al., 2023): "Voyager: An Open-Ended Embodied Agent with Large Language Models"
  → Skill library: simpan successful action sequences sebagai reusable skills
  → Automatic curriculum: tingkatkan difficulty berdasarkan skill yang sudah dikuasai
  → Self-verification: agent verifikasi sendiri apakah skill berhasil
- Reflexion (Shinn et al., 2023): Verbal reinforcement learning
  → Simpan "lessons learned" dari failures
  → Gunakan reflections di future episodes sebagai in-context learning
- SELF-RAG (Asai et al., 2023): Self-reflective retrieval
  → Model belajar kapan dan bagaimana melakukan retrieval
- Self-Play / Constitutional AI: Iterative self-improvement
  → Agent mengkritik dan memperbaiki outputnya sendiri

Komponen:
1. Skill Library: Simpan successful action sequences
2. Failure Analysis: Analisis pola kegagalan
3. Prompt Optimizer: Perbaiki prompts berdasarkan performance
4. Performance Tracker: Track metrics dan progress
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """
    Satu skill dalam skill library.

    Terinspirasi dari Voyager: skill = sekuens actions yang sukses
    untuk menyelesaikan jenis task tertentu.

    Skills bisa di-reuse ketika agent menemukan task yang mirip.
    """
    name: str                           # Nama skill
    description: str                    # Apa yang dilakukan skill ini
    task_type: str                      # Kategori task yang diselesaikan
    action_sequence: List[Dict[str, Any]]  # Urutan actions yang berhasil
    success_count: int = 1              # Berapa kali berhasil digunakan
    failure_count: int = 0              # Berapa kali gagal
    skill_id: str = ""                  # ID unik
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)  # Skills yang dibutuhkan
    average_duration_ms: float = 0.0    # Rata-rata durasi eksekusi
    context_requirements: str = ""      # Konteks yang dibutuhkan untuk skill ini

    def __post_init__(self):
        if not self.skill_id:
            import hashlib
            self.skill_id = "skill_" + hashlib.md5(
                f"{self.name}{self.created_at}".encode()
            ).hexdigest()[:8]

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / max(1, total)

    @property
    def reliability_score(self) -> float:
        """
        Score keandalan berdasarkan success rate dan usage count.

        Skills yang sering berhasil mendapat score lebih tinggi.
        """
        usage_bonus = min(0.2, self.success_count * 0.02)
        return min(1.0, self.success_rate + usage_bonus)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type,
            "action_sequence": self.action_sequence,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": round(self.success_rate, 3),
            "created_at": self.created_at,
            "last_used": self.last_used,
            "tags": self.tags,
        }


@dataclass
class PerformanceMetric:
    """Satu metric performa untuk satu task/session."""
    task: str
    success: bool
    steps_taken: int
    total_tokens: int
    duration_ms: float
    tools_used: List[str]
    timestamp: float = field(default_factory=time.time)
    error_count: int = 0
    skill_reuses: int = 0               # Berapa skill yang di-reuse
    notes: str = ""


class SkillLibrary:
    """
    Skill Library - Simpan dan Retrieve Successful Action Sequences.

    Terinspirasi dari Voyager yang menggunakan skills sebagai
    fundamental building block untuk curriculum learning.

    Skills disimpan ke JSON untuk persistence antar sessions.
    Retrieval menggunakan keyword matching (bisa ditingkatkan ke semantic search).
    """

    def __init__(self, storage_path: str = "./skill_library.json"):
        self.storage_path = storage_path
        self._skills: Dict[str, Skill] = {}  # skill_id → Skill
        self._load()
        logger.info(f"SkillLibrary diinisialisasi: {len(self._skills)} skills")

    def _load(self):
        """Load skills dari file JSON."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for skill_data in data.get("skills", []):
                    skill = Skill(**skill_data)
                    self._skills[skill.skill_id] = skill
        except Exception as e:
            logger.warning(f"Tidak bisa load skill library: {e}")

    def _save(self):
        """Simpan skills ke file JSON."""
        try:
            data = {
                "version": "1.0",
                "saved_at": time.time(),
                "skills": [s.to_dict() for s in self._skills.values()]
            }
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Gagal menyimpan skill library: {e}")

    def store(self, skill: Skill) -> str:
        """Simpan skill baru ke library."""
        # Cek apakah sudah ada skill serupa
        existing = self.find_similar(skill.name, skill.task_type)
        if existing:
            # Update skill yang sudah ada
            existing.success_count += 1
            existing.last_used = time.time()
            existing.action_sequence = skill.action_sequence  # Update dengan sequence terbaru
            self._save()
            logger.debug(f"[SkillLibrary] Updated: {existing.name}")
            return existing.skill_id

        self._skills[skill.skill_id] = skill
        self._save()
        logger.info(f"[SkillLibrary] Stored new skill: {skill.name}")
        return skill.skill_id

    def retrieve(
        self,
        task_description: str,
        task_type: str = "",
        min_success_rate: float = 0.6,
        top_k: int = 3,
    ) -> List[Skill]:
        """
        Retrieve skills relevan untuk task.

        Gunakan skill matching berdasarkan:
        1. Task type (exact match)
        2. Keyword similarity
        3. Success rate filter
        """
        candidates = []

        for skill in self._skills.values():
            if skill.success_rate < min_success_rate:
                continue

            score = 0.0

            # Task type match
            if task_type and skill.task_type == task_type:
                score += 0.4

            # Keyword matching
            task_words = set(task_description.lower().split())
            skill_words = set(
                f"{skill.name} {skill.description}".lower().split()
            )
            overlap = len(task_words & skill_words) / max(len(task_words), 1)
            score += overlap * 0.4

            # Reliability bonus
            score += skill.reliability_score * 0.2

            if score > 0.2:
                candidates.append((skill, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [skill for skill, _ in candidates[:top_k]]

    def find_similar(self, name: str, task_type: str) -> Optional[Skill]:
        """Cari skill yang mirip berdasarkan nama dan type."""
        name_lower = name.lower()
        for skill in self._skills.values():
            if skill.task_type == task_type and \
               (skill.name.lower() == name_lower or
                name_lower in skill.name.lower()):
                return skill
        return None

    def update_usage(self, skill_id: str, success: bool):
        """Update statistik penggunaan skill."""
        if skill_id in self._skills:
            skill = self._skills[skill_id]
            skill.last_used = time.time()
            if success:
                skill.success_count += 1
            else:
                skill.failure_count += 1
            self._save()

    def get_top_skills(self, n: int = 10) -> List[Skill]:
        """Dapatkan top N skills berdasarkan reliability score."""
        sorted_skills = sorted(
            self._skills.values(),
            key=lambda s: s.reliability_score,
            reverse=True,
        )
        return sorted_skills[:n]

    @property
    def size(self) -> int:
        return len(self._skills)

    def summary(self) -> str:
        """Ringkasan skill library."""
        if not self._skills:
            return "Skill library kosong"

        lines = [f"Skill Library ({len(self._skills)} skills):"]
        for skill in self.get_top_skills(5):
            lines.append(
                f"  • {skill.name} ({skill.task_type}) "
                f"- success rate: {skill.success_rate:.0%}"
            )
        return "\n".join(lines)


class SelfImprovementSystem:
    """
    Self-Improvement System - Agent Belajar dari Pengalaman.

    Mengintegrasikan semua komponen self-improvement:
    1. Skill extraction dari successful runs
    2. Failure pattern analysis
    3. Performance tracking
    4. Prompt optimization (menggunakan LLM untuk revisi prompts)

    Terinspirasi dari Voyager dan Reflexion sebagai sistem
    yang memungkinkan agent "belajar" antar sessions.
    """

    def __init__(
        self,
        llm_interface=None,
        storage_dir: str = "./agent_storage",
    ):
        self.llm = llm_interface
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

        self.skill_library = SkillLibrary(
            storage_path=os.path.join(storage_dir, "skills.json")
        )
        self.performance_history: List[PerformanceMetric] = []
        self._load_performance_history()

        logger.info("SelfImprovementSystem diinisialisasi")

    def _load_performance_history(self):
        """Load performance history dari file."""
        path = os.path.join(self.storage_dir, "performance.json")
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                self.performance_history = [
                    PerformanceMetric(**item)
                    for item in data.get("history", [])
                ]
        except Exception as e:
            logger.warning(f"Tidak bisa load performance history: {e}")

    def _save_performance_history(self):
        """Simpan performance history ke file."""
        path = os.path.join(self.storage_dir, "performance.json")
        try:
            data = {
                "history": [
                    {k: v for k, v in asdict(m).items()}
                    for m in self.performance_history[-100:]  # Simpan 100 terakhir
                ]
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Gagal simpan performance history: {e}")

    async def extract_skill(
        self,
        task: str,
        task_type: str,
        action_sequence: List[Dict[str, Any]],
        was_successful: bool,
    ) -> Optional[Skill]:
        """
        Extract skill dari successful action sequence.

        Terinspirasi dari Voyager: setelah task berhasil, ekstrak
        "skill" yang bisa di-reuse untuk task serupa.

        Args:
            task: Deskripsi task yang diselesaikan
            task_type: Kategori task
            action_sequence: Urutan actions yang diambil
            was_successful: Apakah berhasil?

        Returns:
            Skill object atau None jika tidak layak disimpan
        """
        if not was_successful or not action_sequence:
            return None

        # Gunakan LLM untuk generate skill description
        if self.llm:
            try:
                skill_prompt = f"""Berdasarkan task yang berhasil diselesaikan, buat deskripsi skill yang bisa di-reuse.

Task: {task}
Task type: {task_type}
Actions yang diambil:
{json.dumps(action_sequence[:5], ensure_ascii=False, indent=2)[:1000]}

Buat skill description dalam JSON:
{{
    "name": "nama skill yang deskriptif (< 30 karakter)",
    "description": "deskripsi lengkap apa yang dilakukan skill ini",
    "tags": ["tag1", "tag2"],
    "context_requirements": "konteks apa yang dibutuhkan untuk menggunakan skill ini"
}}"""

                response = await self.llm.generate_json(
                    [{"role": "user", "content": skill_prompt}],
                    temperature=0.3,
                )

                skill = Skill(
                    name=response.get("name", f"Skill_{task_type}"),
                    description=response.get("description", task),
                    task_type=task_type,
                    action_sequence=action_sequence,
                    tags=response.get("tags", []),
                    context_requirements=response.get("context_requirements", ""),
                )

                self.skill_library.store(skill)
                logger.info(f"[SelfImprovement] Skill diekstrak: {skill.name}")
                return skill

            except Exception as e:
                logger.error(f"Gagal ekstrak skill dengan LLM: {e}")

        # Fallback: buat skill sederhana
        skill = Skill(
            name=f"{task_type}_{int(time.time())}",
            description=task[:100],
            task_type=task_type,
            action_sequence=action_sequence,
        )
        self.skill_library.store(skill)
        return skill

    async def analyze_failures(
        self, failure_patterns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analisis pola kegagalan untuk mengidentifikasi improvement areas.

        Terinspirasi dari Reflexion: analisis failure patterns
        dan generate concrete suggestions untuk improvement.

        Args:
            failure_patterns: List of {task, action, error, context}

        Returns:
            Analysis dengan root causes dan suggestions
        """
        if not failure_patterns:
            return {"analysis": "Tidak ada failure patterns", "suggestions": []}

        if not self.llm:
            return {
                "analysis": f"{len(failure_patterns)} kegagalan terdeteksi",
                "suggestions": ["Tambahkan error handling yang lebih robust"],
            }

        patterns_text = json.dumps(failure_patterns[:10], ensure_ascii=False)[:2000]

        analysis_prompt = f"""Analisis pola kegagalan agent berikut dan berikan saran perbaikan.

Failure patterns:
{patterns_text}

Analisis dalam JSON:
{{
    "root_causes": ["penyebab utama kegagalan"],
    "common_patterns": ["pola yang sering muncul"],
    "suggestions": ["saran konkrit untuk perbaikan"],
    "priority_fixes": ["perbaikan yang paling penting (urutan prioritas)"]
}}"""

        try:
            response = await self.llm.generate_json(
                [{"role": "user", "content": analysis_prompt}],
                temperature=0.3,
            )
            logger.info(
                f"[SelfImprovement] Failure analysis selesai: "
                f"{len(response.get('suggestions', []))} suggestions"
            )
            return response
        except Exception as e:
            logger.error(f"Gagal analisis failures: {e}")
            return {"analysis": "Analisis gagal", "suggestions": []}

    async def optimize_prompt(
        self,
        current_prompt: str,
        prompt_name: str,
        performance_data: List[Dict[str, Any]],
    ) -> str:
        """
        Optimasi prompt berdasarkan performance data.

        Terinspirasi dari APE (Automatic Prompt Engineering) dan
        Constitutional AI: gunakan LLM untuk memperbaiki prompts.

        Args:
            current_prompt: Prompt yang ingin dioptimasi
            prompt_name: Nama/label prompt
            performance_data: Data performa dengan prompt ini

        Returns:
            Optimized prompt string
        """
        if not self.llm:
            return current_prompt

        success_rate = sum(1 for p in performance_data if p.get("success", False))
        total = len(performance_data)

        if total == 0:
            return current_prompt

        sr_pct = success_rate / total * 100
        logger.info(
            f"[SelfImprovement] Optimasi prompt '{prompt_name}' "
            f"(current success rate: {sr_pct:.0f}%)"
        )

        # Kumpulkan contoh failures
        failures = [p for p in performance_data if not p.get("success")][:5]
        failure_text = json.dumps(failures, ensure_ascii=False)[:1000]

        optimize_prompt = f"""Kamu adalah prompt engineer expert. Optimalkan prompt berikut berdasarkan performance data.

Prompt saat ini:
```
{current_prompt}
```

Performance:
- Success rate: {sr_pct:.0f}%
- Total tests: {total}

Contoh failures:
{failure_text}

Buat versi prompt yang lebih baik yang:
1. Memperbaiki area yang sering gagal
2. Lebih spesifik dan jelas
3. Mempertahankan intent asli
4. Menambahkan contoh jika perlu

Respond dengan JSON:
{{
    "optimized_prompt": "prompt yang sudah dioptimasi",
    "changes_made": ["list perubahan yang dilakukan"],
    "expected_improvement": "perkiraan peningkatan"
}}"""

        try:
            response = await self.llm.generate_json(
                [{"role": "user", "content": optimize_prompt}],
                temperature=0.5,
            )
            optimized = response.get("optimized_prompt", current_prompt)
            changes = response.get("changes_made", [])

            logger.info(
                f"[SelfImprovement] Prompt '{prompt_name}' dioptimasi, "
                f"{len(changes)} perubahan"
            )
            return optimized

        except Exception as e:
            logger.error(f"Gagal optimasi prompt: {e}")
            return current_prompt

    def record_performance(self, metric: PerformanceMetric):
        """Catat metric performa untuk satu task/session."""
        self.performance_history.append(metric)
        self._save_performance_history()

        logger.debug(
            f"[SelfImprovement] Performance recorded: "
            f"task='{metric.task[:50]}', "
            f"success={metric.success}, "
            f"steps={metric.steps_taken}"
        )

    def get_performance_summary(self, last_n: int = 20) -> Dict[str, Any]:
        """Ringkasan performa N session terakhir."""
        recent = self.performance_history[-last_n:]
        if not recent:
            return {"message": "Belum ada data performa"}

        success_count = sum(1 for m in recent if m.success)
        avg_steps = sum(m.steps_taken for m in recent) / len(recent)
        avg_tokens = sum(m.total_tokens for m in recent) / len(recent)
        avg_duration = sum(m.duration_ms for m in recent) / len(recent)

        # Tool usage frequency
        tool_freq: Dict[str, int] = {}
        for m in recent:
            for tool in m.tools_used:
                tool_freq[tool] = tool_freq.get(tool, 0) + 1

        return {
            "sessions_analyzed": len(recent),
            "success_rate": round(success_count / len(recent) * 100, 1),
            "avg_steps_per_task": round(avg_steps, 1),
            "avg_tokens_per_task": round(avg_tokens, 0),
            "avg_duration_sec": round(avg_duration / 1000, 1),
            "most_used_tools": sorted(
                tool_freq.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "skill_library_size": self.skill_library.size,
        }

    def get_improvement_suggestions(self) -> List[str]:
        """Generate saran improvement berdasarkan data historis."""
        if len(self.performance_history) < 5:
            return ["Kumpulkan lebih banyak data untuk analisis yang akurat"]

        suggestions = []
        recent = self.performance_history[-20:]

        # Cek success rate
        sr = sum(1 for m in recent if m.success) / len(recent)
        if sr < 0.5:
            suggestions.append(
                f"Success rate rendah ({sr:.0%}). "
                "Pertimbangkan untuk menyederhanakan task decomposition."
            )

        # Cek step count
        avg_steps = sum(m.steps_taken for m in recent) / len(recent)
        if avg_steps > 12:
            suggestions.append(
                f"Rata-rata {avg_steps:.0f} steps per task - terlalu banyak. "
                "Optimasi planning untuk menghasilkan plan yang lebih efisien."
            )

        # Cek error rate
        avg_errors = sum(m.error_count for m in recent) / len(recent)
        if avg_errors > 2:
            suggestions.append(
                f"Rata-rata {avg_errors:.1f} errors per task. "
                "Perbaiki error handling dan validasi input."
            )

        # Cek skill reuse
        avg_reuse = sum(m.skill_reuses for m in recent) / len(recent)
        if avg_reuse < 0.5 and self.skill_library.size > 3:
            suggestions.append(
                "Skill reuse rendah meski skill library sudah terisi. "
                "Tingkatkan skill retrieval/matching logic."
            )

        if not suggestions:
            suggestions.append(
                f"Performa baik! Success rate: {sr:.0%}. "
                "Pertimbangkan untuk mencoba task yang lebih kompleks."
            )

        return suggestions
