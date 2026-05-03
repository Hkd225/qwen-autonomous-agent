"""
uncertainty.py - Uncertainty Tracking System
=============================================
Sistem untuk melacak dan mengelola uncertainty dalam setiap keputusan agent.
Membantu agent tahu kapan harus exploration vs exploitation.

Referensi Paper:
- Uncertainty in Deep Learning (Gal, 2016): Epistemic vs Aleatoric uncertainty
  → Epistemic: ketidakpastian model (bisa dikurangi dengan data lebih)
  → Aleatoric: ketidakpastian inheren dalam task/dunia
- SAIL (Uncertainty-Driven Exploration): Uncertainty untuk guide exploration
  → Saat uncertainty tinggi → explore (cari informasi)
  → Saat uncertainty rendah → exploit (gunakan pengetahuan)
- Self-Consistency (Wang et al., 2022): Consistency antar multiple samples
  → Ukur uncertainty dari variance antar jawaban berbeda
- AgentBench: Confidence calibration untuk action selection
  → Agent yang well-calibrated tahu "apa yang tidak diketahuinya"

Tipe Uncertainty:
1. Epistemic: "Model tidak tahu" → Bisa dikurangi dengan info lebih banyak
2. Aleatoric: "Task inherently ambigu" → Tidak bisa dikurangi
3. Decision: "Tidak yakin action mana yang terbaik" → Butuh deliberasi lebih
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class UncertaintyType(Enum):
    """Tipe uncertainty yang ditracking."""
    EPISTEMIC = "epistemic"         # Ketidakpastian pengetahuan
    ALEATORIC = "aleatoric"         # Ketidakpastian inheren task
    DECISION = "decision"           # Ketidakpastian dalam memilih action
    FACTUAL = "factual"             # Ketidakpastian tentang fakta spesifik
    TEMPORAL = "temporal"           # Informasi mungkin sudah outdated


@dataclass
class UncertaintyEstimate:
    """
    Estimasi uncertainty untuk satu keputusan/informasi.

    Score: 0.0 = sangat yakin, 1.0 = sangat tidak yakin
    """
    uncertainty_type: UncertaintyType
    score: float                        # 0.0 (certain) → 1.0 (uncertain)
    confidence: float = 0.5             # 1 - uncertainty (convenience property)
    reason: str = ""                    # Alasan uncertainty
    timestamp: float = field(default_factory=time.time)
    context: str = ""                   # Context yang menyebabkan uncertainty
    should_explore: bool = False        # Apakah perlu exploration?
    suggested_action: str = ""          # Saran untuk mengurangi uncertainty

    def __post_init__(self):
        self.confidence = 1.0 - self.score

    @property
    def level(self) -> str:
        """Level uncertainty dalam bahasa natural."""
        if self.score < 0.2:
            return "sangat yakin"
        elif self.score < 0.4:
            return "cukup yakin"
        elif self.score < 0.6:
            return "tidak pasti"
        elif self.score < 0.8:
            return "kurang yakin"
        else:
            return "sangat tidak yakin"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.uncertainty_type.value,
            "score": round(self.score, 3),
            "confidence": round(self.confidence, 3),
            "level": self.level,
            "reason": self.reason,
            "should_explore": self.should_explore,
        }


@dataclass
class ActionUncertainty:
    """Uncertainty untuk satu action candidate."""
    action_name: str
    action_input: Dict[str, Any]
    epistemic: float = 0.5          # Ketidaktahuan tentang outcome
    aleatoric: float = 0.3          # Randomness inheren
    expected_outcome_confidence: float = 0.5  # Seberapa yakin tentang hasil
    risk_score: float = 0.3         # Seberapa berisiko action ini
    notes: str = ""

    @property
    def composite_uncertainty(self) -> float:
        """Composite uncertainty: rata-rata tertimbang."""
        return (0.5 * self.epistemic + 0.3 * self.aleatoric +
                0.2 * (1 - self.expected_outcome_confidence))

    @property
    def should_avoid(self) -> bool:
        """Apakah action ini terlalu berisiko?"""
        return self.composite_uncertainty > 0.8 and self.risk_score > 0.7


class UncertaintyTracker:
    """
    Tracks dan manages uncertainty di setiap step agent.

    Terinspirasi dari uncertainty quantification dalam deep learning:
    - Multiple forward passes untuk estimasi variance (monte carlo dropout analogy)
    - Calibration: pastikan confidence = akurasi sebenarnya
    - Aktif eksplorasi ketika uncertainty tinggi

    Dalam konteks agent:
    - Uncertainty tinggi → Minta klarifikasi, cari info lebih, coba multiple approaches
    - Uncertainty rendah → Eksekusi dengan percaya diri
    """

    def __init__(
        self,
        epistemic_threshold: float = 0.7,   # Threshold untuk flagging epistemic uncertainty
        aleatoric_threshold: float = 0.6,   # Threshold untuk flagging aleatoric uncertainty
        exploration_threshold: float = 0.65, # Threshold untuk trigger exploration
        llm_interface=None,
    ):
        self.epistemic_threshold = epistemic_threshold
        self.aleatoric_threshold = aleatoric_threshold
        self.exploration_threshold = exploration_threshold
        self.llm = llm_interface

        # History uncertainty untuk tracking dan analisis
        self.history: List[UncertaintyEstimate] = []
        self.action_history: List[ActionUncertainty] = []

        # Kalibrasi (berapa banyak confident predictions yang benar)
        self._calibration_data: List[Tuple[float, bool]] = []

        logger.info(
            f"UncertaintyTracker diinisialisasi: "
            f"exploration_threshold={exploration_threshold}"
        )

    # ─────────────────────────────
    # Core Estimation Methods
    # ─────────────────────────────

    async def estimate_epistemic(
        self,
        statement: str,
        context: str = "",
        use_llm: bool = True,
    ) -> UncertaintyEstimate:
        """
        Estimasi epistemic uncertainty dari sebuah statement.

        Epistemic uncertainty mengukur seberapa besar "tidak tahu"
        yang ada dalam knowledge model tentang statement tersebut.

        Cara estimasi:
        1. LLM-based: Minta LLM menilai confidence-nya
        2. Keyword-based: Cari marker uncertainty dalam teks
        3. Temporal: Cek apakah info mungkin sudah outdated
        """
        if use_llm and self.llm:
            return await self._estimate_llm(
                statement, UncertaintyType.EPISTEMIC, context
            )
        else:
            return self._estimate_keyword(
                statement, UncertaintyType.EPISTEMIC, context
            )

    async def estimate_aleatoric(
        self,
        task_description: str,
        context: str = "",
    ) -> UncertaintyEstimate:
        """
        Estimasi aleatoric uncertainty dari sebuah task.

        Aleatoric uncertainty mengukur ketidakpastian inheren dalam task:
        - Task ambiguity (instruksi tidak jelas)
        - Multiple valid outcomes
        - Dependency on external factors
        """
        score = self._compute_aleatoric_score(task_description)
        should_explore = score >= self.aleatoric_threshold

        reasons = []
        if score > 0.7:
            reasons.append("task sangat ambigu")
        if "atau" in task_description.lower() or " or " in task_description.lower():
            reasons.append("ada multiple opsi")
        if "mungkin" in task_description.lower() or "maybe" in task_description.lower():
            reasons.append("kondisi tidak pasti")

        estimate = UncertaintyEstimate(
            uncertainty_type=UncertaintyType.ALEATORIC,
            score=score,
            reason="; ".join(reasons) if reasons else "task relatif jelas",
            context=context,
            should_explore=should_explore,
            suggested_action=(
                "Minta klarifikasi dari user" if score > 0.7
                else "Lanjutkan dengan asumsi terbaik"
            ),
        )

        self.history.append(estimate)
        return estimate

    async def estimate_action_uncertainty(
        self,
        action_name: str,
        action_input: Dict[str, Any],
        task_context: str = "",
    ) -> ActionUncertainty:
        """
        Estimasi uncertainty untuk satu candidate action.

        Menganalisis:
        - Seberapa predictable outcome dari action ini?
        - Seberapa reversible jika salah?
        - Ada alternatif yang lebih certain?

        Args:
            action_name: Nama tool/action
            action_input: Parameter action
            task_context: Konteks task saat ini
        """
        # Estimasi epistemic uncertainty untuk action ini
        action_desc = f"Action: {action_name} dengan input {action_input}"
        epistemic_est = await self.estimate_epistemic(
            action_desc, task_context, use_llm=False
        )

        # Risk assessment berdasarkan tipe action
        risk = self._assess_action_risk(action_name, action_input)

        # Aleatoric uncertainty dari task context
        aleatoric = await self.estimate_aleatoric(task_context)

        action_uncertainty = ActionUncertainty(
            action_name=action_name,
            action_input=action_input,
            epistemic=epistemic_est.score,
            aleatoric=aleatoric.score,
            expected_outcome_confidence=1.0 - epistemic_est.score,
            risk_score=risk,
        )

        self.action_history.append(action_uncertainty)
        return action_uncertainty

    # ─────────────────────────────
    # Exploration vs Exploitation
    # ─────────────────────────────

    def should_explore(
        self,
        task: str,
        current_confidence: float,
        step_count: int = 0,
    ) -> Tuple[bool, str]:
        """
        Putuskan apakah agent harus explore (cari info) atau exploit (execute).

        Terinspirasi dari epsilon-greedy dan UCB strategies:
        - Exploration lebih penting di awal task
        - Exploitation lebih baik ketika confidence tinggi
        - Emergency exploration jika task sangat ambigu

        Args:
            task: Deskripsi task
            current_confidence: Confidence saat ini (0-1)
            step_count: Berapa langkah sudah diambil

        Returns:
            (should_explore: bool, reason: str)
        """
        uncertainty = 1.0 - current_confidence

        # Selalu explore di step pertama jika uncertainty tinggi
        if step_count == 0 and uncertainty > 0.5:
            return True, "Step awal dengan uncertainty tinggi, perlu orientasi"

        # Hitung average uncertainty dari history terbaru
        if self.history:
            recent = self.history[-5:]
            avg_uncertainty = sum(e.score for e in recent) / len(recent)
            if avg_uncertainty > self.exploration_threshold:
                return True, f"Rata-rata uncertainty tinggi ({avg_uncertainty:.2f})"

        # Exploration decay: semakin banyak langkah, semakin fokus eksekusi
        exploration_probability = max(0.1, uncertainty - 0.1 * step_count)

        if exploration_probability > self.exploration_threshold:
            return True, f"Uncertainty {uncertainty:.2f} melebihi threshold"

        return False, f"Confidence {current_confidence:.2f} cukup untuk eksekusi"

    def choose_best_action(
        self,
        candidates: List[ActionUncertainty],
        risk_tolerance: float = 0.5,
    ) -> Optional[ActionUncertainty]:
        """
        Pilih action terbaik berdasarkan uncertainty dan risk.

        Strategy:
        - Hindari action dengan uncertainty > 0.8 DAN risk > 0.7
        - Prefer action dengan expected outcome confidence tertinggi
        - Pertimbangkan risk tolerance

        Args:
            candidates: List candidate actions
            risk_tolerance: Seberapa toleran terhadap risk (0=risk-averse, 1=risk-seeking)

        Returns:
            Action terbaik atau None jika semua terlalu uncertain
        """
        if not candidates:
            return None

        # Filter: hapus action yang terlalu berisiko
        viable = [
            a for a in candidates
            if not (a.composite_uncertainty > 0.8 and a.risk_score > risk_tolerance + 0.3)
        ]

        if not viable:
            # Jika semua berisiko, pilih yang paling aman
            viable = candidates

        # Score berdasarkan expected outcome dan inverse risk
        def score(a: ActionUncertainty) -> float:
            return (
                a.expected_outcome_confidence * 0.6
                - a.risk_score * (1 - risk_tolerance) * 0.4
            )

        return max(viable, key=score)

    # ─────────────────────────────
    # Private Helpers
    # ─────────────────────────────

    async def _estimate_llm(
        self,
        statement: str,
        uncertainty_type: UncertaintyType,
        context: str,
    ) -> UncertaintyEstimate:
        """Estimasi uncertainty menggunakan LLM self-assessment."""
        try:
            prompt = f"""Nilai uncertainty (ketidakpastian) dari statement berikut.

Statement: {statement[:300]}
{f"Konteks: {context[:200]}" if context else ""}

Berikan estimasi dalam JSON:
{{
    "uncertainty_score": <0.0-1.0, dimana 0=sangat yakin, 1=sangat tidak yakin>,
    "reason": "alasan singkat",
    "information_needed": "informasi apa yang dibutuhkan untuk lebih yakin (kosong jika sudah yakin)",
    "suggested_action": "apa yang sebaiknya dilakukan agent"
}}"""

            response = await asyncio.wait_for(
                self.llm.generate_json([{"role": "user", "content": prompt}]),
                timeout=8.0,
            )

            score = float(response.get("uncertainty_score", 0.5))
            score = max(0.0, min(1.0, score))

            estimate = UncertaintyEstimate(
                uncertainty_type=uncertainty_type,
                score=score,
                reason=response.get("reason", ""),
                context=context,
                should_explore=score >= self.exploration_threshold,
                suggested_action=response.get("suggested_action", ""),
            )

            self.history.append(estimate)
            logger.debug(
                f"[Uncertainty] {uncertainty_type.value}: {score:.3f} "
                f"({estimate.level}) - {estimate.reason}"
            )
            return estimate

        except Exception as e:
            logger.debug(f"LLM uncertainty estimation gagal: {e}")
            return self._estimate_keyword(statement, uncertainty_type, context)

    def _estimate_keyword(
        self,
        text: str,
        uncertainty_type: UncertaintyType,
        context: str,
    ) -> UncertaintyEstimate:
        """Estimasi uncertainty berdasarkan marker kata kunci."""
        text_lower = text.lower()

        # Marker uncertainty tinggi
        high_uncertainty_markers = [
            "mungkin", "sepertinya", "kemungkinan", "tidak yakin",
            "maybe", "possibly", "uncertain", "unclear", "unknown",
            "might", "could be", "not sure", "approximately", "roughly",
            "sekitar", "kira-kira", "saya tidak tahu",
        ]
        # Marker uncertainty rendah
        low_uncertainty_markers = [
            "pasti", "jelas", "definitif", "tentu saja",
            "definitely", "certainly", "clearly", "obviously",
            "fact", "proven", "confirmed", "verified",
        ]

        high_count = sum(1 for m in high_uncertainty_markers if m in text_lower)
        low_count = sum(1 for m in low_uncertainty_markers if m in text_lower)

        # Hitung score
        if high_count > low_count:
            score = min(0.9, 0.4 + high_count * 0.1)
        elif low_count > high_count:
            score = max(0.1, 0.4 - low_count * 0.1)
        else:
            score = 0.5  # Default: moderate uncertainty

        estimate = UncertaintyEstimate(
            uncertainty_type=uncertainty_type,
            score=score,
            reason=f"Keyword analysis: {high_count} markers tinggi, {low_count} markers rendah",
            context=context,
            should_explore=score >= self.exploration_threshold,
        )

        self.history.append(estimate)
        return estimate

    def _compute_aleatoric_score(self, task: str) -> float:
        """Hitung aleatoric uncertainty score dari task description."""
        task_lower = task.lower()

        # Faktor yang meningkatkan aleatoric uncertainty
        factors = {
            "atau": 0.15, "or ": 0.15, "either": 0.15,
            "mungkin": 0.1, "maybe": 0.1, "might": 0.1,
            "coba": 0.05, "try": 0.05,
            "beberapa": 0.08, "several": 0.08, "multiple": 0.08,
            "tidak jelas": 0.2, "unclear": 0.2, "ambiguous": 0.2,
            "?": 0.05,
        }

        base_score = 0.2  # Baseline uncertainty
        for marker, weight in factors.items():
            if marker in task_lower:
                base_score += weight

        return min(1.0, base_score)

    def _assess_action_risk(
        self, action_name: str, action_input: Dict[str, Any]
    ) -> float:
        """Nilai risk level dari sebuah action."""
        # Risk berdasarkan tipe action
        risk_levels = {
            "web_search": 0.1,          # Sangat aman
            "calculator": 0.05,         # Hampir tidak ada risk
            "memory": 0.1,              # Aman
            "file_manager": 0.4,        # Moderat (bisa overwrite file)
            "code_executor": 0.5,       # Moderat-tinggi (code execution)
        }

        base_risk = risk_levels.get(action_name, 0.3)

        # Modifikasi berdasarkan input
        if isinstance(action_input, dict):
            action = str(action_input.get("action", "")).lower()
            if action in ("delete", "write", "overwrite"):
                base_risk += 0.2
            if action in ("read", "list", "get"):
                base_risk -= 0.1

        return max(0.0, min(1.0, base_risk))

    # ─────────────────────────────
    # Calibration & Stats
    # ─────────────────────────────

    def update_calibration(self, predicted_confidence: float, was_correct: bool):
        """
        Update kalibrasi dengan data ground truth.

        Semakin banyak data, semakin akurat confidence estimation.
        """
        self._calibration_data.append((predicted_confidence, was_correct))

    def compute_calibration_error(self) -> float:
        """
        Hitung Expected Calibration Error (ECE).

        ECE = 0 berarti perfectly calibrated
        ECE = 1 berarti selalu salah dalam confidence
        """
        if not self._calibration_data:
            return 0.5  # Unknown calibration

        # Bagi ke bucket
        buckets = [[] for _ in range(10)]
        for conf, correct in self._calibration_data:
            bucket_idx = min(9, int(conf * 10))
            buckets[bucket_idx].append((conf, correct))

        ece = 0.0
        n = len(self._calibration_data)
        for bucket in buckets:
            if not bucket:
                continue
            avg_conf = sum(c for c, _ in bucket) / len(bucket)
            accuracy = sum(1 for _, correct in bucket if correct) / len(bucket)
            ece += (len(bucket) / n) * abs(avg_conf - accuracy)

        return ece

    def get_current_state(self) -> Dict[str, Any]:
        """Dapatkan state uncertainty terkini."""
        if not self.history:
            return {"status": "no_data", "avg_uncertainty": 0.5}

        recent = self.history[-10:]
        avg = sum(e.score for e in recent) / len(recent)
        types = {}
        for e in recent:
            types[e.uncertainty_type.value] = types.get(e.uncertainty_type.value, 0) + 1

        return {
            "recent_avg_uncertainty": round(avg, 3),
            "current_level": recent[-1].level if recent else "unknown",
            "should_explore": avg >= self.exploration_threshold,
            "history_length": len(self.history),
            "type_breakdown": types,
            "calibration_error": round(self.compute_calibration_error(), 3),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Statistik lengkap uncertainty tracker."""
        return {
            **self.get_current_state(),
            "total_estimates": len(self.history),
            "total_action_estimates": len(self.action_history),
            "thresholds": {
                "epistemic": self.epistemic_threshold,
                "aleatoric": self.aleatoric_threshold,
                "exploration": self.exploration_threshold,
            },
        }
