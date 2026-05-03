"""
memory/scoring.py - Memory Scoring & Filtering
===============================================
Sistem scoring untuk menentukan relevansi dan kepentingan memori.
Menggabungkan multiple signals untuk menghasilkan composite score.

Referensi Paper:
- Generative Agents (Park et al., 2023): "Generative Agents: Interactive Simulacra of Human Behavior"
  → Tiga dimensi memory scoring: recency, importance, relevance
  → Composite score = recency_score + importance_score + relevance_score
  → "The recency score is computed using an exponential decay function"
- MemGPT (Packer et al., 2023): Memory management dan eviction policy
  → Importance score digunakan untuk menentukan memori mana yang di-swap ke archival
- Reflexion (Shinn et al., 2023): Self-evaluation
  → Agent menilai sendiri kepentingan pengalamannya

Rumus Scoring:
  composite_score = α * recency + β * relevance + γ * importance
  dimana α + β + γ = 1
"""

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MemoryScore:
    """
    Hasil scoring sebuah item memori.

    Mengandung skor individual per dimensi dan composite score final.
    """
    item_id: str                    # ID item yang di-score
    recency_score: float = 0.0      # Score berdasarkan waktu (0-1)
    relevance_score: float = 0.0    # Score berdasarkan similarity ke query (0-1)
    importance_score: float = 0.0   # Score berdasarkan kepentingan (0-1)
    composite_score: float = 0.0    # Score gabungan final (0-1)

    # Metadata scoring
    age_hours: float = 0.0          # Umur item dalam jam
    decay_rate: float = 0.99        # Rate decay yang digunakan
    raw_importance: float = 5.0     # Importance raw (1-10 dari LLM)

    def should_retain(self, threshold: float = 0.3) -> bool:
        """Apakah item ini harus dipertahankan berdasarkan threshold?"""
        return self.composite_score >= threshold

    def to_dict(self) -> Dict[str, float]:
        return {
            "item_id": self.item_id,
            "recency": round(self.recency_score, 4),
            "relevance": round(self.relevance_score, 4),
            "importance": round(self.importance_score, 4),
            "composite": round(self.composite_score, 4),
        }


@dataclass
class ScoringConfig:
    """Konfigurasi untuk sistem scoring memori."""
    # Bobot untuk composite score (harus berjumlah 1.0)
    recency_weight: float = 0.3        # Bobot recency
    relevance_weight: float = 0.4      # Bobot relevance (lebih tinggi karena query-dependent)
    importance_weight: float = 0.3     # Bobot importance

    # Parameter recency decay
    # Terinspirasi dari Generative Agents: "exponential decay with base 0.99"
    decay_base: float = 0.99           # Base decay per jam
    min_recency: float = 0.01          # Minimum recency score

    # Threshold untuk filtering
    retention_threshold: float = 0.2   # Score minimum untuk dipertahankan
    high_importance_threshold: float = 0.8  # Threshold untuk "sangat penting"

    # LLM-based scoring
    use_llm_for_importance: bool = True    # Gunakan LLM untuk rate importance
    llm_importance_timeout: float = 10.0   # Timeout untuk LLM call

    def validate(self):
        """Validasi bobot berjumlah 1.0."""
        total = self.recency_weight + self.relevance_weight + self.importance_weight
        assert abs(total - 1.0) < 0.001, (
            f"Bobot scoring harus berjumlah 1.0, saat ini: {total}"
        )


class MemoryScorer:
    """
    Memory Scoring Engine.

    Implementasi sistem scoring 3 dimensi yang terinspirasi dari
    Generative Agents paper. Setiap dimensi mengukur aspek berbeda
    dari "seberapa berguna" sebuah memori untuk agent saat ini.

    Dimensi Scoring:
    ┌───────────────┬────────────────────────────────────────────┐
    │   Recency     │ Seberapa baru item ini?                    │
    │               │ Decay eksponensial: score = base^(age_hrs) │
    ├───────────────┼────────────────────────────────────────────┤
    │   Relevance   │ Seberapa relevan dengan query/task saat ini?│
    │               │ Cosine similarity antara item & query       │
    ├───────────────┼────────────────────────────────────────────┤
    │   Importance  │ Seberapa penting secara intrinsik?          │
    │               │ Dinilai LLM: 1-10, di-normalize ke 0-1     │
    └───────────────┴────────────────────────────────────────────┘
    """

    def __init__(
        self,
        config: Optional[ScoringConfig] = None,
        llm_interface=None,
    ):
        """
        Args:
            config: Konfigurasi scoring (gunakan default jika None)
            llm_interface: LLM interface untuk importance scoring
        """
        self.config = config or ScoringConfig()
        self.config.validate()
        self.llm = llm_interface

        # Cache untuk importance scores (hindari LLM call berulang)
        self._importance_cache: Dict[str, float] = {}

        # Embedder untuk relevance scoring
        self._embedder = None
        self._embedder_loaded = False

        logger.info(
            f"MemoryScorer diinisialisasi: "
            f"weights=[recency={self.config.recency_weight}, "
            f"relevance={self.config.relevance_weight}, "
            f"importance={self.config.importance_weight}]"
        )

    def _load_embedder(self):
        """Lazy load embedding model."""
        if self._embedder_loaded:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.debug("Embedder dimuat untuk relevance scoring")
        except ImportError:
            logger.warning("Embedder tidak tersedia, relevance scoring menggunakan keyword")
        self._embedder_loaded = True

    # ─────────────────────────────
    # Individual Scorers
    # ─────────────────────────────

    def compute_recency_score(self, timestamp: float) -> Tuple[float, float]:
        """
        Hitung recency score menggunakan exponential decay.

        Formula dari Generative Agents paper:
          recency = decay_base ^ (hours_since_last_access)

        Contoh dengan decay_base=0.99:
          - 0 jam lalu → score = 1.0
          - 1 jam lalu → score = 0.99
          - 24 jam lalu → score = 0.99^24 ≈ 0.79
          - 7 hari lalu → score = 0.99^168 ≈ 0.19
          - 30 hari lalu → score = 0.99^720 ≈ 0.001

        Args:
            timestamp: Unix timestamp ketika item disimpan

        Returns:
            (recency_score, age_hours)
        """
        age_seconds = time.time() - timestamp
        age_hours = max(0.0, age_seconds / 3600.0)

        # Exponential decay
        score = math.pow(self.config.decay_base, age_hours)
        score = max(self.config.min_recency, score)

        return score, age_hours

    def compute_relevance_score(
        self,
        content: str,
        query: str,
    ) -> float:
        """
        Hitung relevance score menggunakan cosine similarity.

        Jika embedder tersedia, gunakan semantic similarity.
        Fallback ke Jaccard similarity (keyword overlap).

        Args:
            content: Konten item memori
            query: Query atau task context saat ini

        Returns:
            Relevance score (0-1)
        """
        if not self._embedder_loaded:
            self._load_embedder()

        if self._embedder is not None:
            try:
                import numpy as np
                embeddings = self._embedder.encode(
                    [content, query], normalize_embeddings=True
                )
                # Cosine similarity (sudah normalize, jadi dot product = cosine sim)
                similarity = float(np.dot(embeddings[0], embeddings[1]))
                # Clip ke [0, 1]
                return max(0.0, min(1.0, similarity))
            except Exception as e:
                logger.debug(f"Embedding similarity error: {e}")

        # Fallback: Jaccard similarity
        content_words = set(content.lower().split())
        query_words = set(query.lower().split())

        if not query_words:
            return 0.0

        intersection = len(content_words & query_words)
        union = len(content_words | query_words)

        return intersection / union if union > 0 else 0.0

    def compute_importance_score_sync(
        self,
        content: str,
        raw_importance: float = 5.0,
    ) -> float:
        """
        Normalize raw importance ke range [0, 1].

        Raw importance adalah angka 1-10 yang bisa di-rate oleh LLM.

        Args:
            content: Konten item (untuk cache key)
            raw_importance: Skor 1-10 dari LLM atau manual

        Returns:
            Normalized importance score (0-1)
        """
        # Normalize dari skala 1-10 ke 0-1
        normalized = (raw_importance - 1) / 9.0
        return max(0.0, min(1.0, normalized))

    async def compute_importance_score_llm(
        self,
        content: str,
        task_context: str = "",
    ) -> Tuple[float, float]:
        """
        Rate importance menggunakan LLM (Generative Agents style).

        LLM diminta untuk menilai kepentingan informasi pada skala 1-10.
        Hasilnya di-cache untuk menghindari LLM call berulang.

        Args:
            content: Konten yang akan di-rate
            task_context: Konteks task saat ini

        Returns:
            (normalized_score, raw_score_1_to_10)
        """
        # Cek cache
        cache_key = hashlib.md5(content[:200].encode()).hexdigest()
        if cache_key in self._importance_cache:
            raw = self._importance_cache[cache_key]
            return self.compute_importance_score_sync(content, raw), raw

        # Default jika LLM tidak tersedia
        if self.llm is None:
            return 0.5, 5.0

        try:
            prompt = f"""Rate kepentingan informasi berikut untuk agent AI yang sedang bekerja pada task.
Skala: 1 (tidak penting/noise) sampai 10 (sangat kritis)

Informasi: {content[:500]}
{f"Task konteks: {task_context[:200]}" if task_context else ""}

Respond dengan JSON: {{"score": <1-10>, "reason": "<alasan singkat>"}}"""

            response = await asyncio.wait_for(
                self.llm.generate_json([{"role": "user", "content": prompt}]),
                timeout=self.config.llm_importance_timeout,
            )

            raw_score = float(response.get("score", 5))
            raw_score = max(1.0, min(10.0, raw_score))

            # Cache hasil
            self._importance_cache[cache_key] = raw_score

            normalized = self.compute_importance_score_sync(content, raw_score)
            logger.debug(
                f"LLM importance score: {raw_score}/10 → {normalized:.3f} "
                f"({response.get('reason', 'N/A')})"
            )
            return normalized, raw_score

        except Exception as e:
            logger.debug(f"LLM importance scoring gagal: {e}")
            return 0.5, 5.0

    # ─────────────────────────────
    # Composite Scoring
    # ─────────────────────────────

    def compute_composite(
        self,
        recency: float,
        relevance: float,
        importance: float,
    ) -> float:
        """
        Hitung composite score dari tiga dimensi.

        Formula:
          composite = α * recency + β * relevance + γ * importance

        Args:
            recency: Recency score (0-1)
            relevance: Relevance score (0-1)
            importance: Importance score (0-1)

        Returns:
            Composite score (0-1)
        """
        composite = (
            self.config.recency_weight * recency
            + self.config.relevance_weight * relevance
            + self.config.importance_weight * importance
        )
        return max(0.0, min(1.0, composite))

    async def score_item(
        self,
        item_id: str,
        content: str,
        timestamp: float,
        query: str,
        raw_importance: float = 5.0,
        use_llm: bool = False,
        task_context: str = "",
    ) -> MemoryScore:
        """
        Hitung composite score untuk satu item memori.

        Args:
            item_id: ID item memori
            content: Konten teks
            timestamp: Waktu item disimpan
            query: Query/konteks saat ini untuk relevance scoring
            raw_importance: Raw importance score (1-10)
            use_llm: Apakah gunakan LLM untuk importance scoring?
            task_context: Konteks task untuk LLM importance scoring

        Returns:
            MemoryScore dengan semua dimensi terisi
        """
        # 1. Recency score
        recency, age_hours = self.compute_recency_score(timestamp)

        # 2. Relevance score
        relevance = self.compute_relevance_score(content, query)

        # 3. Importance score
        if use_llm and self.llm is not None:
            importance, raw_importance = await self.compute_importance_score_llm(
                content, task_context
            )
        else:
            importance = self.compute_importance_score_sync(content, raw_importance)

        # 4. Composite
        composite = self.compute_composite(recency, relevance, importance)

        return MemoryScore(
            item_id=item_id,
            recency_score=recency,
            relevance_score=relevance,
            importance_score=importance,
            composite_score=composite,
            age_hours=age_hours,
            decay_rate=self.config.decay_base,
            raw_importance=raw_importance,
        )

    async def score_and_filter(
        self,
        items: List[Dict[str, Any]],
        query: str,
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        task_context: str = "",
    ) -> Tuple[List[MemoryScore], List[Dict[str, Any]]]:
        """
        Score dan filter daftar item memori.

        Pipeline lengkap:
        1. Score setiap item
        2. Filter berdasarkan threshold
        3. Sort berdasarkan composite score
        4. Ambil top-k

        Args:
            items: List item dengan fields: id, content, timestamp, importance
            query: Query untuk relevance scoring
            threshold: Minimum score (default: config.retention_threshold)
            top_k: Ambil N item terbaik
            task_context: Konteks task

        Returns:
            (scores, filtered_items) - item yang lolos filter
        """
        if not items:
            return [], []

        threshold = threshold or self.config.retention_threshold

        # Score semua item secara concurrent
        score_tasks = [
            self.score_item(
                item_id=item.get("id", str(i)),
                content=item.get("content", ""),
                timestamp=item.get("timestamp", time.time()),
                query=query,
                raw_importance=item.get("importance", 5.0),
                task_context=task_context,
            )
            for i, item in enumerate(items)
        ]

        scores = await asyncio.gather(*score_tasks)

        # Filter dan sort
        scored_pairs = [
            (score, item)
            for score, item in zip(scores, items)
            if score.composite_score >= threshold
        ]
        scored_pairs.sort(key=lambda x: x[0].composite_score, reverse=True)

        if top_k:
            scored_pairs = scored_pairs[:top_k]

        filtered_scores = [pair[0] for pair in scored_pairs]
        filtered_items = [pair[1] for pair in scored_pairs]

        logger.debug(
            f"[MemoryScorer] Scored {len(items)} items, "
            f"filtered to {len(filtered_items)} "
            f"(threshold={threshold:.2f})"
        )

        return filtered_scores, filtered_items

    def explain_score(self, score: MemoryScore) -> str:
        """Buat penjelasan human-readable dari score."""
        age_str = (
            f"{score.age_hours:.1f} jam"
            if score.age_hours < 24
            else f"{score.age_hours/24:.1f} hari"
        )

        return (
            f"Score: {score.composite_score:.3f} | "
            f"Recency: {score.recency_score:.3f} (umur {age_str}) | "
            f"Relevance: {score.relevance_score:.3f} | "
            f"Importance: {score.importance_score:.3f} "
            f"(raw: {score.raw_importance:.1f}/10)"
        )

    def get_config(self) -> Dict[str, Any]:
        """Kembalikan konfigurasi scoring saat ini."""
        return {
            "weights": {
                "recency": self.config.recency_weight,
                "relevance": self.config.relevance_weight,
                "importance": self.config.importance_weight,
            },
            "decay_base": self.config.decay_base,
            "retention_threshold": self.config.retention_threshold,
            "use_llm": self.config.use_llm_for_importance,
        }


# ─────────────────────────────
# Import hashlib (dibutuhkan di atas)
# ─────────────────────────────
import hashlib
