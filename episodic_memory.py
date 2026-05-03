"""
memory/episodic_memory.py - Episodic Long-Term Memory
======================================================
Implementasi episodic memory menggunakan ChromaDB sebagai vector store.
Menyimpan episode/pengalaman agent dan mendukung retrieval berbasis semantik.

Referensi Paper:
- MemGPT (Packer et al., 2023): External storage yang bisa di-query
  → "Archival storage" untuk memori jangka panjang
- Generative Agents (Park et al., 2023): Memory stream
  → Setiap observasi disimpan sebagai memory dengan timestamp
  → Retrieval berdasarkan recency + relevance + importance
- SELF-RAG (Asai et al., 2023): Retrieval-Augmented Generation
  → Query ke memory store untuk enrich context sebelum reasoning
- Voyager (Wang et al., 2023): Skill & experience library
  → Menyimpan episode sukses sebagai referensi masa depan

Komponen:
- ChromaDB sebagai persistent vector database
- sentence-transformers untuk embedding
- Metadata filtering untuk query yang lebih presisi
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """
    Satu episode dalam episodic memory.

    Episode merepresentasikan satu pengalaman/kejadian yang dialami agent.
    Bisa berupa: observasi, action sequence, percakapan, atau task completion.
    """
    content: str                        # Konten episode (teks)
    episode_type: str = "observation"   # Tipe: "observation", "action", "task", "reflection"
    timestamp: float = field(default_factory=time.time)
    importance: float = 5.0             # Skor kepentingan 1-10 (dari LLM rating)
    metadata: Dict[str, Any] = field(default_factory=dict)
    episode_id: str = ""                # ID unik (auto-generated)
    tags: List[str] = field(default_factory=list)
    task_context: str = ""              # Task yang sedang dikerjakan saat episode ini terjadi
    outcome: str = ""                   # Hasil (success/failure/partial)

    def __post_init__(self):
        if not self.episode_id:
            # Generate ID dari hash content + timestamp
            hash_input = f"{self.content[:100]}{self.timestamp}"
            self.episode_id = "ep_" + hashlib.md5(
                hash_input.encode()
            ).hexdigest()[:12]


@dataclass
class RetrievedEpisode:
    """Episode yang sudah diambil dari store, dilengkapi similarity score."""
    episode: Episode
    similarity_score: float    # Cosine similarity dengan query (0-1)
    rank: int                  # Ranking dalam hasil retrieval


class EpisodicMemory:
    """
    Episodic Memory menggunakan ChromaDB + sentence-transformers.

    Episodic memory menyimpan "apa yang terjadi" - pengalaman nyata agent
    dalam urutan kronologis. Bisa di-retrieve dengan similarity search.

    Architecture:
    ┌─────────────────────────────────────────────────┐
    │                   Episode                        │
    │  content (text) → embedding (vector)            │
    │  metadata: timestamp, type, importance, tags    │
    └─────────────────────────────────────────────────┘
              ↓ disimpan di
    ┌─────────────────────────────────────────────────┐
    │              ChromaDB Collection                │
    │  - Persistent vector storage                    │
    │  - Cosine similarity search                     │
    │  - Metadata filtering                           │
    └─────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        collection_name: str = "agent_episodes",
        persist_directory: str = "./memory_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        max_episodes: int = 10000,
    ):
        """
        Args:
            collection_name: Nama collection di ChromaDB
            persist_directory: Direktori untuk persistent storage
            embedding_model: Model sentence-transformer untuk embedding
            max_episodes: Batas maksimal episode yang disimpan
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model
        self.max_episodes = max_episodes

        self._client = None
        self._collection = None
        self._embedder = None
        self._initialized = False

        # Fallback in-memory storage jika ChromaDB tidak tersedia
        self._fallback_store: List[Episode] = []
        self._use_fallback = False

        logger.info(f"EpisodicMemory diinisialisasi: collection='{collection_name}'")

    async def initialize(self):
        """Inisialisasi ChromaDB dan embedding model secara async."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._init_sync)
        self._initialized = True

    def _init_sync(self):
        """Inisialisasi synchronous (dijalankan di thread pool)."""
        try:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(
                f"ChromaDB berhasil diinisialisasi. "
                f"Existing episodes: {self._collection.count()}"
            )
        except ImportError:
            logger.warning(
                "ChromaDB tidak tersedia. Menggunakan in-memory fallback. "
                "Install dengan: pip install chromadb"
            )
            self._use_fallback = True

        # Inisialisasi embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Embedding model '{self.embedding_model_name}' dimuat")
        except ImportError:
            logger.warning(
                "sentence-transformers tidak tersedia. "
                "Retrieval akan menggunakan keyword matching. "
                "Install dengan: pip install sentence-transformers"
            )

    def _embed(self, text: str) -> Optional[List[float]]:
        """Generate embedding untuk teks."""
        if self._embedder is None:
            return None
        return self._embedder.encode(text, normalize_embeddings=True).tolist()

    def _embed_batch(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generate embedding untuk batch teks."""
        if self._embedder is None:
            return None
        return self._embedder.encode(
            texts, normalize_embeddings=True, batch_size=32
        ).tolist()

    # ─────────────────────────────
    # Store Operations
    # ─────────────────────────────

    async def store(self, episode: Episode) -> str:
        """
        Simpan episode ke memory store.

        Args:
            episode: Episode yang akan disimpan

        Returns:
            episode_id dari episode yang disimpan
        """
        if not self._initialized:
            await self.initialize()

        if self._use_fallback:
            return self._store_fallback(episode)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._store_sync, episode)

    def _store_sync(self, episode: Episode) -> str:
        """Synchronous store operation."""
        try:
            # Generate embedding
            embedding = self._embed(episode.content)

            # Flatten metadata untuk ChromaDB (hanya support scalar values)
            flat_metadata = {
                "episode_type": episode.episode_type,
                "timestamp": episode.timestamp,
                "importance": episode.importance,
                "task_context": episode.task_context[:500] if episode.task_context else "",
                "outcome": episode.outcome,
                "tags": json.dumps(episode.tags),
                **{k: str(v) for k, v in episode.metadata.items()
                   if isinstance(v, (str, int, float, bool))},
            }

            kwargs: Dict[str, Any] = {
                "documents": [episode.content],
                "ids": [episode.episode_id],
                "metadatas": [flat_metadata],
            }
            if embedding:
                kwargs["embeddings"] = [embedding]

            self._collection.upsert(**kwargs)

            logger.debug(
                f"[EpisodicMemory] Stored: {episode.episode_id} "
                f"({episode.episode_type}, importance={episode.importance:.1f})"
            )
            return episode.episode_id

        except Exception as e:
            logger.error(f"Gagal menyimpan episode: {e}")
            # Fallback ke in-memory
            return self._store_fallback(episode)

    def _store_fallback(self, episode: Episode) -> str:
        """Simpan ke in-memory list sebagai fallback."""
        self._fallback_store.append(episode)
        # Batas ukuran
        if len(self._fallback_store) > self.max_episodes:
            self._fallback_store.pop(0)
        return episode.episode_id

    # ─────────────────────────────
    # Retrieval Operations
    # ─────────────────────────────

    async def retrieve(
        self,
        query: str,
        n_results: int = 5,
        episode_type: Optional[str] = None,
        min_importance: float = 0.0,
        time_window_hours: Optional[float] = None,
    ) -> List[RetrievedEpisode]:
        """
        Retrieve episode yang relevan dengan query.

        Terinspirasi dari Generative Agents: retrieval berdasarkan
        kombinasi relevance (semantic similarity) + recency + importance.

        Args:
            query: Teks query untuk similarity search
            n_results: Jumlah hasil yang dikembalikan
            episode_type: Filter berdasarkan tipe episode
            min_importance: Minimum importance score
            time_window_hours: Filter hanya episode dalam N jam terakhir

        Returns:
            List RetrievedEpisode diurutkan berdasarkan relevance
        """
        if not self._initialized:
            await self.initialize()

        if self._use_fallback:
            return self._retrieve_fallback(query, n_results, episode_type)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._retrieve_sync,
            query, n_results, episode_type, min_importance, time_window_hours
        )

    def _retrieve_sync(
        self,
        query: str,
        n_results: int,
        episode_type: Optional[str],
        min_importance: float,
        time_window_hours: Optional[float],
    ) -> List[RetrievedEpisode]:
        """Synchronous retrieval operation."""
        try:
            # Build where clause untuk filtering metadata
            where_clause: Optional[Dict] = None
            conditions = []

            if episode_type:
                conditions.append({"episode_type": {"$eq": episode_type}})
            if min_importance > 0:
                conditions.append({"importance": {"$gte": min_importance}})
            if time_window_hours:
                cutoff_time = time.time() - (time_window_hours * 3600)
                conditions.append({"timestamp": {"$gte": cutoff_time}})

            if conditions:
                where_clause = {"$and": conditions} if len(conditions) > 1 else conditions[0]

            # Generate query embedding
            query_embedding = self._embed(query)

            # Query ChromaDB
            kwargs: Dict[str, Any] = {
                "n_results": min(n_results, max(1, self._collection.count())),
                "include": ["documents", "metadatas", "distances", "embeddings"],
            }
            if query_embedding:
                kwargs["query_embeddings"] = [query_embedding]
            else:
                kwargs["query_texts"] = [query]

            if where_clause:
                kwargs["where"] = where_clause

            results = self._collection.query(**kwargs)

            # Parse hasil
            retrieved = []
            if results["ids"] and results["ids"][0]:
                for i, (doc_id, doc, meta, dist) in enumerate(zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )):
                    # ChromaDB cosine distance → similarity
                    similarity = 1 - dist

                    episode = Episode(
                        content=doc,
                        episode_id=doc_id,
                        episode_type=meta.get("episode_type", "unknown"),
                        timestamp=float(meta.get("timestamp", 0)),
                        importance=float(meta.get("importance", 5.0)),
                        task_context=meta.get("task_context", ""),
                        outcome=meta.get("outcome", ""),
                        tags=json.loads(meta.get("tags", "[]")),
                    )

                    retrieved.append(RetrievedEpisode(
                        episode=episode,
                        similarity_score=similarity,
                        rank=i + 1,
                    ))

            logger.debug(
                f"[EpisodicMemory] Retrieved {len(retrieved)} episodes "
                f"untuk query: '{query[:50]}...'"
            )
            return retrieved

        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return self._retrieve_fallback(query, n_results, episode_type)

    def _retrieve_fallback(
        self,
        query: str,
        n_results: int,
        episode_type: Optional[str],
    ) -> List[RetrievedEpisode]:
        """Fallback retrieval menggunakan keyword matching."""
        query_words = set(query.lower().split())
        results = []

        for episode in self._fallback_store:
            if episode_type and episode.episode_type != episode_type:
                continue

            # Simple keyword overlap score
            content_words = set(episode.content.lower().split())
            overlap = len(query_words & content_words)
            score = overlap / max(len(query_words), 1)

            if score > 0:
                results.append(RetrievedEpisode(
                    episode=episode,
                    similarity_score=score,
                    rank=0,
                ))

        # Sort by score + recency
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        results = results[:n_results]

        for i, r in enumerate(results):
            r.rank = i + 1

        return results

    async def get_recent(
        self,
        n: int = 10,
        episode_type: Optional[str] = None,
    ) -> List[Episode]:
        """Ambil N episode terbaru."""
        if not self._initialized:
            await self.initialize()

        if self._use_fallback:
            episodes = self._fallback_store
            if episode_type:
                episodes = [e for e in episodes if e.episode_type == episode_type]
            return sorted(episodes, key=lambda e: e.timestamp, reverse=True)[:n]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_recent_sync, n, episode_type)

    def _get_recent_sync(self, n: int, episode_type: Optional[str]) -> List[Episode]:
        """Synchronous get recent."""
        try:
            where = {"episode_type": {"$eq": episode_type}} if episode_type else None
            results = self._collection.get(
                where=where,
                include=["documents", "metadatas"],
            )

            episodes = []
            if results["ids"]:
                for doc_id, doc, meta in zip(
                    results["ids"], results["documents"], results["metadatas"]
                ):
                    episodes.append(Episode(
                        content=doc,
                        episode_id=doc_id,
                        episode_type=meta.get("episode_type", "unknown"),
                        timestamp=float(meta.get("timestamp", 0)),
                        importance=float(meta.get("importance", 5.0)),
                    ))

            # Sort by timestamp terbaru
            episodes.sort(key=lambda e: e.timestamp, reverse=True)
            return episodes[:n]

        except Exception as e:
            logger.error(f"get_recent error: {e}")
            return []

    async def count(self) -> int:
        """Jumlah total episode yang tersimpan."""
        if not self._initialized:
            await self.initialize()

        if self._use_fallback:
            return len(self._fallback_store)

        try:
            return self._collection.count()
        except Exception:
            return 0

    async def delete(self, episode_id: str) -> bool:
        """Hapus episode berdasarkan ID."""
        if not self._initialized:
            await self.initialize()

        if self._use_fallback:
            original = len(self._fallback_store)
            self._fallback_store = [
                e for e in self._fallback_store if e.episode_id != episode_id
            ]
            return len(self._fallback_store) < original

        try:
            self._collection.delete(ids=[episode_id])
            return True
        except Exception as e:
            logger.error(f"Gagal menghapus episode {episode_id}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Statistik episodic memory."""
        count = 0
        try:
            if not self._use_fallback and self._collection:
                count = self._collection.count()
            else:
                count = len(self._fallback_store)
        except Exception:
            pass

        return {
            "total_episodes": count,
            "backend": "fallback" if self._use_fallback else "chromadb",
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model_name,
            "initialized": self._initialized,
        }
