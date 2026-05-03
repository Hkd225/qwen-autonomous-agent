"""
memory/goal_memory.py - Long-Term Goal Memory
==============================================
Sistem manajemen goals hierarkis untuk agent.
Menyimpan, track, dan prioritize goals agent dari abstrak ke konkrit.

Referensi Paper:
- BabyAGI (Nakajima, 2023): Task list management dengan prioritization
  → "Create new tasks based on the result of last completed task"
  → Goals bisa generate sub-tasks secara dinamis
- AutoGPT (Significant Gravitas, 2023): Goal decomposition
  → Hierarchical goal structure: ultimate → sub-goals → tasks
- Voyager (Wang et al., 2023): Curriculum learning & goal management
  → "Skill prerequisite graph" - goals punya dependencies
  → Progressively harder goals berdasarkan current capabilities
- HTN Planning: Hierarchical Task Network
  → Goals di-decompose secara top-down
  → Tasks adalah "primitive" yang bisa langsung dieksekusi

Goal Hierarchy:
  Level 0: Ultimate Goal (e.g., "Build a web app")
  Level 1: Sub-goals (e.g., "Design DB", "Build backend", "Build frontend")
  Level 2: Tasks (e.g., "Create users table", "Create auth endpoints")
  Level 3: Actions (langsung dieksekusi oleh agent)
"""

import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GoalStatus(Enum):
    """Status lifecycle sebuah goal."""
    PENDING = "pending"         # Belum dimulai
    ACTIVE = "active"           # Sedang dikerjakan
    COMPLETED = "completed"     # Berhasil diselesaikan
    FAILED = "failed"           # Gagal
    PAUSED = "paused"           # Di-pause sementara
    CANCELLED = "cancelled"     # Dibatalkan


class GoalPriority(Enum):
    """Level prioritas goal."""
    CRITICAL = 5    # Harus diselesaikan sekarang
    HIGH = 4        # Penting, kerjakan segera
    MEDIUM = 3      # Normal priority
    LOW = 2         # Bisa ditunda
    MINIMAL = 1     # Opsional / nice-to-have


@dataclass
class Goal:
    """
    Representasi satu goal dalam hierarki.

    Goal bisa merupakan:
    - Ultimate goal (level 0): Objective tertinggi agent
    - Sub-goal (level 1-N): Breakdown dari goal parent
    - Task (leaf node): Unit kerja yang bisa langsung dieksekusi
    """
    title: str                          # Judul singkat
    description: str                    # Deskripsi lengkap
    goal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: Optional[str] = None     # ID parent goal (None = root)
    level: int = 0                      # Depth dalam hierarki (0 = root)
    status: GoalStatus = GoalStatus.PENDING
    priority: GoalPriority = GoalPriority.MEDIUM
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    deadline: Optional[float] = None    # Unix timestamp deadline (opsional)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Progress tracking
    progress: float = 0.0               # Progress 0.0 - 1.0
    attempts: int = 0                   # Berapa kali dicoba
    max_attempts: int = 3               # Batas percobaan sebelum FAILED

    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # ID goals yang harus selesai dulu

    # Context & notes
    context: str = ""                   # Konteks tambahan untuk eksekusi
    failure_reason: str = ""            # Alasan gagal (jika FAILED)
    result: str = ""                    # Hasil ketika COMPLETED

    @property
    def is_leaf(self) -> bool:
        """Apakah ini leaf node (task yang bisa langsung dieksekusi)?"""
        return self.metadata.get("is_leaf", False)

    @property
    def is_overdue(self) -> bool:
        """Apakah goal ini sudah melewati deadline?"""
        if self.deadline is None:
            return False
        return time.time() > self.deadline

    def mark_active(self):
        self.status = GoalStatus.ACTIVE
        self.updated_at = time.time()
        self.attempts += 1

    def mark_completed(self, result: str = ""):
        self.status = GoalStatus.COMPLETED
        self.completed_at = time.time()
        self.updated_at = time.time()
        self.progress = 1.0
        self.result = result

    def mark_failed(self, reason: str = ""):
        self.status = GoalStatus.FAILED
        self.updated_at = time.time()
        self.failure_reason = reason

    def update_progress(self, progress: float):
        self.progress = max(0.0, min(1.0, progress))
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ke dictionary."""
        d = asdict(self)
        d["status"] = self.status.value
        d["priority"] = self.priority.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Goal":
        """Deserialize dari dictionary."""
        data = data.copy()
        data["status"] = GoalStatus(data.get("status", "pending"))
        data["priority"] = GoalPriority(data.get("priority", 3))
        return cls(**data)

    def __repr__(self) -> str:
        indent = "  " * self.level
        status_symbol = {
            GoalStatus.PENDING: "○",
            GoalStatus.ACTIVE: "◉",
            GoalStatus.COMPLETED: "✓",
            GoalStatus.FAILED: "✗",
            GoalStatus.PAUSED: "⏸",
            GoalStatus.CANCELLED: "⊘",
        }.get(self.status, "?")
        return f"{indent}{status_symbol} [{self.goal_id}] {self.title} ({self.status.value})"


class GoalMemory:
    """
    Goal Memory Manager - Mengelola Hierarki Goals Agent.

    Persistent storage menggunakan SQLite. Mendukung:
    - CRUD goals
    - Goal decomposition (parent → children)
    - Priority-based retrieval
    - Progress tracking
    - Dependency management

    Terinspirasi dari BabyAGI dan AutoGPT yang menggunakan
    task list management sebagai inti dari autonomous agent loop.
    """

    def __init__(self, db_path: str = "./goals.db"):
        """
        Args:
            db_path: Path ke file SQLite database
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._initialize_db()
        logger.info(f"GoalMemory diinisialisasi: {db_path}")

    def _initialize_db(self):
        """Buat tabel goals jika belum ada."""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                goal_id TEXT PRIMARY KEY,
                parent_id TEXT,
                title TEXT NOT NULL,
                description TEXT,
                level INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                priority INTEGER DEFAULT 3,
                created_at REAL,
                updated_at REAL,
                completed_at REAL,
                deadline REAL,
                progress REAL DEFAULT 0.0,
                attempts INTEGER DEFAULT 0,
                max_attempts INTEGER DEFAULT 3,
                context TEXT DEFAULT '',
                failure_reason TEXT DEFAULT '',
                result TEXT DEFAULT '',
                tags TEXT DEFAULT '[]',
                depends_on TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}'
            )
        """)
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        """Dapatkan koneksi SQLite (thread-local)."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    # ─────────────────────────────
    # CRUD Operations
    # ─────────────────────────────

    def add_goal(self, goal: Goal) -> str:
        """Tambahkan goal baru ke database."""
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO goals VALUES (
                :goal_id, :parent_id, :title, :description, :level,
                :status, :priority, :created_at, :updated_at, :completed_at,
                :deadline, :progress, :attempts, :max_attempts, :context,
                :failure_reason, :result, :tags, :depends_on, :metadata
            )
        """, {
            "goal_id": goal.goal_id,
            "parent_id": goal.parent_id,
            "title": goal.title,
            "description": goal.description,
            "level": goal.level,
            "status": goal.status.value,
            "priority": goal.priority.value,
            "created_at": goal.created_at,
            "updated_at": goal.updated_at,
            "completed_at": goal.completed_at,
            "deadline": goal.deadline,
            "progress": goal.progress,
            "attempts": goal.attempts,
            "max_attempts": goal.max_attempts,
            "context": goal.context,
            "failure_reason": goal.failure_reason,
            "result": goal.result,
            "tags": json.dumps(goal.tags),
            "depends_on": json.dumps(goal.depends_on),
            "metadata": json.dumps(goal.metadata),
        })
        conn.commit()
        logger.debug(f"[GoalMemory] Goal ditambahkan: {goal.goal_id} '{goal.title}'")
        return goal.goal_id

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Ambil goal berdasarkan ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM goals WHERE goal_id = ?", (goal_id,)
        ).fetchone()
        return self._row_to_goal(row) if row else None

    def update_goal(self, goal: Goal):
        """Update goal yang sudah ada."""
        goal.updated_at = time.time()
        self.add_goal(goal)  # INSERT OR REPLACE

    def delete_goal(self, goal_id: str, cascade: bool = True) -> int:
        """
        Hapus goal berdasarkan ID.

        Args:
            goal_id: ID goal yang akan dihapus
            cascade: Apakah hapus semua child goals juga?

        Returns:
            Jumlah goal yang dihapus
        """
        conn = self._get_conn()
        count = 0

        if cascade:
            # Hapus semua descendants
            children = self.get_children(goal_id)
            for child in children:
                count += self.delete_goal(child.goal_id, cascade=True)

        conn.execute("DELETE FROM goals WHERE goal_id = ?", (goal_id,))
        conn.commit()
        return count + 1

    # ─────────────────────────────
    # Hierarchy Navigation
    # ─────────────────────────────

    def get_root_goals(self) -> List[Goal]:
        """Ambil semua root goals (level 0, tanpa parent)."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM goals WHERE parent_id IS NULL ORDER BY priority DESC, created_at ASC"
        ).fetchall()
        return [self._row_to_goal(r) for r in rows]

    def get_children(self, parent_id: str) -> List[Goal]:
        """Ambil semua child goals dari parent tertentu."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM goals WHERE parent_id = ? ORDER BY priority DESC, created_at ASC",
            (parent_id,)
        ).fetchall()
        return [self._row_to_goal(r) for r in rows]

    def get_tree(self, root_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Ambil seluruh goal tree dalam format nested dict.

        Args:
            root_id: ID root goal (None = ambil semua root goals)

        Returns:
            Nested dict merepresentasikan goal hierarchy
        """
        if root_id:
            root = self.get_goal(root_id)
            if not root:
                return {}
            return self._build_tree_node(root)
        else:
            roots = self.get_root_goals()
            return {
                "roots": [self._build_tree_node(root) for root in roots]
            }

    def _build_tree_node(self, goal: Goal) -> Dict[str, Any]:
        """Helper untuk membangun tree node secara rekursif."""
        node = goal.to_dict()
        children = self.get_children(goal.goal_id)
        if children:
            node["children"] = [self._build_tree_node(c) for c in children]
        return node

    # ─────────────────────────────
    # Status & Priority Management
    # ─────────────────────────────

    def get_active_goals(self) -> List[Goal]:
        """Ambil semua goals yang sedang aktif."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM goals WHERE status = 'active' ORDER BY priority DESC"
        ).fetchall()
        return [self._row_to_goal(r) for r in rows]

    def get_next_goal(self) -> Optional[Goal]:
        """
        Ambil goal berikutnya yang harus dikerjakan.

        Prioritas:
        1. Goals yang sedang ACTIVE
        2. Goals yang PENDING dan semua dependensinya sudah COMPLETED
        3. Goals dengan priority tertinggi
        4. Goals yang dibuat lebih awal (FIFO dalam priority yang sama)
        """
        # Cek yang sedang aktif
        active = self.get_active_goals()
        if active:
            # Ambil yang prioritasnya paling tinggi
            return sorted(active, key=lambda g: g.priority.value, reverse=True)[0]

        # Cari yang pending dan siap dikerjakan
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM goals WHERE status = 'pending' ORDER BY priority DESC, created_at ASC"
        ).fetchall()
        pending_goals = [self._row_to_goal(r) for r in rows]

        for goal in pending_goals:
            if self._are_dependencies_met(goal):
                return goal

        return None

    def _are_dependencies_met(self, goal: Goal) -> bool:
        """Cek apakah semua dependensi goal sudah selesai."""
        for dep_id in goal.depends_on:
            dep = self.get_goal(dep_id)
            if dep is None or dep.status != GoalStatus.COMPLETED:
                return False
        return True

    def mark_goal_status(self, goal_id: str, status: GoalStatus, **kwargs):
        """Update status goal dan propagate ke parent jika perlu."""
        goal = self.get_goal(goal_id)
        if not goal:
            return

        goal.status = status
        if status == GoalStatus.COMPLETED:
            goal.completed_at = time.time()
            goal.progress = 1.0
            goal.result = kwargs.get("result", "")
        elif status == GoalStatus.FAILED:
            goal.failure_reason = kwargs.get("reason", "")
        elif status == GoalStatus.ACTIVE:
            goal.attempts += 1

        self.update_goal(goal)

        # Update parent progress
        if goal.parent_id:
            self._update_parent_progress(goal.parent_id)

        logger.debug(
            f"[GoalMemory] Status diupdate: {goal_id} → {status.value}"
        )

    def _update_parent_progress(self, parent_id: str):
        """Recalculate progress parent berdasarkan progress anak-anaknya."""
        parent = self.get_goal(parent_id)
        if not parent:
            return

        children = self.get_children(parent_id)
        if not children:
            return

        completed = sum(1 for c in children if c.status == GoalStatus.COMPLETED)
        parent.progress = completed / len(children)

        if completed == len(children):
            parent.mark_completed()

        self.update_goal(parent)

        # Propagate ke grandparent
        if parent.parent_id:
            self._update_parent_progress(parent.parent_id)

    def add_subgoal(
        self,
        parent_id: str,
        title: str,
        description: str,
        priority: GoalPriority = GoalPriority.MEDIUM,
        depends_on: Optional[List[str]] = None,
        is_leaf: bool = False,
    ) -> Goal:
        """
        Tambahkan sub-goal ke parent yang ada.

        Otomatis menentukan level berdasarkan parent.
        """
        parent = self.get_goal(parent_id)
        if not parent:
            raise ValueError(f"Parent goal {parent_id} tidak ditemukan")

        child = Goal(
            title=title,
            description=description,
            parent_id=parent_id,
            level=parent.level + 1,
            priority=priority,
            depends_on=depends_on or [],
            metadata={"is_leaf": is_leaf},
        )
        self.add_goal(child)
        return child

    # ─────────────────────────────
    # Search & Query
    # ─────────────────────────────

    def search_goals(
        self,
        keyword: str = "",
        status: Optional[GoalStatus] = None,
        min_priority: GoalPriority = GoalPriority.MINIMAL,
    ) -> List[Goal]:
        """Cari goals berdasarkan keyword dan filter."""
        conn = self._get_conn()

        query = "SELECT * FROM goals WHERE 1=1"
        params = []

        if keyword:
            query += " AND (title LIKE ? OR description LIKE ?)"
            like = f"%{keyword}%"
            params.extend([like, like])

        if status:
            query += " AND status = ?"
            params.append(status.value)

        query += " AND priority >= ?"
        params.append(min_priority.value)

        query += " ORDER BY priority DESC, created_at ASC"

        rows = conn.execute(query, params).fetchall()
        return [self._row_to_goal(r) for r in rows]

    # ─────────────────────────────
    # Reporting & Visualization
    # ─────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Statistik goal management."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM goals").fetchone()[0]

        status_counts = {}
        for row in conn.execute(
            "SELECT status, COUNT(*) as cnt FROM goals GROUP BY status"
        ).fetchall():
            status_counts[row[0]] = row[1]

        return {
            "total_goals": total,
            "by_status": status_counts,
            "completion_rate": (
                round(status_counts.get("completed", 0) / max(1, total) * 100, 1)
            ),
            "active_count": status_counts.get("active", 0),
            "pending_count": status_counts.get("pending", 0),
        }

    def format_tree(self, root_id: Optional[str] = None) -> str:
        """Format goal tree sebagai ASCII tree untuk display."""
        lines = []

        def _format_node(goal: Goal):
            indent = "  " * goal.level
            status_symbol = {
                GoalStatus.PENDING: "○",
                GoalStatus.ACTIVE: "◉",
                GoalStatus.COMPLETED: "✓",
                GoalStatus.FAILED: "✗",
                GoalStatus.PAUSED: "⏸",
                GoalStatus.CANCELLED: "⊘",
            }.get(goal.status, "?")

            progress_bar = ""
            if goal.progress > 0:
                filled = int(goal.progress * 10)
                progress_bar = f" [{'█' * filled}{'░' * (10 - filled)}] {int(goal.progress * 100)}%"

            lines.append(
                f"{indent}{status_symbol} {goal.title}"
                f" (P{goal.priority.value}){progress_bar}"
            )

            for child in self.get_children(goal.goal_id):
                _format_node(child)

        if root_id:
            root = self.get_goal(root_id)
            if root:
                _format_node(root)
        else:
            for root in self.get_root_goals():
                _format_node(root)
                lines.append("")  # Separator antar root goals

        return "\n".join(lines)

    def _row_to_goal(self, row: sqlite3.Row) -> Goal:
        """Konversi SQLite row ke Goal object."""
        return Goal(
            goal_id=row["goal_id"],
            parent_id=row["parent_id"],
            title=row["title"],
            description=row["description"],
            level=row["level"],
            status=GoalStatus(row["status"]),
            priority=GoalPriority(row["priority"]),
            created_at=row["created_at"] or time.time(),
            updated_at=row["updated_at"] or time.time(),
            completed_at=row["completed_at"],
            deadline=row["deadline"],
            progress=row["progress"] or 0.0,
            attempts=row["attempts"] or 0,
            max_attempts=row["max_attempts"] or 3,
            context=row["context"] or "",
            failure_reason=row["failure_reason"] or "",
            result=row["result"] or "",
            tags=json.loads(row["tags"] or "[]"),
            depends_on=json.loads(row["depends_on"] or "[]"),
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def close(self):
        """Tutup koneksi database."""
        if self._conn:
            self._conn.close()
            self._conn = None
