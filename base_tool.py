"""
tools/base_tool.py - Abstract Base Tool
========================================
Abstraksi dasar untuk semua tools yang bisa digunakan agent.
Mendefinisikan interface, metadata, dan registry system.

Referensi Paper:
- ReAct (Yao et al., 2022): "We use a simple API to interact with WikiSearch,
  Calculator, and other tools"
  → Tools diakses lewat uniform interface
- Toolformer (Schick et al., 2023): Tool call format standardization
  → Tools punya nama, description, dan parameter schema
- AgentBench (Liu et al., 2023): Tool abstraction untuk benchmarking
  → Berbagai tools diuji dengan interface yang konsisten

Design Pattern:
- Abstract base class untuk semua tools
- Tool registry untuk dynamic dispatch
- ToolResult sebagai standar return type
- Schema validation untuk tool inputs
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Tool Result
# ─────────────────────────────────────────────

@dataclass
class ToolResult:
    """
    Standar return type untuk semua tool executions.

    Mengandung output, status, metadata eksekusi, dan error handling.
    Desain ini memungkinkan agent membedakan success/failure dan
    mengakses output terstruktur vs raw text.
    """
    success: bool                       # Apakah eksekusi berhasil?
    output: str                         # Output sebagai string (untuk LLM)
    data: Any = None                    # Structured data (opsional)
    error: str = ""                     # Pesan error jika gagal
    tool_name: str = ""                 # Nama tool yang dijalankan
    execution_time_ms: float = 0.0      # Waktu eksekusi
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.success:
            return self.output
        return f"ERROR: {self.error}"

    @property
    def formatted(self) -> str:
        """Format output untuk ditampilkan ke LLM."""
        if self.success:
            return f"[{self.tool_name}] Output:\n{self.output}"
        return f"[{self.tool_name}] Error: {self.error}"


# ─────────────────────────────────────────────
# Base Tool
# ─────────────────────────────────────────────

class BaseTool(ABC):
    """
    Abstract base class untuk semua agent tools.

    Setiap tool harus mengimplementasikan:
    - name: identifier unik tool
    - description: deskripsi untuk LLM (harus jelas dan informatif)
    - parameter_schema: dict mendefinisikan input parameters
    - execute(): logika eksekusi async

    Tools bisa async atau sync, tapi interface luar selalu async.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nama unik tool (lowercase, snake_case)."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Deskripsi tool untuk LLM.

        Harus menjelaskan:
        - Apa yang dilakukan tool
        - Kapan sebaiknya digunakan
        - Format input yang diharapkan
        - Contoh penggunaan
        """
        pass

    @property
    @abstractmethod
    def parameter_schema(self) -> Dict[str, Any]:
        """
        Schema parameter tool dalam format JSON Schema.

        Format:
        {
            "param_name": {
                "type": "string" | "integer" | "boolean" | "array",
                "description": "...",
                "required": True/False,
                "default": ...,
            }
        }
        """
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Jalankan tool dengan parameter yang diberikan.

        Returns:
            ToolResult dengan output dan metadata
        """
        pass

    async def safe_execute(self, **kwargs) -> ToolResult:
        """
        Wrapper aman untuk execute() dengan error handling dan timing.

        Gunakan ini daripada execute() langsung di agent loop.
        """
        start_time = time.time()

        try:
            # Validasi parameter
            validation_error = self._validate_params(kwargs)
            if validation_error:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Parameter tidak valid: {validation_error}",
                    tool_name=self.name,
                )

            # Jalankan dengan timeout
            timeout = self.timeout_seconds
            if timeout:
                result = await asyncio.wait_for(
                    self.execute(**kwargs),
                    timeout=timeout
                )
            else:
                result = await self.execute(**kwargs)

            result.tool_name = self.name
            result.execution_time_ms = (time.time() - start_time) * 1000

            logger.info(
                f"[{self.name}] Eksekusi berhasil "
                f"({result.execution_time_ms:.0f}ms)"
            )
            return result

        except asyncio.TimeoutError:
            error_msg = f"Tool timeout setelah {self.timeout_seconds}s"
            logger.warning(f"[{self.name}] {error_msg}")
            return ToolResult(
                success=False,
                output="",
                error=error_msg,
                tool_name=self.name,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"[{self.name}] Error: {error_msg}", exc_info=True)
            return ToolResult(
                success=False,
                output="",
                error=error_msg,
                tool_name=self.name,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    def _validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        """Validasi parameter berdasarkan schema."""
        schema = self.parameter_schema
        for param_name, param_def in schema.items():
            is_required = param_def.get("required", False)
            if is_required and param_name not in params:
                return f"Parameter wajib '{param_name}' tidak ditemukan"
        return None

    @property
    def timeout_seconds(self) -> Optional[float]:
        """Timeout eksekusi dalam detik (None = tidak ada timeout)."""
        return 30.0

    def format_for_prompt(self) -> str:
        """Format tool description untuk dimasukkan ke system prompt."""
        param_desc = []
        for name, schema in self.parameter_schema.items():
            required = "WAJIB" if schema.get("required") else "opsional"
            param_desc.append(
                f"  - {name} ({schema.get('type', 'any')}, {required}): "
                f"{schema.get('description', '')}"
            )

        params_str = "\n".join(param_desc) if param_desc else "  (tidak ada parameter)"
        return (
            f"Tool: {self.name}\n"
            f"Deskripsi: {self.description}\n"
            f"Parameter:\n{params_str}"
        )

    def __repr__(self) -> str:
        return f"Tool({self.name})"


# ─────────────────────────────────────────────
# Tool Registry
# ─────────────────────────────────────────────

class ToolRegistry:
    """
    Registry untuk mengelola semua tools yang tersedia untuk agent.

    Mendukung:
    - Registrasi tools
    - Dynamic dispatch berdasarkan nama
    - Generate tool descriptions untuk LLM
    - Tool chaining / composition
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self.call_history: List[Dict[str, Any]] = []
        logger.info("ToolRegistry diinisialisasi")

    def register(self, tool: BaseTool):
        """Daftarkan tool ke registry."""
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' sudah terdaftar, ditimpa")
        self._tools[tool.name] = tool
        logger.debug(f"[ToolRegistry] Registered: {tool.name}")

    def unregister(self, tool_name: str) -> bool:
        """Hapus tool dari registry."""
        if tool_name in self._tools:
            del self._tools[tool_name]
            return True
        return False

    def get(self, tool_name: str) -> Optional[BaseTool]:
        """Ambil tool berdasarkan nama."""
        return self._tools.get(tool_name)

    async def call(
        self,
        tool_name: str,
        tool_input: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> ToolResult:
        """
        Panggil tool berdasarkan nama.

        Terinspirasi dari ReAct: "Action: tool_name[input]"
        → Agent hanya perlu tahu nama tool dan inputnya

        Args:
            tool_name: Nama tool yang akan dipanggil
            tool_input: Dict parameter (opsional, bisa pakai **kwargs)
            **kwargs: Parameter langsung

        Returns:
            ToolResult dari eksekusi tool
        """
        tool = self.get(tool_name)

        if tool is None:
            available = list(self._tools.keys())
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Tool '{tool_name}' tidak ditemukan. "
                    f"Tools tersedia: {available}"
                ),
                tool_name=tool_name,
            )

        # Merge tool_input dan kwargs
        params = {**(tool_input or {}), **kwargs}

        # Jalankan tool
        result = await tool.safe_execute(**params)

        # Catat ke history
        self.call_history.append({
            "tool": tool_name,
            "params": params,
            "success": result.success,
            "timestamp": time.time(),
            "execution_time_ms": result.execution_time_ms,
        })

        return result

    @property
    def tools(self) -> List[BaseTool]:
        """List semua tools yang terdaftar."""
        return list(self._tools.values())

    @property
    def tool_names(self) -> List[str]:
        """List nama semua tools."""
        return list(self._tools.keys())

    def get_descriptions(self) -> str:
        """
        Generate deskripsi semua tools untuk system prompt.

        Format dioptimalkan untuk LLM agar mudah memilih tool yang tepat.
        """
        if not self._tools:
            return "Tidak ada tools tersedia."

        sections = ["=== Tools Tersedia ===\n"]
        for tool in self._tools.values():
            sections.append(tool.format_for_prompt())
            sections.append("")  # Blank line separator

        return "\n".join(sections)

    def get_stats(self) -> Dict[str, Any]:
        """Statistik penggunaan tools."""
        tool_usage: Dict[str, int] = {}
        for entry in self.call_history:
            tool_usage[entry["tool"]] = tool_usage.get(entry["tool"], 0) + 1

        success_count = sum(1 for e in self.call_history if e["success"])
        total_count = len(self.call_history)

        return {
            "registered_tools": len(self._tools),
            "total_calls": total_count,
            "success_rate": (
                round(success_count / max(1, total_count) * 100, 1)
            ),
            "tool_usage": tool_usage,
        }
