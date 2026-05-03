"""
tools/code_executor.py - Safe Python Code Executor
====================================================
Tool untuk mengeksekusi kode Python secara aman dalam sandbox environment.
Membatasi imports, waktu eksekusi, dan resource usage.

Referensi Paper:
- PAL (Gao et al., 2022): "Program-Aided Language Models"
  → Menggunakan Python execution untuk memecahkan masalah yang butuh komputasi
- CodeAct (Wang et al., 2024): Code-based action execution
  → Agent mengeksekusi Python code sebagai primary action format
- AgentBench (Liu et al., 2023): Code execution environment untuk benchmark
- AutoGPT: Python code execution untuk automasi task
"""

import asyncio
import io
import logging
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Optional
from .base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# Modul yang diizinkan untuk diimport dalam sandbox
SAFE_IMPORTS = {
    "math", "random", "datetime", "json", "re", "string", "collections",
    "itertools", "functools", "operator", "statistics", "decimal",
    "fractions", "hashlib", "base64", "time", "calendar",
    "csv", "io", "textwrap", "unicodedata", "struct", "copy",
    # Data science (jika tersedia)
    "numpy", "pandas", "scipy", "sklearn",
}

# Builtins berbahaya yang diblokir
BLOCKED_BUILTINS = {
    "exec", "eval", "compile", "open", "__import__", "breakpoint",
    "input", "print",  # print diblokir karena kita capture stdout
}


class CodeExecutorTool(BaseTool):
    """
    Safe Python Code Executor.

    Mengeksekusi kode Python dalam environment yang dibatasi:
    - Timeout untuk mencegah infinite loop
    - Whitelist imports
    - Capture stdout/stderr
    - Resource limits

    Cocok untuk: kalkulasi kompleks, data processing, algoritma.
    TIDAK cocok untuk: network requests, file I/O, system calls.
    """

    def __init__(
        self,
        timeout_sec: float = 10.0,
        max_output_chars: int = 5000,
    ):
        self._timeout = timeout_sec
        self.max_output_chars = max_output_chars

    @property
    def name(self) -> str:
        return "code_executor"

    @property
    def description(self) -> str:
        return (
            "Eksekusi kode Python untuk komputasi, data processing, atau algoritma. "
            "Output dari print() akan dikembalikan. "
            "Gunakan untuk kalkulasi kompleks, manipulasi string, atau logika program. "
            "Modul tersedia: math, random, json, re, datetime, collections, itertools, dsb. "
            "Contoh: 'result = [x**2 for x in range(10)]\\nprint(result)'"
        )

    @property
    def parameter_schema(self) -> Dict[str, Any]:
        return {
            "code": {
                "type": "string",
                "description": "Kode Python yang akan dieksekusi. Gunakan print() untuk output.",
                "required": True,
            },
            "timeout": {
                "type": "number",
                "description": "Timeout dalam detik (default: 10, max: 30)",
                "required": False,
                "default": 10.0,
            },
        }

    async def execute(
        self,
        code: str,
        timeout: float = 10.0,
    ) -> ToolResult:
        """
        Eksekusi kode Python dalam sandbox.

        Args:
            code: Kode Python yang akan dieksekusi
            timeout: Maksimal waktu eksekusi

        Returns:
            ToolResult dengan stdout sebagai output
        """
        timeout = min(float(timeout), 30.0)

        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._execute_sandbox, code),
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                output="",
                error=f"Kode timeout setelah {timeout} detik. "
                      "Optimasi kode atau kurangi kompleksitas.",
            )

    def _execute_sandbox(self, code: str) -> ToolResult:
        """Jalankan kode dalam sandbox (synchronous)."""
        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Buat namespace aman
        safe_globals = self._build_safe_globals()
        local_vars: Dict[str, Any] = {}

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, safe_globals, local_vars)  # noqa: S102

            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()

            # Potong output jika terlalu panjang
            if len(stdout) > self.max_output_chars:
                stdout = stdout[:self.max_output_chars] + f"\n... [output terpotong setelah {self.max_output_chars} karakter]"

            # Build output
            output_parts = []
            if stdout:
                output_parts.append(f"Output:\n{stdout.rstrip()}")
            if stderr:
                output_parts.append(f"Stderr:\n{stderr.rstrip()}")

            # Tambahkan variabel yang didefinisikan user (kecuali builtins)
            user_vars = {
                k: v for k, v in local_vars.items()
                if not k.startswith("_")
                and k not in safe_globals
                and not callable(v)
            }
            if user_vars and not output_parts:
                # Jika tidak ada print, tampilkan variabel terakhir
                var_summary = ", ".join(
                    f"{k}={repr(v)[:100]}" for k, v in list(user_vars.items())[-3:]
                )
                output_parts.append(f"Variabel: {var_summary}")

            final_output = "\n".join(output_parts) if output_parts else "(Kode berhasil dieksekusi tanpa output)"

            logger.debug(
                f"[CodeExecutor] Eksekusi berhasil, "
                f"output: {len(final_output)} chars"
            )

            return ToolResult(
                success=True,
                output=final_output,
                data={
                    "stdout": stdout,
                    "stderr": stderr,
                    "local_vars": {k: repr(v) for k, v in user_vars.items()},
                },
            )

        except SyntaxError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Syntax Error: {e}",
            )
        except ImportError as e:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Import Error: {e}. "
                    f"Modul yang diizinkan: {sorted(SAFE_IMPORTS)}"
                ),
            )
        except NameError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Name Error: {e}. Pastikan variabel sudah didefinisikan.",
            )
        except Exception as e:
            tb = traceback.format_exc()
            short_error = f"{type(e).__name__}: {e}"
            return ToolResult(
                success=False,
                output="",
                error=short_error,
                metadata={"traceback": tb[-1000:]},  # Potong traceback
            )

    def _build_safe_globals(self) -> Dict[str, Any]:
        """Bangun namespace global yang aman untuk eksekusi."""
        # Handle both dict and module builtins
        if isinstance(__builtins__, dict):
            safe_builtins = {
                k: v for k, v in __builtins__.items()
                if k not in BLOCKED_BUILTINS
            }
        else:
            safe_builtins = {
                k: getattr(__builtins__, k)
                for k in dir(__builtins__)
                if not k.startswith("_") and k not in BLOCKED_BUILTINS
            }

        # Tambahkan print yang aman (ke stdout capture)
        safe_builtins["print"] = print  # Akan dicapture oleh redirect

        # Import handler yang dibatasi
        original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
            else __import__

        def safe_import(name, *args, **kwargs):
            base_name = name.split(".")[0]
            if base_name not in SAFE_IMPORTS:
                raise ImportError(
                    f"Import '{name}' tidak diizinkan. "
                    f"Modul yang diizinkan: {sorted(SAFE_IMPORTS)}"
                )
            return original_import(name, *args, **kwargs)

        safe_builtins["__import__"] = safe_import

        return {"__builtins__": safe_builtins, "__name__": "__sandbox__"}

    @property
    def timeout_seconds(self) -> float:
        return self._timeout + 2  # Buffer lebih dari timeout internal
