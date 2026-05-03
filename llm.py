"""
llm.py - LLM Interface untuk Qwen 2.5 7B
=========================================
Modul ini menyediakan abstraksi unified untuk berinteraksi dengan Qwen 2.5 7B
melalui dua backend: Ollama (primary) dan HuggingFace Transformers (fallback).

Referensi Paper:
- ReAct (Yao et al., 2022): "ReAct: Synergizing Reasoning and Acting in Language Models"
  → LLM digunakan sebagai engine untuk reasoning (Thought) sebelum acting (Action)
- Chain-of-Thought (Wei et al., 2022): Mendukung structured prompting untuk CoT
- SELF-RAG (Asai et al., 2023): LLM digunakan untuk generate & critique sekaligus

Fitur:
- Async support dengan asyncio
- Retry logic dengan exponential backoff
- Temperature & sampling parameter control
- Structured JSON output parsing
- Token usage tracking
- Streaming support
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class LLMConfig:
    """Konfigurasi untuk LLM interface."""
    # Model settings
    model_name: str = "qwen2.5:7b"
    ollama_base_url: str = "http://localhost:11434"
    hf_model_id: str = "Qwen/Qwen2.5-7B-Instruct"

    # Sampling parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 2048
    repeat_penalty: float = 1.1

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    timeout: float = 120.0

    # Backend preference
    prefer_ollama: bool = True
    use_json_mode: bool = False


@dataclass
class LLMResponse:
    """Struktur response dari LLM."""
    content: str                        # Teks response
    model: str = ""                     # Nama model yang digunakan
    prompt_tokens: int = 0              # Token input
    completion_tokens: int = 0          # Token output
    total_tokens: int = 0               # Total token
    finish_reason: str = "stop"         # Alasan berhenti
    raw_response: Dict = field(default_factory=dict)  # Response mentah
    latency_ms: float = 0.0            # Latensi dalam milliseconds
    backend: str = "ollama"             # Backend yang digunakan

    @property
    def parsed_json(self) -> Optional[Dict]:
        """Parse content sebagai JSON jika memungkinkan."""
        try:
            # Coba ekstrak JSON dari markdown code block
            content = self.content.strip()
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()
            return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return None


@dataclass
class Message:
    """Pesan dalam format chat."""
    role: str          # "system", "user", atau "assistant"
    content: str       # Isi pesan

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


# ─────────────────────────────────────────────
# Ollama Backend
# ─────────────────────────────────────────────

class OllamaBackend:
    """
    Backend untuk Ollama - menjalankan Qwen 2.5 7B secara lokal.

    Ollama menyediakan REST API yang kompatibel dengan OpenAI format,
    memungkinkan inferensi lokal tanpa perlu API key.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.ollama_base_url
        self.client = httpx.AsyncClient(timeout=config.timeout)

    async def check_availability(self) -> bool:
        """Cek apakah Ollama server berjalan dan model tersedia."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                # Cek apakah model Qwen tersedia
                model_available = any(
                    self.config.model_name.split(":")[0] in m
                    for m in models
                )
                if not model_available:
                    logger.warning(
                        f"Model {self.config.model_name} tidak ditemukan di Ollama. "
                        f"Tersedia: {models}"
                    )
                return True
            return False
        except Exception as e:
            logger.debug(f"Ollama tidak tersedia: {e}")
            return False

    async def generate(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate response dari Ollama."""
        start_time = time.time()

        # Build request payload
        payload = {
            "model": self.config.model_name,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature or self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "num_predict": max_tokens or self.config.max_tokens,
                "repeat_penalty": self.config.repeat_penalty,
            }
        }

        # Aktifkan JSON mode jika diminta
        if json_mode or self.config.use_json_mode:
            payload["format"] = "json"

        response = await self.client.post(
            f"{self.base_url}/api/chat",
            json=payload
        )
        response.raise_for_status()
        data = response.json()

        latency = (time.time() - start_time) * 1000
        content = data.get("message", {}).get("content", "")

        return LLMResponse(
            content=content,
            model=data.get("model", self.config.model_name),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            finish_reason=data.get("done_reason", "stop"),
            raw_response=data,
            latency_ms=latency,
            backend="ollama",
        )

    async def stream(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream response token demi token dari Ollama."""
        payload = {
            "model": self.config.model_name,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
            "options": {
                "temperature": temperature or self.config.temperature,
                "top_p": self.config.top_p,
            }
        }

        async with self.client.stream(
            "POST", f"{self.base_url}/api/chat", json=payload
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

    async def close(self):
        """Tutup HTTP client."""
        await self.client.aclose()


# ─────────────────────────────────────────────
# HuggingFace Backend (Fallback)
# ─────────────────────────────────────────────

class HuggingFaceBackend:
    """
    Backend fallback menggunakan HuggingFace Transformers.

    Digunakan ketika Ollama tidak tersedia. Memuat model langsung
    ke memori menggunakan transformers library.

    Catatan: Membutuhkan GPU dengan minimal 16GB VRAM untuk full precision,
    atau 8GB dengan quantization 8-bit.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._pipeline = None

    def _load_model(self):
        """Lazy load model ke memori (dipanggil pertama kali dibutuhkan)."""
        if self._pipeline is not None:
            return

        logger.info(f"Memuat model HuggingFace: {self.config.hf_model_id}")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.hf_model_id,
                trust_remote_code=True
            )

            # Deteksi device yang tersedia
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            logger.info(f"Menggunakan device: {device}")

            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.hf_model_id,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )

            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                device=0 if device == "cuda" else -1,
            )
            logger.info("Model berhasil dimuat!")

        except ImportError as e:
            raise RuntimeError(
                f"Gagal memuat transformers: {e}. "
                "Install dengan: pip install transformers torch"
            )

    async def generate(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate response menggunakan HuggingFace pipeline."""
        # Load model secara lazy
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)

        start_time = time.time()

        # Format messages sebagai chat template
        chat_messages = [m.to_dict() for m in messages]

        # Jalankan inferensi di thread pool agar tidak block event loop
        def _run_inference():
            text = self._tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            outputs = self._pipeline(
                text,
                max_new_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature or self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            return outputs[0]["generated_text"]

        result = await loop.run_in_executor(None, _run_inference)

        # Ekstrak hanya bagian yang baru digenerate
        # (hapus prompt asli dari output)
        input_text_len = len(self._tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        ))
        content = result[input_text_len:].strip()

        latency = (time.time() - start_time) * 1000

        return LLMResponse(
            content=content,
            model=self.config.hf_model_id,
            latency_ms=latency,
            backend="huggingface",
        )

    async def close(self):
        """Bebaskan memori model."""
        if self._model is not None:
            try:
                import torch
                del self._model
                del self._pipeline
                torch.cuda.empty_cache()
            except Exception:
                pass


# ─────────────────────────────────────────────
# Main LLM Interface
# ─────────────────────────────────────────────

class LLMInterface:
    """
    Interface utama untuk LLM Qwen 2.5 7B.

    Mengabstraksi backend (Ollama/HuggingFace) dan menyediakan:
    - Retry logic dengan exponential backoff
    - Token counting & usage tracking
    - Structured output parsing
    - Prompt template management

    Terinspirasi dari ReAct paper yang menggunakan LLM sebagai
    reasoning engine yang bisa dipanggil berulang kali dengan konteks.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.ollama = OllamaBackend(self.config)
        self.hf = HuggingFaceBackend(self.config)

        # Statistik penggunaan
        self.total_calls: int = 0
        self.total_tokens: int = 0
        self.total_latency_ms: float = 0.0
        self.error_count: int = 0

        # Backend aktif
        self._active_backend: Optional[str] = None

        logger.info(f"LLMInterface diinisialisasi dengan model: {self.config.model_name}")

    async def _get_active_backend(self) -> str:
        """Tentukan backend yang akan digunakan."""
        if self._active_backend:
            return self._active_backend

        if self.config.prefer_ollama:
            if await self.ollama.check_availability():
                self._active_backend = "ollama"
                logger.info("Menggunakan backend: Ollama")
            else:
                self._active_backend = "huggingface"
                logger.info("Ollama tidak tersedia, fallback ke HuggingFace")
        else:
            self._active_backend = "huggingface"

        return self._active_backend

    async def generate(
        self,
        messages: Union[List[Message], List[Dict[str, str]], str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate response dari LLM dengan retry logic.

        Args:
            messages: List pesan atau string prompt sederhana
            temperature: Override temperature (None = gunakan config default)
            max_tokens: Override max tokens
            json_mode: Paksa output dalam format JSON
            system_prompt: System prompt tambahan

        Returns:
            LLMResponse dengan content dan metadata
        """
        # Normalisasi input ke format Message
        normalized = self._normalize_messages(messages, system_prompt)

        # Retry loop dengan exponential backoff
        last_error = None
        delay = self.config.retry_delay

        for attempt in range(self.config.max_retries):
            try:
                backend = await self._get_active_backend()

                if backend == "ollama":
                    response = await self.ollama.generate(
                        normalized, temperature, max_tokens, json_mode
                    )
                else:
                    response = await self.hf.generate(
                        normalized, temperature, max_tokens, json_mode
                    )

                # Update statistik
                self.total_calls += 1
                self.total_tokens += response.total_tokens
                self.total_latency_ms += response.latency_ms

                logger.debug(
                    f"LLM call berhasil | tokens: {response.total_tokens} | "
                    f"latency: {response.latency_ms:.0f}ms"
                )
                return response

            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning(f"HTTP error (attempt {attempt+1}): {e}")
                if e.response.status_code == 429:  # Rate limit
                    delay *= 3  # Tunggu lebih lama
            except Exception as e:
                last_error = e
                logger.warning(f"LLM error (attempt {attempt+1}/{self.config.max_retries}): {e}")
                # Reset backend cache untuk mencoba ulang deteksi
                if attempt > 0:
                    self._active_backend = None

            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(delay)
                delay *= self.config.retry_backoff

        self.error_count += 1
        raise RuntimeError(
            f"LLM gagal setelah {self.config.max_retries} percobaan. "
            f"Error terakhir: {last_error}"
        )

    async def generate_json(
        self,
        messages: Union[List[Message], List[Dict[str, str]], str],
        schema_hint: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate output terstruktur dalam format JSON.

        Menambahkan instruksi JSON ke system prompt dan memparse hasilnya.
        Digunakan untuk structured outputs seperti planning, scoring, dll.

        Args:
            messages: Pesan input
            schema_hint: Deskripsi schema JSON yang diharapkan
            system_prompt: System prompt tambahan

        Returns:
            Dict hasil parsing JSON
        """
        # Tambahkan instruksi JSON ke system prompt
        json_instruction = (
            "Respond ONLY with valid JSON. Do not include markdown code blocks, "
            "explanations, or any text outside the JSON object."
        )
        if schema_hint:
            json_instruction += f"\nExpected schema: {schema_hint}"

        if system_prompt:
            enhanced_system = f"{system_prompt}\n\n{json_instruction}"
        else:
            enhanced_system = json_instruction

        response = await self.generate(
            messages,
            json_mode=True,
            system_prompt=enhanced_system,
            **kwargs,
        )

        # Coba parse JSON
        parsed = response.parsed_json
        if parsed is not None:
            return parsed

        # Jika gagal, kembalikan sebagai dict dengan key 'content'
        logger.warning("Gagal parse JSON response, mengembalikan raw content")
        return {"content": response.content, "parse_error": True}

    async def stream(
        self,
        messages: Union[List[Message], List[Dict[str, str]], str],
        temperature: Optional[float] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream response token demi token.

        Berguna untuk output panjang yang ingin ditampilkan real-time.
        """
        normalized = self._normalize_messages(messages)
        backend = await self._get_active_backend()

        if backend == "ollama":
            async for token in self.ollama.stream(normalized, temperature):
                yield token
        else:
            # HuggingFace tidak mendukung streaming langsung, fallback ke batch
            response = await self.hf.generate(normalized, temperature)
            yield response.content

    def _normalize_messages(
        self,
        messages: Union[List[Message], List[Dict[str, str]], str],
        system_prompt: Optional[str] = None,
    ) -> List[Message]:
        """Normalisasi berbagai format input ke List[Message]."""
        result: List[Message] = []

        # Tambahkan system prompt jika ada
        if system_prompt:
            result.append(Message(role="system", content=system_prompt))

        if isinstance(messages, str):
            result.append(Message(role="user", content=messages))
        elif isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, Message):
                    result.append(msg)
                elif isinstance(msg, dict):
                    result.append(Message(
                        role=msg.get("role", "user"),
                        content=msg.get("content", "")
                    ))
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Kembalikan statistik penggunaan LLM."""
        avg_latency = (
            self.total_latency_ms / self.total_calls
            if self.total_calls > 0 else 0
        )
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": round(avg_latency, 2),
            "error_count": self.error_count,
            "active_backend": self._active_backend,
        }

    async def close(self):
        """Tutup semua koneksi dan bebaskan resource."""
        await self.ollama.close()
        await self.hf.close()
        logger.info("LLMInterface ditutup")


# ─────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────

class PromptTemplates:
    """
    Kumpulan template prompt yang terinspirasi dari berbagai paper.

    Setiap template dirancang untuk use case spesifik dalam agentic pipeline.
    """

    REACT_SYSTEM = """Kamu adalah agen AI yang menggunakan pola ReAct (Reasoning + Acting).
Untuk setiap langkah, kamu HARUS mengikuti format berikut:

Thought: [Analisis situasi dan putuskan apa yang perlu dilakukan]
Action: [Nama tool yang akan digunakan]
Action Input: [Input untuk tool tersebut dalam format JSON]

Setelah menerima Observation dari tool, lanjutkan cycle ini sampai task selesai.
Ketika sudah selesai, gunakan:
Final Answer: [Jawaban final atau hasil task]

Tools yang tersedia: {tools}
"""

    REFLEXION_CRITIC = """Kamu adalah critic yang menganalisis performa agen AI.
Analisis langkah-langkah berikut dan berikan critique yang konstruktif:

Task: {task}
Langkah yang diambil: {steps}
Hasil: {outcome}

Berikan reflection dalam format JSON:
{{
    "success": bool,
    "what_went_well": ["list hal yang berjalan baik"],
    "what_went_wrong": ["list kesalahan atau inefficiency"],
    "lessons_learned": ["list pelajaran untuk iterasi berikutnya"],
    "suggested_improvements": ["list saran konkrit"],
    "confidence_score": float (0-1)
}}
"""

    PLANNING_SYSTEM = """Kamu adalah perencana AI yang menggunakan Tree of Thoughts.
Buat rencana hierarkis untuk menyelesaikan task berikut.

Task: {task}
Konteks: {context}
Tools tersedia: {tools}

Buat rencana dalam format JSON:
{{
    "goal": "deskripsi goal utama",
    "sub_goals": [
        {{
            "id": "sg_1",
            "description": "sub-goal 1",
            "tasks": [
                {{
                    "id": "t_1",
                    "action": "nama tool",
                    "description": "deskripsi task",
                    "depends_on": [],
                    "priority": 1
                }}
            ]
        }}
    ],
    "estimated_steps": int,
    "feasibility_score": float (0-1)
}}
"""

    IMPORTANCE_SCORER = """Rate kepentingan informasi berikut untuk agent dalam menyelesaikan task-nya.
Berikan skor 1-10 dimana:
1-3: Tidak penting / noise
4-6: Mungkin relevan
7-9: Penting
10: Sangat kritis

Informasi: {content}
Task konteks: {context}

Respond dengan JSON: {{"score": int, "reason": "penjelasan singkat"}}
"""

    COT_SYSTEM = """Kamu adalah problem solver yang menggunakan Chain-of-Thought reasoning.
Sebelum memberikan jawaban, SELALU:
1. Uraikan masalah menjadi komponen kecil
2. Analisis setiap komponen secara sistematis
3. Sintesis solusi dari analisis tersebut
4. Verifikasi solusi sebelum menyimpulkan

Gunakan format:
<thinking>
[Langkah-langkah reasoning kamu]
</thinking>

<answer>
[Jawaban final]
</answer>
"""
