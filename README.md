# 🤖 Autonomous Agent - LLM Qwen 2.5 7B

Implementasi lengkap **Autonomous Agent** berbasis LLM Qwen 2.5 7B yang terinspirasi dari paper-paper AGI terbaik. Menggabungkan ReAct, Reflexion, MemGPT, Tree of Thoughts, dan Voyager dalam satu sistem yang coherent.

---

## 📄 Paper Referensi

| Paper | Kontribusi ke Agent |
|-------|---------------------|
| **ReAct** (Yao et al., 2022) | Core Think→Act→Observe loop |
| **Reflexion** (Shinn et al., 2023) | Post-action self-reflection & verbal RL |
| **MemGPT** (Packer et al., 2023) | Hierarchical memory management |
| **Generative Agents** (Park et al., 2023) | Memory scoring (recency × relevance × importance) |
| **Tree of Thoughts** (Yao et al., 2023) | Multi-path planning |
| **Voyager** (Wang et al., 2023) | Skill library & curriculum learning |
| **SELF-RAG** (Asai et al., 2023) | Self-reflective RAG pipeline |
| **BabyAGI** (Nakajima, 2023) | Goal management & task prioritization |

---

## 🏗️ Arsitektur

```
autonomous_agent/
├── main.py              # Entry point & demo modes
├── agent.py             # Main agent loop (ReAct + integrations)
├── llm.py               # LLM interface (Qwen 2.5 7B via Ollama/HF)
├── planning.py          # Tree of Thoughts + HTN planning
├── decision.py          # ReAct decision making + Reflexion
├── uncertainty.py       # Epistemic & aleatoric uncertainty tracking
├── self_improvement.py  # Skill library + performance tracking
├── memory/
│   ├── working_memory.py    # Short-term context management (MemGPT-style)
│   ├── episodic_memory.py   # Long-term experience storage (ChromaDB)
│   ├── semantic_memory.py   # Factual knowledge (FAISS)
│   ├── scoring.py           # Memory scoring (Generative Agents formula)
│   └── goal_memory.py       # Hierarchical goal tracking (SQLite)
└── tools/
    ├── base_tool.py          # Abstract tool interface
    ├── web_search.py         # DuckDuckGo/SerpAPI search
    ├── calculator.py         # Safe math evaluator
    ├── file_manager.py       # Sandboxed file operations
    ├── code_executor.py      # Safe Python execution
    └── memory_tool.py        # Memory query/store tool
```

---

## ⚙️ Setup & Instalasi

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) (untuk inferensi lokal)

### 2. Install Ollama & Model

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull model Qwen 2.5 7B
ollama pull qwen2.5:7b

# Verifikasi
ollama list
ollama run qwen2.5 "Hello!"
```

### 3. Install Python Dependencies

```bash
# Buat virtual environment (direkomendasikan)
python -m venv venv
source venv/bin/activate   # Linux/macOS
# atau: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verifikasi instalasi
python -c "import chromadb, sentence_transformers; print('OK')"
```

### 4. (Opsional) Konfigurasi

```python
# Edit di main.py atau buat config sendiri:
config = AgentConfig(
    model_name="qwen2.5:7b",       # Nama model Ollama
    ollama_url="http://localhost:11434",  # URL Ollama
    max_iterations=15,              # Batas langkah per task
    storage_dir="./agent_storage",  # Direktori penyimpanan
    use_planning=True,              # Aktifkan planning
    use_reflection=True,            # Aktifkan Reflexion
    use_skill_library=True,         # Aktifkan skill reuse
)
```

---

## 🚀 Penggunaan

### Mode 1: Single Task

```bash
# Task default (demo)
python main.py

# Custom task
python main.py --task "Cari 5 paper terbaru tentang LLM agents"

# Dengan verbose logging
python main.py --task "Hitung compound interest 10% selama 20 tahun" --verbose
```

### Mode 2: Demo Semua Tasks

```bash
python main.py --mode demo
```

Menjalankan 5 demo tasks berbeda:
- Riset web
- Kalkulasi matematika
- Analisis konseptual
- Coding task
- Goal planning

### Mode 3: Interactive

```bash
python main.py --mode interactive
```

Mode chat interaktif. Ketik task dan agent akan menyelesaikannya.
Perintah khusus: `status`, `skills`, `goals`, `quit`.

### Mode 4: Goal Pursuit

```bash
python main.py --mode goals
```

Demo goal hierarchy: set ultimate goal → decompose → pursuit otomatis.

### Opsi CLI Lengkap

```
--task TEXT          Task yang akan diselesaikan
--mode MODE          single | demo | interactive | goals
--model MODEL        Nama model Ollama (default: qwen2.5:7b)
--ollama-url URL     URL Ollama server
--max-steps N        Maksimal langkah (default: 15)
--storage-dir DIR    Direktori storage (default: ./agent_storage)
--verbose            Verbose logging
--no-plan            Disable planning system
--no-reflect         Disable Reflexion
```

---

## 🔧 Penggunaan sebagai Library

```python
import asyncio
from agent import AutonomousAgent, AgentConfig

async def main():
    config = AgentConfig(
        model_name="qwen2.5:7b",
        max_iterations=10,
    )
    agent = AutonomousAgent(config)

    # Single task
    trace = await agent.run("Jelaskan konsep transformer architecture")
    print(trace.final_answer)
    print(f"Success: {trace.success}, Steps: {len(trace.steps)}")

    # Dengan goal management
    goal = await agent.set_goal(
        title="Research AI Trends",
        description="Kumpulkan dan analisis tren AI terkini 2024-2025",
    )

    # Auto-decompose dan pursue
    subgoals = await agent.decompose_goal(goal, n_subgoals=3)
    traces = await agent.pursue_goals(max_goals=3)

    # Status
    status = agent.get_status()
    print(f"LLM calls: {status['llm']['total_calls']}")
    print(f"Memory: {status['working_memory']}")

    await agent.close()

asyncio.run(main())
```

### Custom Tool

```python
from tools.base_tool import BaseTool, ToolResult

class MyCustomTool(BaseTool):
    @property
    def name(self): return "my_tool"

    @property
    def description(self): return "Deskripsi tool untuk LLM"

    @property
    def parameter_schema(self):
        return {
            "input": {"type": "string", "required": True, "description": "Input data"}
        }

    async def execute(self, input: str) -> ToolResult:
        result = f"Processed: {input}"
        return ToolResult(success=True, output=result)

# Register ke agent
agent.tool_registry.register(MyCustomTool())
```

---

## 🧠 Komponen Detail

### Memory System (MemGPT-inspired)

```
Working Memory (RAM analog)
├── Max 6000 tokens
├── FIFO eviction (importance-protected)
└── Context window untuk LLM

Episodic Memory (Harddisk analog)
├── ChromaDB vector store
├── Semantic similarity search
└── Time-decay retrieval scoring

Semantic Memory (Knowledge Base)
├── FAISS index
├── Facts & concepts
└── Category-based organization

Goal Memory (Task Manager)
├── SQLite persistence
├── Hierarchical goals
└── Progress tracking
```

### Memory Scoring Formula (Generative Agents)

```
composite_score = 0.3 × recency + 0.4 × relevance + 0.3 × importance

recency = decay_base^(hours_since_creation)  # default decay_base = 0.99
relevance = cosine_similarity(memory_embedding, query_embedding)
importance = LLM_rating(1-10) normalized to [0, 1]
```

### ReAct Loop

```
Untuk setiap langkah:
1. PERCEIVE  → Kumpulkan working memory + retrieved memories
2. THINK     → Chain-of-Thought reasoning (JSON structured)
3. DECIDE    → Pilih action atau FINISH
4. ACT       → Eksekusi tool dengan safe_execute()
5. OBSERVE   → Proses hasil, tambah ke working memory
6. REFLECT   → Evaluasi apakah langkah berhasil (Reflexion)
7. UPDATE    → Simpan ke episodic memory jika penting
```

---

## 🔍 Troubleshooting

### "Ollama tidak tersedia"
```bash
# Cek Ollama berjalan
curl http://localhost:11434/api/tags

# Start Ollama jika belum
ollama serve

# Pastikan model sudah di-pull
ollama pull qwen2.5:7b
```

### "ChromaDB error" atau "FAISS not available"
```bash
# Reinstall dengan versi yang kompatibel
pip install chromadb>=0.4.24 faiss-cpu>=1.8.0 --force-reinstall
```

### "sentence-transformers slow"
Model embedding akan di-download pertama kali (~90MB).
Pastikan koneksi internet tersedia untuk download pertama.

### Memory error / GPU OOM (jika pakai HuggingFace)
```python
config = AgentConfig(
    model_name="qwen2.5:7b",
    prefer_ollama=True,  # Selalu prefer Ollama (lebih efisien memory)
)
```

### Agent loop terlalu lama
```python
config = AgentConfig(
    max_iterations=8,       # Kurangi iterasi
    use_planning=False,     # Disable planning untuk task sederhana
    use_reflection=False,   # Disable reflection untuk kecepatan
)
```

---

## 📊 Performance Tips

1. **Gunakan Ollama** daripada HuggingFace untuk inference yang lebih cepat
2. **Kurangi `max_iterations`** untuk task sederhana (5-8 cukup)
3. **Aktifkan skill library** untuk menghemat tokens pada task repetitif
4. **Gunakan temperature rendah** (0.3-0.5) untuk reasoning yang lebih konsisten
5. **Gunakan `qwen2.5:3b`** jika RAM terbatas (lebih cepat, akurasi sedikit lebih rendah)

---

## 📚 Referensi

- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [Reflexion Paper](https://arxiv.org/abs/2303.11366)
- [MemGPT Paper](https://arxiv.org/abs/2310.08560)
- [Generative Agents Paper](https://arxiv.org/abs/2304.03442)
- [Tree of Thoughts Paper](https://arxiv.org/abs/2305.10601)
- [Voyager Paper](https://arxiv.org/abs/2305.16291)
- [SELF-RAG Paper](https://arxiv.org/abs/2310.11511)
- [Qwen 2.5 Technical Report](https://arxiv.org/abs/2412.15115)

---

## 📝 Lisensi

MIT License - bebas digunakan untuk keperluan riset dan pengembangan.
