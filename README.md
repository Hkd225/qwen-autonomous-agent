# 🤖 Autonomous Agent - LLM Qwen 2.5 7B

A complete implementation of an **Autonomous Agent** based on the Qwen 2.5 7B LLM, inspired by the best AGI papers. Integrating ReAct, Reflexion, MemGPT, Tree of Thoughts, and Voyager into a single coherent system.

---

## 📄 Reference Papers

| Paper | Contribution to Agent |
|-------|-----------------------|
| **ReAct** (Yao et al., 2022) | Core Think→Act→Observe loop |
| **Reflexion** (Shinn et al., 2023) | Post-action self-reflection & verbal RL |
| **MemGPT** (Packer et al., 2023) | Hierarchical memory management |
| **Generative Agents** (Park et al., 2023) | Memory scoring (recency × relevance × importance) |
| **Tree of Thoughts** (Yao et al., 2023) | Multi-path planning |
| **Voyager** (Wang et al., 2023) | Skill library & curriculum learning |
| **SELF-RAG** (Asai et al., 2023) | Self-reflective RAG pipeline |
| **BabyAGI** (Nakajima, 2023) | Goal management & task prioritization |

---

## 🏗️ Architecture

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

## ⚙️ Setup & Installation

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) (for local inference)

### 2. Install Ollama & Model

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Qwen 2.5 7B model
ollama pull qwen2.5:7b

# Verification
ollama list
ollama run qwen2.5 "Hello!"
```

### 3. Install Python Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import chromadb, sentence_transformers; print('OK')"
```

### 4. (Optional) Configuration

```python
# Edit in main.py or create your own config:
config = AgentConfig(
    model_name="qwen2.5:7b",       # Ollama model name
    ollama_url="http://localhost:11434",  # Ollama URL
    max_iterations=15,              # Step limit per task
    storage_dir="./agent_storage",  # Storage directory
    use_planning=True,              # Enable planning
    use_reflection=True,            # Enable Reflexion
    use_skill_library=True,         # Enable skill reuse
)
```

---

## 🚀 Usage

### Mode 1: Single Task

```bash
# Default task (demo)
python main.py

# Custom task
python main.py --task "Find 5 recent papers about LLM agents"

# With verbose logging
python main.py --task "Calculate 10% compound interest for 20 years" --verbose
```

### Mode 2: Demo All Tasks

```bash
python main.py --mode demo
```

Runs 5 different demo tasks:
- Web research
- Mathematical calculation
- Conceptual analysis
- Coding task
- Goal planning

### Mode 3: Interactive

```bash
python main.py --mode interactive
```

Interactive chat mode. Type a task and the agent will complete it.
Special commands: `status`, `skills`, `goals`, `quit`.

### Mode 4: Goal Pursuit

```bash
python main.py --mode goals
```

Goal hierarchy demo: set ultimate goal → decompose → automatic pursuit.

### Complete CLI Options

```
--task TEXT          Task to be completed
--mode MODE          single | demo | interactive | goals
--model MODEL        Ollama model name (default: qwen2.5:7b)
--ollama-url URL     Ollama server URL
--max-steps N        Maximum steps (default: 15)
--storage-dir DIR    Storage directory (default: ./agent_storage)
--verbose            Verbose logging
--no-plan            Disable planning system
--no-reflect         Disable Reflexion
```

---

## 🔧 Usage as a Library

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
    trace = await agent.run("Explain the transformer architecture concept")
    print(trace.final_answer)
    print(f"Success: {trace.success}, Steps: {len(trace.steps)}")

    # With goal management
    goal = await agent.set_goal(
        title="Research AI Trends",
        description="Collect and analyze the latest AI trends 2024-2025",
    )

    # Auto-decompose and pursue
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
    def description(self): return "Tool description for the LLM"

    @property
    def parameter_schema(self):
        return {
            "input": {"type": "string", "required": True, "description": "Input data"}
        }

    async def execute(self, input: str) -> ToolResult:
        result = f"Processed: {input}"
        return ToolResult(success=True, output=result)

# Register to the agent
agent.tool_registry.register(MyCustomTool())
```

---

## 🧠 Detailed Components

### Memory System (MemGPT-inspired)

```
Working Memory (RAM analogue)
├── Max 6000 tokens
├── FIFO eviction (importance-protected)
└── Context window for the LLM

Episodic Memory (Hard disk analogue)
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
For each step:
1. PERCEIVE  → Collect working memory + retrieved memories
2. THINK     → Chain-of-Thought reasoning (JSON structured)
3. DECIDE    → Choose action or FINISH
4. ACT       → Execute tool with safe_execute()
5. OBSERVE   → Process results, add to working memory
6. REFLECT   → Evaluate if the step was successful (Reflexion)
7. UPDATE    → Save to episodic memory if important
```

---

## 🔍 Troubleshooting

### "Ollama is not available"
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve

# Ensure the model is pulled
ollama pull qwen2.5:7b
```

### "ChromaDB error" or "FAISS not available"
```bash
# Reinstall with a compatible version
pip install chromadb>=0.4.24 faiss-cpu>=1.8.0 --force-reinstall
```

### "sentence-transformers slow"
The embedding model will be downloaded the first time (~90MB).
Ensure an internet connection is available for the initial download.

### Memory error / GPU OOM (if using HuggingFace)
```python
config = AgentConfig(
    model_name="qwen2.5:7b",
    prefer_ollama=True,  # Always prefer Ollama (more memory efficient)
)
```

### Agent loop takes too long
```python
config = AgentConfig(
    max_iterations=8,       # Reduce iterations
    use_planning=False,     # Disable planning for simple tasks
    use_reflection=False,   # Disable reflection for speed
)
```

---

## 📊 Performance Tips

1. **Use Ollama** instead of HuggingFace for faster inference.
2. **Reduce `max_iterations`** for simple tasks (5-8 is sufficient).
3. **Enable the skill library** to save tokens on repetitive tasks.
4. **Use lower temperature** (0.3-0.5) for more consistent reasoning.
5. **Use `qwen2.5:3b`** if RAM is limited (faster, slightly lower accuracy).

---

## 📚 References

- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [Reflexion Paper](https://arxiv.org/abs/2303.11366)
- [MemGPT Paper](https://arxiv.org/abs/2310.08560)
- [Generative Agents Paper](https://arxiv.org/abs/2304.03442)
- [Tree of Thoughts Paper](https://arxiv.org/abs/2305.10601)
- [Voyager Paper](https://arxiv.org/abs/2305.16291)
- [SELF-RAG Paper](https://arxiv.org/abs/2310.11511)
- [Qwen 2.5 Technical Report](https://arxiv.org/abs/2412.15115)

---

## 📝 License

MIT License - free to use for research and development purposes.
