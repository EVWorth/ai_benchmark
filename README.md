# ai_benchmark

A lightweight tool for benchmarking **local AI models** on tokens-per-second throughput and reasoning quality.  Designed to squeeze maximum performance out of models running on consumer hardware (e.g. AMD RX 7900 XTX, 24 GB VRAM).

---

## Goals

| Goal | How |
|------|-----|
| Measure raw inference speed | Wall-clock tokens/sec per prompt |
| Evaluate reasoning quality | Curated prompt suite with auto-scoring |
| Compare backends | Ollama vs. llama.cpp (and future backends) |
| Tune for best TPS/quality balance | Save JSON results and run `compare` across configs |

---

## Supported backends

| Backend | Default URL | Notes |
|---------|-------------|-------|
| **Ollama** | `http://localhost:11434` | `ollama serve` |
| **llama.cpp** | `http://localhost:8080` | `llama-server -m model.gguf` |

---

## Quick start

### 1 – Install

```bash
git clone https://github.com/EVWorth/ai_benchmark.git
cd ai_benchmark
pip install -e ".[dev]"
```

### 2 – Start your backend

**Ollama**
```bash
ollama serve
ollama pull llama3.2       # or any model you want to test
```

**llama.cpp**
```bash
llama-server -m /path/to/model.gguf --port 8080 --n-gpu-layers 999
```

### 3 – Run the benchmark

```bash
# Ollama (default backend)
ai-benchmark run --model llama3.2

# llama.cpp
ai-benchmark run --backend llamacpp --model mistral-7b-q4 --output results/llamacpp_mistral.json

# Restrict to specific prompt categories
ai-benchmark run --model llama3.2 --categories math --categories reasoning

# Tune generation parameters
ai-benchmark run --model llama3.2 --temperature 0.0 --max-tokens 256
```

### 4 – Save and compare results

```bash
# Benchmark several models, saving each run
ai-benchmark run --model llama3.2         --output results/llama3.2.json
ai-benchmark run --model mistral          --output results/mistral.json
ai-benchmark run --backend llamacpp --model mistral-q4_k_m --output results/llamacpp_q4.json

# Summarise all saved runs side-by-side
ai-benchmark compare --results-dir results/
```

---

## CLI reference

```
ai-benchmark run        Run the benchmark suite against a single model
ai-benchmark compare    Compare results from multiple saved JSON files
ai-benchmark list-models  List models available in the specified backend
```

### `run` options

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` / `-b` | `ollama` | Backend: `ollama` or `llamacpp` |
| `--host` | _(backend default)_ | Override backend URL |
| `--model` / `-m` | _(required)_ | Model name to benchmark |
| `--categories` / `-c` | _(all)_ | Filter prompt categories (repeatable) |
| `--output` / `-o` | _(none)_ | Save full results to JSON file |
| `--temperature` | `0.0` | Sampling temperature |
| `--max-tokens` | `256` | Max completion tokens per prompt |

---

## Benchmark prompt suite

20 prompts across 4 categories, all auto-scored:

| Category | Prompts | Scored by |
|----------|---------|-----------|
| `math` | Arithmetic, word problems, geometry | Exact number extraction |
| `reasoning` | Logic, syllogisms, lateral thinking | Keyword / phrase match |
| `knowledge` | Science, history, programming facts | Keyword / phrase match |
| `coding` | Python functions, complexity, git | Phrase / code fragment match |

Add your own prompts by editing `prompts/benchmark_prompts.json`.  Each entry needs:

```json
{
  "id":         "my_001",
  "category":   "custom",
  "prompt":     "Question text here.",
  "expected":   "expected answer",
  "match_type": "contains_phrase"
}
```

Supported `match_type` values: `exact_number`, `exact_word`, `contains_word`, `contains_phrase`.

---

## Output

### Console (per-run)

```
ollama / llama3.2 – Prompt Results
 ID             Category   TPS    Tokens  Pass  Expected   Got
 math_001       math       48.2   3       ✓     408        408
 reasoning_001  reasoning  52.1   2       ✓     YES        YES
 ...

Avg tokens/sec        48.7
Avg completion tokens 4.2
Quality score         90.0%  (18/20)
  coding              80.0%
  knowledge           100.0%
  math                100.0%
  reasoning           80.0%
```

### Console (comparison)

```
Model Comparison
╔══════════╤══════════════════╤═══════╤═════════╤═════════╤════════╗
║ Backend  │ Model            │ Avg TPS│ Quality │ Prompts │ Errors ║
╠══════════╪══════════════════╪═══════╪═════════╪═════════╪════════╣
║ llamacpp │ mistral-q4_k_m   │  71.3  │  85.0%  │  20     │  0     ║
║ ollama   │ llama3.2         │  48.7  │  90.0%  │  20     │  0     ║
╚══════════╧══════════════════╧═══════╧═════════╧═════════╧════════╝
```

### JSON (saved with `--output`)

```json
{
  "timestamp": "2026-05-06T01:01:01Z",
  "model": "llama3.2",
  "backend": "ollama",
  "summary": {
    "avg_tokens_per_second": 48.7,
    "quality_score": 0.9,
    "quality_by_category": { "math": 1.0, "reasoning": 0.8, ... },
    "total_prompts": 20,
    "passed": 18,
    "errors": 0
  },
  "results": [ ... ],
  "scores":  [ ... ]
}
```

---

## Development

```bash
# Run tests
pytest

# Run a specific test module
pytest tests/test_scoring.py -v
```

### Project layout

```
benchmark/
├── cli.py           – Click CLI entry point
├── engine.py        – Orchestrates prompt execution
├── scoring.py       – Scores responses against expected answers
├── reporter.py      – Console tables and JSON output
└── runners/
    ├── base.py           – Abstract runner + InferenceResult dataclass
    ├── ollama_runner.py  – Ollama /api/generate backend
    └── llamacpp_runner.py– llama.cpp /completion backend
prompts/
└── benchmark_prompts.json – 20 curated benchmark prompts
tests/
└── test_*.py        – Unit tests (51 tests, fully offline/mocked)
```

---

## Tips for maximising tokens/sec on AMD RX 7900 XTX

- **Use GGUF Q4_K_M quantisation** – best TPS/quality balance for 7B–13B models at 24 GB.
- **llama.cpp with ROCm** – build with `make LLAMA_HIP=1` for full GPU offload; use `--n-gpu-layers 999`.
- **Ollama** – supports ROCm natively; set `OLLAMA_NUM_GPU=1` if needed.
- **Batch size** – increase `--parallel` in Ollama or `-np` in llama.cpp for multi-user scenarios.
- **Flash attention** – enable with `--flash-attn` in llama.cpp server for longer contexts.
- **Context length** – shorter context = faster prefill; use `--ctx-size 2048` for benchmarking.

Run the same model at different quantisation levels and compare with:
```bash
ai-benchmark compare --results-dir results/
```

---

## License

MIT – see [LICENSE](LICENSE).
