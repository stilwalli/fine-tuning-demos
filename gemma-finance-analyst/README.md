# Gemma Finance Analyst

A hands-on demo that explores three progressive approaches to building an AI-powered financial analyst using **Google's Gemma 2 (9B)** running locally via **Ollama**. The project takes a sample earnings-call transcript and extracts structured financial insights (revenue, EPS, sentiment, risk flags, etc.) as JSON.

## What This Does

The demo feeds a fictional **ACME Financial Corp Q3 2024 earnings-call transcript** to a locally-running Gemma 2 model and asks it to return a structured JSON summary containing:

- Actual revenue & next-quarter guidance
- EPS actual and vs. estimate
- Management sentiment
- Growth drivers
- Risk flags

The goal is to show how the same task can be tackled with increasing levels of sophistication across three phases.

## Three-Phase Roadmap

| Phase | Directory | Approach | Status |
|-------|-----------|----------|--------|
| **1 — Prompt Engineering** | `01_prompt_engineering/` | Zero-shot prompting with a system prompt and JSON template | ✅ Complete |
| **2 — RAG** | `02_rag/` | Retrieval-Augmented Generation for richer context | 🔜 Upcoming |
| **3 — Fine-Tuning** | `03_fine_tuning/` | Fine-tune Gemma on domain-specific financial data | 🔜 Upcoming |

---

### Phase 1 — Prompt Engineering (`01_prompt_engineering/`)

This phase demonstrates **pure prompt engineering** — no retrieval, no fine-tuning. The script:

1. Loads the earnings transcript from `data/earnings_transcript.txt`.
2. Sends it to Gemma 2 (9B) via the Ollama Python client with:
   - A **system prompt** that instructs the model to behave as a senior financial analyst and respond in raw JSON only.
   - A **user prompt** containing a JSON template and the transcript.
3. Cleans up any accidental markdown fences from the response.
4. Parses and pretty-prints the resulting JSON.

**Run it:**

```bash
python 01_prompt_engineering/pe_analyst.py
```

---

## Dependencies

### 1. UV (Python Package Manager)

This project uses [UV](https://docs.astral.sh/uv/) for Python version and dependency management.

**Install UV:**

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via Homebrew
brew install uv
```

### 2. Ollama (Local LLM Runtime)

[Ollama](https://ollama.com/) is used to run the Gemma model locally.

**Install Ollama:**

```bash
# macOS — download from https://ollama.com/download or:
brew install ollama
```

**Start the Ollama server:**

```bash
ollama serve
```

**Pull the Gemma 2 9B model:**

```bash
ollama pull gemma2:9b
```

### 3. Python Dependencies

The project requires **Python ≥ 3.14** and the following package (defined in `pyproject.toml`):

- `ollama` — Python client for the Ollama API

**Install with UV:**

```bash
uv sync
```

---

## Quick Start

```bash
# 1. Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install Ollama and pull the model
brew install ollama
ollama serve          # keep this running in a separate terminal
ollama pull gemma2:9b

# 3. Clone the repo and set up the environment
cd gemma-finance-analyst
uv sync

# 4. Activate the virtual environment
source .venv/bin/activate

# 5. Run Phase 1 — Prompt Engineering
python 01_prompt_engineering/pe_analyst.py
```

## Project Structure

```
gemma-finance-analyst/
├── 01_prompt_engineering/
│   └── pe_analyst.py          # Phase 1: prompt-only approach
├── 02_rag/                    # Phase 2: RAG (upcoming)
├── 03_fine_tuning/            # Phase 3: fine-tuning (upcoming)
├── data/
│   └── earnings_transcript.txt
├── pyproject.toml
├── uv.lock
└── README.md
```
