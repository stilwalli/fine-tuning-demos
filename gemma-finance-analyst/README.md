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
| **2 — RAG** | `02_rag/` | Retrieval-Augmented Generation for richer context | ✅ Complete |
| **3 — Fine-Tuning** | `03_fine_tuning/` | Fine-tune Gemma on domain-specific financial data | ✅ Complete |

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

### Phase 2 — RAG (`02_rag/`)

This phase adds **Retrieval-Augmented Generation** so the model can answer questions across multiple earnings transcripts (Q1–Q3 2024) instead of just one.

It consists of two scripts:

1. **`ingest.py`** — Ingestion pipeline
   - Loads all transcript files from `data/`.
   - Generates embeddings using the `all-MiniLM-L6-v2` sentence-transformer model.
   - Stores documents and embeddings in a persistent **ChromaDB** vector database (`02_rag/db/`).

2. **`rag_analyst.py`** — Query pipeline
   - Encodes the user's question into an embedding.
   - Retrieves the top-3 most relevant transcripts from ChromaDB.
   - Sends the retrieved context + question to Gemma 2 (9B) via Ollama.
   - Prints the model's answer grounded in the retrieved documents.

**Run it:**

```bash
# Step 1: Ingest the transcripts into the vector DB
python 02_rag/ingest.py

# Step 2: Ask a question across all transcripts
python 02_rag/rag_analyst.py
```

---

### Phase 3 — Fine-Tuning (`03_fine_tuning/`)

This phase **fine-tunes Gemma 2 (2B)** on domain-specific financial earnings data using **LoRA** (Low-Rank Adaptation) via Hugging Face PEFT and TRL. It requires a Hugging Face API token for model access.

It consists of two scripts:

1. **`finetune.py`** — Training pipeline
   - Loads prompt/completion pairs from `data/training_data.jsonl`.
   - Downloads the `google/gemma-2-2b` base model from Hugging Face.
   - Applies LoRA adapters to the attention layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`) with rank 16.
   - Trains for 10 epochs using `SFTTrainer` from TRL.
   - Saves the fine-tuned LoRA adapter to `03_fine_tuning/output/`.

2. **`inference.py`** — Inference pipeline
   - Loads the base Gemma 2 (2B) model and applies the saved LoRA adapter.
   - Feeds a test earnings transcript and generates a structured JSON analysis.
   - Demonstrates the fine-tuned model's ability to extract revenue, EPS, sentiment, and risk flags.

**Setup:**

Create a `.env` file with your Hugging Face token:

```bash
HF_TOKEN=hf_your_token_here
```

**Run it:**

```bash
# Step 1: Fine-tune the model (takes a few minutes on CPU)
python 03_fine_tuning/finetune.py

# Step 2: Run inference with the fine-tuned model
python 03_fine_tuning/inference.py
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

### 3. Hugging Face Token (Phase 3 only)

Phase 3 requires access to the gated `google/gemma-2-2b` model on Hugging Face. Create a `.env` file in the project root:

```bash
HF_TOKEN=hf_your_token_here
```

### 4. Python Dependencies

The project requires **Python ≥ 3.14** and the following packages (defined in `pyproject.toml`):

- `ollama` — Python client for the Ollama API (Phases 1 & 2)
- `chromadb` — Vector database for storing and querying document embeddings (Phase 2)
- `sentence-transformers` — Embedding model (`all-MiniLM-L6-v2`) for encoding text (Phase 2)
- `torch` — PyTorch
- `transformers` — Hugging Face Transformers for model loading (Phase 3)
- `peft` — Parameter-Efficient Fine-Tuning / LoRA (Phase 3)
- `trl` — Transformer Reinforcement Learning / SFTTrainer (Phase 3)
- `accelerate` — Hugging Face training accelerator (Phase 3)
- `python-dotenv` — Load environment variables from `.env` (Phase 3)

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

# 6. Run Phase 2 — RAG
python 02_rag/ingest.py        # ingest transcripts into vector DB
python 02_rag/rag_analyst.py   # ask questions across all transcripts

# 7. Run Phase 3 — Fine-Tuning (requires HF_TOKEN in .env)
python 03_fine_tuning/finetune.py   # fine-tune Gemma 2 (2B) with LoRA
python 03_fine_tuning/inference.py  # run inference with the fine-tuned model
```

## Project Structure

```
gemma-finance-analyst/
├── 00_scratch/
│   └── scratch.py             # Scratch / experimentation
├── 01_prompt_engineering/
│   └── pe_analyst.py          # Phase 1: prompt-only approach
├── 02_rag/
│   ├── ingest.py              # Phase 2: ingest transcripts into ChromaDB
│   └── rag_analyst.py         # Phase 2: RAG-powered Q&A
├── 03_fine_tuning/
│   ├── finetune.py            # Phase 3: LoRA fine-tuning script
│   ├── inference.py           # Phase 3: inference with fine-tuned model
│   └── output/                # Saved LoRA adapter weights
├── data/
│   ├── earnings_transcript.txt
│   ├── q1_2024.txt
│   ├── q2_2024.txt
│   ├── q3_2024.txt
│   └── training_data.jsonl    # Fine-tuning training examples
├── .env                       # HF_TOKEN (not committed)
├── pyproject.toml
├── uv.lock
└── README.md
```
