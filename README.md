# Fine-Tuning Demos

A collection of hands-on demos exploring progressive approaches to building AI-powered applications with open-weight LLMs — from basic prompting to retrieval-augmented generation (RAG) to fine-tuning.

## Projects

### [Gemma Finance Analyst](gemma-finance-analyst/)

An AI-powered financial analyst that extracts structured insights from earnings-call transcripts. The project demonstrates three progressive levels of LLM customization:

| Phase | Approach | Description |
|-------|----------|-------------|
| **1 — Prompt Engineering** | Zero-shot prompting | System prompt + JSON template with Gemma 2 (9B) via Ollama |
| **2 — RAG** | Retrieval-Augmented Generation | ChromaDB vector store for multi-transcript Q&A |
| **3 — Fine-Tuning** | LoRA fine-tuning | Fine-tune Gemma 2 (2B) with PEFT/TRL on domain-specific financial data |

Each phase builds on the previous one, showing the trade-offs between simplicity, flexibility, and domain specialization.

## Getting Started

See the [Gemma Finance Analyst README](gemma-finance-analyst/README.md) for detailed setup instructions, dependencies, and usage.

## Tech Stack

- **Models**: [Google Gemma 2](https://ai.google.dev/gemma) (9B for prompting/RAG, 2B for fine-tuning)
- **Local Inference**: [Ollama](https://ollama.com/) (Phases 1 & 2)
- **Fine-Tuning**: [Hugging Face Transformers](https://huggingface.co/docs/transformers), [PEFT](https://huggingface.co/docs/peft), [TRL](https://huggingface.co/docs/trl)
- **Vector DB**: [ChromaDB](https://www.trychroma.com/) (Phase 2)
- **Embeddings**: [Sentence Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`)
- **Package Management**: [UV](https://docs.astral.sh/uv/)

## License

This project is for educational and demonstration purposes.
