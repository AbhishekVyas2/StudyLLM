# StudyLLM

A private, local AI that learns from the files in your `data/` folder.

## Overview

StudyLLM is an open-source, self-hosted/offline local AI knowledge engine that automatically indexes documents in your `data/` folder and allows you to ask questions about them using Retrieval-Augmented Generation (RAG).

## Core Philosophy

The user should only need to understand:
```
models/ = AI model
data/   = AI knowledge
```

Everything else is an implementation detail.

## Features

- 🔒 **Private & Local**: All processing happens on your machine
- 📁 **Automatic Indexing**: New, modified, and deleted files are handled automatically
- 🖥️ **Offline Operation**: No internet required after initial setup
- 🎯 **Hardware Aware**: Automatically detects your system capabilities and recommends appropriate models
- 📚 **Multiple Formats**: Supports PDF, DOCX, TXT, Markdown, PPTX, XLSX, CSV, and more
- 📝 **Source Citations**: Answers include references to the original documents

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/studyllm.git
cd studyyllm

# Core (CLI + hardware detection)
pip install -e .

# Full local AI stack: embeddings, vector DB, parsing, live updates
pip install -e .[ai]
```

### Dependencies

| Package | Required | Purpose |
|---|---|---|
| typer, rich | yes | CLI and terminal UI |
| llama-cpp-python | for answers | GGUF model inference |
| sentence-transformers | for indexing | BGE-M3 embeddings (+ optional reranker) |
| qdrant-client | for indexing | Local vector database |
| docling | PDF/DOCX/PPTX/XLSX | Document parsing |
| watchdog | live updates | Filesystem change detection |

Missing optional dependencies degrade gracefully: StudyLLM still runs, tells you what's missing, and skips the affected feature.

## Quick Start

1. Run StudyLLM for the first time:
   ```bash
   study-llm
   ```

2. Follow the prompts to:
   - Let StudyLLM detect your hardware
   - Download the recommended GGUF model
   - Place it in the `models/` directory

3. Add your documents to the `data/` folder:
   ```bash
   cp your-documents/* data/
   ```

4. Ask questions:
   ```bash
   study-llm ask "What is the main topic of my documents?"
   ```

   Or start an interactive chat:
   ```bash
   study-llm chat
   ```

## Supported Document Types

- PDF
- DOCX
- TXT
- Markdown
- PPTX
- XLSX
- CSV
- Common image formats (with OCR)

## Hardware Tiers & Recommended Models

StudyLLM detects your hardware and recommends a model size. Download the recommended GGUF (quantization `Q4_K_M` unless noted) from Hugging Face and place it in `models/`.

| Tier | Hardware | RAM | VRAM | Recommended model size | Example models |
|---|---|---|---|---|---|
| 1 | Older laptop / netbook | < 8 GB | any | ~1–2B | Qwen3 1.7B, Llama 3.2 1B, Gemma 3 1B |
| 2 | Entry desktop / laptop | ≥ 8 GB | < 6 GB | ~3B | Qwen3 4B, Llama 3.2 3B, Phi-4-mini |
| 3 | Mainstream | ≥ 16 GB | ≥ 6 GB | ~7B | Qwen3 8B (Q4), Llama 3.1 8B, Gemma 3 12B |
| 4 | High-end | ≥ 32 GB | ≥ 10 GB | ~13B | Qwen3 14B, Phi-4 14B |
| 5 | Enthusiast | ≥ 32 GB | ≥ 20 GB | ~30B+ | Qwen3 30B-A3B, Gemma 3 27B |

Tier 3 requires a GPU that can actually hold the recommended model — machines with a weak GPU (like a 4 GB card) are classified Tier 2 and pointed at 3–4B models instead, which run comfortably on CPU alone.

Any compatible GGUF works — StudyLLM discovers models by reading their metadata, not by filename. If several models are present it picks the one closest to your tier's recommendation.

## Performance Profiles

Runtime settings adapt automatically to your tier:

| Profile | Tier | Context | RAG top-k | Context budget | Reranker |
|---|---|---|---|---|---|
| lite | 1–2 | 2048 | 5–6 | 800–1000 tok | off |
| balanced | 3 | 4096 | 8 | 1500 tok | optional |
| quality | 4–5 | 8192–16384 | 10 | 2500–3000 tok | on |

Explicit values in `config/settings.json` always override these defaults.

## Configuration

StudyLLM aims for zero-configuration usage. Advanced settings can be found in `config/settings.json`.

## Development

See [HANDOVER.md](HANDOVER.md) for detailed development instructions and architecture.

## License

MIT

## Privacy

StudyLLM is designed for private documents:
- Documents remain on your machine
- Embeddings remain on your machine
- Vector database remains on your machine
- LLM inference remains on your machine
- No document upload
- No cloud inference
- No telemetry by default