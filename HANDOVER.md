# StudyLLM — Claude Code Handover

## 1. Project Overview

Build **StudyLLM**, a public, open-source, self-hosted/offline local AI knowledge engine.

The core idea:

- The user puts documents into `data/`.
- StudyLLM automatically detects new, changed, and deleted files.
- New files are parsed, chunked, embedded, and indexed automatically.
- The local LLM uses RAG to answer questions from the indexed documents.
- If a document is deleted from `data/`, its knowledge is automatically removed from the index.
- If a document changes, the old index is removed and the new version is indexed.
- The underlying LLM is **not retrained**. RAG/indexing is the mechanism used to give the model document knowledge.
- Everything should run locally/offline after initial installation and model acquisition.
- The project will be publicly available on GitHub.
- The application should be extremely easy to use. Users should not need to understand RAG, embeddings, vector databases, GGUF, quantization, etc.

### Product philosophy

The user should only need to understand:

```text
models/ = AI model
data/   = AI knowledge
```

Everything else is an implementation detail.

The project should feel like a local AI appliance rather than a developer-only RAG framework.

---

# 2. Core User Experience

The ideal workflow is:

1. Clone/install StudyLLM.
2. Run StudyLLM.
3. StudyLLM detects the user's hardware.
4. StudyLLM determines a hardware performance tier from 1–5.
5. StudyLLM recommends a suitable GGUF model.
6. The README and/or first-run CLI provide a link/instructions for downloading that model.
7. User places the `.gguf` file in `models/`.
8. User runs StudyLLM again.
9. StudyLLM detects and validates the GGUF automatically.
10. User puts PDFs/DOCX/TXT/etc. into `data/`.
11. StudyLLM automatically indexes them.
12. User asks questions.
13. Answers should cite the relevant source document and page/section where available.

No manual `index` command should be required for normal use.

An optional `study-llm index` command can exist for debugging/manual synchronization, but automatic indexing is the default.

---

# 3. Example User Experience

First run:

```text
$ study-llm

╔══════════════════════════════════════════════════╗
║                    StudyLLM                         ║
║        Private • Local • Offline AI              ║
╚══════════════════════════════════════════════════╝

First-time setup detected.

Checking your hardware...

CPU       : Ryzen 5 5600H
RAM       : 16 GB
GPU       : RTX 3050
VRAM      : 4 GB

Performance Tier: 3 / 5

Recommended model:

  Qwen3 8B
  Quantization: Q4_K_M
  Format: GGUF

Download the recommended model and place the
.gguf file in:

  ./models/

Waiting for model...
```

After the model is installed:

```text
$ study-llm

✓ Hardware detected
✓ Model detected
✓ Model compatible
✓ Embedding model ready
✓ Vector database ready

Model:
  Qwen3 8B Q4_K_M

Knowledge base:
  0 documents

Put files into:

  ./data/

StudyLLM will automatically index them.
```

When a PDF is added:

```text
New document detected:

physics-class-12.pdf

Extracting text...        ✓
Analyzing structure...    ✓
Creating chunks...        ✓
Generating embeddings...  ✓
Updating knowledge base...✓

642 pages
4,821 chunks

✓ physics-class-12.pdf is now available to StudyLLM.
```

Then:

```text
$ study-llm ask "Explain Newton's second law"
```

Example:

```text
Newton's second law states that the acceleration
of an object is directly proportional to the net
force acting on it and inversely proportional to
its mass.

F = ma

Sources:
  • physics-class-12.pdf — Page 87
  • physics-class-12.pdf — Page 88
```

Interactive chat should also be available:

```text
$ study-llm chat
```

---

# 4. File Lifecycle / Source of Truth

The `data/` directory is the **source of truth**.

Example:

```text
data/
├── physics.pdf
├── chemistry.pdf
└── notes.docx
```

## New file

```text
data/new.pdf
       ↓
detect
       ↓
parse
       ↓
chunk
       ↓
embed
       ↓
store in vector DB
       ↓
available to RAG
```

## Modified file

Use a content hash such as SHA-256.

```text
physics.pdf
OLD HASH = ABC123
NEW HASH = XYZ789
```

Then:

```text
delete old document index
       ↓
parse new file
       ↓
chunk
       ↓
embed
       ↓
insert new version
```

Do not duplicate chunks.

## Deleted file

If the user deletes:

```text
data/physics.pdf
```

StudyLLM must detect the deletion and remove all chunks/vectors associated with that document.

The result:

```text
physics.pdf
      ↓
knowledge removed
```

The underlying LLM is not modified. The document is simply no longer retrievable.

## Files changed while StudyLLM is closed

On startup, perform a synchronization/reconciliation between `data/` and the metadata database.

Example:

```text
Knowledge base synchronization

+ physics.pdf       NEW
+ chemistry.pdf     NEW
- old-notes.pdf     REMOVED
~ textbook.pdf      CHANGED
```

Then automatically bring the index into exact agreement with `data/`.

This ensures `data/` always remains authoritative.

---

# 5. Automatic File Watching

When StudyLLM is running, watch `data/` for:

- file creation
- file modification
- file deletion
- renames/moves where detectable

Use a robust file watcher, initially **watchdog**.

Do not rely only on file watcher events. Always perform startup reconciliation as well.

This gives two layers:

```text
Running:
    watchdog → immediate indexing/removal

Startup:
    filesystem reconciliation → correctness
```

---

# 6. Supported Document Types

Initial target:

- PDF
- DOCX
- TXT
- Markdown
- PPTX
- XLSX
- CSV
- common image formats where OCR is useful

Design the ingestion system with a clean parser abstraction so additional formats can be added later.

The primary document-processing choice is **Docling**.

Docling should be evaluated/used for structured document extraction because textbooks and technical PDFs can contain:

- headings
- paragraphs
- tables
- equations
- figures
- multi-column layouts
- page boundaries
- OCR/scanned pages

Do not reduce every PDF to plain text if structured information can be preserved.

---

# 7. Recommended Technology Stack

## Language

**Python 3.12+**

Reason:

- mature AI ecosystem
- excellent document processing libraries
- straightforward CLI development
- easy packaging
- cross-platform

Keep application architecture modular so performance-critical pieces can later be optimized if necessary.

---

## LLM inference

Primary choice:

**llama.cpp**

Use GGUF models.

Reasons:

- fully local
- open source
- mature
- CPU/GPU support
- quantized model ecosystem
- suitable for standalone deployment
- avoids making Ollama a mandatory dependency

Ollama can potentially be supported later as an optional backend, but it should NOT be a hard dependency of the core architecture.

---

## LLM model format

**GGUF**

The repository should NOT contain large model files.

Users download a compatible GGUF separately and place it into:

```text
models/
```

StudyLLM should automatically discover GGUF files and inspect their metadata.

The model system should eventually support multiple installed GGUFs.

---

## Embeddings

Initial target:

**BGE-M3**

Use a local embedding model.

Embedding inference must also work offline.

Do not require a cloud embedding API.

Create an embedding abstraction so the embedding model can be replaced later.

---

## Vector database

Primary choice:

**Qdrant**

Use local/self-hosted Qdrant storage.

Store useful metadata alongside vectors, including at minimum:

- document ID
- file path
- filename
- page
- section/heading when available
- chunk ID
- content hash
- text/chunk metadata

Document-level filtering must be easy.

Deletion must be deterministic using `document_id`.

---

## Reranking

Use a BGE-family reranker where appropriate.

Reranking should be optional/adaptive rather than mandatory for every query.

For small knowledge bases and low-end hardware, a lightweight retrieval path should be possible without an expensive reranker.

---

## OCR

Use Docling's capabilities where possible and Tesseract as an additional local OCR option if needed.

OCR must remain offline.

---

## Metadata database

Use **SQLite**.

SQLite stores document/index state, not the actual semantic vector search responsibility.

Suggested document metadata:

```text
document_id
relative_path
filename
file_extension
sha256
file_size
modified_time
indexed_at
status
parser_version
chunking_version
embedding_model
```

This allows reliable incremental indexing and future migrations.

---

## File watching

Use **watchdog**.

---

## CLI

Use:

- **Typer** for CLI command structure
- **Rich** for terminal UI/progress/status tables

The CLI should be polished and friendly.

---

# 8. RAG Architecture

The core query pipeline:

```text
User question
      ↓
Query preprocessing
      ↓
Embedding model
      ↓
Vector search
      ↓
Candidate chunks
      ↓
Optional reranker
      ↓
Top relevant chunks
      ↓
Context builder
      ↓
Local LLM via llama.cpp
      ↓
Answer
      ↓
Source citations
```

The LLM should be instructed to answer from retrieved context and avoid inventing information.

If the evidence is insufficient, the model should say so rather than hallucinating.

---

# 9. Adaptive Retrieval

Do NOT blindly use the same expensive RAG pipeline for every machine or every knowledge base.

The system should adapt based on:

- hardware tier
- number of documents
- number of indexed chunks
- query complexity
- available memory/VRAM

Example:

### Small knowledge base

```text
50 chunks
   ↓
vector search
   ↓
top 3
   ↓
LLM
```

### Large knowledge base

```text
millions of chunks
   ↓
vector search
   ↓
top 20–50
   ↓
reranker
   ↓
top 4–8
   ↓
context control
   ↓
LLM
```

Do not send entire documents to the LLM.

---

# 10. Context Management

A critical goal is to minimize context sent to the LLM.

For example:

```text
10,000 pages
      ↓
retrieve relevant candidates
      ↓
rerank
      ↓
select best chunks
      ↓
deduplicate
      ↓
fit context budget
      ↓
LLM
```

This improves:

- response latency
- memory usage
- low-end PC usability
- answer quality
- token processing efficiency

The number of retrieved chunks and context budget should be adaptive.

---

# 11. Low-End PC Support

The application must be designed so low-end PCs can run it.

Important distinction:

**Small data does not inherently increase the LLM's generation tokens/sec.**

Generation speed mainly depends on:

- model size
- quantization
- CPU/GPU
- RAM/VRAM
- inference backend
- context length

However, small data can make:

- indexing faster
- retrieval faster
- context smaller
- total response latency lower

Therefore optimize the entire pipeline, not just retrieval.

---

# 12. Five Hardware Performance Tiers

Implement automatic hardware detection with tiers 1–5.

The tiers represent hardware capability, NOT document count.

Initial conceptual tiers:

```text
Tier 1
Very low-end
~8 GB RAM
CPU-only / weak integrated GPU
1–3B quantized models

Tier 2
Low-end
~8–16 GB RAM
CPU / integrated GPU
3–4B quantized models

Tier 3
Mainstream
~16 GB RAM
~4–8 GB VRAM
7–8B quantized models

Tier 4
High-end
~32 GB RAM
~8–16 GB VRAM
8–14B quantized models

Tier 5
Enthusiast
32+ GB RAM
16–24+ GB VRAM or equivalent
14–32B+ quantized models
```

These are starting guidelines, NOT hard guarantees.

Actual recommendation should consider:

- system RAM
- available RAM
- VRAM
- CPU core/thread count
- GPU
- GPU memory
- operating system
- free disk space

If feasible, run a small benchmark after hardware detection and use that to refine the recommendation.

---

# 13. Adaptive Model Selection

On first launch:

```text
hardware detection
       ↓
estimated tier
       ↓
candidate models
       ↓
memory/compatibility checks
       ↓
optional quick benchmark
       ↓
recommended GGUF
```

Example:

```text
Detected: Tier 3

Candidate:

Qwen3 4B Q4
  estimated: fast

Qwen3 8B Q4_K_M
  estimated: balanced

Qwen3 14B Q4
  estimated: slow

Recommended:
Qwen3 8B Q4_K_M
```

The application should explain the recommendation in simple language.

---

# 14. Model Directory

The user-facing model directory is:

```text
models/
```

Example:

```text
models/
├── Qwen3-8B-Q4_K_M.gguf
└── another-model.gguf
```

StudyLLM should automatically:

- discover `.gguf` files
- inspect GGUF metadata
- identify architecture/model name where available
- identify parameter count where available
- identify quantization
- check compatibility
- estimate memory requirements
- mark models as ready/not ready

No manual config should be required just to use a model.

---

# 15. Model Catalog in README

The README should contain a clear model recommendation table organized by Tier 1–5.

Important:

- Do NOT commit GGUF files to the repository.
- Provide links to appropriate model-hosting pages.
- Verify model licenses and redistribution terms before release.
- Do not make unsupported claims about exact tokens/sec.
- Use approximate hardware compatibility guidance.
- The actual model list should be kept current at release time.

The application itself should also be able to tell the user which model to download.

---

# 16. Zero-Configuration Philosophy

The user should NOT have to manually configure:

- RAG parameters
- chunk size
- embedding dimensions
- vector database settings
- GPU layers
- CPU threads
- context length
- reranking
- model paths

The application should choose sensible defaults based on hardware and knowledge-base size.

Advanced configuration can exist, but normal users should never need it.

---

# 17. Fast / Lite / Balanced / Quality Behavior

Consider supporting performance profiles:

```text
Lite
Balanced
Quality
```

Possible behavior:

### Lite

- smaller model
- smaller context
- fewer retrieved chunks
- no reranker or lightweight reranker
- conservative RAM/VRAM usage

### Balanced

- recommended model
- normal retrieval
- moderate context
- reranker when beneficial

### Quality

- larger compatible model
- more retrieval candidates
- reranking
- larger context
- higher resource usage

Automatic hardware tiering should select a sensible default.

Do not make users configure this during normal setup.

---

# 18. Automatic Indexing Architecture

Implement a `DocumentManager` / `KnowledgeBaseManager`.

It should have:

```text
startup_sync()
watch_data_folder()
handle_created()
handle_modified()
handle_deleted()
index_document()
remove_document()
reindex_document()
```

Recommended flow:

```text
                   ./data/
                      │
          ┌───────────┴───────────┐
          │                       │
     StudyLLM running            StudyLLM starts
          │                       │
      watchdog                 full sync
          │                       │
          └───────────┬───────────┘
                      ↓
              Document Manager
                      │
          ┌───────────┼───────────┐
          ↓           ↓           ↓
         NEW       CHANGED      DELETED
          │           │           │
          ↓           ↓           ↓
       Index       Reindex      Remove
```

---

# 19. Indexing Queue

Do not process a large number of files simultaneously on low-end systems.

Use a queue:

```text
[1] physics.pdf       INDEXING
[2] chemistry.pdf     QUEUED
[3] biology.pdf       QUEUED
[4] maths.pdf         QUEUED
```

Only use concurrency appropriate for the hardware tier.

Indexing should happen in the background when possible.

Already-indexed documents should remain available while new documents are being indexed.

---

# 20. Partial Indexing

Large documents should not freeze the application.

Show progress:

```text
physics.pdf

Pages: 238 / 642
Chunks: 1,842
Progress: 37%

██████████████░░░░░░░░░░
```

If a document is deleted during indexing:

```text
physics.pdf removed.

Cancelling indexing...
Cleaning temporary data...

✓ Removed.
```

No partial knowledge from the deleted file should remain.

Use temporary/staging state so a document is only marked fully indexed after successful completion.

---

# 21. File Hashing and Incremental Indexing

Use SHA-256 or an equally reliable content identity mechanism.

Never re-index unchanged documents.

Example:

```text
physics.pdf
hash = ABC123

database:
ABC123 ✓

→ skip
```

If the file changes:

```text
OLD = ABC123
NEW = XYZ789

→ remove old index
→ index new version
```

This is essential for performance.

---

# 22. Source Citations

Every chunk should retain provenance.

At minimum:

```text
filename
page number where available
section/heading where available
chunk ID
document ID
```

Answers should cite sources.

Example:

```text
Sources:
  • physics.pdf — Page 87
  • physics.pdf — Page 88
```

Do not fabricate page numbers.

If page information is unavailable, cite the filename and available section/chunk metadata instead.

---

# 23. Knowledge Isolation / Scopes

Design the architecture so document collections can eventually be scoped.

Example:

```text
data/
├── Physics/
│   ├── textbook.pdf
│   └── notes.pdf
├── Chemistry/
│   └── chemistry.pdf
└── General/
    └── notes.txt
```

Potential future usage:

```bash
study-llm ask "Explain electromagnetic induction" --scope Physics
```

and:

```bash
study-llm chat --scope Physics
```

This does NOT need to be a major v1 feature, but the metadata architecture should not prevent it.

---

# 24. Suggested Repository Structure

Target structure:

```text
StudyLLM/
│
├── data/
│   └── .gitkeep
│
├── models/
│   └── .gitkeep
│
├── storage/
│   └── .gitkeep
│
├── config/
│   └── ...
│
├── src/
│   └── study-llm/
│       ├── __init__.py
│       ├── cli/
│       ├── core/
│       ├── config/
│       ├── hardware/
│       ├── models/
│       ├── inference/
│       ├── documents/
│       ├── ingestion/
│       ├── chunking/
│       ├── embeddings/
│       ├── retrieval/
│       ├── reranking/
│       ├── rag/
│       ├── storage/
│       ├── knowledge/
│       ├── citations/
│       └── utils/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── scripts/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
└── HANDOVER.md
```

Adjust the exact structure if needed, but maintain clear separation of concerns.

---

# 25. Architecture Principles

## Keep components replaceable

Use interfaces/protocols for:

- LLM inference
- embedding model
- vector store
- document parser
- reranker
- hardware detection

Example conceptual interfaces:

```text
LLMProvider
EmbeddingProvider
VectorStore
DocumentParser
Reranker
HardwareDetector
```

This prevents the project from being permanently tied to one implementation.

---

# 26. Do NOT Start With LangChain/LlamaIndex

Do not introduce LangChain or LlamaIndex into the core unless a concrete need emerges.

Reason:

- the project should control its own RAG behavior
- fewer dependencies
- easier debugging
- easier offline packaging
- easier lifecycle management
- clearer architecture

Use direct libraries/components where practical.

A framework can be evaluated later if it solves a real problem.

---

# 27. Offline Requirement

After initial setup/model acquisition, the application must be able to operate without internet.

Runtime should NOT require:

- OpenAI API
- Anthropic API
- cloud embeddings
- cloud vector DB
- cloud OCR
- cloud document parsing
- telemetry

Default behavior should be local/offline.

If any optional online feature is ever added, it must be explicitly opt-in.

---

# 28. Privacy

The project is intended to be suitable for private documents.

Default behavior:

- documents remain on the user's machine
- embeddings remain on the user's machine
- vector database remains on the user's machine
- LLM inference remains on the user's machine
- no document upload
- no cloud inference
- no telemetry by default

Document this clearly in the README.

---

# 29. CLI Commands

Keep the normal command set small.

Suggested:

```bash
study-llm
study-llm ask "question"
study-llm chat
study-llm models
study-llm status
study-llm index
study-llm config
```

`study-llm index` is an explicit/manual synchronization command, but normal users should not need it.

`study-llm` itself should automatically:

- check hardware
- detect model
- synchronize `data/`
- process pending indexing
- start chat/interactive mode when ready

---

# 30. Startup State Machine

When running:

```bash
study-llm
```

the application should determine its state.

## State A — first run

```text
No model found
→ detect hardware
→ recommend model
→ explain where to put GGUF
→ exit/wait
```

## State B — model exists, data empty

```text
Model ready
→ explain data/ folder
→ start interactive mode
```

## State C — model exists, new files exist

```text
Detect new files
→ automatically index
→ show progress
→ start chat
```

## State D — model exists, modified/deleted files exist

```text
Synchronize filesystem
→ remove stale knowledge
→ reindex changed files
→ start chat
```

## State E — everything ready

```text
Start chat immediately
```

---

# 31. Model Discovery

If multiple GGUFs exist:

```text
models/
├── model-a.gguf
├── model-b.gguf
└── model-c.gguf
```

StudyLLM should list them and choose a sensible default.

Example:

```text
Installed Models

[1] Qwen3 8B Q4_K_M
    Tier 3
    READY

[2] Qwen3 4B Q4
    Tier 2
    READY

[3] Another Model
    Tier 4
    READY

Recommended:
    Qwen3 8B Q4_K_M
```

Allow explicit model selection later.

---

# 32. Model Compatibility

Before loading a GGUF:

- validate file
- inspect metadata
- detect architecture
- inspect quantization
- estimate memory
- compare with hardware
- warn if unsuitable

Example:

```text
Model:
Qwen3 32B Q4

Detected hardware:
Tier 2

WARNING:
This model may exceed available memory.

Recommended:
Qwen3 4B Q4
```

Do not prevent advanced users from overriding this unless loading would clearly be impossible.

---

# 33. No GitHub Model Files

Never commit large `.gguf` model files into the source repository.

The repository should contain:

- code
- configuration
- documentation
- tests
- small fixtures only

Model downloads should be external.

README should contain model recommendations organized by tiers.

---

# 34. Licensing

Before publishing:

- verify the license of every dependency
- verify the license of every recommended model
- verify whether model links/download mechanisms are permitted
- include required notices
- choose an appropriate license for StudyLLM itself

Do not assume all "open" models have the same redistribution rights.

---

# 35. Quality Requirements

The project should prioritize correctness over flashy features.

Important tests:

## Document lifecycle

- add file → indexed
- unchanged file → not reindexed
- modified file → old version removed, new version indexed
- deleted file → all associated vectors removed
- application restart → synchronization correct
- file added while app closed → indexed on startup
- file deleted while app closed → removed on startup
- duplicate filenames in different folders → handled correctly
- indexing failure → no corrupt/partial final index

## RAG

- relevant chunks retrieved
- irrelevant chunks rejected
- citations point to real documents
- page numbers are correct when available
- insufficient evidence produces a safe response
- context budget respected

## Hardware

- low-memory system handled gracefully
- CPU-only system works
- GPU system works
- unsupported GPU falls back to CPU
- model too large produces clear guidance

---

# 36. Low-End Performance Requirements

The application should avoid unnecessary resource usage.

Implement:

- adaptive worker count
- adaptive indexing concurrency
- bounded queues
- incremental indexing
- cached embeddings
- cached parsed documents where useful
- minimal context retrieval
- optional reranking
- quantized GGUF models
- CPU/GPU-aware llama.cpp settings

Never assume the user has a dedicated GPU.

---

# 37. Important Technical Distinction

This project uses:

**RAG, not continual model training.**

The model remains unchanged.

The knowledge lifecycle is:

```text
document
   ↓
parse
   ↓
chunk
   ↓
embed
   ↓
vector database
   ↓
retrieve
   ↓
LLM
```

Deleting a document means:

```text
delete document metadata
+
delete its chunks
+
delete its vectors
```

It does NOT mean retraining the model.

This is intentional and should be clearly documented.

---

# 38. Future Features — Do Not Prioritize for MVP

Possible later features:

- optional LoRA/QLoRA fine-tuning
- multiple knowledge scopes
- web UI
- local REST API
- plugin system
- multimodal models
- image understanding
- advanced OCR
- document summaries
- automatic question generation
- study mode
- source highlighting
- conversation memory
- model benchmarking
- automatic model downloading
- model switching
- Linux/macOS installers
- Windows packaged executable
- GPU-specific optimization
- distributed/local network inference

Do not let these delay the core MVP.

---

# 39. MVP Definition

MVP is complete when all of the following work:

1. User installs/clones StudyLLM.
2. `study-llm` starts.
3. Hardware is detected.
4. Tier 1–5 is calculated.
5. A suitable GGUF is recommended.
6. User puts GGUF in `models/`.
7. StudyLLM automatically discovers it.
8. Local LLM inference works.
9. User puts a PDF into `data/`.
10. StudyLLM automatically detects it.
11. PDF is parsed.
12. PDF is chunked.
13. Embeddings are generated locally.
14. Vectors are stored locally.
15. User can ask questions through CLI.
16. Relevant context is retrieved.
17. LLM answers using retrieved context.
18. Source citations are shown.
19. User deletes the PDF.
20. StudyLLM automatically removes its knowledge.
21. User replaces/edits the PDF.
22. StudyLLM automatically reindexes the changed version.
23. Restarting StudyLLM correctly reconciles the filesystem.
24. No internet is required during normal runtime.
25. Tests cover the document lifecycle.

---

# 40. Development Strategy

Build in this order:

## Phase 1 — Project foundation

- repository
- Python packaging
- configuration
- CLI skeleton
- logging
- tests
- `data/`, `models/`, `storage/`

## Phase 2 — Hardware detection

- CPU
- RAM
- GPU
- VRAM
- tier calculation
- model recommendation engine

## Phase 3 — GGUF inference

- llama.cpp integration
- GGUF discovery
- metadata inspection
- model loading
- basic prompt → answer

## Phase 4 — Document ingestion

- Docling integration
- PDF/DOCX/TXT/etc.
- normalized document representation
- page/section metadata

## Phase 5 — Chunking + embeddings

- chunker
- BGE-M3
- caching
- embedding storage

## Phase 6 — Qdrant

- local vector DB
- document metadata
- chunk metadata
- deterministic deletion

## Phase 7 — RAG

- query embedding
- retrieval
- reranking
- context builder
- citations
- hallucination-resistant prompting

## Phase 8 — Automatic document lifecycle

- watchdog
- add
- modify
- delete
- startup reconciliation
- indexing queue
- background indexing
- cancellation
- failure recovery

## Phase 9 — Adaptive performance

- hardware-aware settings
- retrieval adaptation
- context budgets
- lite/balanced/quality profiles
- CPU/GPU optimization

## Phase 10 — UX and release

- polished CLI
- README
- model tier table
- setup instructions
- license
- privacy statement
- dependency notices
- tests
- release packaging

---

# 41. Definition of "Easy"

A technically knowledgeable developer may know:

```text
RAG
Qdrant
BGE-M3
GGUF
llama.cpp
Docling
```

The end user should NOT need to.

The user experience should be:

```text
Download StudyLLM
       ↓
Run StudyLLM
       ↓
Get model recommendation
       ↓
Download model
       ↓
Put it in models/
       ↓
Put files in data/
       ↓
Ask questions
```

Everything else is automatic.

---

# 42. Important Product Rule

Do not turn this into a generic RAG framework.

The goal is a **complete local AI application**.

The user should not have to assemble:

```text
Ollama
+
Qdrant
+
LangChain
+
embedding model
+
document parser
+
configuration
```

The user installs **StudyLLM**.

StudyLLM orchestrates the components internally.

---

# 43. Suggested README Tagline

Possible project positioning:

> **StudyLLM — A private, local AI that learns from the files in your `data/` folder.**

Alternative:

> **Drop in your documents. Ask your local AI. Nothing leaves your machine.**

Avoid claiming the model is literally retrained.

Use wording such as:

- "learns from your documents"
- "indexes your knowledge"
- "local document knowledge"
- "RAG-powered local AI"

---

# 44. Final Architecture

```text
                         USER
                           │
                           ▼
                    ┌─────────────┐
                    │     CLI     │
                    └──────┬──────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   StudyLLM Core    │
                  └───────┬─────────┘
                          │
             ┌────────────┼─────────────┐
             │            │             │
             ▼            ▼             ▼
        Hardware       Model        Knowledge
        Manager        Manager       Manager
             │            │             │
             │            ▼             │
             │       llama.cpp          │
             │            │             │
             │            ▼             │
             │       Local GGUF         │
             │                          │
             │                    ┌─────┴─────┐
             │                    │           │
             │                 ./data/    SQLite
             │                    │           │
             │                 watchdog      │
             │                    │           │
             │                    ▼           │
             │               Docling          │
             │                    │           │
             │                 Chunker        │
             │                    │           │
             │                 BGE-M3         │
             │                    │           │
             │                    ▼           │
             │                  Qdrant ◄──────┘
             │                    │
             │               Retrieval
             │                    │
             │                Reranker
             │                    │
             │              Context Builder
             │                    │
             └────────────────────┼───────────┘
                                  │
                                  ▼
                              llama.cpp
                                  │
                                  ▼
                               ANSWER
                                  │
                                  ▼
                              CITATIONS
```

---

# 45. Claude Code Instructions

You are taking over development of this project.

Do NOT blindly implement everything in one pass.

First:

1. Inspect the repository.
2. Read this `HANDOVER.md`.
3. Determine what already exists.
4. Create a concise implementation plan.
5. Identify dependency/licensing concerns.
6. Implement the MVP in logical phases.
7. Keep the architecture modular.
8. Add tests alongside functionality.
9. Run formatting, linting, type checking and tests regularly.
10. Do not introduce unnecessary frameworks.
11. Do not add LangChain/LlamaIndex unless there is a demonstrated requirement.
12. Do not require cloud APIs.
13. Do not commit model files.
14. Do not hard-code one specific GGUF model as the only supported model.
15. Make model discovery metadata-driven.
16. Make `data/` the source of truth.
17. Make automatic indexing the default behavior.
18. Make deletion remove all knowledge associated with the deleted document.
19. Make startup reconciliation mandatory.
20. Keep low-end hardware support in mind throughout implementation.

When making architectural decisions not explicitly specified here, prefer:

- offline
- local
- open source
- minimal dependencies
- modularity
- correctness
- low resource usage
- easy user experience
- cross-platform compatibility
- deterministic document lifecycle

Do not over-engineer features that are outside the MVP.

The final application should feel like:

```text
Put GGUF in models/
Put documents in data/
Run StudyLLM
Ask questions
```

That is the core product.
