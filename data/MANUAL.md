# StudyLLM — Simple User Manual

Your personal AI that knows what's inside your own documents.
Everything stays on your computer. Nothing is uploaded anywhere.

---

## The two folders that matter

```
models/   ← put the AI brain here (one .gguf file)
data/     ← put YOUR documents here
```

That's really all you need to know to use it.

---

## First-time setup (once)

**Step 1 — Install**

```bash
pip install -e .
```

**Step 2 — Get an AI model**

1. Go to huggingface.co
2. Search: `Qwen3 4B GGUF` (good for normal laptops; use **Qwen3 1.7B** if yours is older/weaker)
3. Download the file ending in `.gguf` (pick **Q4_K_M** if offered)
4. Put that file into your `models/` folder

That's it. No accounts, no API keys, no internet needed afterwards.

---

## Daily use

### 1. Add knowledge

Drop any of these into `data/`:

PDF • Word (.docx) • PowerPoint (.pptx) • Excel (.xlsx)
TXT • Markdown (.md) • CSV • HTML

- New file? It gets learned automatically.
- Changed a file? It re-learns just that one.
- Deleted a file? Its knowledge disappears too.

You never have to press a "sync" button — but if you want to force one:

```bash
study-llm index
```

### 2. Ask questions

One question at a time:

```bash
study-llm ask "What did chapter 3 say about photosynthesis?"
```

Or chat back and forth:

```bash
study-llm chat
```

(then just type questions; type `exit` or `quit` to leave)

### 3. Read the sources

Every answer ends with numbered sources, like:

```
Sources:
[1] biology-notes.pdf p.12
[2] summary.md §Overview
```

Those tell you exactly where the answer came from, so you can double-check it.

---

## Important rule

The AI only answers from your documents.
If the answer isn't in there, it will say:

> I don't know based on your documents.

This is on purpose — it won't guess or make things up.

---

## Handy commands

| Command | What it does |
|---|---|
| `study-llm` | Start interactive mode (checks everything first) |
| `study-llm ask "..."` | Ask one question |
| `study-llm chat` | Chat mode |
| `study-llm index` | Re-scan data/ now |
| `study-llm status` | See if model & data are ready |
| `study-llm models` | List models found in models/ |
| `study-llm config` | Show current settings |

---

## Where is my data kept?

Everything lives in these folders next to the program:

| Folder | Contains |
|---|---|
| `data/` | Your original documents (untouched) |
| `storage/` | The search index it builds from them |
| `models/` | The AI brain |

Nothing leaves your machine. Ever.

---

## Problems?

| Symptom | Fix |
|---|---|
| Says no model found | Put a `.gguf` file in `models/` |
| Answers are slow | Use a smaller model (1–3B size) |
| A file isn't learned | Check it's a supported type; run `study-llm index` |
| Answer says "I don't know" | The info genuinely isn't in your files |
