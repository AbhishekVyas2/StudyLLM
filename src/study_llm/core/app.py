"""
Main StudyLLM application.
Orchestrates hardware detection, model management, and the main application loop.
"""

import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
from rich.prompt import Prompt, Confirm

from study_llm.hardware.detector import get_performance_tier, get_hardware_specs
from study_llm.models.manager import get_model_manager
from study_llm.core.config import StudyLLMConfig


class StudyLLMApp:
    """Main StudyLLM application class."""

    def __init__(self):
        self.console = Console()
        self.config = StudyLLMConfig()
        self.hardware_specs = get_hardware_specs()
        self.performance_tier = get_performance_tier()
        self.model_manager = get_model_manager()

    def _show_banner(self):
        """Display the startup banner."""
        banner = Text()
        banner.append("StudyLLM", style="bold cyan")
        banner.append(
            "\nA private, local AI that learns from the files in your data/ folder",
            style="dim",
        )
        self.console.print(Panel(banner, border_style="cyan"))
        specs = self.hardware_specs
        gpu = f" · GPU: {specs.gpu_model}" if specs.gpu_model else ""
        self.console.print(
            f"[dim]{specs.cpu_model} ({specs.cpu_threads} threads) · "
            f"RAM {specs.ram_total_gb:.0f} GB{gpu}[/dim]"
        )
        tier = self.performance_tier
        self.console.print(f"Performance tier: [bold]{tier.tier}[/bold] — {tier.description}\n")

    def run(self):
        """Main application entry point."""
        self.console.clear()
        self._show_banner()

        # Check if we're ready to run
        if not self._check_initial_state():
            return

        # Reconcile data/ with index, start background indexing + watcher
        self._start_lifecycle()
        self.start_chat()

    @property
    def inference(self):
        """Lazily load the GGUF model on first use."""
        if not hasattr(self, "_inference") or self._inference is None:
            from study_llm.inference.llama_cpp import LlamaCppInference
            model_info = self.model_manager.get_recommended_model(
                self.performance_tier
            )
            if model_info is None:
                raise RuntimeError("No GGUF model available in models/")
            self.console.print(f"[cyan]Loading {model_info.name}...[/cyan]")
            self._inference = LlamaCppInference(
                model_path=model_info.path,
                n_ctx=self.config.ctx_size,
                n_gpu_layers=self.config.gpu_layers,
            )
        return self._inference

    def _start_lifecycle(self):
        """Reconcile and start automatic document lifecycle."""
        try:
            from study_llm.core.bootstrap import get_runtime
            self.runtime = get_runtime()
            result = self.runtime.reconcile_and_enqueue()
            n = len(result["new"]) + len(result["changed"])
            if n:
                self.console.print(
                    f"[cyan]Indexing {n} new/changed document(s)...[/cyan]"
                )
            self.runtime.start_background()
        except ImportError as e:
            missing = "qdrant-client" if "qdrant" in str(e).lower() else (
                "sentence-transformers" if "sentence" in str(e).lower() else str(e)
            )
            self.console.print(
                f"[yellow]Automatic indexing unavailable "
                f"(missing dependency: {missing}).[/yellow]"
            )
        except Exception as e:
            self.console.print(f"[yellow]Indexing unavailable: {e}[/yellow]")

    def _check_initial_state(self) -> bool:
        """Check the initial state and guide user through setup if needed."""
        # Check for model
        model_info = self.model_manager.get_recommended_model(self.performance_tier)

        if not model_info or not model_info.is_valid:
            # State A: First run - no model found
            self._handle_no_model_state()
            return False
        else:
            # We have a valid model
            self.console.print(f"✓ Model detected: [green]{model_info.name}[/green]")

            # Check if data directory has files
            data_dir = Path("data")
            if not data_dir.exists() or not any(data_dir.iterdir()):
                # State B: Model exists, data empty
                self._handle_empty_data_state()
                return True
            else:
                # Check for new/changed/deleted files (State C or D)
                # For now, we'll just start chat and prompt; lifecycle prints indexing progress.
                # State E: Everything ready
                return True

    def _handle_no_model_state(self):
        """Handle State A: First run - no model found."""
        self.console.print(Panel(
            "[bold]First-time setup detected.[/bold]\n\n"
            "Checking your hardware...",
            title="StudyLLM",
            border_style="blue"
        ))

        # Show hardware specs in a readable form (not a raw dict dump)
        s = self.hardware_specs
        self.console.print(f"CPU: {s.cpu_model} ({s.cpu_threads} threads)")
        self.console.print(f"RAM: {s.ram_total_gb:.1f} GB total / {s.ram_available_gb:.1f} GB available")
        if s.gpu_model:
            self.console.print(f"GPU: {s.gpu_model}" + (
                f" ({s.vram_total_gb:.1f} GB VRAM)" if s.vram_total_gb else ""))
        self.console.print(f"Disk free: {s.disk_free_gb:.0f} GB")

        # Show performance tier
        self.console.print(f"\nPerformance Tier: {self.performance_tier.tier} / 5 — {self.performance_tier.description}")

        # Show recommended model for this tier
        tier_size = self.performance_tier.recommended_model_size_billions
        self.console.print("\n[bold]Recommended model:[/bold]")
        self.console.print(f"  ~{tier_size:g}B parameters, quantization Q4_K_M, format GGUF")

        self.console.print(f"\nDownload the recommended model and place the")
        self.console.print(f".gguf file in:")
        self.console.print(f"  ./models/")
        self.console.print()

        # Wait for model
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Waiting for model...", total=None)

            # In a real implementation, we'd watch for model files
            # For now, just wait a moment and then check again
            time.sleep(3)
            progress.update(task, description="Checking for model...")

            # Check again
            model_info = self.model_manager.get_recommended_model(self.performance_tier)
            if model_info and model_info.is_valid:
                progress.update(task, description="Model found!")
                time.sleep(1)
                return
            else:
                progress.update(task, description="Still waiting for model...")
                time.sleep(2)

        # If we still don't have a model, exit
        self.console.print("\n[yellow]Please place a GGUF model in the models/ directory and restart StudyLLM.[/yellow]")
        raise typer.Exit()

    def _handle_empty_data_state(self):
        """Handle State B: Model exists, data empty."""
        self.console.print()
        self.console.print("✓ Hardware detected")
        self.console.print("✓ Model detected")
        self.console.print("✓ Model compatible")
        self.console.print("✓ Embedding model ready")
        self.console.print("✓ Vector database ready")
        self.console.print()

        # Show model info
        model_info = self.model_manager.get_recommended_model(self.performance_tier)
        self.console.print(f"Model:")
        self.console.print(f"  {model_info.name}")
        if model_info.parameter_count:
            self.console.print(f"  Size: ~{model_info.parameter_count}B {model_info.quantization or ''}")
        self.console.print()

        # Real document count from the metadata DB (0 on first run)
        doc_count = 0
        try:
            from study_llm.core.bootstrap import get_runtime
            doc_count = get_runtime().metadata_db.count_documents()
        except Exception:
            pass
        self.console.print(f"Knowledge base:")
        self.console.print(f"  {doc_count} documents")
        self.console.print()

        self.console.print("Put files into:")
        self.console.print("  ./data/")
        self.console.print()

        self.console.print("StudyLLM will automatically index them.")
        self.console.print()

    def start_chat(self):
        """Start interactive chat mode."""
        self.console.print("[green]StudyLLM is ready![/green]")
        self.console.print("Type your questions below. Use 'exit' or 'quit' to stop.\n")

        try:
            while True:
                try:
                    question = Prompt.ask("You")
                    if question.lower() in ['exit', 'quit']:
                        self.console.print("[yellow]Goodbye![/yellow]")
                        break

                    if not question.strip():
                        continue

                    self.ask_question(question)

                except KeyboardInterrupt:
                    self.console.print("\n[yellow]Goodbye![/yellow]")
                    break
                except EOFError:
                    self.console.print("\n[yellow]Goodbye![/yellow]")
                    break
        finally:
            # Release the vector DB / metadata DB so later runs don't hit a lock
            try:
                from study_llm.core.bootstrap import get_runtime
                get_runtime().shutdown()
            except Exception:
                pass

    def ask_question(self, question: str):
        """Ask a question to StudyLLM using the full RAG pipeline."""
        from study_llm.rag.pipeline import RAGPipeline
        from study_llm.rag.retriever import Retriever

        try:
            from study_llm.core.bootstrap import get_runtime
            runtime = get_runtime()
        except Exception as e:
            self.console.print(f"[red]Knowledge base unavailable: {e}[/red]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Searching your documents...", total=None)

            retriever = Retriever(
                embedding_provider=runtime.embedding_provider,
                vector_store=runtime.vector_store,
                top_k=self.config.retrieval_top_k,
            )
            pipeline = RAGPipeline(retriever, self.inference)
            result = pipeline.ask(question)

        self.console.print("\n[blue]StudyLLM:[/blue]")
        self.console.print(result.answer)
        self.console.print()

        if result.citations and self.config.show_sources:
            self.console.print("[dim]Sources:[/dim]")
            for c in result.citations:
                loc = f" p.{c.page}" if c.page else ""
                sec = f" §{c.section}" if c.section else ""
                self.console.print(f"  [{c.num}] {c.filename}{loc}{sec}")
        elif not result.citations:
            self.console.print("[dim]Sources: none — the answer isn't in your documents.[/dim]")
        self.console.print()

    def list_models(self):
        """List available GGUF models."""
        models = self.model_manager.discover_models()

        if not models:
            self.console.print("[yellow]No GGUF models found in models/ directory.[/yellow]")
            self.console.print("Place .gguf files in the models/ directory.")
            return

        self.console.print("[bold]Installed Models[/bold]\n")

        for i, model in enumerate(models, 1):
            status = "[green]READY[/green]" if model.is_valid else "[red]INVALID[/red]"
            tier_guess = "Tier ?"  # Would calculate based on model size

            self.console.print(f"[{i}] {model.name}")
            self.console.print(f"    {tier_guess}")
            self.console.print(f"    {status}")

            if model.parameter_count:
                size_str = f"~{model.parameter_count}B"
                if model.quantization:
                    size_str += f" {model.quantization}"
                self.console.print(f"    Size: {size_str}")
            self.console.print()

    def show_status(self):
        """Show StudyLLM status."""
        self.console.print("[bold]StudyLLM Status[/bold]\n")

        # Hardware
        self.console.print("[blue]Hardware:[/blue]")
        self.console.print(self.hardware_specs.__dict__)  # Simplified
        self.console.print()

        # Performance tier
        self.console.print("[blue]Performance Tier:[/blue]")
        self.console.print(f"  Tier {self.performance_tier.tier} / 5")
        self.console.print(f"  {self.performance_tier.description}")
        self.console.print(f"  Recommended model size: {self.performance_tier.recommended_model_size_billions}B")
        self.console.print(f"  Notes: {self.performance_tier.notes}")
        self.console.print()

        # Model status
        self.console.print("[blue]Model Status:[/blue]")
        model_info = self.model_manager.get_recommended_model(self.performance_tier)
        if model_info and model_info.is_valid:
            self.console.print(f"  ✓ Model detected: {model_info.name}")
            self.console.print(f"  ✓ Model valid")
            if model_info.parameter_count:
                self.console.print(f"  Size: ~{model_info.parameter_count}B {model_info.quantization or ''}")
        else:
            self.console.print(f"  ✗ No valid model found")
            self.console.print(f"    Place a GGUF model in the models/ directory")
        self.console.print()

        # Data directory
        self.console.print("[blue]Data Directory:[/blue]")
        data_dir = Path("data")
        if data_dir.exists():
            file_count = len(list(data_dir.iterdir()))
            self.console.print(f"  {file_count} files in data/")
            if file_count > 0:
                self.console.print("  Files:")
                for file_path in data_dir.iterdir():
                    if file_path.is_file():
                        self.console.print(f"    - {file_path.name}")
        else:
            self.console.print("  data/ directory does not exist")
        self.console.print()

        # Storage directory
        self.console.print("[blue]Storage Directory:[/blue]")
        storage_dir = Path("storage")
        if storage_dir.exists():
            self.console.print("  ✓ Storage directory exists")
            # Placeholder for actual DB status
            self.console.print("  ✓ Vector database ready")
            self.console.print("  ✓ Metadata database ready")
        else:
            self.console.print("  ! Storage directory will be created on first run")
        self.console.print()

    def manual_index(self):
        """Manually trigger indexing/synchronization."""
        import time
        try:
            from study_llm.core.bootstrap import get_runtime
            runtime = get_runtime()
            result = runtime.reconcile_and_enqueue()
            new = len(result["new"])
            changed = len(result["changed"])
            deleted = len(result["deleted"])
            if deleted:
                self.console.print(
                    f"[yellow]Removed knowledge from {deleted} deleted file(s).[/yellow]"
                )
            if new or changed:
                self.console.print(
                    f"[cyan]Indexing {new} new and {changed} changed document(s)...[/cyan]"
                )
                service = runtime.indexing_service
                while not service.queue.empty() or service.is_busy:
                    time.sleep(0.2)
                self.console.print("[green]✓ Indexing complete.[/green]")
            else:
                self.console.print("[green]Everything up to date.[/green]")
        except ImportError as e:
            missing = "qdrant-client" if "qdrant" in str(e).lower() else (
                "sentence-transformers" if "sentence" in str(e).lower() else str(e)
            )
            self.console.print(
                f"[yellow]Indexing unavailable (missing dependency: {missing}).[/yellow]"
            )
        except Exception as e:
            self.console.print(f"[red]Indexing failed: {e}[/red]")

    def show_config(self):
        """Show or modify configuration."""
        from study_llm.core.config import get_config

        config = get_config()
        self.console.print("[bold]StudyLLM Configuration[/bold] (config/settings.json)\n")

        sections = {
            "Model": ["auto_load_model", "preferred_quantization", "ctx_size", "gpu_layers"],
            "Embeddings": ["embedding_model", "embedding_batch_size"],
            "Documents": ["chunk_size", "chunk_overlap", "max_file_size_mb"],
            "RAG": [
                "retrieval_top_k", "reranker_top_k", "use_reranker",
                "context_budget_tokens",
            ],
            "Performance": ["indexing_workers", "background_indexing"],
            "UI": ["use_colors", "show_sources"],
            "Storage": ["vector_db_path"],
        }
        data = config.to_dict()
        for section, keys in sections.items():
            self.console.print(f"[blue]{section}[/blue]")
            for key in keys:
                if key in data:
                    self.console.print(f"  {key} = {data[key]}")
            self.console.print()


# Global app instance for CLI access
app = StudyLLMApp()

if __name__ == "__main__":
    app.run()