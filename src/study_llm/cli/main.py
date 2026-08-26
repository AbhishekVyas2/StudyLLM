#!/usr/bin/env python3
"""StudyLLM CLI - Main entry point."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from study_llm.core.app import StudyLLMApp

app = typer.Typer(
    name="study-llm",
    help="A private, local AI that learns from the files in your data/ folder",
    no_args_is_help=True,
)
console = Console()


@app.command()
def main():
    """Start StudyLLM - checks hardware, model, and data then starts interactive mode."""
    app_instance = StudyLLMApp()
    app_instance.run()


@app.command()
def ask(question: str):
    """Ask a question to StudyLLM."""
    app_instance = StudyLLMApp()
    app_instance.ask_question(question)


@app.command()
def chat():
    """Start interactive chat mode."""
    app_instance = StudyLLMApp()
    app_instance.start_chat()


@app.command()
def models():
    """List available GGUF models."""
    app_instance = StudyLLMApp()
    app_instance.list_models()


@app.command()
def status():
    """Show StudyLLM status."""
    app_instance = StudyLLMApp()
    app_instance.show_status()


@app.command()
def index():
    """Manually trigger indexing/synchronization."""
    app_instance = StudyLLMApp()
    app_instance.manual_index()


@app.command()
def config():
    """Show or modify configuration."""
    app_instance = StudyLLMApp()
    app_instance.show_config()


if __name__ == "__main__":
    app()