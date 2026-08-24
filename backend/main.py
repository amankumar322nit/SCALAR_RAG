#!/usr/bin/env python3
"""
Scaler Learner Support RAG CLI and Service Entrypoint.
"""

import sys
from pathlib import Path

# Add backend directory to Python path
BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(ROOT_DIR))

import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.pipeline import RAGPipeline
from src.config import TOP_K

console = Console()


def print_banner():
    console.print(Panel(
        "[bold cyan]🚀 SCALER LEARNER SUPPORT RAG SYSTEM[/bold cyan]\n"
        "[dim]Production RAG Pipeline with Grounding, Citations & Structured Observability[/dim]",
        border_style="cyan"
    ))


def format_cli_output(response: dict):
    answer = response.get("answer", "")
    sources = response.get("sources", [])
    latency = response.get("latency_ms", 0.0)
    citations = response.get("citations", [])

    console.print("\n[bold green]💡 Grounded Answer:[/bold green]")
    console.print(Panel(answer, border_style="green"))

    if sources:
        table = Table(title="📑 Retrieved Grounding Sources & Citations", border_style="blue")
        table.add_column("Tag", style="bold cyan", width=10)
        table.add_column("Score", justify="right", style="magenta", width=8)
        table.add_column("Document", style="yellow", width=28)
        table.add_column("Section", style="white")

        for s in sources:
            is_cited = s["chunk_tag"] in citations
            tag_display = f"{s['chunk_tag']} {'⭐' if is_cited else ''}"
            table.add_row(
                tag_display,
                f"{s['similarity_score']:.3f}",
                s["doc_path"],
                s["section_path"]
            )
        console.print(table)

    console.print(f"[dim]⏱️ End-to-end Latency: {latency:.2f} ms | Citations: {', '.join(citations) if citations else 'None'}[/dim]\n")


def interactive_mode(rag: RAGPipeline):
    print_banner()
    console.print("[bold yellow]Interactive Session Started. Type your question or 'exit'/'quit' to end.[/bold yellow]\n")

    while True:
        try:
            query = console.input("[bold cyan]Ask Scaler AI > [/bold cyan]").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                console.print("[bold yellow]Exiting interactive session. Goodbye![/bold yellow]")
                break

            response = rag.query(query, emit_stdout=True)
            format_cli_output(response)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]Session ended.[/bold yellow]")
            break


def main():
    parser = argparse.ArgumentParser(description="Scaler Learner Support RAG System")
    parser.add_argument("--query", "-q", type=str, help="Natural language query to ask")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive chat CLI")
    parser.add_argument("--ingest", action="store_true", help="Re-index the entire corpus")
    parser.add_argument("--server", action="store_true", help="Start the REST API and Web UI server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on (default: 8000)")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Number of chunks to retrieve")

    args = parser.parse_args()

    rag = RAGPipeline()

    if args.ingest:
        count = rag.ingest()
        console.print(f"[bold green]Successfully indexed {count} chunks into vector store.[/bold green]")
        return

    if args.server:
        import uvicorn
        from src.api import app
        console.print(f"[bold cyan]Starting Scaler RAG API & Web Server on http://0.0.0.0:{args.port}[/bold cyan]")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
        return

    if args.query:
        response = rag.query(args.query, top_k=args.top_k, emit_stdout=True)
        format_cli_output(response)
        return

    if args.interactive or len(sys.argv) == 1:
        interactive_mode(rag)


if __name__ == "__main__":
    main()
