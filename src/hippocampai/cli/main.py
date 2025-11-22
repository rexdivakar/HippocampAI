"""CLI interface using Typer."""

import json
import sys
from typing import Optional

import typer
from rich import print
from rich.table import Table

from hippocampai.api.app import run_server
from hippocampai.client import MemoryClient
from hippocampai.config import get_config

app = typer.Typer(name="hippocampai", help="HippocampAI memory engine")


@app.command()
def init() -> None:
    """Initialize HippocampAI (create collections)."""
    try:
        config = get_config()
        client = MemoryClient(config=config)  # noqa: F841
        print("[green]✓[/green] HippocampAI initialized successfully")
        print(f"  Qdrant: {config.qdrant_url}")
        print(f"  Collections: {config.collection_facts}, {config.collection_prefs}")
    except Exception as e:
        print(f"[red]✗[/red] Initialization failed: {e}")
        raise typer.Exit(1)


@app.command()
def remember(
    user: str = typer.Option(..., "--user", "-u", help="User ID"),
    text: Optional[str] = typer.Option(None, "--text", "-t", help="Memory text"),
    type: str = typer.Option("fact", "--type", help="Memory type"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID"),
) -> None:
    """Store a memory."""
    if not text:
        # Read from stdin
        text = sys.stdin.read().strip()

    if not text:
        print("[red]✗[/red] No text provided")
        raise typer.Exit(1)

    try:
        client = MemoryClient()
        memory = client.remember(text=text, user_id=user, session_id=session, type=type)
        print(f"[green]✓[/green] Stored memory: {memory.id}")
        print(f"  Text: {memory.text}")
        print(f"  Type: {memory.type.value}")
        print(f"  Importance: {memory.importance:.1f}/10")
    except Exception as e:
        print(f"[red]✗[/red] Failed: {e}")
        raise typer.Exit(1)


@app.command()
def recall(
    user: str = typer.Option(..., "--user", "-u", help="User ID"),
    query: str = typer.Option(..., "--query", "-q", help="Search query"),
    k: int = typer.Option(5, "-k", help="Number of results"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Session ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Retrieve memories."""
    try:
        client = MemoryClient()
        results = client.recall(query=query, user_id=user, session_id=session, k=k)

        if json_output:
            data = [
                {
                    "id": r.memory.id,
                    "text": r.memory.text,
                    "score": r.score,
                    "breakdown": r.breakdown,
                }
                for r in results
            ]
            print(json.dumps(data, indent=2))
        else:
            if not results:
                print("[yellow]No memories found[/yellow]")
                return

            table = Table(title=f"Memories for: {query}")
            table.add_column("Score", style="cyan", width=8)
            table.add_column("Text", style="white")
            table.add_column("Type", style="green", width=12)

            for r in results:
                table.add_row(f"{r.score:.3f}", r.memory.text[:80], r.memory.type.value)

            print(table)

    except Exception as e:
        print(f"[red]✗[/red] Failed: {e}")
        raise typer.Exit(1)


@app.command(name="api")
def run_api(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind"),
) -> None:
    """Run FastAPI server."""
    print(f"[green]Starting HippocampAI API server at http://{host}:{port}[/green]")
    run_server(host=host, port=port)


if __name__ == "__main__":
    app()
