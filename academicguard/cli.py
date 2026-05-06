"""AcademicGuard CLI -- single entry point for all analysis modules."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import print as rprint
from rich.markup import escape

from academicguard import __version__
from academicguard.core.document import Document
from academicguard.core.report import AnalysisReport
from academicguard.detectors.ai_detector import AIDetector
from academicguard.detectors.plagiarism import PlagiarismDetector
from academicguard.detectors.grammar import GrammarChecker
from academicguard.style import VENUE_REGISTRY, get_style_checker
from academicguard.integrations.external import check_available_services, ENV_SETUP_GUIDE

app = typer.Typer(
    name="academicguard",
    help="Open-source academic writing integrity toolkit.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()

VENUE_CHOICES = sorted(VENUE_REGISTRY.keys())

# ------------------------------------------------------------------ #
# Main analyze command
# ------------------------------------------------------------------ #

@app.command(name="analyze")
def analyze(
    file: Path = typer.Argument(..., help="Input file (.txt, .docx, .pdf, .tex)"),
    venue: str = typer.Option(
        "ieee",
        "--venue", "-v",
        help=f"Target venue: {', '.join(VENUE_CHOICES)}",
    ),
    output_json: Optional[Path] = typer.Option(None, "--json", "-j", help="Save JSON report"),
    output_html: Optional[Path] = typer.Option(None, "--html", "-H", help="Save HTML report"),
    corpus_dir: Optional[Path] = typer.Option(None, "--corpus", "-c", help="Local corpus directory for plagiarism check"),
    skip_ai: bool = typer.Option(False, "--skip-ai", help="Skip AI detection (faster)"),
    skip_plagiarism: bool = typer.Option(False, "--skip-plagiarism", help="Skip plagiarism check"),
    skip_grammar: bool = typer.Option(False, "--skip-grammar", help="Skip grammar check"),
    skip_style: bool = typer.Option(False, "--skip-style", help="Skip venue style check"),
    use_transformer: bool = typer.Option(True, "--transformer/--no-transformer", help="Use GPT-2 for AI detection"),
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Show all findings including info-level"),
):
    """Analyze an academic document for AI content, plagiarism, grammar, and style."""

    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    venue_key = venue.lower().strip()
    if venue_key not in VENUE_REGISTRY:
        console.print(f"[red]Unknown venue '{venue}'. Choose from: {', '.join(VENUE_CHOICES)}[/red]")
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"[bold blue]AcademicGuard v{__version__}[/bold blue]\n"
        f"File: [cyan]{file}[/cyan]  |  Venue: [yellow]{venue.upper()}[/yellow]",
        title="Academic Writing Integrity Toolkit",
    ))

    # Load document
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as prog:
        t = prog.add_task("Loading document...", total=None)
        try:
            doc = Document.from_file(file)
        except Exception as e:
            console.print(f"[red]Error loading document: {e}[/red]")
            raise typer.Exit(1)
        prog.update(t, description=f"Loaded: {doc.word_count} words, {len(doc.sections)} sections")

    console.print(f"[green]Document:[/green] {doc.title or '(untitled)'} -- {doc.word_count} words, {len(doc.sections)} sections")

    report = AnalysisReport(document_title=doc.title or file.name, venue=venue.upper())

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        transient=False,
        console=console,
    ) as prog:

        # AI Detection
        if not skip_ai:
            t = prog.add_task("[cyan]AI Detection...", total=None)
            detector = AIDetector(use_transformer=use_transformer)
            result = detector.analyze(doc)
            report.modules.append(result)
            prog.update(t, description=f"[cyan]AI Detection[/cyan]  [{_label_color(result.label)}]{result.label}[/{_label_color(result.label)}] {result.score:.0%}")
            prog.stop_task(t)

        # Plagiarism
        if not skip_plagiarism:
            t = prog.add_task("[magenta]Plagiarism Check...", total=None)
            pd = PlagiarismDetector(corpus_dir=corpus_dir)
            result = pd.analyze(doc)
            report.modules.append(result)
            prog.update(t, description=f"[magenta]Plagiarism[/magenta]     [{_label_color(result.label)}]{result.label}[/{_label_color(result.label)}] {result.score:.0%}")
            prog.stop_task(t)

        # Grammar
        if not skip_grammar:
            t = prog.add_task("[yellow]Grammar Check...", total=None)
            gc = GrammarChecker()
            result = gc.analyze(doc)
            report.modules.append(result)
            prog.update(t, description=f"[yellow]Grammar[/yellow]        [{_label_color(result.label)}]{result.label}[/{_label_color(result.label)}] {result.score:.0%}")
            prog.stop_task(t)
            gc.close()

        # Style
        if not skip_style:
            t = prog.add_task(f"[green]{venue.upper()} Style...", total=None)
            checker = get_style_checker(venue)
            result = checker.analyze(doc)
            report.modules.append(result)
            prog.update(t, description=f"[green]{venue.upper()} Style[/green]        [{_label_color(result.label)}]{result.label}[/{_label_color(result.label)}] {result.score:.0%}")
            prog.stop_task(t)

    report.compute_overall()

    # Print summary table
    _print_summary(report, verbose)

    # Save outputs
    if output_json:
        report.save_json(output_json)
        console.print(f"[green]JSON report saved:[/green] {output_json}")
    if output_html:
        report.save_html(output_html)
        console.print(f"[green]HTML report saved:[/green] {output_html}")

    # Exit code reflects overall status
    if report.overall_label == "FAIL":
        raise typer.Exit(2)
    if report.overall_label == "WARN":
        raise typer.Exit(1)


# ------------------------------------------------------------------ #
# Quick sub-commands
# ------------------------------------------------------------------ #

@app.command(name="ai")
def detect_ai(
    file: Path = typer.Argument(..., help="Input file"),
    no_transformer: bool = typer.Option(False, "--no-transformer", help="Use heuristic mode only"),
):
    """Run AI content detection only."""
    doc = _load(file)
    detector = AIDetector(use_transformer=not no_transformer)
    result = detector.analyze(doc)
    _print_module(result)


@app.command(name="plagiarism")
def check_plagiarism(
    file: Path = typer.Argument(..., help="Input file"),
    corpus: Optional[Path] = typer.Option(None, "--corpus", "-c", help="Local corpus directory"),
):
    """Run plagiarism check only."""
    doc = _load(file)
    pd = PlagiarismDetector(corpus_dir=corpus)
    result = pd.analyze(doc)
    _print_module(result)


@app.command(name="grammar")
def check_grammar(
    file: Path = typer.Argument(..., help="Input file"),
    lang: str = typer.Option("en-US", "--lang", "-l", help="Language code (en-US, en-GB)"),
):
    """Run grammar and academic register check only."""
    doc = _load(file)
    gc = GrammarChecker(language=lang)
    result = gc.analyze(doc)
    _print_module(result)
    gc.close()


@app.command(name="style")
def check_style(
    file: Path = typer.Argument(..., help="Input file"),
    venue: str = typer.Option("ieee", "--venue", "-v", help=f"Venue: {', '.join(VENUE_CHOICES)}"),
):
    """Run venue style check only."""
    doc = _load(file)
    checker = get_style_checker(venue)
    result = checker.analyze(doc)
    _print_module(result)


@app.command(name="venues")
def list_venues():
    """List all supported publication venues."""
    table = Table(title="Supported Publication Venues", show_header=True, header_style="bold blue")
    table.add_column("Key", style="cyan")
    table.add_column("Venue")
    table.add_column("URL", style="dim")

    from academicguard.style import (
        IEEEStyleChecker, ElsevierStyleChecker, ACMStyleChecker,
        IETStyleChecker, BCSStyleChecker
    )
    checkers = [IEEEStyleChecker(), ElsevierStyleChecker(), ACMStyleChecker(),
                IETStyleChecker(), BCSStyleChecker()]
    for c in checkers:
        key = c.venue_name.split("/")[0].lower().strip()
        table.add_row(key, c.venue_name, c.venue_url)
    console.print(table)


@app.command(name="services")
def show_services():
    """Show available external service integrations."""
    services = check_available_services()
    table = Table(title="External Service Integrations", show_header=True, header_style="bold blue")
    table.add_column("Service")
    table.add_column("Status")
    table.add_column("Auth")
    table.add_column("Note", style="dim")

    for s in services:
        status = "[green]ENABLED[/green]" if s.available else "[red]NOT SET[/red]"
        table.add_row(s.name, status, s.auth_method, s.note)
    console.print(table)
    console.print()
    console.print("[dim]Run: academicguard setup-env  to see configuration instructions[/dim]")


@app.command(name="setup-env")
def setup_env():
    """Print environment variable setup instructions for external services."""
    console.print(Panel(ENV_SETUP_GUIDE, title="Environment Setup", border_style="blue"))


@app.command(name="version")
def version():
    """Show AcademicGuard version."""
    console.print(f"AcademicGuard v{__version__}")


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _load(file: Path) -> Document:
    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)
    try:
        return Document.from_file(file)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def _label_color(label: str) -> str:
    return {"PASS": "green", "WARN": "yellow", "FAIL": "red"}.get(label, "white")


def _print_module(result) -> None:
    color = _label_color(result.label)
    console.print(f"\n[bold]{result.module}[/bold]: [{color}]{result.label}[/{color}] ({result.score:.0%})")
    console.print(f"  {result.summary}")
    if result.findings:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Severity")
        table.add_column("Message")
        table.add_column("Suggestion", style="dim")
        for f in result.findings[:20]:
            sev_color = {"error": "red", "warning": "yellow", "info": "blue"}.get(f.severity, "white")
            table.add_row(
                f"[{sev_color}]{f.severity.upper()}[/{sev_color}]",
                escape(f.message[:100]),
                escape(f.suggestion[:80]),
            )
        console.print(table)


def _print_summary(report: AnalysisReport, verbose: bool) -> None:
    color = _label_color(report.overall_label)
    console.print()
    console.print(Panel.fit(
        f"Overall: [{color}]{report.overall_label}[/{color}]  ({report.overall_score:.0%})",
        title="Analysis Complete",
        border_style=color,
    ))

    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Module", style="cyan")
    table.add_column("Score")
    table.add_column("Status")
    table.add_column("Summary")

    for m in report.modules:
        c = _label_color(m.label)
        score_bar = _score_bar(m.score)
        table.add_row(m.module, score_bar, f"[{c}]{m.label}[/{c}]", m.summary[:80])
    console.print(table)

    # Show findings
    severity_filter = {"error", "warning"} if not verbose else {"error", "warning", "info"}
    all_findings = [f for m in report.modules for f in m.findings if f.severity in severity_filter]
    if all_findings:
        console.print(f"\n[bold]Findings ({len(all_findings)}):[/bold]")
        ftable = Table(show_header=True, header_style="bold")
        ftable.add_column("Sev.")
        ftable.add_column("Module")
        ftable.add_column("Message")
        ftable.add_column("Suggestion", style="dim")
        for m in report.modules:
            for f in m.findings:
                if f.severity not in severity_filter:
                    continue
                sev_color = {"error": "red", "warning": "yellow", "info": "blue"}.get(f.severity, "white")
                ftable.add_row(
                    f"[{sev_color}]{f.severity[0].upper()}[/{sev_color}]",
                    m.module,
                    escape(f.message[:100]),
                    escape(f.suggestion[:70]),
                )
        console.print(ftable)


def _score_bar(score: float, width: int = 10) -> str:
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    pct = f"{score:.0%}"
    color = "green" if score >= 0.80 else ("yellow" if score >= 0.55 else "red")
    return f"[{color}]{bar}[/{color}] {pct}"


if __name__ == "__main__":
    app()
