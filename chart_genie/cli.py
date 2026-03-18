"""CLI entry point for chart_genie.

This module implements the command-line interface using argparse and Rich
formatting. It wires together the full chart_genie pipeline:

1. Parse CLI arguments (data file, prompt, output path, backend, model).
2. Load and validate the input data (CSV or JSON, file or stdin).
3. Call the LLM backend to generate a ChartConfig from the user prompt.
4. Render the ChartConfig to a self-contained HTML file.
5. Display progress and results with Rich formatting.

Entry point::

    chart-genie --data sales.csv --prompt "Show monthly sales as a bar chart" --output chart.html

Or via Python::

    from chart_genie.cli import main
    main()
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from chart_genie import __version__
from chart_genie.chart_config import ChartConfig, ChartConfigError
from chart_genie.data_loader import (
    DataLoadError,
    UnsupportedFormatError,
    get_column_names,
    infer_column_types,
    load_data,
)
from chart_genie.llm_client import (
    LLMBackendError,
    LLMClient,
    LLMResponseParseError,
)
from chart_genie.renderer import RendererError, save_chart

# ---------------------------------------------------------------------------
# Rich console setup
# ---------------------------------------------------------------------------

_THEME = Theme(
    {
        "info": "cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "highlight": "bold magenta",
        "muted": "dim white",
        "step": "bold blue",
    }
)

#: Main Rich console used for output.
console = Console(theme=_THEME, stderr=False)

#: Error console writing to stderr.
err_console = Console(theme=_THEME, stderr=True)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    """Configure root logging level based on verbosity flag.

    Args:
        verbose: If True, set logging to DEBUG; otherwise WARNING.
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argparse argument parser.

    Returns:
        A configured :class:`argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        prog="chart-genie",
        description=(
            "Generate interactive HTML charts from data and a natural language prompt.\n"
            "Powered by Chart.js and an LLM backend (OpenAI or Ollama)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  chart-genie --data sales.csv --prompt \"Monthly sales as a bar chart\"\n"
            "  chart-genie --data temps.json --prompt \"Line chart of temperature trends\" -o temps.html\n"
            "  cat data.csv | chart-genie --prompt \"Pie chart by region\" -o regions.html\n"
            "\nEnvironment variables:\n"
            "  OPENAI_API_KEY          OpenAI API key (required for openai backend)\n"
            "  CHART_GENIE_LLM_BACKEND LLM backend: openai or ollama (default: openai)\n"
            "  CHART_GENIE_MODEL       Model name (default: gpt-4o / llama3)\n"
            "  CHART_GENIE_OLLAMA_URL  Ollama server URL (default: http://localhost:11434)\n"
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"chart-genie {__version__}",
    )

    parser.add_argument(
        "--data",
        "-d",
        metavar="FILE",
        default=None,
        help=(
            "Path to input data file (.csv or .json). "
            "Reads from stdin if omitted or set to '-'."
        ),
    )

    parser.add_argument(
        "--prompt",
        "-p",
        metavar="TEXT",
        default=None,
        help=(
            "Natural language description of the desired chart "
            "(e.g. 'Show monthly sales as a bar chart with tooltips'). "
            "If omitted, you will be prompted interactively."
        ),
    )

    parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        default="chart.html",
        help="Path for the output HTML file. (default: chart.html)",
    )

    parser.add_argument(
        "--backend",
        choices=["openai", "ollama"],
        default=None,
        help=(
            "LLM backend to use. Overrides CHART_GENIE_LLM_BACKEND env var. "
            "(default: openai)"
        ),
    )

    parser.add_argument(
        "--model",
        "-m",
        metavar="NAME",
        default=None,
        help=(
            "Model name to use (e.g. gpt-4o, gpt-4-turbo, llama3). "
            "Overrides CHART_GENIE_MODEL env var."
        ),
    )

    parser.add_argument(
        "--ollama-url",
        metavar="URL",
        default=None,
        help=(
            "Ollama server base URL. "
            "Overrides CHART_GENIE_OLLAMA_URL env var. "
            "(default: http://localhost:11434)"
        ),
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose/debug output.",
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        default=False,
        help="Disable Rich color output (plain text mode).",
    )

    return parser


# ---------------------------------------------------------------------------
# Rich UI helpers
# ---------------------------------------------------------------------------


def _print_banner(con: Console) -> None:
    """Print the chart_genie welcome banner.

    Args:
        con: The Rich console to print to.
    """
    title = Text()
    title.append(" chart_genie ", style="bold white on blue")
    title.append(f" v{__version__}", style="muted")
    con.print()
    con.print(title, justify="center")
    con.print(
        Text(
            "Transform data + natural language → interactive HTML charts",
            style="muted",
        ),
        justify="center",
    )
    con.print()


def _print_data_summary(
    con: Console,
    records: list[dict],
    source_label: str,
) -> None:
    """Print a Rich table summarising the loaded dataset.

    Args:
        con: The Rich console to print to.
        records: The loaded records list.
        source_label: Human-readable data source description (filename or 'stdin').
    """
    column_names = get_column_names(records)
    column_types = infer_column_types(records)

    table = Table(
        title=f"Dataset: [highlight]{source_label}[/highlight]",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        expand=False,
    )
    table.add_column("Column", style="bold", no_wrap=True)
    table.add_column("Type", style="info")
    table.add_column("Sample Values", style="muted", max_width=60)

    for col in column_names:
        col_type = column_types.get(col, "string")
        sample_vals = [
            str(r[col]) for r in records[:3] if r.get(col) is not None
        ]
        sample_str = ", ".join(sample_vals[:3])
        if len(records) > 3:
            sample_str += ", ..."
        table.add_row(col, col_type, sample_str)

    con.print(table)
    con.print(
        f"  [muted]Loaded [bold]{len(records)}[/bold] records, "
        f"[bold]{len(column_names)}[/bold] columns.[/muted]"
    )
    con.print()


def _print_config_summary(con: Console, config: ChartConfig) -> None:
    """Print a summary of the generated chart configuration.

    Args:
        con: The Rich console to print to.
        config: The generated chart configuration.
    """
    table = Table(
        title="Generated Chart Configuration",
        show_header=False,
        border_style="dim",
        expand=False,
    )
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Chart Type", config.chart_type.upper())
    table.add_row("Title", config.title)
    table.add_row("Datasets", str(len(config.datasets)))
    dataset_labels = ", ".join(ds.label for ds in config.datasets)
    table.add_row("Series", dataset_labels)
    table.add_row("Data Points", str(len(config.labels) or len(config.datasets[0].data)))
    if config.x_axis_label:
        table.add_row("X Axis", config.x_axis_label)
    if config.y_axis_label:
        table.add_row("Y Axis", config.y_axis_label)
    table.add_row("Legend", "✓" if config.show_legend else "✗")
    table.add_row("Tooltips", "✓" if config.show_tooltips else "✗")
    table.add_row("Responsive", "✓" if config.responsive else "✗")

    con.print(table)
    con.print()


def _prompt_user_interactively(con: Console) -> str:
    """Prompt the user interactively for a chart description.

    Used when --prompt is not provided on the command line.

    Args:
        con: The Rich console to use for the prompt.

    Returns:
        The user-entered prompt string.

    Raises:
        SystemExit: If the user enters an empty prompt or presses Ctrl+C.
    """
    con.print(
        Panel(
            "[info]No --prompt provided.[/info]\n"
            "Describe the chart you want to generate in plain English.\n"
            "[muted]Example: \"Show monthly sales and returns as a grouped bar chart\"[/muted]",
            title="Chart Description",
            border_style="blue",
        )
    )
    try:
        user_input = con.input("[bold cyan]Your prompt:[/bold cyan] ").strip()
    except (KeyboardInterrupt, EOFError):
        con.print("\n[warning]Aborted.[/warning]")
        raise SystemExit(1)

    if not user_input:
        err_console.print("[error]Error:[/error] Prompt cannot be empty.")
        raise SystemExit(1)

    return user_input


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def run(
    data_source: str | None,
    prompt: str,
    output_path: str | Path,
    backend: str | None = None,
    model: str | None = None,
    ollama_url: str | None = None,
    verbose: bool = False,
    con: Console | None = None,
) -> Path:
    """Execute the full chart generation pipeline.

    This function is the programmatic API equivalent of the CLI command. It
    loads data, generates a chart config via LLM, renders the HTML, and saves
    the output file.

    Args:
        data_source: Path to a CSV/JSON file, ``None``, or ``"-"`` for stdin.
        prompt: Natural language chart description.
        output_path: Destination path for the HTML output.
        backend: LLM backend override (``"openai"`` or ``"ollama"``).
        model: LLM model name override.
        ollama_url: Ollama server URL override.
        verbose: If True, print verbose progress information.
        con: Rich :class:`~rich.console.Console` to use. Creates a default
            console if not provided.

    Returns:
        The resolved :class:`pathlib.Path` where the HTML was saved.

    Raises:
        DataLoadError: If the input data cannot be loaded or parsed.
        LLMBackendError: If the LLM API call fails.
        LLMResponseParseError: If the LLM response cannot be parsed.
        RendererError: If the HTML rendering or file write fails.
        SystemExit: On unrecoverable errors (exits with code 1).
    """
    if con is None:
        con = Console(theme=_THEME)

    # --- Step 1: Load data ---------------------------------------------------
    source_label = Path(data_source).name if data_source and data_source != "-" else "stdin"
    with Progress(
        SpinnerColumn(),
        TextColumn("[step]Loading data...[/step] {task.description}"),
        TimeElapsedColumn(),
        console=con,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Reading from {source_label}")
        try:
            records = load_data(data_source)
        except (DataLoadError, UnsupportedFormatError, FileNotFoundError) as exc:
            err_console.print(f"[error]Data load error:[/error] {exc}")
            raise SystemExit(1) from exc

    con.print(f"  [success]✓[/success] Loaded [bold]{len(records)}[/bold] records from [highlight]{source_label}[/highlight]")

    if not records:
        err_console.print("[error]Error:[/error] The data source is empty — no records to chart.")
        raise SystemExit(1)

    if verbose:
        _print_data_summary(con, records, source_label)

    # --- Step 2: Generate chart config via LLM ------------------------------
    resolved_backend = backend or os.environ.get("CHART_GENIE_LLM_BACKEND", "openai")
    resolved_model = model or os.environ.get(
        "CHART_GENIE_MODEL",
        "gpt-4o" if resolved_backend == "openai" else "llama3",
    )

    con.print(
        f"  [step]→[/step] Calling [highlight]{resolved_backend}[/highlight] "
        f"([muted]{resolved_model}[/muted]) to generate chart config..."
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[step]Generating chart configuration...[/step]"),
        TimeElapsedColumn(),
        console=con,
        transient=True,
    ) as progress:
        progress.add_task("LLM request")
        try:
            client = LLMClient(
                backend=backend,
                model=model,
                ollama_url=ollama_url,
            )
            config = client.get_chart_config(
                user_description=prompt,
                records=records,
            )
        except LLMBackendError as exc:
            err_console.print(f"[error]LLM backend error:[/error] {exc}")
            raise SystemExit(1) from exc
        except LLMResponseParseError as exc:
            err_console.print(f"[error]LLM response parse error:[/error] {exc}")
            raise SystemExit(1) from exc
        except ValueError as exc:
            err_console.print(f"[error]Configuration error:[/error] {exc}")
            raise SystemExit(1) from exc

    con.print(
        f"  [success]✓[/success] Chart config generated: "
        f"[bold]{config.chart_type}[/bold] chart — [highlight]{config.title!r}[/highlight]"
    )

    if verbose:
        _print_config_summary(con, config)

    # --- Step 3: Render HTML -------------------------------------------------
    with Progress(
        SpinnerColumn(),
        TextColumn("[step]Rendering HTML...[/step]"),
        TimeElapsedColumn(),
        console=con,
        transient=True,
    ) as progress:
        progress.add_task("Rendering")
        try:
            output_file = save_chart(config, output_path=output_path)
        except RendererError as exc:
            err_console.print(f"[error]Render error:[/error] {exc}")
            raise SystemExit(1) from exc
        except OSError as exc:
            err_console.print(f"[error]File write error:[/error] {exc}")
            raise SystemExit(1) from exc

    con.print(
        f"  [success]✓[/success] HTML written to [bold green]{output_file}[/bold green]"
    )

    return output_file


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entry point for chart_genie.

    Parses command-line arguments, runs the chart generation pipeline, and
    prints the result with Rich formatting.

    Args:
        argv: Optional list of argument strings (for testing). If ``None``,
            reads from :data:`sys.argv`.

    Returns:
        Exit code: ``0`` on success, ``1`` on error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Configure logging before anything else.
    _configure_logging(args.verbose)

    # Set up the Rich console (disable color if requested).
    if args.no_color:
        global console, err_console  # noqa: PLW0603
        console = Console(theme=_THEME, no_color=True)
        err_console = Console(theme=_THEME, stderr=True, no_color=True)

    # Print the banner.
    _print_banner(console)

    # --- Resolve prompt -------------------------------------------------------
    prompt = args.prompt
    if not prompt:
        # Check if stdin has data piped (non-interactive) — in that case we
        # cannot interactively prompt for the description.
        if not sys.stdin.isatty() and args.data is None:
            err_console.print(
                "[error]Error:[/error] When reading data from stdin, --prompt is required.\n"
                "  Usage: cat data.csv | chart-genie --prompt \"Describe your chart\" -o out.html"
            )
            return 1
        prompt = _prompt_user_interactively(console)

    prompt = prompt.strip()
    if not prompt:
        err_console.print("[error]Error:[/error] --prompt cannot be empty.")
        return 1

    # --- Echo the request back to the user ------------------------------------
    console.print(
        Panel(
            f"[white]{prompt}[/white]",
            title="[bold blue]Chart Request[/bold blue]",
            border_style="blue",
            expand=False,
        )
    )
    console.print()

    # --- Run pipeline ---------------------------------------------------------
    start_time = time.monotonic()

    try:
        output_file = run(
            data_source=args.data,
            prompt=prompt,
            output_path=args.output,
            backend=args.backend,
            model=args.model,
            ollama_url=args.ollama_url,
            verbose=args.verbose,
            con=console,
        )
    except SystemExit as exc:
        # run() already printed the error.
        return int(exc.code) if exc.code is not None else 1

    elapsed = time.monotonic() - start_time

    # --- Success message ------------------------------------------------------
    console.print()
    console.print(Rule(style="green"))
    console.print(
        Panel(
            f"[success]Chart generated successfully in {elapsed:.1f}s[/success]\n\n"
            f"  Output file: [bold green]{output_file}[/bold green]\n"
            f"  Open in any browser to view your interactive chart.",
            title="[bold green]✓ Done[/bold green]",
            border_style="green",
            expand=False,
        )
    )
    console.print()

    logger.debug("chart_genie pipeline completed in %.2fs", elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
