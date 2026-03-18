"""chart_genie - Transform natural language descriptions and raw data into interactive HTML charts.

This package provides a CLI tool and programmatic API for generating self-contained,
interactive HTML chart files powered by Chart.js. Users describe charts in plain English
and provide CSV or JSON data; an LLM backend interprets the intent and produces a
fully configured, embeddable visualization.

Example usage (programmatic)::

    from chart_genie import generate_chart

    html = generate_chart(
        data_path="sales.csv",
        prompt="Show monthly sales as a bar chart with tooltips",
        output_path="sales_chart.html",
    )

Example usage (CLI)::

    chart-genie --data sales.csv --prompt "Show monthly sales as a bar chart" --output chart.html
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Chart Genie Contributors"
__license__ = "MIT"

__all__ = [
    "__version__",
    "__author__",
    "__license__",
]
