"""Prompt templates for chart_genie LLM interactions.

This module provides the system and user prompt templates sent to the LLM
backend (OpenAI or Ollama) when requesting chart configuration extraction.
Prompts are designed to elicit a structured JSON response that can be parsed
directly into a ChartConfig object.

Typical usage::

    from chart_genie.prompts import build_system_prompt, build_user_prompt

    system_msg = build_system_prompt()
    user_msg = build_user_prompt(
        user_description="Show monthly sales as a bar chart",
        column_names=["month", "sales", "returns"],
        column_types={"month": "string", "sales": "number", "returns": "number"},
        sample_records=records[:3],
        num_records=12,
    )
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Supported chart types listed in prompts so the LLM knows the valid set.
_SUPPORTED_CHART_TYPES = ["bar", "line", "pie", "doughnut", "scatter", "radar"]

#: The JSON schema description embedded in the system prompt.
_CHART_CONFIG_SCHEMA = """
{
  "chart_type": "<string: one of bar|line|pie|doughnut|scatter|radar>",
  "title": "<string: descriptive chart title>",
  "labels": ["<string>", "..."],
  "datasets": [
    {
      "label": "<string: series name>",
      "data": ["<number or {x, y} object>", "..."],
      "background_color": "<CSS color string or list of CSS color strings>",
      "border_color": "<CSS color string or list of CSS color strings>",
      "border_width": "<integer, default 1>",
      "fill": "<boolean, default false>",
      "tension": "<float 0.0-1.0, default 0.3, for line charts>",
      "point_radius": "<integer, default 3>"
    }
  ],
  "x_axis_label": "<string: optional x-axis label>",
  "y_axis_label": "<string: optional y-axis label>",
  "show_legend": "<boolean, default true>",
  "show_tooltips": "<boolean, default true>",
  "responsive": "<boolean, default true>",
  "maintain_aspect_ratio": "<boolean, default false>"
}
""".strip()

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a data visualization expert and assistant for the chart_genie tool.
Your task is to analyze a dataset and a user's natural language description, \
then produce a complete Chart.js chart configuration in JSON format.

SUPPORTED CHART TYPES: {supported_types}

You MUST respond with ONLY a valid JSON object matching this schema exactly:
{schema}

CRITICAL RULES:
1. Respond with ONLY the raw JSON object. No markdown fences, no explanation, no commentary.
2. The JSON must be parseable by Python's json.loads().
3. Always choose the most appropriate chart type for the data and description.
4. For bar and line charts: 'labels' should be the x-axis categories (e.g. month names).
5. For pie and doughnut charts: 'labels' are the slice names, and there is typically ONE dataset.
6. For scatter charts: each data point in 'data' must be an object with 'x' and 'y' keys.
7. For radar charts: 'labels' are the axis spoke names.
8. Colors should be in rgba() format for best visual results, e.g. "rgba(54, 162, 235, 0.7)".
9. Provide a border_color that is the fully opaque version of background_color (alpha=1.0).
10. If the user mentions specific colors, use those; otherwise choose a pleasant, distinct palette.
11. The 'data' arrays must contain actual numbers extracted from the dataset — NOT column names.
12. Include ALL data rows in the datasets (not just the sample shown).
13. The 'title' should be descriptive and reflect the user's intent.
14. For multi-series charts, create one dataset per numeric column being visualized.
15. Only include numeric columns in datasets; use string columns for labels.
"""

# ---------------------------------------------------------------------------
# User prompt
# ---------------------------------------------------------------------------

_USER_PROMPT_TEMPLATE = """\
USER REQUEST:
{user_description}

DATASET INFORMATION:
- Total records: {num_records}
- Columns: {column_names_str}
- Column types: {column_types_str}

SAMPLE DATA (first {num_sample} of {num_records} records):
{sample_json}

FULL DATA (all {num_records} records):
{full_data_json}

Based on the user's request and the dataset above, generate the complete \
Chart.js configuration JSON.
Remember: respond with ONLY the raw JSON object.
"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_system_prompt() -> str:
    """Build the system prompt sent to the LLM.

    The system prompt instructs the LLM to act as a chart configuration
    generator and specifies the exact JSON schema it must produce.

    Returns:
        A formatted system prompt string.
    """
    return _SYSTEM_PROMPT_TEMPLATE.format(
        supported_types=", ".join(_SUPPORTED_CHART_TYPES),
        schema=_CHART_CONFIG_SCHEMA,
    )


def build_user_prompt(
    user_description: str,
    column_names: list[str],
    column_types: dict[str, str],
    sample_records: list[dict[str, Any]],
    num_records: int,
    all_records: list[dict[str, Any]] | None = None,
) -> str:
    """Build the user prompt containing data context and the user's request.

    Constructs a detailed prompt that includes column metadata, a sample of
    the data (for brevity), and the full dataset so the LLM can extract all
    data values for the chart configuration.

    Args:
        user_description: The plain-English chart description from the user.
        column_names: Ordered list of column names in the dataset.
        column_types: Mapping of column name to inferred type string
            (``"number"``, ``"boolean"``, or ``"string"``).
        sample_records: A small subset of records to show as a preview
            (typically the first 5).
        num_records: Total number of records in the full dataset.
        all_records: All records in the dataset. If ``None``, only the sample
            is used (not recommended for accuracy).

    Returns:
        A formatted user prompt string ready to send to the LLM.
    """
    column_names_str = ", ".join(column_names)
    column_types_str = ", ".join(
        f"{col}: {typ}" for col, typ in column_types.items()
    )

    num_sample = len(sample_records)
    sample_json = json.dumps(sample_records, indent=2, default=str)

    records_to_embed = all_records if all_records is not None else sample_records
    full_data_json = json.dumps(records_to_embed, indent=2, default=str)

    return _USER_PROMPT_TEMPLATE.format(
        user_description=user_description.strip(),
        num_records=num_records,
        column_names_str=column_names_str,
        column_types_str=column_types_str,
        num_sample=num_sample,
        sample_json=sample_json,
        full_data_json=full_data_json,
    )


def format_messages(
    user_description: str,
    column_names: list[str],
    column_types: dict[str, str],
    sample_records: list[dict[str, Any]],
    num_records: int,
    all_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Build a complete messages list for the LLM chat API.

    Combines the system prompt and user prompt into the standard
    ``[{"role": ..., "content": ...}]`` format used by both OpenAI and
    Ollama chat completions APIs.

    Args:
        user_description: The plain-English chart description from the user.
        column_names: Ordered list of column names.
        column_types: Mapping of column name to inferred type string.
        sample_records: A small preview subset of the records.
        num_records: Total record count.
        all_records: All records; passed through to :func:`build_user_prompt`.

    Returns:
        A list of message dicts suitable for passing as the ``messages``
        argument to an OpenAI-compatible chat completions call.
    """
    return [
        {
            "role": "system",
            "content": build_system_prompt(),
        },
        {
            "role": "user",
            "content": build_user_prompt(
                user_description=user_description,
                column_names=column_names,
                column_types=column_types,
                sample_records=sample_records,
                num_records=num_records,
                all_records=all_records,
            ),
        },
    ]
