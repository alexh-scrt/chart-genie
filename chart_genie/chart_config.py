"""Chart configuration dataclass with validation for chart_genie.

This module defines the ChartConfig dataclass, which holds and validates
all parameters that describe a chart: type, title, axes, datasets, colors,
and rendering options. It is populated from LLM-returned JSON and consumed
by the renderer to produce the final Chart.js configuration.

Typical usage::

    from chart_genie.chart_config import ChartConfig, DatasetConfig

    config = ChartConfig.from_dict({
        "chart_type": "bar",
        "title": "Monthly Sales",
        "x_axis_label": "Month",
        "y_axis_label": "Sales (USD)",
        "labels": ["Jan", "Feb", "Mar"],
        "datasets": [
            {
                "label": "Sales",
                "data": [15200, 18400, 21000],
                "background_color": "rgba(54, 162, 235, 0.7)",
                "border_color": "rgba(54, 162, 235, 1.0)",
            }
        ],
    })
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: All chart types recognised by Chart.js and supported by chart_genie.
SUPPORTED_CHART_TYPES: frozenset[str] = frozenset(
    {"bar", "line", "pie", "doughnut", "scatter", "radar"}
)

#: Default background colors applied when a dataset does not specify its own.
DEFAULT_BACKGROUND_COLORS: list[str] = [
    "rgba(54, 162, 235, 0.7)",
    "rgba(255, 99, 132, 0.7)",
    "rgba(75, 192, 192, 0.7)",
    "rgba(255, 206, 86, 0.7)",
    "rgba(153, 102, 255, 0.7)",
    "rgba(255, 159, 64, 0.7)",
    "rgba(199, 199, 199, 0.7)",
    "rgba(83, 102, 255, 0.7)",
]

#: Matching border colors (fully opaque) for the defaults above.
DEFAULT_BORDER_COLORS: list[str] = [
    "rgba(54, 162, 235, 1.0)",
    "rgba(255, 99, 132, 1.0)",
    "rgba(75, 192, 192, 1.0)",
    "rgba(255, 206, 86, 1.0)",
    "rgba(153, 102, 255, 1.0)",
    "rgba(255, 159, 64, 1.0)",
    "rgba(199, 199, 199, 1.0)",
    "rgba(83, 102, 255, 1.0)",
]

# Regex for a basic CSS color: hex (#rgb / #rrggbb / #rrggbbaa) or rgba?(...)
_CSS_COLOR_RE = re.compile(
    r"^("
    r"#([0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})"
    r"|"
    r"rgba?\(\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}(\s*,\s*[\d.]+)?\s*\)"
    r"|"
    r"hsl\(\s*\d{1,3}\s*,\s*\d{1,3}%\s*,\s*\d{1,3}%\s*\)"
    r"|"
    r"[a-zA-Z]{3,30}"
    r")$"
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ChartConfigError(ValueError):
    """Raised when a ChartConfig cannot be constructed due to invalid data."""


# ---------------------------------------------------------------------------
# Helper validators
# ---------------------------------------------------------------------------


def _validate_chart_type(value: str) -> str:
    """Validate and normalise a chart type string.

    Args:
        value: The chart type string to validate (case-insensitive).

    Returns:
        The lower-cased, validated chart type.

    Raises:
        ChartConfigError: If the value is not a supported chart type.
    """
    normalised = value.strip().lower()
    if normalised not in SUPPORTED_CHART_TYPES:
        supported = ", ".join(sorted(SUPPORTED_CHART_TYPES))
        raise ChartConfigError(
            f"Unsupported chart type '{value}'. "
            f"Supported types are: {supported}."
        )
    return normalised


def _validate_color(value: str, field_name: str = "color") -> str:
    """Validate that a string looks like a recognisable CSS color.

    Accepts hex codes, rgb/rgba/hsl functions, and named CSS colors.  The
    check is intentionally lenient — it rejects obvious non-colors while
    still permitting unusual-but-valid values the LLM might return.

    Args:
        value: The color string to validate.
        field_name: Human-readable field name for error messages.

    Returns:
        The stripped color string.

    Raises:
        ChartConfigError: If the value does not look like a CSS color.
    """
    stripped = value.strip()
    if not stripped:
        raise ChartConfigError(f"'{field_name}' must not be an empty string.")
    if not _CSS_COLOR_RE.match(stripped):
        raise ChartConfigError(
            f"'{field_name}' value '{stripped}' does not appear to be a valid "
            "CSS color (expected hex, rgb/rgba, hsl, or named color)."
        )
    return stripped


def _validate_nonempty_string(value: str, field_name: str) -> str:
    """Validate that a string is non-empty after stripping whitespace.

    Args:
        value: The string to validate.
        field_name: Human-readable field name for error messages.

    Returns:
        The stripped string.

    Raises:
        ChartConfigError: If the string is blank.
    """
    stripped = str(value).strip()
    if not stripped:
        raise ChartConfigError(f"'{field_name}' must not be an empty string.")
    return stripped


# ---------------------------------------------------------------------------
# DatasetConfig
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Configuration for a single dataset within a chart.

    Attributes:
        label: Human-readable series name shown in the legend.
        data: The numeric (or x/y pair) data points for this series.
        background_color: Fill color for bars, pie slices, or area fills.
            May be a single color string or a list of color strings (one per
            data point, required for pie/doughnut charts).
        border_color: Border/stroke color for bars or lines.  Same scalar-or-
            list semantics as *background_color*.
        border_width: Stroke width in pixels.
        fill: Whether to fill the area under a line chart series.
        tension: Bezier curve tension for line charts (0 = straight lines).
        point_radius: Radius of data-point markers on line/radar charts.
    """

    label: str
    data: list[float | int | dict[str, float]]
    background_color: str | list[str] = field(default="")
    border_color: str | list[str] = field(default="")
    border_width: int = 1
    fill: bool = False
    tension: float = 0.3
    point_radius: int = 3

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        index: int = 0,
    ) -> "DatasetConfig":
        """Construct a DatasetConfig from a raw dictionary (e.g. LLM output).

        Missing optional fields are filled in with sensible defaults.  The
        *index* parameter is used to cycle through the default color palette
        when colors are not provided.

        Args:
            raw: Dictionary representation of the dataset.
            index: Zero-based position of this dataset within the chart, used
                to pick a default color from the palette.

        Returns:
            A validated DatasetConfig instance.

        Raises:
            ChartConfigError: If required fields are missing or invalid.
        """
        if "label" not in raw:
            raise ChartConfigError(
                "Each dataset must include a 'label' field."
            )
        label = _validate_nonempty_string(raw["label"], "dataset.label")

        if "data" not in raw:
            raise ChartConfigError(
                f"Dataset '{label}' is missing the required 'data' field."
            )
        data = _parse_data_points(raw["data"], label)

        # Colors — validate only when explicitly provided.
        palette_idx = index % len(DEFAULT_BACKGROUND_COLORS)
        bg_color_raw = raw.get("background_color", "")
        border_color_raw = raw.get("border_color", "")

        bg_color: str | list[str]
        border_color: str | list[str]

        if isinstance(bg_color_raw, list):
            bg_color = [
                _validate_color(c, f"dataset[{label}].background_color[{i}]")
                for i, c in enumerate(bg_color_raw)
            ]
        elif bg_color_raw:
            bg_color = _validate_color(bg_color_raw, f"dataset[{label}].background_color")
        else:
            bg_color = DEFAULT_BACKGROUND_COLORS[palette_idx]

        if isinstance(border_color_raw, list):
            border_color = [
                _validate_color(c, f"dataset[{label}].border_color[{i}]")
                for i, c in enumerate(border_color_raw)
            ]
        elif border_color_raw:
            border_color = _validate_color(
                border_color_raw, f"dataset[{label}].border_color"
            )
        else:
            border_color = DEFAULT_BORDER_COLORS[palette_idx]

        border_width = int(raw.get("border_width", 1))
        if border_width < 0:
            raise ChartConfigError(
                f"Dataset '{label}': 'border_width' must be a non-negative integer."
            )

        fill = bool(raw.get("fill", False))

        tension_raw = raw.get("tension", 0.3)
        try:
            tension = float(tension_raw)
        except (TypeError, ValueError) as exc:
            raise ChartConfigError(
                f"Dataset '{label}': 'tension' must be a float, got '{tension_raw}'."
            ) from exc
        if not 0.0 <= tension <= 1.0:
            raise ChartConfigError(
                f"Dataset '{label}': 'tension' must be between 0.0 and 1.0."
            )

        point_radius = int(raw.get("point_radius", 3))
        if point_radius < 0:
            raise ChartConfigError(
                f"Dataset '{label}': 'point_radius' must be a non-negative integer."
            )

        return cls(
            label=label,
            data=data,
            background_color=bg_color,
            border_color=border_color,
            border_width=border_width,
            fill=fill,
            tension=tension,
            point_radius=point_radius,
        )

    def to_chartjs_dict(self) -> dict[str, Any]:
        """Serialise this dataset to a Chart.js-compatible dictionary.

        Returns:
            A dictionary suitable for embedding in a Chart.js ``datasets``
            array.
        """
        return {
            "label": self.label,
            "data": self.data,
            "backgroundColor": self.background_color,
            "borderColor": self.border_color,
            "borderWidth": self.border_width,
            "fill": self.fill,
            "tension": self.tension,
            "pointRadius": self.point_radius,
        }


def _parse_data_points(
    raw_data: Any,
    dataset_label: str,
) -> list[float | int | dict[str, float]]:
    """Parse and validate a raw data sequence into a typed list.

    Accepts lists of numbers or lists of ``{x, y}`` dicts (for scatter charts).

    Args:
        raw_data: The raw value from the input dictionary.
        dataset_label: Used for error messages.

    Returns:
        A list of numeric values or x/y dictionaries.

    Raises:
        ChartConfigError: If the data cannot be interpreted.
    """
    if not isinstance(raw_data, list):
        raise ChartConfigError(
            f"Dataset '{dataset_label}': 'data' must be a list, "
            f"got {type(raw_data).__name__}."
        )
    parsed: list[float | int | dict[str, float]] = []
    for i, item in enumerate(raw_data):
        if isinstance(item, dict):
            # Scatter / bubble point — require at least x and y.
            if "x" not in item or "y" not in item:
                raise ChartConfigError(
                    f"Dataset '{dataset_label}': data point {i} is a dict but "
                    "is missing required 'x' or 'y' keys."
                )
            try:
                parsed.append({"x": float(item["x"]), "y": float(item["y"])})
            except (TypeError, ValueError) as exc:
                raise ChartConfigError(
                    f"Dataset '{dataset_label}': data point {i} has non-numeric "
                    "x/y values."
                ) from exc
        elif item is None:
            # Chart.js accepts null gaps in line charts.
            parsed.append(None)  # type: ignore[arg-type]
        else:
            try:
                numeric = float(item)
                # Preserve int representation where possible.
                parsed.append(int(numeric) if numeric == int(numeric) else numeric)
            except (TypeError, ValueError) as exc:
                raise ChartConfigError(
                    f"Dataset '{dataset_label}': data point {i} ('{item}') is not "
                    "numeric and not a valid x/y dict."
                ) from exc
    return parsed


# ---------------------------------------------------------------------------
# ChartConfig
# ---------------------------------------------------------------------------


@dataclass
class ChartConfig:
    """Top-level configuration for a chart_genie chart.

    Holds all parameters needed to render a Chart.js chart, including chart
    type, title, axis labels, data labels (x-axis ticks for bar/line charts),
    and the list of datasets.

    Attributes:
        chart_type: One of the supported Chart.js chart types (bar, line, etc.).
        title: Human-readable chart title displayed above the chart.
        labels: Category labels for the x-axis (bar, line, radar, pie, doughnut).
            For scatter charts this is typically empty.
        datasets: One or more DatasetConfig objects.
        x_axis_label: Optional label for the x-axis.
        y_axis_label: Optional label for the y-axis.
        show_legend: Whether to display the chart legend.
        show_tooltips: Whether to enable Chart.js tooltips on hover.
        responsive: Whether the chart canvas should be responsive.
        maintain_aspect_ratio: Whether Chart.js should maintain the aspect ratio.
        width: Optional explicit canvas width in pixels (ignored when responsive).
        height: Optional explicit canvas height in pixels.
        extra_options: Arbitrary additional Chart.js options dict merged into
            the top-level ``options`` object before rendering.
    """

    chart_type: str
    title: str
    labels: list[str]
    datasets: list[DatasetConfig]
    x_axis_label: str = ""
    y_axis_label: str = ""
    show_legend: bool = True
    show_tooltips: bool = True
    responsive: bool = True
    maintain_aspect_ratio: bool = False
    width: int | None = None
    height: int | None = None
    extra_options: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Run field-level validation immediately after construction."""
        self.chart_type = _validate_chart_type(self.chart_type)
        self.title = _validate_nonempty_string(self.title, "title")

        if not isinstance(self.labels, list):
            raise ChartConfigError("'labels' must be a list of strings.")
        self.labels = [str(lbl) for lbl in self.labels]

        if not self.datasets:
            raise ChartConfigError(
                "'datasets' must contain at least one DatasetConfig."
            )
        if not all(isinstance(ds, DatasetConfig) for ds in self.datasets):
            raise ChartConfigError(
                "All items in 'datasets' must be DatasetConfig instances."
            )

        if self.width is not None and self.width <= 0:
            raise ChartConfigError("'width' must be a positive integer.")
        if self.height is not None and self.height <= 0:
            raise ChartConfigError("'height' must be a positive integer.")

        if not isinstance(self.extra_options, dict):
            raise ChartConfigError("'extra_options' must be a dictionary.")

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ChartConfig":
        """Construct a ChartConfig from a raw dictionary (e.g. LLM JSON output).

        All required fields are validated; optional fields fall back to
        sensible defaults.

        Required keys:
            - ``chart_type`` (str)
            - ``title`` (str)
            - ``datasets`` (list of dicts)

        Optional keys:
            - ``labels`` (list of str; defaults to [])
            - ``x_axis_label`` (str)
            - ``y_axis_label`` (str)
            - ``show_legend`` (bool)
            - ``show_tooltips`` (bool)
            - ``responsive`` (bool)
            - ``maintain_aspect_ratio`` (bool)
            - ``width`` (int)
            - ``height`` (int)
            - ``extra_options`` (dict)

        Args:
            raw: A dictionary, typically parsed from LLM JSON output.

        Returns:
            A fully validated ChartConfig instance.

        Raises:
            ChartConfigError: If any required field is missing or invalid.
        """
        # --- required fields ---
        for required in ("chart_type", "title", "datasets"):
            if required not in raw:
                raise ChartConfigError(
                    f"Missing required field '{required}' in chart config."
                )

        raw_datasets = raw["datasets"]
        if not isinstance(raw_datasets, list) or not raw_datasets:
            raise ChartConfigError(
                "'datasets' must be a non-empty list of dataset objects."
            )

        datasets = [
            DatasetConfig.from_dict(ds, index=i)
            for i, ds in enumerate(raw_datasets)
        ]

        # --- optional fields with defaults ---
        labels_raw = raw.get("labels", [])
        if not isinstance(labels_raw, list):
            raise ChartConfigError("'labels' must be a list.")
        labels = [str(lbl) for lbl in labels_raw]

        width_raw = raw.get("width")
        height_raw = raw.get("height")
        width = int(width_raw) if width_raw is not None else None
        height = int(height_raw) if height_raw is not None else None

        extra_options = raw.get("extra_options", {})
        if not isinstance(extra_options, dict):
            raise ChartConfigError("'extra_options' must be a dictionary.")

        return cls(
            chart_type=raw["chart_type"],
            title=raw["title"],
            labels=labels,
            datasets=datasets,
            x_axis_label=str(raw.get("x_axis_label", "")),
            y_axis_label=str(raw.get("y_axis_label", "")),
            show_legend=bool(raw.get("show_legend", True)),
            show_tooltips=bool(raw.get("show_tooltips", True)),
            responsive=bool(raw.get("responsive", True)),
            maintain_aspect_ratio=bool(raw.get("maintain_aspect_ratio", False)),
            width=width,
            height=height,
            extra_options=extra_options,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_chartjs_config(self) -> dict[str, Any]:
        """Produce a complete Chart.js configuration dictionary.

        The returned dict is suitable for direct JSON serialisation and
        embedding inside a ``new Chart(ctx, <config>)`` call.

        Returns:
            A Chart.js config dict with ``type``, ``data``, and ``options``
            top-level keys.
        """
        options: dict[str, Any] = {
            "responsive": self.responsive,
            "maintainAspectRatio": self.maintain_aspect_ratio,
            "plugins": {
                "title": {
                    "display": bool(self.title),
                    "text": self.title,
                    "font": {"size": 18},
                },
                "legend": {
                    "display": self.show_legend,
                },
                "tooltip": {
                    "enabled": self.show_tooltips,
                },
            },
        }

        # Axis labels are only relevant for cartesian chart types.
        if self.chart_type in {"bar", "line", "scatter"}:
            scales: dict[str, Any] = {}
            if self.x_axis_label:
                scales["x"] = {
                    "title": {
                        "display": True,
                        "text": self.x_axis_label,
                    }
                }
            if self.y_axis_label:
                scales["y"] = {
                    "title": {
                        "display": True,
                        "text": self.y_axis_label,
                    }
                }
            if scales:
                options["scales"] = scales

        # Deep-merge extra_options on top of the built options dict.
        _deep_merge(options, self.extra_options)

        config: dict[str, Any] = {
            "type": self.chart_type,
            "data": {
                "labels": self.labels,
                "datasets": [ds.to_chartjs_dict() for ds in self.datasets],
            },
            "options": options,
        }
        return config

    def to_dict(self) -> dict[str, Any]:
        """Serialise this ChartConfig to a plain Python dictionary.

        Useful for debugging, caching, or passing between processes.

        Returns:
            A JSON-serialisable dictionary representation.
        """
        return {
            "chart_type": self.chart_type,
            "title": self.title,
            "labels": self.labels,
            "datasets": [
                {
                    "label": ds.label,
                    "data": ds.data,
                    "background_color": ds.background_color,
                    "border_color": ds.border_color,
                    "border_width": ds.border_width,
                    "fill": ds.fill,
                    "tension": ds.tension,
                    "point_radius": ds.point_radius,
                }
                for ds in self.datasets
            ],
            "x_axis_label": self.x_axis_label,
            "y_axis_label": self.y_axis_label,
            "show_legend": self.show_legend,
            "show_tooltips": self.show_tooltips,
            "responsive": self.responsive,
            "maintain_aspect_ratio": self.maintain_aspect_ratio,
            "width": self.width,
            "height": self.height,
            "extra_options": self.extra_options,
        }

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_cartesian(self) -> bool:
        """True if the chart type uses Cartesian (x/y) axes."""
        return self.chart_type in {"bar", "line", "scatter"}

    @property
    def is_radial(self) -> bool:
        """True if the chart type uses radial / polar axes."""
        return self.chart_type in {"pie", "doughnut", "radar"}

    def __repr__(self) -> str:  # pragma: no cover
        dataset_labels = [ds.label for ds in self.datasets]
        return (
            f"ChartConfig(chart_type={self.chart_type!r}, title={self.title!r}, "
            f"datasets={dataset_labels!r})"
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Recursively merge *override* into *base*, modifying *base* in place.

    Nested dicts are merged rather than replaced; all other value types in
    *override* overwrite the corresponding key in *base*.

    Args:
        base: The dictionary to merge into (mutated in place).
        override: The dictionary whose values take precedence.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
