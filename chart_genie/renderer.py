"""HTML renderer for chart_genie: produces self-contained Chart.js HTML files.

This module uses Jinja2 to render a self-contained HTML file that embeds
Chart.js (from CDN), the chart configuration, and all data. The output is
a single portable HTML file that can be opened in any modern browser with
no additional dependencies.

Typical usage::

    from chart_genie.renderer import render_chart, save_chart
    from chart_genie.chart_config import ChartConfig

    config = ChartConfig.from_dict({...})

    # Render to string
    html = render_chart(config)

    # Render and save to file
    output_path = save_chart(config, output_path="chart.html")
    print(f"Chart saved to {output_path}")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, TemplateNotFound, select_autoescape

from chart_genie.chart_config import ChartConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Absolute path to the templates directory bundled with chart_genie.
_TEMPLATES_DIR: Path = Path(__file__).parent / "templates"

#: Default template filename within the templates directory.
_DEFAULT_TEMPLATE = "chart.html.j2"

#: Chart.js CDN URL used in the generated HTML.
CHARTJS_CDN_URL = "https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"

#: Default canvas width (pixels) used when responsive=False and no width is specified.
_DEFAULT_CANVAS_WIDTH = 900

#: Default canvas height (pixels) used when responsive=False and no height is specified.
_DEFAULT_CANVAS_HEIGHT = 500


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RendererError(RuntimeError):
    """Raised when the HTML renderer encounters an unrecoverable error."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_chart(
    config: ChartConfig,
    chartjs_cdn_url: str = CHARTJS_CDN_URL,
    extra_context: dict[str, Any] | None = None,
) -> str:
    """Render a ChartConfig into a self-contained HTML string.

    Uses the bundled Jinja2 template (``chart.html.j2``) to produce a fully
    self-contained HTML document. Chart.js is loaded from CDN; all chart
    configuration and data are embedded as inline JavaScript.

    Args:
        config: A validated :class:`~chart_genie.chart_config.ChartConfig`
            instance containing all chart parameters and data.
        chartjs_cdn_url: URL for the Chart.js CDN script tag. Override to
            use a different version or a local copy.
        extra_context: Optional dict of additional Jinja2 template variables
            merged into the render context. Keys override built-in context
            values, so use with care.

    Returns:
        A complete HTML document as a string, ready to be written to a file
        or served over HTTP.

    Raises:
        RendererError: If the Jinja2 template cannot be found or rendered.
        TypeError: If *config* is not a :class:`~chart_genie.chart_config.ChartConfig`.
    """
    if not isinstance(config, ChartConfig):
        raise TypeError(
            f"config must be a ChartConfig instance, got {type(config).__name__}."
        )

    env = _build_jinja_env()
    try:
        template = env.get_template(_DEFAULT_TEMPLATE)
    except TemplateNotFound as exc:
        raise RendererError(
            f"Chart template '{_DEFAULT_TEMPLATE}' not found in '{_TEMPLATES_DIR}'. "
            "Ensure the chart_genie package is installed correctly."
        ) from exc

    context = _build_render_context(config, chartjs_cdn_url)
    if extra_context:
        context.update(extra_context)

    try:
        html = template.render(**context)
    except Exception as exc:
        raise RendererError(
            f"Jinja2 template rendering failed: {exc}"
        ) from exc

    logger.debug(
        "Rendered chart HTML: type=%s, title=%r, size=%d bytes",
        config.chart_type,
        config.title,
        len(html),
    )
    return html


def save_chart(
    config: ChartConfig,
    output_path: str | Path = "chart.html",
    chartjs_cdn_url: str = CHARTJS_CDN_URL,
    extra_context: dict[str, Any] | None = None,
    encoding: str = "utf-8",
) -> Path:
    """Render a ChartConfig and save the HTML to a file.

    Renders the chart using :func:`render_chart` and writes the result to the
    specified output path. Parent directories are created automatically if
    they do not exist.

    Args:
        config: A validated :class:`~chart_genie.chart_config.ChartConfig`.
        output_path: Destination file path for the HTML output. Defaults to
            ``"chart.html"`` in the current working directory.
        chartjs_cdn_url: URL for the Chart.js CDN script tag.
        extra_context: Optional additional Jinja2 template variables.
        encoding: File encoding for the written HTML. Defaults to ``"utf-8"``.

    Returns:
        The resolved absolute :class:`pathlib.Path` of the written file.

    Raises:
        RendererError: If rendering fails or the file cannot be written.
        TypeError: If *config* is not a :class:`~chart_genie.chart_config.ChartConfig`.
    """
    html = render_chart(
        config,
        chartjs_cdn_url=chartjs_cdn_url,
        extra_context=extra_context,
    )

    resolved = Path(output_path).resolve()
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(html, encoding=encoding)
    except OSError as exc:
        raise RendererError(
            f"Cannot write chart HTML to '{resolved}': {exc}"
        ) from exc

    logger.info("Chart saved to '%s' (%d bytes)", resolved, len(html))
    return resolved


def get_template_path() -> Path:
    """Return the absolute path to the bundled Jinja2 chart template.

    Returns:
        A :class:`pathlib.Path` pointing to ``chart.html.j2``.
    """
    return _TEMPLATES_DIR / _DEFAULT_TEMPLATE


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_jinja_env() -> Environment:
    """Create and return a Jinja2 Environment configured for HTML rendering.

    The environment uses :class:`jinja2.FileSystemLoader` pointed at the
    bundled templates directory and enables autoescaping for HTML/XML files.

    Returns:
        A configured :class:`jinja2.Environment` instance.
    """
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _build_render_context(
    config: ChartConfig,
    chartjs_cdn_url: str,
) -> dict[str, Any]:
    """Build the Jinja2 template context dictionary from a ChartConfig.

    Serialises the Chart.js configuration to JSON (safe for embedding in a
    ``<script>`` tag) and computes canvas size hints.

    Args:
        config: The chart configuration to serialise.
        chartjs_cdn_url: URL for the Chart.js CDN script.

    Returns:
        A dict of template variables ready for Jinja2 rendering.
    """
    chartjs_config = config.to_chartjs_config()

    # Serialise to JSON with indentation for readability in generated source.
    # Use a custom encoder to handle any non-standard types gracefully.
    chartjs_config_json = json.dumps(
        chartjs_config,
        indent=2,
        ensure_ascii=False,
        cls=_SafeJsonEncoder,
    )

    # Determine canvas dimensions.
    canvas_width: int | None
    canvas_height: int | None
    if config.responsive:
        # Responsive canvas — let CSS / Chart.js control sizing.
        canvas_width = None
        canvas_height = None
    else:
        canvas_width = config.width if config.width is not None else _DEFAULT_CANVAS_WIDTH
        canvas_height = config.height if config.height is not None else _DEFAULT_CANVAS_HEIGHT

    return {
        # Core chart data
        "chart_title": config.title,
        "chart_type": config.chart_type,
        "chartjs_config_json": chartjs_config_json,
        # CDN
        "chartjs_cdn_url": chartjs_cdn_url,
        # Canvas sizing
        "responsive": config.responsive,
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        # Feature flags (for potential conditional template logic)
        "show_legend": config.show_legend,
        "show_tooltips": config.show_tooltips,
        # Package metadata
        "generator": "chart_genie",
    }


class _SafeJsonEncoder(json.JSONEncoder):
    """A JSON encoder that converts non-serialisable types to safe equivalents.

    Handles:
    * ``None`` → ``null`` (default JSON encoder behaviour)
    * Any other non-serialisable object → its ``str()`` representation
    """

    def default(self, obj: Any) -> Any:
        """Return a JSON-serialisable version of *obj*.

        Args:
            obj: The object that could not be serialised by the default encoder.

        Returns:
            A JSON-serialisable representation.
        """
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)
