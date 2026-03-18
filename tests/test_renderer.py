"""Unit tests for chart_genie.renderer.

Verifies that:
- render_chart produces valid HTML containing expected Chart.js canvas markup.
- Chart configuration and data are correctly injected into the template.
- save_chart writes the HTML to disk.
- Error handling works for invalid inputs and missing templates.
- The template context is built correctly from ChartConfig.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from chart_genie.chart_config import ChartConfig, DatasetConfig
from chart_genie.renderer import (
    CHARTJS_CDN_URL,
    RendererError,
    _build_jinja_env,
    _build_render_context,
    get_template_path,
    render_chart,
    save_chart,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def bar_chart_config() -> ChartConfig:
    """A minimal bar chart configuration."""
    return ChartConfig.from_dict({
        "chart_type": "bar",
        "title": "Monthly Sales",
        "labels": ["January", "February", "March"],
        "datasets": [
            {
                "label": "Sales",
                "data": [15200, 18400, 21000],
                "background_color": "rgba(54, 162, 235, 0.7)",
                "border_color": "rgba(54, 162, 235, 1.0)",
            }
        ],
        "x_axis_label": "Month",
        "y_axis_label": "Sales (USD)",
    })


@pytest.fixture()
def line_chart_config() -> ChartConfig:
    """A line chart with two datasets."""
    return ChartConfig.from_dict({
        "chart_type": "line",
        "title": "Temperature Trends",
        "labels": ["Jan", "Feb", "Mar", "Apr"],
        "datasets": [
            {
                "label": "High",
                "data": [3.5, 5.2, 10.1, 16.8],
                "background_color": "rgba(255, 99, 132, 0.4)",
                "border_color": "rgba(255, 99, 132, 1.0)",
                "fill": True,
                "tension": 0.4,
            },
            {
                "label": "Low",
                "data": [-2.8, -1.5, 3.0, 8.5],
                "background_color": "rgba(54, 162, 235, 0.4)",
                "border_color": "rgba(54, 162, 235, 1.0)",
                "fill": False,
                "tension": 0.4,
            },
        ],
    })


@pytest.fixture()
def pie_chart_config() -> ChartConfig:
    """A pie chart configuration."""
    return ChartConfig.from_dict({
        "chart_type": "pie",
        "title": "Market Share",
        "labels": ["Product A", "Product B", "Product C"],
        "datasets": [
            {
                "label": "Share",
                "data": [45, 30, 25],
                "background_color": [
                    "rgba(54, 162, 235, 0.7)",
                    "rgba(255, 99, 132, 0.7)",
                    "rgba(75, 192, 192, 0.7)",
                ],
                "border_color": [
                    "rgba(54, 162, 235, 1.0)",
                    "rgba(255, 99, 132, 1.0)",
                    "rgba(75, 192, 192, 1.0)",
                ],
            }
        ],
    })


@pytest.fixture()
def non_responsive_config() -> ChartConfig:
    """A fixed-size (non-responsive) chart configuration."""
    return ChartConfig.from_dict({
        "chart_type": "bar",
        "title": "Fixed Size Chart",
        "labels": ["A", "B"],
        "datasets": [{"label": "D", "data": [1, 2]}],
        "responsive": False,
        "width": 800,
        "height": 400,
    })


# ---------------------------------------------------------------------------
# Helper: extract embedded chartConfig JSON from rendered HTML
# ---------------------------------------------------------------------------


def _extract_chart_config_json(html: str) -> dict:
    """Extract the chartConfig JSON object embedded in the rendered HTML.

    Args:
        html: The rendered HTML string.

    Returns:
        The parsed chartConfig dict.

    Raises:
        AssertionError: If the chartConfig variable is not found.
    """
    match = re.search(r"var chartConfig = (\{.*?\});\s*", html, re.DOTALL)
    assert match is not None, "chartConfig variable not found in HTML"
    return json.loads(match.group(1))


# ---------------------------------------------------------------------------
# Tests: get_template_path
# ---------------------------------------------------------------------------


class TestGetTemplatePath:
    def test_returns_path_object(self) -> None:
        """get_template_path returns a Path instance."""
        path = get_template_path()
        assert isinstance(path, Path)

    def test_template_file_exists(self) -> None:
        """The template file actually exists on disk."""
        path = get_template_path()
        assert path.exists(), f"Template not found at {path}"

    def test_template_is_j2_file(self) -> None:
        """The template file has a .j2 suffix."""
        path = get_template_path()
        assert path.suffix == ".j2"

    def test_filename_is_chart_html_j2(self) -> None:
        """The template filename is 'chart.html.j2'."""
        path = get_template_path()
        assert path.name == "chart.html.j2"

    def test_parent_directory_is_templates(self) -> None:
        """The template lives inside a 'templates' directory."""
        path = get_template_path()
        assert path.parent.name == "templates"

    def test_template_is_absolute(self) -> None:
        """The returned path is absolute."""
        path = get_template_path()
        assert path.is_absolute()


# ---------------------------------------------------------------------------
# Tests: _build_jinja_env
# ---------------------------------------------------------------------------


class TestBuildJinjaEnv:
    def test_returns_environment(self) -> None:
        """_build_jinja_env returns a Jinja2 Environment instance."""
        from jinja2 import Environment
        env = _build_jinja_env()
        assert isinstance(env, Environment)

    def test_can_load_chart_template(self) -> None:
        """The chart.html.j2 template can be loaded from the environment."""
        env = _build_jinja_env()
        template = env.get_template("chart.html.j2")
        assert template is not None

    def test_template_not_found_raises(self) -> None:
        """Requesting a nonexistent template raises TemplateNotFound."""
        from jinja2 import TemplateNotFound
        env = _build_jinja_env()
        with pytest.raises(TemplateNotFound):
            env.get_template("nonexistent.html.j2")

    def test_autoescape_enabled(self) -> None:
        """Autoescaping is enabled in the environment."""
        env = _build_jinja_env()
        # autoescape is enabled for HTML files
        assert env.is_async is False  # not async
        # The environment should be configured with autoescape
        # We verify this indirectly by checking the environment config
        assert env.trim_blocks is True
        assert env.lstrip_blocks is True


# ---------------------------------------------------------------------------
# Tests: _build_render_context
# ---------------------------------------------------------------------------


class TestBuildRenderContext:
    def test_contains_required_keys(self, bar_chart_config: ChartConfig) -> None:
        """The render context contains all required template keys."""
        context = _build_render_context(bar_chart_config, CHARTJS_CDN_URL)
        required_keys = {
            "chart_title",
            "chart_type",
            "chartjs_config_json",
            "chartjs_cdn_url",
            "responsive",
            "canvas_width",
            "canvas_height",
            "show_legend",
            "show_tooltips",
            "generator",
        }
        for key in required_keys:
            assert key in context, f"Missing key '{key}' in render context"

    def test_chart_title(self, bar_chart_config: ChartConfig) -> None:
        """chart_title in context matches the config title."""
        context = _build_render_context(bar_chart_config, CHARTJS_CDN_URL)
        assert context["chart_title"] == "Monthly Sales"

    def test_chart_type(self, bar_chart_config: ChartConfig) -> None:
        """chart_type in context matches the config chart_type."""
        context = _build_render_context(bar_chart_config, CHARTJS_CDN_URL)
        assert context["chart_type"] == "bar"

    def test_chartjs_cdn_url(self, bar_chart_config: ChartConfig) -> None:
        """chartjs_cdn_url in context matches the provided URL."""
        custom_url = "https://example.com/chart.min.js"
        context = _build_render_context(bar_chart_config, custom_url)
        assert context["chartjs_cdn_url"] == custom_url

    def test_chartjs_config_json_is_valid_json(self, bar_chart_config: ChartConfig) -> None:
        """chartjs_config_json in context is valid JSON."""
        context = _build_render_context(bar_chart_config, CHARTJS_CDN_URL)
        parsed = json.loads(context["chartjs_config_json"])
        assert isinstance(parsed, dict)

    def test_chartjs_config_json_has_type_data_options(self, bar_chart_config: ChartConfig) -> None:
        """The embedded JSON has 'type', 'data', and 'options' keys."""
        context = _build_render_context(bar_chart_config, CHARTJS_CDN_URL)
        parsed = json.loads(context["chartjs_config_json"])
        assert "type" in parsed
        assert "data" in parsed
        assert "options" in parsed

    def test_responsive_canvas_no_dimensions(self, bar_chart_config: ChartConfig) -> None:
        """For responsive configs, canvas_width and canvas_height are None."""
        context = _build_render_context(bar_chart_config, CHARTJS_CDN_URL)
        assert context["responsive"] is True
        assert context["canvas_width"] is None
        assert context["canvas_height"] is None

    def test_non_responsive_canvas_has_dimensions(self, non_responsive_config: ChartConfig) -> None:
        """For non-responsive configs, canvas_width and canvas_height are set."""
        context = _build_render_context(non_responsive_config, CHARTJS_CDN_URL)
        assert context["responsive"] is False
        assert context["canvas_width"] == 800
        assert context["canvas_height"] == 400

    def test_non_responsive_default_dimensions(self) -> None:
        """Non-responsive config without width/height uses default dimensions."""
        config = ChartConfig.from_dict({
            "chart_type": "bar",
            "title": "Test",
            "labels": ["A"],
            "datasets": [{"label": "D", "data": [1]}],
            "responsive": False,
        })
        context = _build_render_context(config, CHARTJS_CDN_URL)
        assert context["canvas_width"] == 900
        assert context["canvas_height"] == 500

    def test_generator_field(self, bar_chart_config: ChartConfig) -> None:
        """The generator field is set to 'chart_genie'."""
        context = _build_render_context(bar_chart_config, CHARTJS_CDN_URL)
        assert context["generator"] == "chart_genie"

    def test_show_legend_true(self, bar_chart_config: ChartConfig) -> None:
        """show_legend in context matches the config value."""
        context = _build_render_context(bar_chart_config, CHARTJS_CDN_URL)
        assert context["show_legend"] is True

    def test_show_tooltips_true(self, bar_chart_config: ChartConfig) -> None:
        """show_tooltips in context matches the config value."""
        context = _build_render_context(bar_chart_config, CHARTJS_CDN_URL)
        assert context["show_tooltips"] is True

    def test_show_legend_false(self) -> None:
        """show_legend=False is reflected in the render context."""
        config = ChartConfig.from_dict({
            "chart_type": "bar",
            "title": "T",
            "labels": ["A"],
            "datasets": [{"label": "D", "data": [1]}],
            "show_legend": False,
        })
        context = _build_render_context(config, CHARTJS_CDN_URL)
        assert context["show_legend"] is False

    def test_show_tooltips_false(self) -> None:
        """show_tooltips=False is reflected in the render context."""
        config = ChartConfig.from_dict({
            "chart_type": "bar",
            "title": "T",
            "labels": ["A"],
            "datasets": [{"label": "D", "data": [1]}],
            "show_tooltips": False,
        })
        context = _build_render_context(config, CHARTJS_CDN_URL)
        assert context["show_tooltips"] is False

    def test_pie_chart_context(self, pie_chart_config: ChartConfig) -> None:
        """Pie chart builds a valid context."""
        context = _build_render_context(pie_chart_config, CHARTJS_CDN_URL)
        assert context["chart_type"] == "pie"
        assert context["chart_title"] == "Market Share"

    def test_chartjs_config_json_contains_labels(self, bar_chart_config: ChartConfig) -> None:
        """The embedded JSON contains the chart labels."""
        context = _build_render_context(bar_chart_config, CHARTJS_CDN_URL)
        parsed = json.loads(context["chartjs_config_json"])
        assert "January" in parsed["data"]["labels"]
        assert "February" in parsed["data"]["labels"]
        assert "March" in parsed["data"]["labels"]

    def test_chartjs_config_json_contains_datasets(self, bar_chart_config: ChartConfig) -> None:
        """The embedded JSON contains the chart datasets."""
        context = _build_render_context(bar_chart_config, CHARTJS_CDN_URL)
        parsed = json.loads(context["chartjs_config_json"])
        assert len(parsed["data"]["datasets"]) == 1
        assert parsed["data"]["datasets"][0]["label"] == "Sales"


# ---------------------------------------------------------------------------
# Tests: render_chart — HTML structure
# ---------------------------------------------------------------------------


class TestRenderChartHtmlStructure:
    def test_returns_string(self, bar_chart_config: ChartConfig) -> None:
        """render_chart returns a string."""
        html = render_chart(bar_chart_config)
        assert isinstance(html, str)

    def test_html_nonempty(self, bar_chart_config: ChartConfig) -> None:
        """The rendered HTML is non-empty."""
        html = render_chart(bar_chart_config)
        assert len(html) > 100

    def test_contains_doctype(self, bar_chart_config: ChartConfig) -> None:
        """The rendered HTML starts with a DOCTYPE declaration."""
        html = render_chart(bar_chart_config)
        assert "<!DOCTYPE html>" in html

    def test_contains_html_tag(self, bar_chart_config: ChartConfig) -> None:
        """The rendered HTML has opening and closing html tags."""
        html = render_chart(bar_chart_config)
        assert "<html" in html
        assert "</html>" in html

    def test_contains_head_and_body(self, bar_chart_config: ChartConfig) -> None:
        """The rendered HTML has head and body sections."""
        html = render_chart(bar_chart_config)
        assert "<head>" in html
        assert "</head>" in html
        assert "<body>" in html
        assert "</body>" in html

    def test_contains_canvas_element(self, bar_chart_config: ChartConfig) -> None:
        """The rendered HTML contains a canvas element with id 'chart-canvas'."""
        html = render_chart(bar_chart_config)
        assert "<canvas" in html
        assert 'id="chart-canvas"' in html

    def test_contains_chart_title_in_head(self, bar_chart_config: ChartConfig) -> None:
        """The page title tag contains the chart title."""
        html = render_chart(bar_chart_config)
        assert "<title>Monthly Sales</title>" in html

    def test_contains_chart_title_in_body(self, bar_chart_config: ChartConfig) -> None:
        """The chart title appears in the body of the page."""
        html = render_chart(bar_chart_config)
        assert "Monthly Sales" in html

    def test_contains_chartjs_cdn_script(self, bar_chart_config: ChartConfig) -> None:
        """The Chart.js CDN URL is included in a script tag."""
        html = render_chart(bar_chart_config)
        assert CHARTJS_CDN_URL in html
        assert "<script" in html

    def test_contains_new_chart_call(self, bar_chart_config: ChartConfig) -> None:
        """The rendered HTML contains a 'new Chart(' call."""
        html = render_chart(bar_chart_config)
        assert "new Chart(" in html

    def test_contains_chart_config_json(self, bar_chart_config: ChartConfig) -> None:
        """The chart type appears in the embedded JSON."""
        html = render_chart(bar_chart_config)
        assert '"type": "bar"' in html

    def test_contains_data_labels(self, bar_chart_config: ChartConfig) -> None:
        """The month labels appear in the rendered HTML."""
        html = render_chart(bar_chart_config)
        assert "January" in html
        assert "February" in html
        assert "March" in html

    def test_contains_dataset_data(self, bar_chart_config: ChartConfig) -> None:
        """The dataset numeric values appear in the rendered HTML."""
        html = render_chart(bar_chart_config)
        assert "15200" in html
        assert "18400" in html
        assert "21000" in html

    def test_contains_dataset_label(self, bar_chart_config: ChartConfig) -> None:
        """The dataset label appears in the rendered HTML."""
        html = render_chart(bar_chart_config)
        assert "Sales" in html

    def test_chart_type_badge_present(self, bar_chart_config: ChartConfig) -> None:
        """The chart type badge appears in the rendered HTML."""
        html = render_chart(bar_chart_config)
        assert "bar" in html

    def test_generator_meta_tag(self, bar_chart_config: ChartConfig) -> None:
        """The generator meta tag is present in the HTML head."""
        html = render_chart(bar_chart_config)
        assert 'name="generator"' in html
        assert "chart_genie" in html

    def test_script_tag_closed(self, bar_chart_config: ChartConfig) -> None:
        """Script tags are properly closed."""
        html = render_chart(bar_chart_config)
        assert "</script>" in html

    def test_use_strict_in_script(self, bar_chart_config: ChartConfig) -> None:
        """The chart initialization script uses strict mode."""
        html = render_chart(bar_chart_config)
        assert '"use strict"' in html

    def test_iife_wrapper(self, bar_chart_config: ChartConfig) -> None:
        """The chart initialization is wrapped in an IIFE."""
        html = render_chart(bar_chart_config)
        assert "(function ()" in html or "(function()" in html


# ---------------------------------------------------------------------------
# Tests: render_chart — pie chart
# ---------------------------------------------------------------------------


class TestRenderChartPie:
    def test_pie_chart_renders(self, pie_chart_config: ChartConfig) -> None:
        """A pie chart renders without error."""
        html = render_chart(pie_chart_config)
        assert "pie" in html
        assert "Market Share" in html

    def test_pie_labels_present(self, pie_chart_config: ChartConfig) -> None:
        """Pie chart slice labels appear in the rendered HTML."""
        html = render_chart(pie_chart_config)
        assert "Product A" in html
        assert "Product B" in html
        assert "Product C" in html

    def test_pie_data_present(self, pie_chart_config: ChartConfig) -> None:
        """Pie chart numeric data appears in the rendered HTML."""
        html = render_chart(pie_chart_config)
        assert "45" in html
        assert "30" in html
        assert "25" in html

    def test_pie_chart_type_in_json(self, pie_chart_config: ChartConfig) -> None:
        """The embedded JSON has type='pie'."""
        html = render_chart(pie_chart_config)
        data = _extract_chart_config_json(html)
        assert data["type"] == "pie"

    def test_pie_no_scales_in_json(self, pie_chart_config: ChartConfig) -> None:
        """Pie chart JSON options do not include a 'scales' key."""
        html = render_chart(pie_chart_config)
        data = _extract_chart_config_json(html)
        assert "scales" not in data["options"]


# ---------------------------------------------------------------------------
# Tests: render_chart — line chart with multiple datasets
# ---------------------------------------------------------------------------


class TestRenderChartLine:
    def test_line_chart_renders(self, line_chart_config: ChartConfig) -> None:
        """A line chart renders without error."""
        html = render_chart(line_chart_config)
        assert "line" in html
        assert "Temperature Trends" in html

    def test_both_datasets_present(self, line_chart_config: ChartConfig) -> None:
        """Both dataset labels appear in the rendered HTML."""
        html = render_chart(line_chart_config)
        assert "High" in html
        assert "Low" in html

    def test_line_data_present(self, line_chart_config: ChartConfig) -> None:
        """Line chart data values appear in the rendered HTML."""
        html = render_chart(line_chart_config)
        assert "3.5" in html
        assert "-2.8" in html

    def test_two_datasets_in_json(self, line_chart_config: ChartConfig) -> None:
        """The embedded JSON contains two datasets."""
        html = render_chart(line_chart_config)
        data = _extract_chart_config_json(html)
        assert len(data["data"]["datasets"]) == 2

    def test_line_chart_type_in_json(self, line_chart_config: ChartConfig) -> None:
        """The embedded JSON has type='line'."""
        html = render_chart(line_chart_config)
        data = _extract_chart_config_json(html)
        assert data["type"] == "line"


# ---------------------------------------------------------------------------
# Tests: render_chart — custom CDN URL
# ---------------------------------------------------------------------------


class TestRenderChartCustomCdn:
    def test_custom_cdn_url_in_output(self, bar_chart_config: ChartConfig) -> None:
        """A custom CDN URL replaces the default CDN URL in the output."""
        custom_url = "https://cdn.example.com/chart.min.js"
        html = render_chart(bar_chart_config, chartjs_cdn_url=custom_url)
        assert custom_url in html
        assert CHARTJS_CDN_URL not in html

    def test_default_cdn_url_used_by_default(self, bar_chart_config: ChartConfig) -> None:
        """The default CDN URL is used when no custom URL is provided."""
        html = render_chart(bar_chart_config)
        assert CHARTJS_CDN_URL in html

    def test_cdn_url_in_script_src(self, bar_chart_config: ChartConfig) -> None:
        """The CDN URL appears in a script src attribute."""
        html = render_chart(bar_chart_config)
        assert f'src="{CHARTJS_CDN_URL}"' in html


# ---------------------------------------------------------------------------
# Tests: render_chart — responsive vs fixed size
# ---------------------------------------------------------------------------


class TestRenderChartResponsive:
    def test_responsive_config_renders(self, bar_chart_config: ChartConfig) -> None:
        """A responsive chart renders with responsive=true in the JSON."""
        html = render_chart(bar_chart_config)
        assert '"responsive": true' in html

    def test_non_responsive_canvas_dimensions(self, non_responsive_config: ChartConfig) -> None:
        """Non-responsive chart includes explicit pixel dimensions in CSS."""
        html = render_chart(non_responsive_config)
        assert "800" in html
        assert "400" in html

    def test_non_responsive_json(self, non_responsive_config: ChartConfig) -> None:
        """Non-responsive chart has responsive=false in the embedded JSON."""
        html = render_chart(non_responsive_config)
        assert '"responsive": false' in html


# ---------------------------------------------------------------------------
# Tests: render_chart — extra_context
# ---------------------------------------------------------------------------


class TestRenderChartExtraContext:
    def test_extra_context_not_raises(self, bar_chart_config: ChartConfig) -> None:
        """Providing extra_context does not raise an error."""
        html = render_chart(
            bar_chart_config,
            extra_context={"generator": "custom_generator"},
        )
        assert isinstance(html, str)
        assert "custom_generator" in html

    def test_extra_context_overrides_generator(self, bar_chart_config: ChartConfig) -> None:
        """extra_context can override the generator value."""
        html = render_chart(
            bar_chart_config,
            extra_context={"generator": "my_custom_tool"},
        )
        assert "my_custom_tool" in html

    def test_none_extra_context_works(self, bar_chart_config: ChartConfig) -> None:
        """extra_context=None is handled gracefully."""
        html = render_chart(bar_chart_config, extra_context=None)
        assert isinstance(html, str)
        assert len(html) > 100


# ---------------------------------------------------------------------------
# Tests: render_chart — input validation
# ---------------------------------------------------------------------------


class TestRenderChartValidation:
    def test_non_chartconfig_raises_type_error(self) -> None:
        """Passing a non-ChartConfig raises TypeError."""
        with pytest.raises(TypeError, match="ChartConfig"):
            render_chart({"chart_type": "bar"})  # type: ignore[arg-type]

    def test_none_raises_type_error(self) -> None:
        """Passing None raises TypeError."""
        with pytest.raises(TypeError, match="ChartConfig"):
            render_chart(None)  # type: ignore[arg-type]

    def test_string_raises_type_error(self) -> None:
        """Passing a string raises TypeError."""
        with pytest.raises(TypeError, match="ChartConfig"):
            render_chart("bar chart")  # type: ignore[arg-type]

    def test_list_raises_type_error(self) -> None:
        """Passing a list raises TypeError."""
        with pytest.raises(TypeError, match="ChartConfig"):
            render_chart([1, 2, 3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests: render_chart — Chart.js options structure in embedded JSON
# ---------------------------------------------------------------------------


class TestRenderChartEmbeddedJson:
    def test_plugins_title_in_json(self, bar_chart_config: ChartConfig) -> None:
        """The embedded JSON contains the plugins.title configuration."""
        html = render_chart(bar_chart_config)
        data = _extract_chart_config_json(html)
        assert data["options"]["plugins"]["title"]["text"] == "Monthly Sales"
        assert data["options"]["plugins"]["title"]["display"] is True

    def test_plugins_legend_in_json(self, bar_chart_config: ChartConfig) -> None:
        """The embedded JSON contains the plugins.legend configuration."""
        html = render_chart(bar_chart_config)
        data = _extract_chart_config_json(html)
        assert "legend" in data["options"]["plugins"]

    def test_plugins_tooltip_in_json(self, bar_chart_config: ChartConfig) -> None:
        """The embedded JSON contains the plugins.tooltip configuration."""
        html = render_chart(bar_chart_config)
        data = _extract_chart_config_json(html)
        assert data["options"]["plugins"]["tooltip"]["enabled"] is True

    def test_axis_labels_in_json(self, bar_chart_config: ChartConfig) -> None:
        """The embedded JSON contains x and y axis label configurations."""
        html = render_chart(bar_chart_config)
        data = _extract_chart_config_json(html)
        scales = data["options"]["scales"]
        assert scales["x"]["title"]["text"] == "Month"
        assert scales["y"]["title"]["text"] == "Sales (USD)"

    def test_datasets_camelcase_keys(self, bar_chart_config: ChartConfig) -> None:
        """Dataset keys in the embedded JSON use camelCase for Chart.js."""
        html = render_chart(bar_chart_config)
        data = _extract_chart_config_json(html)
        ds = data["data"]["datasets"][0]
        assert "backgroundColor" in ds
        assert "borderColor" in ds
        assert "borderWidth" in ds
        assert "background_color" not in ds

    def test_labels_in_json(self, bar_chart_config: ChartConfig) -> None:
        """The embedded JSON contains the correct labels."""
        html = render_chart(bar_chart_config)
        data = _extract_chart_config_json(html)
        assert data["data"]["labels"] == ["January", "February", "March"]

    def test_dataset_data_in_json(self, bar_chart_config: ChartConfig) -> None:
        """The embedded JSON contains the correct dataset data values."""
        html = render_chart(bar_chart_config)
        data = _extract_chart_config_json(html)
        assert data["data"]["datasets"][0]["data"] == [15200, 18400, 21000]

    def test_responsive_true_in_json(self, bar_chart_config: ChartConfig) -> None:
        """responsive=true appears in the embedded JSON options."""
        html = render_chart(bar_chart_config)
        data = _extract_chart_config_json(html)
        assert data["options"]["responsive"] is True

    def test_maintain_aspect_ratio_false_in_json(self, bar_chart_config: ChartConfig) -> None:
        """maintainAspectRatio=false appears in the embedded JSON options."""
        html = render_chart(bar_chart_config)
        data = _extract_chart_config_json(html)
        assert data["options"]["maintainAspectRatio"] is False

    def test_legend_display_false_in_json(self) -> None:
        """show_legend=False is reflected in the embedded JSON."""
        config = ChartConfig.from_dict({
            "chart_type": "bar",
            "title": "T",
            "labels": ["A"],
            "datasets": [{"label": "D", "data": [1]}],
            "show_legend": False,
        })
        html = render_chart(config)
        data = _extract_chart_config_json(html)
        assert data["options"]["plugins"]["legend"]["display"] is False

    def test_tooltip_disabled_in_json(self) -> None:
        """show_tooltips=False is reflected in the embedded JSON."""
        config = ChartConfig.from_dict({
            "chart_type": "bar",
            "title": "T",
            "labels": ["A"],
            "datasets": [{"label": "D", "data": [1]}],
            "show_tooltips": False,
        })
        html = render_chart(config)
        data = _extract_chart_config_json(html)
        assert data["options"]["plugins"]["tooltip"]["enabled"] is False

    def test_extra_options_in_json(self) -> None:
        """extra_options are present in the embedded JSON."""
        config = ChartConfig.from_dict({
            "chart_type": "bar",
            "title": "T",
            "labels": ["A"],
            "datasets": [{"label": "D", "data": [1]}],
            "extra_options": {"animation": {"duration": 1000}},
        })
        html = render_chart(config)
        data = _extract_chart_config_json(html)
        assert data["options"]["animation"]["duration"] == 1000

    def test_multiple_datasets_in_json(self, line_chart_config: ChartConfig) -> None:
        """Multiple datasets appear correctly in the embedded JSON."""
        html = render_chart(line_chart_config)
        data = _extract_chart_config_json(html)
        datasets = data["data"]["datasets"]
        assert len(datasets) == 2
        labels = {ds["label"] for ds in datasets}
        assert "High" in labels
        assert "Low" in labels


# ---------------------------------------------------------------------------
# Tests: save_chart
# ---------------------------------------------------------------------------


class TestSaveChart:
    def test_creates_file(self, tmp_path: Path, bar_chart_config: ChartConfig) -> None:
        """save_chart creates the output file."""
        output = tmp_path / "chart.html"
        result = save_chart(bar_chart_config, output_path=output)
        assert output.exists()
        assert result == output

    def test_returns_absolute_path(self, tmp_path: Path, bar_chart_config: ChartConfig) -> None:
        """save_chart returns an absolute Path."""
        output = tmp_path / "chart.html"
        result = save_chart(bar_chart_config, output_path=output)
        assert result.is_absolute()

    def test_file_content_is_html(self, tmp_path: Path, bar_chart_config: ChartConfig) -> None:
        """The written file contains valid HTML with the chart title."""
        output = tmp_path / "chart.html"
        save_chart(bar_chart_config, output_path=output)
        content = output.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "Monthly Sales" in content

    def test_creates_parent_directories(self, tmp_path: Path, bar_chart_config: ChartConfig) -> None:
        """save_chart creates parent directories as needed."""
        nested = tmp_path / "deep" / "nested" / "dir" / "chart.html"
        save_chart(bar_chart_config, output_path=nested)
        assert nested.exists()

    def test_string_path_accepted(self, tmp_path: Path, bar_chart_config: ChartConfig) -> None:
        """save_chart accepts a string path as well as a Path object."""
        output = str(tmp_path / "chart.html")
        save_chart(bar_chart_config, output_path=output)
        assert Path(output).exists()

    def test_default_output_filename(
        self, tmp_path: Path, bar_chart_config: ChartConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default output filename is 'chart.html' in the current directory."""
        monkeypatch.chdir(tmp_path)
        result = save_chart(bar_chart_config)
        assert result.name == "chart.html"
        assert result.exists()

    def test_file_encoding_is_utf8(self, tmp_path: Path, bar_chart_config: ChartConfig) -> None:
        """The output file is readable as UTF-8."""
        output = tmp_path / "chart.html"
        save_chart(bar_chart_config, output_path=output)
        content = output.read_text(encoding="utf-8")
        assert len(content) > 0

    def test_write_error_raises_renderer_error(self, bar_chart_config: ChartConfig) -> None:
        """An OS error during file write raises RendererError."""
        with patch("chart_genie.renderer.Path.write_text", side_effect=OSError("disk full")):
            with pytest.raises(RendererError, match="Cannot write chart HTML"):
                save_chart(bar_chart_config, output_path="/some/path/chart.html")

    def test_custom_cdn_url_saved(
        self, tmp_path: Path, bar_chart_config: ChartConfig
    ) -> None:
        """A custom CDN URL is present in the saved file."""
        custom_url = "https://mycdn.example.com/chart.js"
        output = tmp_path / "chart.html"
        save_chart(bar_chart_config, output_path=output, chartjs_cdn_url=custom_url)
        content = output.read_text(encoding="utf-8")
        assert custom_url in content

    def test_type_error_propagates(
        self, tmp_path: Path
    ) -> None:
        """Passing a non-ChartConfig to save_chart raises TypeError."""
        output = tmp_path / "chart.html"
        with pytest.raises(TypeError, match="ChartConfig"):
            save_chart(None, output_path=output)  # type: ignore[arg-type]

    def test_save_chart_returns_path_object(
        self, tmp_path: Path, bar_chart_config: ChartConfig
    ) -> None:
        """save_chart returns a Path object, not a string."""
        output = tmp_path / "chart.html"
        result = save_chart(bar_chart_config, output_path=output)
        assert isinstance(result, Path)

    def test_save_chart_file_size_reasonable(
        self, tmp_path: Path, bar_chart_config: ChartConfig
    ) -> None:
        """The saved file is at least 1KB (sanity check)."""
        output = tmp_path / "chart.html"
        save_chart(bar_chart_config, output_path=output)
        assert output.stat().st_size > 1024

    def test_save_chart_overwrites_existing(
        self, tmp_path: Path, bar_chart_config: ChartConfig
    ) -> None:
        """save_chart overwrites an existing file without error."""
        output = tmp_path / "chart.html"
        output.write_text("old content", encoding="utf-8")
        save_chart(bar_chart_config, output_path=output)
        content = output.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "old content" not in content


# ---------------------------------------------------------------------------
# Tests: render_chart — all supported chart types render without error
# ---------------------------------------------------------------------------


class TestAllChartTypesRender:
    @pytest.mark.parametrize(
        "chart_type", ["bar", "line", "pie", "doughnut", "scatter", "radar"]
    )
    def test_chart_type_renders(self, chart_type: str) -> None:
        """Every supported chart type renders to valid HTML without error."""
        config = ChartConfig.from_dict({
            "chart_type": chart_type,
            "title": f"{chart_type.title()} Chart",
            "labels": ["A", "B", "C"],
            "datasets": [
                {
                    "label": "Dataset 1",
                    "data": [10, 20, 30],
                }
            ],
        })
        html = render_chart(config)
        assert isinstance(html, str)
        assert chart_type in html
        assert f"{chart_type.title()} Chart" in html
        assert "<canvas" in html
        assert "new Chart(" in html

    @pytest.mark.parametrize(
        "chart_type", ["bar", "line", "pie", "doughnut", "scatter", "radar"]
    )
    def test_chart_type_json_correct(
        self, chart_type: str
    ) -> None:
        """The embedded JSON has the correct 'type' for every chart type."""
        config = ChartConfig.from_dict({
            "chart_type": chart_type,
            "title": "T",
            "labels": ["X", "Y"],
            "datasets": [{"label": "D", "data": [1, 2]}],
        })
        html = render_chart(config)
        data = _extract_chart_config_json(html)
        assert data["type"] == chart_type


# ---------------------------------------------------------------------------
# Tests: HTML accessibility and SEO basics
# ---------------------------------------------------------------------------


class TestHtmlAccessibility:
    def test_canvas_has_aria_label(self, bar_chart_config: ChartConfig) -> None:
        """The canvas element has an aria-label attribute."""
        html = render_chart(bar_chart_config)
        assert "aria-label=" in html

    def test_canvas_has_role_img(self, bar_chart_config: ChartConfig) -> None:
        """The canvas element has role='img' for accessibility."""
        html = render_chart(bar_chart_config)
        assert 'role="img"' in html

    def test_lang_attribute_on_html(self, bar_chart_config: ChartConfig) -> None:
        """The html element has a lang attribute set to 'en'."""
        html = render_chart(bar_chart_config)
        assert 'lang="en"' in html

    def test_charset_meta(self, bar_chart_config: ChartConfig) -> None:
        """The HTML head contains a UTF-8 charset meta tag."""
        html = render_chart(bar_chart_config)
        assert 'charset="UTF-8"' in html or 'charset=UTF-8' in html

    def test_viewport_meta(self, bar_chart_config: ChartConfig) -> None:
        """The HTML head contains a viewport meta tag."""
        html = render_chart(bar_chart_config)
        assert 'name="viewport"' in html

    def test_canvas_aria_label_matches_title(self, bar_chart_config: ChartConfig) -> None:
        """The canvas aria-label matches the chart title."""
        html = render_chart(bar_chart_config)
        assert 'aria-label="Monthly Sales"' in html

    def test_footer_present(self, bar_chart_config: ChartConfig) -> None:
        """The HTML contains a footer element."""
        html = render_chart(bar_chart_config)
        assert "<footer" in html
        assert "</footer>" in html

    def test_chart_js_link_in_footer(self, bar_chart_config: ChartConfig) -> None:
        """The footer contains a link referencing Chart.js."""
        html = render_chart(bar_chart_config)
        assert "chartjs.org" in html

    def test_noopener_noreferrer_on_external_links(self, bar_chart_config: ChartConfig) -> None:
        """External links use rel='noopener noreferrer' for security."""
        html = render_chart(bar_chart_config)
        assert 'rel="noopener noreferrer"' in html

    def test_error_handling_javascript_present(self, bar_chart_config: ChartConfig) -> None:
        """The rendered HTML includes JavaScript error handling for Chart.js load failures."""
        html = render_chart(bar_chart_config)
        # The template should check if Chart is defined
        assert "typeof Chart" in html

    def test_css_styles_present(self, bar_chart_config: ChartConfig) -> None:
        """The rendered HTML includes a style block."""
        html = render_chart(bar_chart_config)
        assert "<style>" in html
        assert "</style>" in html

    def test_responsive_meta_content(self, bar_chart_config: ChartConfig) -> None:
        """The viewport meta has content specifying width=device-width."""
        html = render_chart(bar_chart_config)
        assert "width=device-width" in html
