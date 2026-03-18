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
from pathlib import Path
from unittest.mock import patch

import pytest

from chart_genie.chart_config import ChartConfig, DatasetConfig
from chart_genie.renderer import (
    CHARTJS_CDN_URL,
    RendererError,
    _build_render_context,
    _build_jinja_env,
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
# Tests: get_template_path
# ---------------------------------------------------------------------------


class TestGetTemplatePath:
    def test_returns_path_object(self) -> None:
        path = get_template_path()
        assert isinstance(path, Path)

    def test_template_file_exists(self) -> None:
        path = get_template_path()
        assert path.exists(), f"Template not found at {path}"

    def test_template_is_j2_file(self) -> None:
        path = get_template_path()
        assert path.suffix == ".j2"

    def test_filename_is_chart_html_j2(self) -> None:
        path = get_template_path()
        assert path.name == "chart.html.j2"


# ---------------------------------------------------------------------------
# Tests: _build_jinja_env
# ---------------------------------------------------------------------------


class TestBuildJinjaEnv:
    def test_returns_environment(self) -> None:
        from jinja2 import Environment
        env = _build_jinja_env()
        assert isinstance(env, Environment)

    def test_can_load_chart_template(self) -> None:
        env = _build_jinja_env()
        template = env.get_template("chart.html.j2")
        assert template is not None

    def test_template_not_found_raises(self) -> None:
        from jinja2 import TemplateNotFound
        env = _build_jinja_env()
        with pytest.raises(TemplateNotFound):
            env.get_template("nonexistent.html.j2")


# ---------------------------------------------------------------------------
# Tests: _build_render_context
# ---------------------------------------------------------------------------


class TestBuildRenderContext:
    def test_contains_required_keys(self, bar_chart_config: ChartConfig) -> None:
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
        context = _build_render_context(bar_chart_config, CHARTJS_CDN_URL)
        assert context["chart_title"] == "Monthly Sales"

    def test_chart_type(self, bar_chart_config: ChartConfig) -> None:
        context = _build_render_context(bar_chart_config, CHARTJS_CDN_URL)
        assert context["chart_type"] == "bar"

    def test_chartjs_cdn_url(self, bar_chart_config: ChartConfig) -> None:
        custom_url = "https://example.com/chart.min.js"
        context = _build_render_context(bar_chart_config, custom_url)
        assert context["chartjs_cdn_url"] == custom_url

    def test_chartjs_config_json_is_valid_json(self, bar_chart_config: ChartConfig) -> None:
        context = _build_render_context(bar_chart_config, CHARTJS_CDN_URL)
        # Should not raise
        parsed = json.loads(context["chartjs_config_json"])
        assert isinstance(parsed, dict)

    def test_chartjs_config_json_has_type_data_options(self, bar_chart_config: ChartConfig) -> None:
        context = _build_render_context(bar_chart_config, CHARTJS_CDN_URL)
        parsed = json.loads(context["chartjs_config_json"])
        assert "type" in parsed
        assert "data" in parsed
        assert "options" in parsed

    def test_responsive_canvas_no_dimensions(self, bar_chart_config: ChartConfig) -> None:
        # bar_chart_config is responsive by default
        context = _build_render_context(bar_chart_config, CHARTJS_CDN_URL)
        assert context["responsive"] is True
        assert context["canvas_width"] is None
        assert context["canvas_height"] is None

    def test_non_responsive_canvas_has_dimensions(
        self, non_responsive_config: ChartConfig
    ) -> None:
        context = _build_render_context(non_responsive_config, CHARTJS_CDN_URL)
        assert context["responsive"] is False
        assert context["canvas_width"] == 800
        assert context["canvas_height"] == 400

    def test_non_responsive_default_dimensions(self) -> None:
        config = ChartConfig.from_dict({
            "chart_type": "bar",
            "title": "Test",
            "labels": ["A"],
            "datasets": [{"label": "D", "data": [1]}],
            "responsive": False,
            # No width/height specified — should fall back to defaults
        })
        context = _build_render_context(config, CHARTJS_CDN_URL)
        assert context["canvas_width"] == 900
        assert context["canvas_height"] == 500

    def test_generator_field(self, bar_chart_config: ChartConfig) -> None:
        context = _build_render_context(bar_chart_config, CHARTJS_CDN_URL)
        assert context["generator"] == "chart_genie"


# ---------------------------------------------------------------------------
# Tests: render_chart — HTML structure
# ---------------------------------------------------------------------------


class TestRenderChartHtmlStructure:
    def test_returns_string(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        assert isinstance(html, str)

    def test_html_nonempty(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        assert len(html) > 100

    def test_contains_doctype(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        assert "<!DOCTYPE html>" in html

    def test_contains_html_tag(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        assert "<html" in html
        assert "</html>" in html

    def test_contains_head_and_body(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        assert "<head>" in html
        assert "</head>" in html
        assert "<body>" in html
        assert "</body>" in html

    def test_contains_canvas_element(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        assert "<canvas" in html
        assert "id=\"chart-canvas\"" in html

    def test_contains_chart_title_in_head(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        assert "<title>Monthly Sales</title>" in html

    def test_contains_chart_title_in_body(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        assert "Monthly Sales" in html

    def test_contains_chartjs_cdn_script(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        assert CHARTJS_CDN_URL in html
        assert "<script" in html

    def test_contains_new_chart_call(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        assert "new Chart(" in html

    def test_contains_chart_config_json(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        # The chart type should appear in the embedded JSON
        assert '"type": "bar"' in html

    def test_contains_data_labels(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        assert "January" in html
        assert "February" in html
        assert "March" in html

    def test_contains_dataset_data(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        assert "15200" in html
        assert "18400" in html
        assert "21000" in html

    def test_contains_dataset_label(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        assert "Sales" in html

    def test_chart_type_badge_present(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        # The badge spans the chart type text
        assert "bar" in html


# ---------------------------------------------------------------------------
# Tests: render_chart — pie chart
# ---------------------------------------------------------------------------


class TestRenderChartPie:
    def test_pie_chart_renders(self, pie_chart_config: ChartConfig) -> None:
        html = render_chart(pie_chart_config)
        assert "pie" in html
        assert "Market Share" in html

    def test_pie_labels_present(self, pie_chart_config: ChartConfig) -> None:
        html = render_chart(pie_chart_config)
        assert "Product A" in html
        assert "Product B" in html
        assert "Product C" in html

    def test_pie_data_present(self, pie_chart_config: ChartConfig) -> None:
        html = render_chart(pie_chart_config)
        assert "45" in html
        assert "30" in html
        assert "25" in html


# ---------------------------------------------------------------------------
# Tests: render_chart — line chart with multiple datasets
# ---------------------------------------------------------------------------


class TestRenderChartLine:
    def test_line_chart_renders(self, line_chart_config: ChartConfig) -> None:
        html = render_chart(line_chart_config)
        assert "line" in html
        assert "Temperature Trends" in html

    def test_both_datasets_present(self, line_chart_config: ChartConfig) -> None:
        html = render_chart(line_chart_config)
        assert "High" in html
        assert "Low" in html

    def test_line_data_present(self, line_chart_config: ChartConfig) -> None:
        html = render_chart(line_chart_config)
        assert "3.5" in html
        assert "-2.8" in html


# ---------------------------------------------------------------------------
# Tests: render_chart — custom CDN URL
# ---------------------------------------------------------------------------


class TestRenderChartCustomCdn:
    def test_custom_cdn_url_in_output(self, bar_chart_config: ChartConfig) -> None:
        custom_url = "https://cdn.example.com/chart.min.js"
        html = render_chart(bar_chart_config, chartjs_cdn_url=custom_url)
        assert custom_url in html
        assert CHARTJS_CDN_URL not in html


# ---------------------------------------------------------------------------
# Tests: render_chart — responsive vs fixed size
# ---------------------------------------------------------------------------


class TestRenderChartResponsive:
    def test_responsive_config_renders(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        # responsive=True by default; config JSON should reflect this
        assert '"responsive": true' in html

    def test_non_responsive_canvas_dimensions(self, non_responsive_config: ChartConfig) -> None:
        html = render_chart(non_responsive_config)
        # Width and height should appear in the CSS
        assert "800" in html
        assert "400" in html


# ---------------------------------------------------------------------------
# Tests: render_chart — extra_context
# ---------------------------------------------------------------------------


class TestRenderChartExtraContext:
    def test_extra_context_not_raises(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(
            bar_chart_config,
            extra_context={"generator": "custom_generator"},
        )
        assert isinstance(html, str)
        assert "custom_generator" in html


# ---------------------------------------------------------------------------
# Tests: render_chart — input validation
# ---------------------------------------------------------------------------


class TestRenderChartValidation:
    def test_non_chartconfig_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="ChartConfig"):
            render_chart({"chart_type": "bar"})  # type: ignore[arg-type]

    def test_none_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="ChartConfig"):
            render_chart(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests: render_chart — Chart.js options structure in embedded JSON
# ---------------------------------------------------------------------------


class TestRenderChartEmbeddedJson:
    def test_plugins_title_in_json(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        # Extract the embedded JSON from the page
        import re
        match = re.search(r"var chartConfig = (\{.*?\});\s*", html, re.DOTALL)
        assert match is not None, "chartConfig variable not found in HTML"
        data = json.loads(match.group(1))
        assert data["options"]["plugins"]["title"]["text"] == "Monthly Sales"
        assert data["options"]["plugins"]["title"]["display"] is True

    def test_plugins_legend_in_json(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        import re
        match = re.search(r"var chartConfig = (\{.*?\});\s*", html, re.DOTALL)
        assert match is not None
        data = json.loads(match.group(1))
        assert "legend" in data["options"]["plugins"]

    def test_plugins_tooltip_in_json(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        import re
        match = re.search(r"var chartConfig = (\{.*?\});\s*", html, re.DOTALL)
        assert match is not None
        data = json.loads(match.group(1))
        assert data["options"]["plugins"]["tooltip"]["enabled"] is True

    def test_axis_labels_in_json(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        import re
        match = re.search(r"var chartConfig = (\{.*?\});\s*", html, re.DOTALL)
        assert match is not None
        data = json.loads(match.group(1))
        scales = data["options"]["scales"]
        assert scales["x"]["title"]["text"] == "Month"
        assert scales["y"]["title"]["text"] == "Sales (USD)"

    def test_datasets_camelcase_keys(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        import re
        match = re.search(r"var chartConfig = (\{.*?\});\s*", html, re.DOTALL)
        assert match is not None
        data = json.loads(match.group(1))
        ds = data["data"]["datasets"][0]
        assert "backgroundColor" in ds
        assert "borderColor" in ds
        assert "borderWidth" in ds
        assert "background_color" not in ds


# ---------------------------------------------------------------------------
# Tests: save_chart
# ---------------------------------------------------------------------------


class TestSaveChart:
    def test_creates_file(self, tmp_path: Path, bar_chart_config: ChartConfig) -> None:
        output = tmp_path / "chart.html"
        result = save_chart(bar_chart_config, output_path=output)
        assert output.exists()
        assert result == output

    def test_returns_absolute_path(self, tmp_path: Path, bar_chart_config: ChartConfig) -> None:
        output = tmp_path / "chart.html"
        result = save_chart(bar_chart_config, output_path=output)
        assert result.is_absolute()

    def test_file_content_is_html(self, tmp_path: Path, bar_chart_config: ChartConfig) -> None:
        output = tmp_path / "chart.html"
        save_chart(bar_chart_config, output_path=output)
        content = output.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "Monthly Sales" in content

    def test_creates_parent_directories(self, tmp_path: Path, bar_chart_config: ChartConfig) -> None:
        nested = tmp_path / "deep" / "nested" / "dir" / "chart.html"
        save_chart(bar_chart_config, output_path=nested)
        assert nested.exists()

    def test_string_path_accepted(self, tmp_path: Path, bar_chart_config: ChartConfig) -> None:
        output = str(tmp_path / "chart.html")
        result = save_chart(bar_chart_config, output_path=output)
        assert Path(output).exists()

    def test_default_output_filename(self, tmp_path: Path, bar_chart_config: ChartConfig, monkeypatch: pytest.MonkeyPatch) -> None:
        # Change cwd so the default 'chart.html' lands in tmp_path
        monkeypatch.chdir(tmp_path)
        result = save_chart(bar_chart_config)
        assert result.name == "chart.html"
        assert result.exists()

    def test_file_encoding_is_utf8(self, tmp_path: Path, bar_chart_config: ChartConfig) -> None:
        output = tmp_path / "chart.html"
        save_chart(bar_chart_config, output_path=output)
        # Should be readable as UTF-8
        content = output.read_text(encoding="utf-8")
        assert len(content) > 0

    def test_write_error_raises_renderer_error(self, bar_chart_config: ChartConfig) -> None:
        with patch("chart_genie.renderer.Path.write_text", side_effect=OSError("disk full")):
            with pytest.raises(RendererError, match="Cannot write chart HTML"):
                save_chart(bar_chart_config, output_path="/some/path/chart.html")

    def test_custom_cdn_url_saved(self, tmp_path: Path, bar_chart_config: ChartConfig) -> None:
        custom_url = "https://mycdn.example.com/chart.js"
        output = tmp_path / "chart.html"
        save_chart(bar_chart_config, output_path=output, chartjs_cdn_url=custom_url)
        content = output.read_text(encoding="utf-8")
        assert custom_url in content


# ---------------------------------------------------------------------------
# Tests: render_chart — all supported chart types render without error
# ---------------------------------------------------------------------------


class TestAllChartTypesRender:
    @pytest.mark.parametrize("chart_type", ["bar", "line", "pie", "doughnut", "scatter", "radar"])
    def test_chart_type_renders(
        self,
        chart_type: str,
    ) -> None:
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


# ---------------------------------------------------------------------------
# Tests: HTML accessibility and SEO basics
# ---------------------------------------------------------------------------


class TestHtmlAccessibility:
    def test_canvas_has_aria_label(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        assert 'aria-label=' in html

    def test_canvas_has_role_img(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        assert 'role="img"' in html

    def test_lang_attribute_on_html(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        assert 'lang="en"' in html

    def test_charset_meta(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        assert 'charset="UTF-8"' in html or 'charset=UTF-8' in html

    def test_viewport_meta(self, bar_chart_config: ChartConfig) -> None:
        html = render_chart(bar_chart_config)
        assert 'name="viewport"' in html
