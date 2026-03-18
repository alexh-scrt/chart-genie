"""Unit tests for chart_genie.chart_config.

Covers ChartConfig and DatasetConfig construction, validation, defaults,
serialisation, and rejection of invalid inputs.
"""

from __future__ import annotations

import pytest

from chart_genie.chart_config import (
    DEFAULT_BACKGROUND_COLORS,
    DEFAULT_BORDER_COLORS,
    SUPPORTED_CHART_TYPES,
    ChartConfig,
    ChartConfigError,
    DatasetConfig,
    _deep_merge,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_dataset_dict() -> dict:
    """Minimal valid dataset dictionary."""
    return {"label": "Sales", "data": [100, 200, 300]}


@pytest.fixture()
def full_dataset_dict() -> dict:
    """Dataset dictionary with all optional fields specified."""
    return {
        "label": "Revenue",
        "data": [500, 600, 700],
        "background_color": "rgba(54, 162, 235, 0.7)",
        "border_color": "rgba(54, 162, 235, 1.0)",
        "border_width": 2,
        "fill": True,
        "tension": 0.4,
        "point_radius": 5,
    }


@pytest.fixture()
def minimal_config_dict(minimal_dataset_dict: dict) -> dict:
    """Minimal valid ChartConfig dictionary."""
    return {
        "chart_type": "bar",
        "title": "Monthly Sales",
        "labels": ["Jan", "Feb", "Mar"],
        "datasets": [minimal_dataset_dict],
    }


@pytest.fixture()
def full_config_dict(full_dataset_dict: dict) -> dict:
    """ChartConfig dictionary with all optional fields specified."""
    return {
        "chart_type": "line",
        "title": "Revenue Trend",
        "labels": ["Q1", "Q2", "Q3"],
        "datasets": [full_dataset_dict],
        "x_axis_label": "Quarter",
        "y_axis_label": "Revenue (USD)",
        "show_legend": False,
        "show_tooltips": False,
        "responsive": False,
        "maintain_aspect_ratio": True,
        "width": 800,
        "height": 400,
        "extra_options": {"animation": False},
    }


# ---------------------------------------------------------------------------
# Tests: SUPPORTED_CHART_TYPES constant
# ---------------------------------------------------------------------------


class TestSupportedChartTypes:
    def test_contains_expected_types(self) -> None:
        """SUPPORTED_CHART_TYPES contains all six expected chart types."""
        expected = {"bar", "line", "pie", "doughnut", "scatter", "radar"}
        assert expected == SUPPORTED_CHART_TYPES

    def test_is_frozenset(self) -> None:
        """SUPPORTED_CHART_TYPES is a frozenset."""
        assert isinstance(SUPPORTED_CHART_TYPES, frozenset)

    def test_has_six_types(self) -> None:
        """SUPPORTED_CHART_TYPES has exactly six entries."""
        assert len(SUPPORTED_CHART_TYPES) == 6


# ---------------------------------------------------------------------------
# Tests: DatasetConfig.from_dict
# ---------------------------------------------------------------------------


class TestDatasetConfigFromDict:
    def test_minimal_valid(self, minimal_dataset_dict: dict) -> None:
        """A minimal valid dataset dict produces the expected DatasetConfig."""
        ds = DatasetConfig.from_dict(minimal_dataset_dict)
        assert ds.label == "Sales"
        assert ds.data == [100, 200, 300]

    def test_defaults_applied(self, minimal_dataset_dict: dict) -> None:
        """Default values are applied when optional fields are absent."""
        ds = DatasetConfig.from_dict(minimal_dataset_dict, index=0)
        assert ds.background_color == DEFAULT_BACKGROUND_COLORS[0]
        assert ds.border_color == DEFAULT_BORDER_COLORS[0]
        assert ds.border_width == 1
        assert ds.fill is False
        assert ds.tension == 0.3
        assert ds.point_radius == 3

    def test_default_colors_cycle_by_index(self, minimal_dataset_dict: dict) -> None:
        """Default colors cycle through the palette by index."""
        ds1 = DatasetConfig.from_dict(minimal_dataset_dict, index=1)
        assert ds1.background_color == DEFAULT_BACKGROUND_COLORS[1]
        ds_wrap = DatasetConfig.from_dict(
            minimal_dataset_dict, index=len(DEFAULT_BACKGROUND_COLORS)
        )
        assert ds_wrap.background_color == DEFAULT_BACKGROUND_COLORS[0]

    def test_full_fields(self, full_dataset_dict: dict) -> None:
        """A full dataset dict produces a DatasetConfig with all fields set."""
        ds = DatasetConfig.from_dict(full_dataset_dict)
        assert ds.label == "Revenue"
        assert ds.data == [500, 600, 700]
        assert ds.background_color == "rgba(54, 162, 235, 0.7)"
        assert ds.border_color == "rgba(54, 162, 235, 1.0)"
        assert ds.border_width == 2
        assert ds.fill is True
        assert ds.tension == 0.4
        assert ds.point_radius == 5

    def test_missing_label_raises(self) -> None:
        """A dataset missing the 'label' key raises ChartConfigError."""
        with pytest.raises(ChartConfigError, match="label"):
            DatasetConfig.from_dict({"data": [1, 2, 3]})

    def test_empty_label_raises(self) -> None:
        """A dataset with a blank 'label' raises ChartConfigError."""
        with pytest.raises(ChartConfigError, match="label"):
            DatasetConfig.from_dict({"label": "  ", "data": [1, 2, 3]})

    def test_missing_data_raises(self) -> None:
        """A dataset missing the 'data' key raises ChartConfigError."""
        with pytest.raises(ChartConfigError, match="data"):
            DatasetConfig.from_dict({"label": "Test"})

    def test_data_not_list_raises(self) -> None:
        """A dataset where 'data' is not a list raises ChartConfigError."""
        with pytest.raises(ChartConfigError, match="list"):
            DatasetConfig.from_dict({"label": "Test", "data": 42})

    def test_negative_border_width_raises(self) -> None:
        """A negative border_width raises ChartConfigError."""
        with pytest.raises(ChartConfigError, match="border_width"):
            DatasetConfig.from_dict(
                {"label": "Test", "data": [1], "border_width": -1}
            )

    def test_tension_out_of_range_raises(self) -> None:
        """A tension value > 1.0 raises ChartConfigError."""
        with pytest.raises(ChartConfigError, match="tension"):
            DatasetConfig.from_dict(
                {"label": "Test", "data": [1], "tension": 1.5}
            )

    def test_tension_negative_raises(self) -> None:
        """A negative tension value raises ChartConfigError."""
        with pytest.raises(ChartConfigError, match="tension"):
            DatasetConfig.from_dict(
                {"label": "Test", "data": [1], "tension": -0.1}
            )

    def test_negative_point_radius_raises(self) -> None:
        """A negative point_radius raises ChartConfigError."""
        with pytest.raises(ChartConfigError, match="point_radius"):
            DatasetConfig.from_dict(
                {"label": "Test", "data": [1], "point_radius": -2}
            )

    def test_color_list_accepted(self) -> None:
        """A list of colors for background_color is accepted."""
        ds = DatasetConfig.from_dict(
            {
                "label": "Pie",
                "data": [30, 40, 30],
                "background_color": ["#ff0000", "#00ff00", "#0000ff"],
            }
        )
        assert isinstance(ds.background_color, list)
        assert len(ds.background_color) == 3

    def test_invalid_color_raises(self) -> None:
        """An invalid color string raises ChartConfigError."""
        with pytest.raises(ChartConfigError, match="color"):
            DatasetConfig.from_dict(
                {"label": "Test", "data": [1], "background_color": "not-a-color!!"}
            )

    def test_scatter_xy_data(self) -> None:
        """Scatter chart x/y dict data points are accepted."""
        ds = DatasetConfig.from_dict(
            {
                "label": "Scatter",
                "data": [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}],
            }
        )
        assert ds.data == [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]

    def test_scatter_xy_missing_key_raises(self) -> None:
        """A scatter data point missing the 'y' key raises ChartConfigError."""
        with pytest.raises(ChartConfigError, match="missing"):
            DatasetConfig.from_dict(
                {"label": "Scatter", "data": [{"x": 1.0}]}
            )

    def test_none_data_points_preserved(self) -> None:
        """None values in data (gap markers) are preserved."""
        ds = DatasetConfig.from_dict(
            {"label": "Gappy", "data": [1, None, 3]}
        )
        assert ds.data[1] is None

    def test_string_numbers_coerced(self) -> None:
        """Numeric strings in data are coerced to numbers."""
        ds = DatasetConfig.from_dict(
            {"label": "Str", "data": ["10", "20.5", "30"]}
        )
        assert ds.data == [10, 20.5, 30]

    def test_zero_border_width_accepted(self) -> None:
        """A border_width of 0 is valid."""
        ds = DatasetConfig.from_dict(
            {"label": "Test", "data": [1], "border_width": 0}
        )
        assert ds.border_width == 0

    def test_tension_zero_accepted(self) -> None:
        """A tension of 0.0 (straight lines) is valid."""
        ds = DatasetConfig.from_dict(
            {"label": "Test", "data": [1], "tension": 0.0}
        )
        assert ds.tension == 0.0

    def test_tension_one_accepted(self) -> None:
        """A tension of 1.0 is valid (max bezier curve)."""
        ds = DatasetConfig.from_dict(
            {"label": "Test", "data": [1], "tension": 1.0}
        )
        assert ds.tension == 1.0

    def test_hex_color_accepted(self) -> None:
        """A hex color string is accepted."""
        ds = DatasetConfig.from_dict(
            {"label": "Test", "data": [1], "background_color": "#ff6384"}
        )
        assert ds.background_color == "#ff6384"

    def test_named_color_accepted(self) -> None:
        """A named CSS color is accepted."""
        ds = DatasetConfig.from_dict(
            {"label": "Test", "data": [1], "background_color": "red"}
        )
        assert ds.background_color == "red"

    def test_border_color_list_accepted(self) -> None:
        """A list of border colors is accepted."""
        ds = DatasetConfig.from_dict(
            {
                "label": "Test",
                "data": [1, 2],
                "border_color": ["rgba(255,0,0,1.0)", "rgba(0,255,0,1.0)"],
            }
        )
        assert isinstance(ds.border_color, list)
        assert len(ds.border_color) == 2

    def test_empty_data_list_accepted(self) -> None:
        """An empty data list is accepted."""
        ds = DatasetConfig.from_dict({"label": "Empty", "data": []})
        assert ds.data == []

    def test_index_out_of_palette_wraps(self, minimal_dataset_dict: dict) -> None:
        """Index values larger than the palette wrap around correctly."""
        palette_len = len(DEFAULT_BACKGROUND_COLORS)
        for i in range(palette_len * 2):
            ds = DatasetConfig.from_dict(minimal_dataset_dict, index=i)
            assert ds.background_color == DEFAULT_BACKGROUND_COLORS[i % palette_len]


# ---------------------------------------------------------------------------
# Tests: DatasetConfig.to_chartjs_dict
# ---------------------------------------------------------------------------


class TestDatasetConfigToChartjsDict:
    def test_keys_camelcase(self, minimal_dataset_dict: dict) -> None:
        """to_chartjs_dict uses camelCase keys for Chart.js compatibility."""
        ds = DatasetConfig.from_dict(minimal_dataset_dict)
        result = ds.to_chartjs_dict()
        assert "backgroundColor" in result
        assert "borderColor" in result
        assert "borderWidth" in result
        assert "pointRadius" in result
        assert "background_color" not in result

    def test_data_preserved(self, minimal_dataset_dict: dict) -> None:
        """Data and label are preserved in the Chart.js dict."""
        ds = DatasetConfig.from_dict(minimal_dataset_dict)
        result = ds.to_chartjs_dict()
        assert result["data"] == [100, 200, 300]
        assert result["label"] == "Sales"

    def test_fill_key_present(self, minimal_dataset_dict: dict) -> None:
        """The 'fill' key is present in the Chart.js dict."""
        ds = DatasetConfig.from_dict(minimal_dataset_dict)
        result = ds.to_chartjs_dict()
        assert "fill" in result

    def test_tension_key_present(self, minimal_dataset_dict: dict) -> None:
        """The 'tension' key is present in the Chart.js dict."""
        ds = DatasetConfig.from_dict(minimal_dataset_dict)
        result = ds.to_chartjs_dict()
        assert "tension" in result

    def test_all_expected_keys_present(self, minimal_dataset_dict: dict) -> None:
        """All expected Chart.js dataset keys are present."""
        ds = DatasetConfig.from_dict(minimal_dataset_dict)
        result = ds.to_chartjs_dict()
        expected_keys = {
            "label", "data", "backgroundColor", "borderColor",
            "borderWidth", "fill", "tension", "pointRadius",
        }
        for key in expected_keys:
            assert key in result, f"Missing key '{key}' in Chart.js dict"

    def test_full_dataset_to_chartjs(self, full_dataset_dict: dict) -> None:
        """A full DatasetConfig serialises correctly to Chart.js format."""
        ds = DatasetConfig.from_dict(full_dataset_dict)
        result = ds.to_chartjs_dict()
        assert result["label"] == "Revenue"
        assert result["data"] == [500, 600, 700]
        assert result["backgroundColor"] == "rgba(54, 162, 235, 0.7)"
        assert result["borderColor"] == "rgba(54, 162, 235, 1.0)"
        assert result["borderWidth"] == 2
        assert result["fill"] is True
        assert result["tension"] == 0.4
        assert result["pointRadius"] == 5


# ---------------------------------------------------------------------------
# Tests: ChartConfig.from_dict
# ---------------------------------------------------------------------------


class TestChartConfigFromDict:
    def test_minimal_valid(self, minimal_config_dict: dict) -> None:
        """A minimal config dict produces a valid ChartConfig."""
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert cfg.chart_type == "bar"
        assert cfg.title == "Monthly Sales"
        assert cfg.labels == ["Jan", "Feb", "Mar"]
        assert len(cfg.datasets) == 1

    def test_full_valid(self, full_config_dict: dict) -> None:
        """A full config dict produces a ChartConfig with all fields set."""
        cfg = ChartConfig.from_dict(full_config_dict)
        assert cfg.chart_type == "line"
        assert cfg.x_axis_label == "Quarter"
        assert cfg.y_axis_label == "Revenue (USD)"
        assert cfg.show_legend is False
        assert cfg.show_tooltips is False
        assert cfg.responsive is False
        assert cfg.maintain_aspect_ratio is True
        assert cfg.width == 800
        assert cfg.height == 400
        assert cfg.extra_options == {"animation": False}

    def test_chart_type_case_insensitive(self, minimal_config_dict: dict) -> None:
        """chart_type is normalised to lowercase regardless of input case."""
        minimal_config_dict["chart_type"] = "BAR"
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert cfg.chart_type == "bar"

    def test_chart_type_mixed_case(self, minimal_config_dict: dict) -> None:
        """Mixed-case chart_type is normalised to lowercase."""
        minimal_config_dict["chart_type"] = "Line"
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert cfg.chart_type == "line"

    def test_all_supported_chart_types(self) -> None:
        """All supported chart types can be used to create a ChartConfig."""
        for chart_type in SUPPORTED_CHART_TYPES:
            cfg = ChartConfig.from_dict(
                {
                    "chart_type": chart_type,
                    "title": f"{chart_type} chart",
                    "labels": ["A", "B"],
                    "datasets": [{"label": "S", "data": [1, 2]}],
                }
            )
            assert cfg.chart_type == chart_type

    def test_unsupported_chart_type_raises(self, minimal_config_dict: dict) -> None:
        """An unsupported chart_type raises ChartConfigError."""
        minimal_config_dict["chart_type"] = "treemap"
        with pytest.raises(ChartConfigError, match="Unsupported chart type"):
            ChartConfig.from_dict(minimal_config_dict)

    def test_missing_chart_type_raises(self, minimal_config_dict: dict) -> None:
        """A config missing 'chart_type' raises ChartConfigError."""
        del minimal_config_dict["chart_type"]
        with pytest.raises(ChartConfigError, match="chart_type"):
            ChartConfig.from_dict(minimal_config_dict)

    def test_missing_title_raises(self, minimal_config_dict: dict) -> None:
        """A config missing 'title' raises ChartConfigError."""
        del minimal_config_dict["title"]
        with pytest.raises(ChartConfigError, match="title"):
            ChartConfig.from_dict(minimal_config_dict)

    def test_empty_title_raises(self, minimal_config_dict: dict) -> None:
        """A config with a blank 'title' raises ChartConfigError."""
        minimal_config_dict["title"] = "   "
        with pytest.raises(ChartConfigError, match="title"):
            ChartConfig.from_dict(minimal_config_dict)

    def test_missing_datasets_raises(self, minimal_config_dict: dict) -> None:
        """A config missing 'datasets' raises ChartConfigError."""
        del minimal_config_dict["datasets"]
        with pytest.raises(ChartConfigError, match="datasets"):
            ChartConfig.from_dict(minimal_config_dict)

    def test_empty_datasets_raises(self, minimal_config_dict: dict) -> None:
        """A config with an empty 'datasets' list raises ChartConfigError."""
        minimal_config_dict["datasets"] = []
        with pytest.raises(ChartConfigError, match="datasets"):
            ChartConfig.from_dict(minimal_config_dict)

    def test_labels_defaults_to_empty_list(self, minimal_config_dict: dict) -> None:
        """When 'labels' is absent, it defaults to an empty list."""
        del minimal_config_dict["labels"]
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert cfg.labels == []

    def test_labels_coerced_to_strings(self, minimal_config_dict: dict) -> None:
        """Non-string label values are coerced to strings."""
        minimal_config_dict["labels"] = [1, 2, 3]
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert cfg.labels == ["1", "2", "3"]

    def test_invalid_width_raises(self, minimal_config_dict: dict) -> None:
        """A non-positive 'width' raises ChartConfigError."""
        minimal_config_dict["width"] = -10
        with pytest.raises(ChartConfigError, match="width"):
            ChartConfig.from_dict(minimal_config_dict)

    def test_invalid_height_raises(self, minimal_config_dict: dict) -> None:
        """A height of 0 raises ChartConfigError."""
        minimal_config_dict["height"] = 0
        with pytest.raises(ChartConfigError, match="height"):
            ChartConfig.from_dict(minimal_config_dict)

    def test_extra_options_not_dict_raises(self, minimal_config_dict: dict) -> None:
        """A non-dict 'extra_options' raises ChartConfigError."""
        minimal_config_dict["extra_options"] = "invalid"
        with pytest.raises(ChartConfigError, match="extra_options"):
            ChartConfig.from_dict(minimal_config_dict)

    def test_multiple_datasets(self, minimal_config_dict: dict) -> None:
        """Multiple datasets are all parsed and stored."""
        minimal_config_dict["datasets"] = [
            {"label": "A", "data": [1, 2, 3]},
            {"label": "B", "data": [4, 5, 6]},
        ]
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert len(cfg.datasets) == 2
        assert cfg.datasets[0].label == "A"
        assert cfg.datasets[1].label == "B"

    def test_show_legend_defaults_true(self, minimal_config_dict: dict) -> None:
        """show_legend defaults to True when not specified."""
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert cfg.show_legend is True

    def test_show_tooltips_defaults_true(self, minimal_config_dict: dict) -> None:
        """show_tooltips defaults to True when not specified."""
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert cfg.show_tooltips is True

    def test_responsive_defaults_true(self, minimal_config_dict: dict) -> None:
        """responsive defaults to True when not specified."""
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert cfg.responsive is True

    def test_maintain_aspect_ratio_defaults_false(self, minimal_config_dict: dict) -> None:
        """maintain_aspect_ratio defaults to False when not specified."""
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert cfg.maintain_aspect_ratio is False

    def test_width_defaults_to_none(self, minimal_config_dict: dict) -> None:
        """width defaults to None when not specified."""
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert cfg.width is None

    def test_height_defaults_to_none(self, minimal_config_dict: dict) -> None:
        """height defaults to None when not specified."""
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert cfg.height is None

    def test_extra_options_defaults_to_empty(self, minimal_config_dict: dict) -> None:
        """extra_options defaults to an empty dict when not specified."""
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert cfg.extra_options == {}

    def test_x_axis_label_defaults_to_empty_string(self, minimal_config_dict: dict) -> None:
        """x_axis_label defaults to empty string when not specified."""
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert cfg.x_axis_label == ""

    def test_y_axis_label_defaults_to_empty_string(self, minimal_config_dict: dict) -> None:
        """y_axis_label defaults to empty string when not specified."""
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert cfg.y_axis_label == ""

    def test_width_one_is_valid(self, minimal_config_dict: dict) -> None:
        """A width of 1 pixel is the minimum valid positive value."""
        minimal_config_dict["width"] = 1
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert cfg.width == 1

    def test_height_one_is_valid(self, minimal_config_dict: dict) -> None:
        """A height of 1 pixel is the minimum valid positive value."""
        minimal_config_dict["height"] = 1
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert cfg.height == 1

    def test_datasets_not_list_raises(self, minimal_config_dict: dict) -> None:
        """A non-list 'datasets' value raises ChartConfigError."""
        minimal_config_dict["datasets"] = {"label": "oops", "data": [1]}
        with pytest.raises(ChartConfigError):
            ChartConfig.from_dict(minimal_config_dict)

    def test_labels_not_list_raises(self, minimal_config_dict: dict) -> None:
        """A non-list 'labels' value raises ChartConfigError."""
        minimal_config_dict["labels"] = "not a list"
        with pytest.raises(ChartConfigError, match="labels"):
            ChartConfig.from_dict(minimal_config_dict)


# ---------------------------------------------------------------------------
# Tests: ChartConfig.to_chartjs_config
# ---------------------------------------------------------------------------


class TestChartConfigToChartjsConfig:
    def test_top_level_keys(self, minimal_config_dict: dict) -> None:
        """The Chart.js config has 'type', 'data', and 'options' keys."""
        cfg = ChartConfig.from_dict(minimal_config_dict)
        result = cfg.to_chartjs_config()
        assert "type" in result
        assert "data" in result
        assert "options" in result

    def test_type_field(self, minimal_config_dict: dict) -> None:
        """The 'type' field in the Chart.js config matches the chart_type."""
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert cfg.to_chartjs_config()["type"] == "bar"

    def test_data_labels_and_datasets(self, minimal_config_dict: dict) -> None:
        """The 'data' field contains 'labels' and 'datasets'."""
        cfg = ChartConfig.from_dict(minimal_config_dict)
        result = cfg.to_chartjs_config()
        assert result["data"]["labels"] == ["Jan", "Feb", "Mar"]
        assert len(result["data"]["datasets"]) == 1

    def test_title_in_plugins(self, minimal_config_dict: dict) -> None:
        """The chart title appears in the plugins.title config."""
        cfg = ChartConfig.from_dict(minimal_config_dict)
        plugins = cfg.to_chartjs_config()["options"]["plugins"]
        assert plugins["title"]["text"] == "Monthly Sales"
        assert plugins["title"]["display"] is True

    def test_legend_display(self, minimal_config_dict: dict) -> None:
        """show_legend=False is reflected in the Chart.js legend config."""
        minimal_config_dict["show_legend"] = False
        cfg = ChartConfig.from_dict(minimal_config_dict)
        plugins = cfg.to_chartjs_config()["options"]["plugins"]
        assert plugins["legend"]["display"] is False

    def test_legend_display_true(self, minimal_config_dict: dict) -> None:
        """show_legend=True is reflected in the Chart.js legend config."""
        minimal_config_dict["show_legend"] = True
        cfg = ChartConfig.from_dict(minimal_config_dict)
        plugins = cfg.to_chartjs_config()["options"]["plugins"]
        assert plugins["legend"]["display"] is True

    def test_tooltip_enabled(self, minimal_config_dict: dict) -> None:
        """show_tooltips=True is reflected in the Chart.js tooltip config."""
        minimal_config_dict["show_tooltips"] = True
        cfg = ChartConfig.from_dict(minimal_config_dict)
        plugins = cfg.to_chartjs_config()["options"]["plugins"]
        assert plugins["tooltip"]["enabled"] is True

    def test_tooltip_disabled(self, minimal_config_dict: dict) -> None:
        """show_tooltips=False is reflected in the Chart.js tooltip config."""
        minimal_config_dict["show_tooltips"] = False
        cfg = ChartConfig.from_dict(minimal_config_dict)
        plugins = cfg.to_chartjs_config()["options"]["plugins"]
        assert plugins["tooltip"]["enabled"] is False

    def test_axis_labels_for_cartesian(self, full_config_dict: dict) -> None:
        """x/y axis labels are included in 'scales' for cartesian chart types."""
        cfg = ChartConfig.from_dict(full_config_dict)
        options = cfg.to_chartjs_config()["options"]
        assert "scales" in options
        assert options["scales"]["x"]["title"]["text"] == "Quarter"
        assert options["scales"]["y"]["title"]["text"] == "Revenue (USD)"

    def test_no_scales_for_pie(self) -> None:
        """Pie charts do not have a 'scales' key even if x_axis_label is set."""
        cfg = ChartConfig.from_dict(
            {
                "chart_type": "pie",
                "title": "Pie Test",
                "labels": ["A", "B"],
                "datasets": [{"label": "Data", "data": [60, 40]}],
                "x_axis_label": "should be ignored",
            }
        )
        options = cfg.to_chartjs_config()["options"]
        assert "scales" not in options

    def test_no_scales_for_doughnut(self) -> None:
        """Doughnut charts do not have a 'scales' key."""
        cfg = ChartConfig.from_dict(
            {
                "chart_type": "doughnut",
                "title": "D",
                "labels": ["X", "Y"],
                "datasets": [{"label": "D", "data": [50, 50]}],
            }
        )
        options = cfg.to_chartjs_config()["options"]
        assert "scales" not in options

    def test_no_scales_for_radar(self) -> None:
        """Radar charts do not have a 'scales' key."""
        cfg = ChartConfig.from_dict(
            {
                "chart_type": "radar",
                "title": "R",
                "labels": ["A", "B", "C"],
                "datasets": [{"label": "D", "data": [1, 2, 3]}],
            }
        )
        options = cfg.to_chartjs_config()["options"]
        assert "scales" not in options

    def test_extra_options_merged(self, minimal_config_dict: dict) -> None:
        """extra_options are merged into the Chart.js options dict."""
        minimal_config_dict["extra_options"] = {"animation": {"duration": 500}}
        cfg = ChartConfig.from_dict(minimal_config_dict)
        options = cfg.to_chartjs_config()["options"]
        assert options["animation"]["duration"] == 500

    def test_extra_options_deep_merge(self, minimal_config_dict: dict) -> None:
        """extra_options are deep-merged, preserving existing nested keys."""
        minimal_config_dict["extra_options"] = {
            "plugins": {"legend": {"position": "bottom"}}
        }
        cfg = ChartConfig.from_dict(minimal_config_dict)
        options = cfg.to_chartjs_config()["options"]
        # Deep merge: original keys preserved, new key added.
        assert options["plugins"]["legend"]["display"] is True
        assert options["plugins"]["legend"]["position"] == "bottom"

    def test_responsive_in_options(self, minimal_config_dict: dict) -> None:
        """The 'responsive' flag appears in the Chart.js options."""
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert cfg.to_chartjs_config()["options"]["responsive"] is True

    def test_maintain_aspect_ratio_in_options(self, minimal_config_dict: dict) -> None:
        """The 'maintainAspectRatio' flag appears in the Chart.js options."""
        minimal_config_dict["maintain_aspect_ratio"] = True
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert cfg.to_chartjs_config()["options"]["maintainAspectRatio"] is True

    def test_no_scales_when_no_axis_labels(self, minimal_config_dict: dict) -> None:
        """When neither axis label is set, 'scales' is absent from options."""
        # minimal_config_dict has no axis labels
        cfg = ChartConfig.from_dict(minimal_config_dict)
        options = cfg.to_chartjs_config()["options"]
        assert "scales" not in options

    def test_only_x_axis_label_produces_scales_x(self, minimal_config_dict: dict) -> None:
        """When only x_axis_label is set, scales contains only 'x'."""
        minimal_config_dict["x_axis_label"] = "Month"
        cfg = ChartConfig.from_dict(minimal_config_dict)
        options = cfg.to_chartjs_config()["options"]
        assert "scales" in options
        assert "x" in options["scales"]
        assert "y" not in options["scales"]

    def test_only_y_axis_label_produces_scales_y(self, minimal_config_dict: dict) -> None:
        """When only y_axis_label is set, scales contains only 'y'."""
        minimal_config_dict["y_axis_label"] = "Value"
        cfg = ChartConfig.from_dict(minimal_config_dict)
        options = cfg.to_chartjs_config()["options"]
        assert "scales" in options
        assert "y" in options["scales"]
        assert "x" not in options["scales"]

    def test_datasets_camelcase_keys(self, minimal_config_dict: dict) -> None:
        """Dataset keys in Chart.js output are camelCase."""
        cfg = ChartConfig.from_dict(minimal_config_dict)
        result = cfg.to_chartjs_config()
        ds = result["data"]["datasets"][0]
        assert "backgroundColor" in ds
        assert "borderColor" in ds
        assert "background_color" not in ds

    def test_multiple_datasets_in_chartjs_config(self) -> None:
        """Multiple datasets appear in the Chart.js data.datasets list."""
        cfg = ChartConfig.from_dict({
            "chart_type": "line",
            "title": "Multi",
            "labels": ["A", "B"],
            "datasets": [
                {"label": "S1", "data": [1, 2]},
                {"label": "S2", "data": [3, 4]},
                {"label": "S3", "data": [5, 6]},
            ],
        })
        result = cfg.to_chartjs_config()
        assert len(result["data"]["datasets"]) == 3


# ---------------------------------------------------------------------------
# Tests: ChartConfig.to_dict
# ---------------------------------------------------------------------------


class TestChartConfigToDict:
    def test_round_trip(self, minimal_config_dict: dict) -> None:
        """to_dict / from_dict round-trips produce equivalent configs."""
        cfg = ChartConfig.from_dict(minimal_config_dict)
        serialised = cfg.to_dict()
        cfg2 = ChartConfig.from_dict(serialised)
        assert cfg2.chart_type == cfg.chart_type
        assert cfg2.title == cfg.title
        assert cfg2.labels == cfg.labels

    def test_contains_expected_keys(self, minimal_config_dict: dict) -> None:
        """to_dict contains all expected top-level keys."""
        cfg = ChartConfig.from_dict(minimal_config_dict)
        d = cfg.to_dict()
        for key in (
            "chart_type",
            "title",
            "labels",
            "datasets",
            "x_axis_label",
            "y_axis_label",
            "show_legend",
            "show_tooltips",
            "responsive",
            "maintain_aspect_ratio",
            "width",
            "height",
            "extra_options",
        ):
            assert key in d

    def test_datasets_serialised_as_list_of_dicts(self, minimal_config_dict: dict) -> None:
        """Datasets in to_dict output are dicts, not DatasetConfig objects."""
        cfg = ChartConfig.from_dict(minimal_config_dict)
        d = cfg.to_dict()
        assert isinstance(d["datasets"], list)
        assert isinstance(d["datasets"][0], dict)

    def test_dataset_dict_has_snake_case_keys(self, minimal_config_dict: dict) -> None:
        """Dataset dicts in to_dict output use snake_case keys."""
        cfg = ChartConfig.from_dict(minimal_config_dict)
        ds_dict = cfg.to_dict()["datasets"][0]
        assert "background_color" in ds_dict
        assert "border_color" in ds_dict
        assert "backgroundColor" not in ds_dict

    def test_full_round_trip(self, full_config_dict: dict) -> None:
        """A full config round-trips correctly."""
        cfg = ChartConfig.from_dict(full_config_dict)
        d = cfg.to_dict()
        cfg2 = ChartConfig.from_dict(d)
        assert cfg2.chart_type == cfg.chart_type
        assert cfg2.x_axis_label == cfg.x_axis_label
        assert cfg2.y_axis_label == cfg.y_axis_label
        assert cfg2.width == cfg.width
        assert cfg2.height == cfg.height
        assert cfg2.extra_options == cfg.extra_options

    def test_none_width_height_preserved(self, minimal_config_dict: dict) -> None:
        """None width and height are preserved in to_dict output."""
        cfg = ChartConfig.from_dict(minimal_config_dict)
        d = cfg.to_dict()
        assert d["width"] is None
        assert d["height"] is None


# ---------------------------------------------------------------------------
# Tests: ChartConfig properties
# ---------------------------------------------------------------------------


class TestChartConfigProperties:
    @pytest.mark.parametrize("chart_type", ["bar", "line", "scatter"])
    def test_is_cartesian_true(self, chart_type: str) -> None:
        """Cartesian chart types report is_cartesian=True and is_radial=False."""
        cfg = ChartConfig.from_dict(
            {
                "chart_type": chart_type,
                "title": "T",
                "labels": [],
                "datasets": [{"label": "D", "data": [1, 2]}],
            }
        )
        assert cfg.is_cartesian is True
        assert cfg.is_radial is False

    @pytest.mark.parametrize("chart_type", ["pie", "doughnut", "radar"])
    def test_is_radial_true(self, chart_type: str) -> None:
        """Radial chart types report is_radial=True and is_cartesian=False."""
        cfg = ChartConfig.from_dict(
            {
                "chart_type": chart_type,
                "title": "T",
                "labels": ["A", "B"],
                "datasets": [{"label": "D", "data": [10, 20]}],
            }
        )
        assert cfg.is_radial is True
        assert cfg.is_cartesian is False

    def test_is_cartesian_bar(self) -> None:
        """Bar chart is classified as cartesian."""
        cfg = ChartConfig.from_dict({
            "chart_type": "bar",
            "title": "T",
            "labels": ["A"],
            "datasets": [{"label": "D", "data": [1]}],
        })
        assert cfg.is_cartesian is True

    def test_is_radial_pie(self) -> None:
        """Pie chart is classified as radial."""
        cfg = ChartConfig.from_dict({
            "chart_type": "pie",
            "title": "T",
            "labels": ["A"],
            "datasets": [{"label": "D", "data": [100]}],
        })
        assert cfg.is_radial is True


# ---------------------------------------------------------------------------
# Tests: ChartConfig post-init validation
# ---------------------------------------------------------------------------


class TestChartConfigPostInit:
    def test_labels_not_list_raises(self) -> None:
        """Constructing ChartConfig with non-list labels raises ChartConfigError."""
        with pytest.raises(ChartConfigError, match="labels"):
            ChartConfig(
                chart_type="bar",
                title="Test",
                labels="not a list",  # type: ignore[arg-type]
                datasets=[
                    DatasetConfig(label="D", data=[1])
                ],
            )

    def test_empty_datasets_raises(self) -> None:
        """Constructing ChartConfig with an empty datasets list raises ChartConfigError."""
        with pytest.raises(ChartConfigError, match="datasets"):
            ChartConfig(
                chart_type="bar",
                title="Test",
                labels=[],
                datasets=[],
            )

    def test_datasets_wrong_type_raises(self) -> None:
        """Constructing ChartConfig with non-DatasetConfig in datasets raises."""
        with pytest.raises(ChartConfigError, match="DatasetConfig"):
            ChartConfig(
                chart_type="bar",
                title="Test",
                labels=[],
                datasets=[{"label": "oops", "data": [1]}],  # type: ignore[list-item]
            )

    def test_zero_width_raises(self) -> None:
        """width=0 raises ChartConfigError."""
        with pytest.raises(ChartConfigError, match="width"):
            ChartConfig(
                chart_type="bar",
                title="Test",
                labels=[],
                datasets=[DatasetConfig(label="D", data=[1])],
                width=0,
            )

    def test_zero_height_raises(self) -> None:
        """height=0 raises ChartConfigError."""
        with pytest.raises(ChartConfigError, match="height"):
            ChartConfig(
                chart_type="bar",
                title="Test",
                labels=[],
                datasets=[DatasetConfig(label="D", data=[1])],
                height=0,
            )

    def test_negative_width_raises(self) -> None:
        """A negative width raises ChartConfigError."""
        with pytest.raises(ChartConfigError, match="width"):
            ChartConfig(
                chart_type="bar",
                title="Test",
                labels=[],
                datasets=[DatasetConfig(label="D", data=[1])],
                width=-5,
            )

    def test_extra_options_not_dict_raises(self) -> None:
        """extra_options that is not a dict raises ChartConfigError."""
        with pytest.raises(ChartConfigError, match="extra_options"):
            ChartConfig(
                chart_type="bar",
                title="Test",
                labels=[],
                datasets=[DatasetConfig(label="D", data=[1])],
                extra_options=["wrong"],  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Tests: _deep_merge utility
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_simple_merge(self) -> None:
        """Simple key merge updates base with override values."""
        base = {"a": 1, "b": 2}
        _deep_merge(base, {"b": 99, "c": 3})
        assert base == {"a": 1, "b": 99, "c": 3}

    def test_nested_merge(self) -> None:
        """Nested dicts are merged recursively."""
        base = {"plugins": {"legend": {"display": True}, "title": {"text": "T"}}}
        _deep_merge(base, {"plugins": {"legend": {"position": "top"}}})
        assert base["plugins"]["legend"]["display"] is True
        assert base["plugins"]["legend"]["position"] == "top"
        assert base["plugins"]["title"]["text"] == "T"

    def test_override_replaces_non_dict(self) -> None:
        """Non-dict values in override replace base values."""
        base = {"key": "old"}
        _deep_merge(base, {"key": "new"})
        assert base["key"] == "new"

    def test_empty_override_no_change(self) -> None:
        """An empty override dict leaves base unchanged."""
        base = {"a": 1}
        _deep_merge(base, {})
        assert base == {"a": 1}

    def test_new_nested_key_added(self) -> None:
        """New keys in override are added to base."""
        base = {"a": {"b": 1}}
        _deep_merge(base, {"a": {"c": 2}})
        assert base == {"a": {"b": 1, "c": 2}}

    def test_empty_base_gets_override(self) -> None:
        """An empty base dict is populated with override values."""
        base: dict = {}
        _deep_merge(base, {"x": 1, "y": 2})
        assert base == {"x": 1, "y": 2}

    def test_deeply_nested_merge(self) -> None:
        """Deeply nested dicts are merged at all levels."""
        base = {"a": {"b": {"c": {"d": 1}}}}
        _deep_merge(base, {"a": {"b": {"c": {"e": 2}}}})
        assert base["a"]["b"]["c"] == {"d": 1, "e": 2}

    def test_override_dict_replaces_non_dict_base(self) -> None:
        """If base has a scalar and override has a dict, override wins."""
        base = {"key": 42}
        _deep_merge(base, {"key": {"nested": True}})
        assert base["key"] == {"nested": True}

    def test_non_dict_override_replaces_dict_base(self) -> None:
        """If base has a dict and override has a scalar, override wins."""
        base = {"key": {"nested": True}}
        _deep_merge(base, {"key": 99})
        assert base["key"] == 99

    def test_list_values_replaced_not_merged(self) -> None:
        """List values in override replace (not extend) list values in base."""
        base = {"items": [1, 2, 3]}
        _deep_merge(base, {"items": [4, 5]})
        assert base["items"] == [4, 5]

    def test_modifies_base_in_place(self) -> None:
        """_deep_merge mutates the base dict in place."""
        base = {"x": 1}
        original_id = id(base)
        _deep_merge(base, {"y": 2})
        assert id(base) == original_id
        assert base["y"] == 2
