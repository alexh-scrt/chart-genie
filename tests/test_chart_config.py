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
        expected = {"bar", "line", "pie", "doughnut", "scatter", "radar"}
        assert expected == SUPPORTED_CHART_TYPES

    def test_is_frozenset(self) -> None:
        assert isinstance(SUPPORTED_CHART_TYPES, frozenset)


# ---------------------------------------------------------------------------
# Tests: DatasetConfig.from_dict
# ---------------------------------------------------------------------------


class TestDatasetConfigFromDict:
    def test_minimal_valid(self, minimal_dataset_dict: dict) -> None:
        ds = DatasetConfig.from_dict(minimal_dataset_dict)
        assert ds.label == "Sales"
        assert ds.data == [100, 200, 300]

    def test_defaults_applied(self, minimal_dataset_dict: dict) -> None:
        ds = DatasetConfig.from_dict(minimal_dataset_dict, index=0)
        assert ds.background_color == DEFAULT_BACKGROUND_COLORS[0]
        assert ds.border_color == DEFAULT_BORDER_COLORS[0]
        assert ds.border_width == 1
        assert ds.fill is False
        assert ds.tension == 0.3
        assert ds.point_radius == 3

    def test_default_colors_cycle_by_index(self, minimal_dataset_dict: dict) -> None:
        ds1 = DatasetConfig.from_dict(minimal_dataset_dict, index=1)
        assert ds1.background_color == DEFAULT_BACKGROUND_COLORS[1]
        ds_wrap = DatasetConfig.from_dict(
            minimal_dataset_dict, index=len(DEFAULT_BACKGROUND_COLORS)
        )
        assert ds_wrap.background_color == DEFAULT_BACKGROUND_COLORS[0]

    def test_full_fields(self, full_dataset_dict: dict) -> None:
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
        with pytest.raises(ChartConfigError, match="label"):
            DatasetConfig.from_dict({"data": [1, 2, 3]})

    def test_empty_label_raises(self) -> None:
        with pytest.raises(ChartConfigError, match="label"):
            DatasetConfig.from_dict({"label": "  ", "data": [1, 2, 3]})

    def test_missing_data_raises(self) -> None:
        with pytest.raises(ChartConfigError, match="data"):
            DatasetConfig.from_dict({"label": "Test"})

    def test_data_not_list_raises(self) -> None:
        with pytest.raises(ChartConfigError, match="list"):
            DatasetConfig.from_dict({"label": "Test", "data": 42})

    def test_negative_border_width_raises(self) -> None:
        with pytest.raises(ChartConfigError, match="border_width"):
            DatasetConfig.from_dict(
                {"label": "Test", "data": [1], "border_width": -1}
            )

    def test_tension_out_of_range_raises(self) -> None:
        with pytest.raises(ChartConfigError, match="tension"):
            DatasetConfig.from_dict(
                {"label": "Test", "data": [1], "tension": 1.5}
            )

    def test_negative_point_radius_raises(self) -> None:
        with pytest.raises(ChartConfigError, match="point_radius"):
            DatasetConfig.from_dict(
                {"label": "Test", "data": [1], "point_radius": -2}
            )

    def test_color_list_accepted(self) -> None:
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
        with pytest.raises(ChartConfigError, match="color"):
            DatasetConfig.from_dict(
                {"label": "Test", "data": [1], "background_color": "not-a-color!!"}
            )

    def test_scatter_xy_data(self) -> None:
        ds = DatasetConfig.from_dict(
            {
                "label": "Scatter",
                "data": [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}],
            }
        )
        assert ds.data == [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]

    def test_scatter_xy_missing_key_raises(self) -> None:
        with pytest.raises(ChartConfigError, match="missing"):
            DatasetConfig.from_dict(
                {"label": "Scatter", "data": [{"x": 1.0}]}
            )

    def test_none_data_points_preserved(self) -> None:
        ds = DatasetConfig.from_dict(
            {"label": "Gappy", "data": [1, None, 3]}
        )
        assert ds.data[1] is None

    def test_string_numbers_coerced(self) -> None:
        ds = DatasetConfig.from_dict(
            {"label": "Str", "data": ["10", "20.5", "30"]}
        )
        assert ds.data == [10, 20.5, 30]


# ---------------------------------------------------------------------------
# Tests: DatasetConfig.to_chartjs_dict
# ---------------------------------------------------------------------------


class TestDatasetConfigToChartjsDict:
    def test_keys_camelcase(self, minimal_dataset_dict: dict) -> None:
        ds = DatasetConfig.from_dict(minimal_dataset_dict)
        result = ds.to_chartjs_dict()
        assert "backgroundColor" in result
        assert "borderColor" in result
        assert "borderWidth" in result
        assert "pointRadius" in result
        assert "background_color" not in result

    def test_data_preserved(self, minimal_dataset_dict: dict) -> None:
        ds = DatasetConfig.from_dict(minimal_dataset_dict)
        result = ds.to_chartjs_dict()
        assert result["data"] == [100, 200, 300]
        assert result["label"] == "Sales"


# ---------------------------------------------------------------------------
# Tests: ChartConfig.from_dict
# ---------------------------------------------------------------------------


class TestChartConfigFromDict:
    def test_minimal_valid(self, minimal_config_dict: dict) -> None:
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert cfg.chart_type == "bar"
        assert cfg.title == "Monthly Sales"
        assert cfg.labels == ["Jan", "Feb", "Mar"]
        assert len(cfg.datasets) == 1

    def test_full_valid(self, full_config_dict: dict) -> None:
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
        minimal_config_dict["chart_type"] = "BAR"
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert cfg.chart_type == "bar"

    def test_all_supported_chart_types(self, minimal_dataset_dict: dict) -> None:
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
        minimal_config_dict["chart_type"] = "treemap"
        with pytest.raises(ChartConfigError, match="Unsupported chart type"):
            ChartConfig.from_dict(minimal_config_dict)

    def test_missing_chart_type_raises(self, minimal_config_dict: dict) -> None:
        del minimal_config_dict["chart_type"]
        with pytest.raises(ChartConfigError, match="chart_type"):
            ChartConfig.from_dict(minimal_config_dict)

    def test_missing_title_raises(self, minimal_config_dict: dict) -> None:
        del minimal_config_dict["title"]
        with pytest.raises(ChartConfigError, match="title"):
            ChartConfig.from_dict(minimal_config_dict)

    def test_empty_title_raises(self, minimal_config_dict: dict) -> None:
        minimal_config_dict["title"] = "   "
        with pytest.raises(ChartConfigError, match="title"):
            ChartConfig.from_dict(minimal_config_dict)

    def test_missing_datasets_raises(self, minimal_config_dict: dict) -> None:
        del minimal_config_dict["datasets"]
        with pytest.raises(ChartConfigError, match="datasets"):
            ChartConfig.from_dict(minimal_config_dict)

    def test_empty_datasets_raises(self, minimal_config_dict: dict) -> None:
        minimal_config_dict["datasets"] = []
        with pytest.raises(ChartConfigError, match="datasets"):
            ChartConfig.from_dict(minimal_config_dict)

    def test_labels_defaults_to_empty_list(self, minimal_config_dict: dict) -> None:
        del minimal_config_dict["labels"]
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert cfg.labels == []

    def test_labels_coerced_to_strings(self, minimal_config_dict: dict) -> None:
        minimal_config_dict["labels"] = [1, 2, 3]
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert cfg.labels == ["1", "2", "3"]

    def test_invalid_width_raises(self, minimal_config_dict: dict) -> None:
        minimal_config_dict["width"] = -10
        with pytest.raises(ChartConfigError, match="width"):
            ChartConfig.from_dict(minimal_config_dict)

    def test_invalid_height_raises(self, minimal_config_dict: dict) -> None:
        minimal_config_dict["height"] = 0
        with pytest.raises(ChartConfigError, match="height"):
            ChartConfig.from_dict(minimal_config_dict)

    def test_extra_options_not_dict_raises(self, minimal_config_dict: dict) -> None:
        minimal_config_dict["extra_options"] = "invalid"
        with pytest.raises(ChartConfigError, match="extra_options"):
            ChartConfig.from_dict(minimal_config_dict)

    def test_multiple_datasets(self, minimal_config_dict: dict) -> None:
        minimal_config_dict["datasets"] = [
            {"label": "A", "data": [1, 2, 3]},
            {"label": "B", "data": [4, 5, 6]},
        ]
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert len(cfg.datasets) == 2
        assert cfg.datasets[0].label == "A"
        assert cfg.datasets[1].label == "B"


# ---------------------------------------------------------------------------
# Tests: ChartConfig.to_chartjs_config
# ---------------------------------------------------------------------------


class TestChartConfigToChartjsConfig:
    def test_top_level_keys(self, minimal_config_dict: dict) -> None:
        cfg = ChartConfig.from_dict(minimal_config_dict)
        result = cfg.to_chartjs_config()
        assert "type" in result
        assert "data" in result
        assert "options" in result

    def test_type_field(self, minimal_config_dict: dict) -> None:
        cfg = ChartConfig.from_dict(minimal_config_dict)
        assert cfg.to_chartjs_config()["type"] == "bar"

    def test_data_labels_and_datasets(self, minimal_config_dict: dict) -> None:
        cfg = ChartConfig.from_dict(minimal_config_dict)
        result = cfg.to_chartjs_config()
        assert result["data"]["labels"] == ["Jan", "Feb", "Mar"]
        assert len(result["data"]["datasets"]) == 1

    def test_title_in_plugins(self, minimal_config_dict: dict) -> None:
        cfg = ChartConfig.from_dict(minimal_config_dict)
        plugins = cfg.to_chartjs_config()["options"]["plugins"]
        assert plugins["title"]["text"] == "Monthly Sales"
        assert plugins["title"]["display"] is True

    def test_legend_display(self, minimal_config_dict: dict) -> None:
        minimal_config_dict["show_legend"] = False
        cfg = ChartConfig.from_dict(minimal_config_dict)
        plugins = cfg.to_chartjs_config()["options"]["plugins"]
        assert plugins["legend"]["display"] is False

    def test_tooltip_enabled(self, minimal_config_dict: dict) -> None:
        minimal_config_dict["show_tooltips"] = True
        cfg = ChartConfig.from_dict(minimal_config_dict)
        plugins = cfg.to_chartjs_config()["options"]["plugins"]
        assert plugins["tooltip"]["enabled"] is True

    def test_axis_labels_for_cartesian(self, full_config_dict: dict) -> None:
        cfg = ChartConfig.from_dict(full_config_dict)
        options = cfg.to_chartjs_config()["options"]
        assert "scales" in options
        assert options["scales"]["x"]["title"]["text"] == "Quarter"
        assert options["scales"]["y"]["title"]["text"] == "Revenue (USD)"

    def test_no_scales_for_pie(self) -> None:
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

    def test_extra_options_merged(self, minimal_config_dict: dict) -> None:
        minimal_config_dict["extra_options"] = {"animation": {"duration": 500}}
        cfg = ChartConfig.from_dict(minimal_config_dict)
        options = cfg.to_chartjs_config()["options"]
        assert options["animation"]["duration"] == 500

    def test_extra_options_deep_merge(self, minimal_config_dict: dict) -> None:
        minimal_config_dict["extra_options"] = {
            "plugins": {"legend": {"position": "bottom"}}
        }
        cfg = ChartConfig.from_dict(minimal_config_dict)
        options = cfg.to_chartjs_config()["options"]
        # Deep merge: original keys preserved, new key added.
        assert options["plugins"]["legend"]["display"] is True
        assert options["plugins"]["legend"]["position"] == "bottom"


# ---------------------------------------------------------------------------
# Tests: ChartConfig.to_dict
# ---------------------------------------------------------------------------


class TestChartConfigToDict:
    def test_round_trip(self, minimal_config_dict: dict) -> None:
        cfg = ChartConfig.from_dict(minimal_config_dict)
        serialised = cfg.to_dict()
        cfg2 = ChartConfig.from_dict(serialised)
        assert cfg2.chart_type == cfg.chart_type
        assert cfg2.title == cfg.title
        assert cfg2.labels == cfg.labels

    def test_contains_expected_keys(self, minimal_config_dict: dict) -> None:
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


# ---------------------------------------------------------------------------
# Tests: ChartConfig properties
# ---------------------------------------------------------------------------


class TestChartConfigProperties:
    @pytest.mark.parametrize("chart_type", ["bar", "line", "scatter"])
    def test_is_cartesian_true(self, chart_type: str) -> None:
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


# ---------------------------------------------------------------------------
# Tests: _deep_merge utility
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_simple_merge(self) -> None:
        base = {"a": 1, "b": 2}
        _deep_merge(base, {"b": 99, "c": 3})
        assert base == {"a": 1, "b": 99, "c": 3}

    def test_nested_merge(self) -> None:
        base = {"plugins": {"legend": {"display": True}, "title": {"text": "T"}}}
        _deep_merge(base, {"plugins": {"legend": {"position": "top"}}})
        assert base["plugins"]["legend"]["display"] is True
        assert base["plugins"]["legend"]["position"] == "top"
        assert base["plugins"]["title"]["text"] == "T"

    def test_override_replaces_non_dict(self) -> None:
        base = {"key": "old"}
        _deep_merge(base, {"key": "new"})
        assert base["key"] == "new"

    def test_empty_override_no_change(self) -> None:
        base = {"a": 1}
        _deep_merge(base, {})
        assert base == {"a": 1}

    def test_new_nested_key_added(self) -> None:
        base = {"a": {"b": 1}}
        _deep_merge(base, {"a": {"c": 2}})
        assert base == {"a": {"b": 1, "c": 2}}
