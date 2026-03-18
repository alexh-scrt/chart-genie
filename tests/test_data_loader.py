"""Unit tests for chart_genie.data_loader.

Covers CSV and JSON parsing, type inference, null handling, edge cases,
and the public helper functions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chart_genie.data_loader import (
    DataLoadError,
    UnsupportedFormatError,
    _coerce_value,
    _is_integer_string,
    get_column_names,
    get_column_values,
    infer_column_types,
    load_csv,
    load_data,
    load_json,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
 def sales_csv_path() -> Path:
    return FIXTURES_DIR / "sales.csv"


@pytest.fixture()
def temperatures_json_path() -> Path:
    return FIXTURES_DIR / "temperatures.json"


@pytest.fixture()
def simple_csv_text() -> str:
    return "name,age,score\nAlice,30,9.5\nBob,25,8.0\nCarol,,7.5\n"


@pytest.fixture()
def simple_json_text() -> str:
    return json.dumps([
        {"name": "Alice", "age": 30, "score": 9.5},
        {"name": "Bob", "age": 25, "score": 8.0},
        {"name": "Carol", "age": None, "score": 7.5},
    ])


# ---------------------------------------------------------------------------
# Tests: load_csv
# ---------------------------------------------------------------------------


class TestLoadCsv:
    def test_basic_parsing(self, simple_csv_text: str) -> None:
        records = load_csv(simple_csv_text)
        assert len(records) == 3
        assert records[0]["name"] == "Alice"

    def test_numeric_coercion(self, simple_csv_text: str) -> None:
        records = load_csv(simple_csv_text)
        assert records[0]["age"] == 30
        assert isinstance(records[0]["age"], int)
        assert records[0]["score"] == 9.5
        assert isinstance(records[0]["score"], float)

    def test_empty_field_becomes_none(self, simple_csv_text: str) -> None:
        records = load_csv(simple_csv_text)
        # Carol has no age
        assert records[2]["age"] is None

    def test_uniform_keys_across_rows(self, simple_csv_text: str) -> None:
        records = load_csv(simple_csv_text)
        keys_set = {frozenset(r.keys()) for r in records}
        assert len(keys_set) == 1, "All rows must have the same keys"

    def test_returns_list_of_dicts(self, simple_csv_text: str) -> None:
        records = load_csv(simple_csv_text)
        assert isinstance(records, list)
        assert all(isinstance(r, dict) for r in records)

    def test_empty_csv_returns_empty_list(self) -> None:
        records = load_csv("name,age\n")
        assert records == []

    def test_csv_with_only_header_returns_empty_list(self) -> None:
        records = load_csv("col1,col2,col3\n")
        assert records == []

    def test_boolean_values_coerced(self) -> None:
        text = "item,active\nWidget,true\nGadget,false\n"
        records = load_csv(text)
        assert records[0]["active"] is True
        assert records[1]["active"] is False

    def test_null_string_values_coerced(self) -> None:
        text = "item,value\nA,null\nB,N/A\nC,none\n"
        records = load_csv(text)
        assert all(r["value"] is None for r in records)

    def test_nan_string_becomes_none(self) -> None:
        text = "x,y\n1.0,NaN\n2.0,3.0\n"
        records = load_csv(text)
        assert records[0]["y"] is None

    def test_integer_strings_parsed(self) -> None:
        text = "a,b\n10,20\n30,40\n"
        records = load_csv(text)
        assert records[0]["a"] == 10
        assert isinstance(records[0]["a"], int)

    def test_float_strings_parsed(self) -> None:
        text = "x\n1.5\n2.7\n"
        records = load_csv(text)
        assert records[0]["x"] == 1.5

    def test_whitespace_stripped_from_strings(self) -> None:
        text = "name\n  Alice  \nBob\n"
        records = load_csv(text)
        assert records[0]["name"] == "Alice"

    def test_multicolumn(self) -> None:
        text = "a,b,c,d\n1,2,3,4\n5,6,7,8\n"
        records = load_csv(text)
        assert records[0] == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_missing_columns_filled_with_none(self) -> None:
        # Simulate a row with fewer columns than the header by using csv
        # extra values — DictReader handles this with restval.
        text = "a,b,c\n1,2,3\n4,5\n"
        records = load_csv(text)
        # Second row is missing 'c' — DictReader uses None as restval by default
        assert records[1]["c"] is None

    def test_yes_no_boolean(self) -> None:
        text = "flag\nyes\nno\n"
        records = load_csv(text)
        assert records[0]["flag"] is True
        assert records[1]["flag"] is False

    def test_on_off_boolean(self) -> None:
        text = "enabled\non\noff\n"
        records = load_csv(text)
        assert records[0]["enabled"] is True
        assert records[1]["enabled"] is False


# ---------------------------------------------------------------------------
# Tests: load_json
# ---------------------------------------------------------------------------


class TestLoadJson:
    def test_array_of_objects(self, simple_json_text: str) -> None:
        records = load_json(simple_json_text)
        assert len(records) == 3
        assert records[0]["name"] == "Alice"

    def test_none_values_preserved(self, simple_json_text: str) -> None:
        records = load_json(simple_json_text)
        assert records[2]["age"] is None

    def test_numeric_values_preserved(self, simple_json_text: str) -> None:
        records = load_json(simple_json_text)
        assert records[0]["age"] == 30
        assert records[0]["score"] == 9.5

    def test_boolean_values_preserved(self) -> None:
        data = json.dumps([{"active": True}, {"active": False}])
        records = load_json(data)
        assert records[0]["active"] is True
        assert records[1]["active"] is False

    def test_wrapped_in_data_key(self) -> None:
        data = json.dumps({"data": [{"x": 1}, {"x": 2}]})
        records = load_json(data)
        assert len(records) == 2

    def test_wrapped_in_records_key(self) -> None:
        data = json.dumps({"records": [{"a": 1}]})
        records = load_json(data)
        assert records[0]["a"] == 1

    def test_single_list_key_unwrapped(self) -> None:
        data = json.dumps({"items": [{"v": 10}, {"v": 20}]})
        records = load_json(data)
        assert len(records) == 2
        assert records[0]["v"] == 10

    def test_empty_array(self) -> None:
        records = load_json("[]")
        assert records == []

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(DataLoadError, match="Failed to parse JSON"):
            load_json("{not valid json")

    def test_non_list_non_dict_raises(self) -> None:
        with pytest.raises(DataLoadError, match="Expected a JSON array"):
            load_json('"just a string"')

    def test_array_of_non_dicts_raises(self) -> None:
        with pytest.raises(DataLoadError):
            load_json(json.dumps([1, 2, 3]))

    def test_array_of_arrays(self) -> None:
        data = json.dumps([["name", "score"], ["Alice", 90], ["Bob", 85]])
        records = load_json(data)
        assert len(records) == 2
        assert records[0]["name"] == "Alice"
        assert records[0]["score"] == 90

    def test_array_of_arrays_missing_values(self) -> None:
        data = json.dumps([["a", "b", "c"], [1, 2], [3, 4, 5]])
        records = load_json(data)
        assert records[0]["c"] is None
        assert records[1]["c"] == 5

    def test_uniform_keys_across_rows(self) -> None:
        data = json.dumps([{"a": 1, "b": 2}, {"a": 3}, {"b": 4, "c": 5}])
        records = load_json(data)
        key_sets = {frozenset(r.keys()) for r in records}
        assert len(key_sets) == 1

    def test_missing_keys_filled_with_none(self) -> None:
        data = json.dumps([{"a": 1, "b": 2}, {"a": 3}])
        records = load_json(data)
        assert records[1]["b"] is None


# ---------------------------------------------------------------------------
# Tests: load_data with file paths
# ---------------------------------------------------------------------------


class TestLoadDataFiles:
    def test_loads_csv_fixture(self, sales_csv_path: Path) -> None:
        records = load_data(sales_csv_path)
        assert len(records) == 12
        assert records[0]["month"] == "January"

    def test_loads_json_fixture(self, temperatures_json_path: Path) -> None:
        records = load_data(temperatures_json_path)
        assert len(records) == 24  # 12 months * 2 cities
        assert records[0]["city"] == "New York"

    def test_csv_sales_numeric_fields(self, sales_csv_path: Path) -> None:
        records = load_data(sales_csv_path)
        assert isinstance(records[0]["sales"], int)
        assert isinstance(records[0]["returns"], int)

    def test_csv_sales_boolean_field(self, sales_csv_path: Path) -> None:
        records = load_data(sales_csv_path)
        assert records[0]["active"] is True
        assert records[6]["active"] is False

    def test_json_temperatures_numeric_fields(self, temperatures_json_path: Path) -> None:
        records = load_data(temperatures_json_path)
        assert isinstance(records[0]["high_c"], float)
        assert isinstance(records[0]["precipitation_mm"], float)

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "data.xlsx"
        f.write_text("dummy")
        with pytest.raises(UnsupportedFormatError, match="Unsupported file extension"):
            load_data(f)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_data(tmp_path / "nonexistent.csv")

    def test_string_path_accepted(self, sales_csv_path: Path) -> None:
        records = load_data(str(sales_csv_path))
        assert len(records) == 12

    def test_tmp_csv_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.csv"
        f.write_text("x,y\n1,2\n3,4\n")
        records = load_data(f)
        assert len(records) == 2
        assert records[0] == {"x": 1, "y": 2}

    def test_tmp_json_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.json"
        f.write_text(json.dumps([{"a": 10}, {"a": 20}]))
        records = load_data(f)
        assert len(records) == 2
        assert records[0]["a"] == 10


# ---------------------------------------------------------------------------
# Tests: infer_column_types
# ---------------------------------------------------------------------------


class TestInferColumnTypes:
    def test_number_columns(self) -> None:
        records = load_csv("a,b\n1,2.5\n3,4.0\n")
        types = infer_column_types(records)
        assert types["a"] == "number"
        assert types["b"] == "number"

    def test_string_column(self) -> None:
        records = load_csv("name\nAlice\nBob\n")
        types = infer_column_types(records)
        assert types["name"] == "string"

    def test_boolean_column(self) -> None:
        records = load_csv("flag\ntrue\nfalse\ntrue\n")
        types = infer_column_types(records)
        assert types["flag"] == "boolean"

    def test_all_null_column_inferred_as_string(self) -> None:
        records = load_csv("x,y\n1,\n2,\n")
        types = infer_column_types(records)
        assert types["y"] == "string"

    def test_mixed_type_column_inferred_as_string(self) -> None:
        records = [{"col": 1}, {"col": "text"}, {"col": 3}]
        types = infer_column_types(records)
        assert types["col"] == "string"

    def test_empty_records_returns_empty(self) -> None:
        assert infer_column_types([]) == {}

    def test_sales_fixture_types(self, sales_csv_path: Path) -> None:
        records = load_data(sales_csv_path)
        types = infer_column_types(records)
        assert types["month"] == "string"
        assert types["sales"] == "number"
        assert types["returns"] == "number"
        assert types["active"] == "boolean"

    def test_temperature_fixture_types(self, temperatures_json_path: Path) -> None:
        records = load_data(temperatures_json_path)
        types = infer_column_types(records)
        assert types["city"] == "string"
        assert types["month"] == "string"
        assert types["high_c"] == "number"
        assert types["low_c"] == "number"
        assert types["precipitation_mm"] == "number"


# ---------------------------------------------------------------------------
# Tests: get_column_names
# ---------------------------------------------------------------------------


class TestGetColumnNames:
    def test_returns_ordered_list(self) -> None:
        records = load_csv("a,b,c\n1,2,3\n")
        assert get_column_names(records) == ["a", "b", "c"]

    def test_empty_records(self) -> None:
        assert get_column_names([]) == []

    def test_union_of_all_row_keys(self) -> None:
        records = [{"a": 1}, {"b": 2}, {"a": 3, "c": 4}]
        cols = get_column_names(records)
        assert set(cols) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# Tests: get_column_values
# ---------------------------------------------------------------------------


class TestGetColumnValues:
    def test_basic_extraction(self) -> None:
        records = load_csv("x,y\n1,4\n2,5\n3,6\n")
        assert get_column_values(records, "x") == [1, 2, 3]

    def test_none_for_missing_in_row(self) -> None:
        records = [{"a": 1, "b": 2}, {"a": 3}]
        # b is missing in second row after normalization; load_json normalizes it
        # but direct list won't — use normalized structure:
        normalized = [{"a": 1, "b": 2}, {"a": 3, "b": None}]
        vals = get_column_values(normalized, "b")
        assert vals == [2, None]

    def test_missing_column_raises(self) -> None:
        records = load_csv("x\n1\n2\n")
        with pytest.raises(DataLoadError, match="not found"):
            get_column_values(records, "nonexistent")

    def test_empty_records(self) -> None:
        with pytest.raises(DataLoadError):
            get_column_values([], "col")


# ---------------------------------------------------------------------------
# Tests: _coerce_value internals
# ---------------------------------------------------------------------------


class TestCoerceValue:
    @pytest.mark.parametrize("val", [None])
    def test_none_passthrough(self, val: None) -> None:
        assert _coerce_value(val) is None

    @pytest.mark.parametrize("val,expected", [
        (True, True),
        (False, False),
        (42, 42),
        (3.14, 3.14),
    ])
    def test_native_types_passthrough(self, val, expected) -> None:
        assert _coerce_value(val) == expected

    @pytest.mark.parametrize("s", ["null", "None", "NULL", "n/a", "N/A", "nan", "", "NaN", "nil"])
    def test_null_strings(self, s: str) -> None:
        assert _coerce_value(s) is None

    @pytest.mark.parametrize("s", ["true", "True", "TRUE", "yes", "YES", "1", "on", "ON"])
    def test_true_strings(self, s: str) -> None:
        assert _coerce_value(s) is True

    @pytest.mark.parametrize("s", ["false", "False", "FALSE", "no", "NO", "0", "off", "OFF"])
    def test_false_strings(self, s: str) -> None:
        assert _coerce_value(s) is False

    @pytest.mark.parametrize("s,expected", [
        ("42", 42),
        ("-10", -10),
        ("+5", 5),
        ("1000", 1000),
    ])
    def test_integer_strings(self, s: str, expected: int) -> None:
        result = _coerce_value(s)
        assert result == expected
        assert isinstance(result, int)

    @pytest.mark.parametrize("s,expected", [
        ("3.14", 3.14),
        ("-0.5", -0.5),
        ("1.0e3", 1000.0),
    ])
    def test_float_strings(self, s: str, expected: float) -> None:
        result = _coerce_value(s)
        assert result == pytest.approx(expected)
        assert isinstance(result, float)

    def test_plain_string_returned_stripped(self) -> None:
        assert _coerce_value("  hello world  ") == "hello world"

    def test_non_numeric_string_returned(self) -> None:
        assert _coerce_value("January") == "January"


# ---------------------------------------------------------------------------
# Tests: _is_integer_string
# ---------------------------------------------------------------------------


class TestIsIntegerString:
    @pytest.mark.parametrize("s", ["0", "42", "-10", "+5", "1_000", "1,000"])
    def test_valid(self, s: str) -> None:
        assert _is_integer_string(s) is True

    @pytest.mark.parametrize("s", ["3.14", "1.0", "abc", "", "1e5"])
    def test_invalid(self, s: str) -> None:
        assert _is_integer_string(s) is False
