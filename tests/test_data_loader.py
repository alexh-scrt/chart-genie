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
    """Path to the sales CSV fixture."""
    return FIXTURES_DIR / "sales.csv"


@pytest.fixture()
def temperatures_json_path() -> Path:
    """Path to the temperatures JSON fixture."""
    return FIXTURES_DIR / "temperatures.json"


@pytest.fixture()
def simple_csv_text() -> str:
    """A simple CSV string with name, age, score columns."""
    return "name,age,score\nAlice,30,9.5\nBob,25,8.0\nCarol,,7.5\n"


@pytest.fixture()
def simple_json_text() -> str:
    """A simple JSON array-of-objects string."""
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
        """Basic CSV parsing returns the expected number of records."""
        records = load_csv(simple_csv_text)
        assert len(records) == 3
        assert records[0]["name"] == "Alice"

    def test_numeric_coercion(self, simple_csv_text: str) -> None:
        """Numeric strings are coerced to int or float."""
        records = load_csv(simple_csv_text)
        assert records[0]["age"] == 30
        assert isinstance(records[0]["age"], int)
        assert records[0]["score"] == 9.5
        assert isinstance(records[0]["score"], float)

    def test_empty_field_becomes_none(self, simple_csv_text: str) -> None:
        """Empty CSV fields are converted to None."""
        records = load_csv(simple_csv_text)
        # Carol has no age
        assert records[2]["age"] is None

    def test_uniform_keys_across_rows(self, simple_csv_text: str) -> None:
        """All rows must have the same set of keys."""
        records = load_csv(simple_csv_text)
        keys_set = {frozenset(r.keys()) for r in records}
        assert len(keys_set) == 1, "All rows must have the same keys"

    def test_returns_list_of_dicts(self, simple_csv_text: str) -> None:
        """load_csv returns a list of dicts."""
        records = load_csv(simple_csv_text)
        assert isinstance(records, list)
        assert all(isinstance(r, dict) for r in records)

    def test_empty_csv_returns_empty_list(self) -> None:
        """CSV with only a header row returns an empty list."""
        records = load_csv("name,age\n")
        assert records == []

    def test_csv_with_only_header_returns_empty_list(self) -> None:
        """CSV with a three-column header and no data rows returns empty list."""
        records = load_csv("col1,col2,col3\n")
        assert records == []

    def test_boolean_values_coerced(self) -> None:
        """'true' and 'false' strings are coerced to Python booleans."""
        text = "item,active\nWidget,true\nGadget,false\n"
        records = load_csv(text)
        assert records[0]["active"] is True
        assert records[1]["active"] is False

    def test_null_string_values_coerced(self) -> None:
        """Common null-like strings are converted to None."""
        text = "item,value\nA,null\nB,N/A\nC,none\n"
        records = load_csv(text)
        assert all(r["value"] is None for r in records)

    def test_nan_string_becomes_none(self) -> None:
        """'NaN' string is converted to None."""
        text = "x,y\n1.0,NaN\n2.0,3.0\n"
        records = load_csv(text)
        assert records[0]["y"] is None

    def test_integer_strings_parsed(self) -> None:
        """Integer strings are parsed as int."""
        text = "a,b\n10,20\n30,40\n"
        records = load_csv(text)
        assert records[0]["a"] == 10
        assert isinstance(records[0]["a"], int)

    def test_float_strings_parsed(self) -> None:
        """Float strings are parsed as float."""
        text = "x\n1.5\n2.7\n"
        records = load_csv(text)
        assert records[0]["x"] == 1.5

    def test_whitespace_stripped_from_strings(self) -> None:
        """Leading/trailing whitespace is stripped from string values."""
        text = "name\n  Alice  \nBob\n"
        records = load_csv(text)
        assert records[0]["name"] == "Alice"

    def test_multicolumn(self) -> None:
        """Four-column CSV is parsed correctly."""
        text = "a,b,c,d\n1,2,3,4\n5,6,7,8\n"
        records = load_csv(text)
        assert records[0] == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_missing_columns_filled_with_none(self) -> None:
        """Rows with fewer columns than the header have missing keys as None."""
        # DictReader uses None as restval by default for missing trailing fields.
        text = "a,b,c\n1,2,3\n4,5\n"
        records = load_csv(text)
        # Second row is missing 'c'
        assert records[1]["c"] is None

    def test_yes_no_boolean(self) -> None:
        """'yes' and 'no' strings are coerced to True and False."""
        text = "flag\nyes\nno\n"
        records = load_csv(text)
        assert records[0]["flag"] is True
        assert records[1]["flag"] is False

    def test_on_off_boolean(self) -> None:
        """'on' and 'off' strings are coerced to True and False."""
        text = "enabled\non\noff\n"
        records = load_csv(text)
        assert records[0]["enabled"] is True
        assert records[1]["enabled"] is False

    def test_negative_integer_parsed(self) -> None:
        """Negative integer strings are parsed as negative ints."""
        text = "val\n-5\n-100\n"
        records = load_csv(text)
        assert records[0]["val"] == -5
        assert isinstance(records[0]["val"], int)

    def test_multiple_rows(self) -> None:
        """CSV with multiple rows returns all rows."""
        text = "x\n1\n2\n3\n4\n5\n"
        records = load_csv(text)
        assert len(records) == 5

    def test_nil_string_becomes_none(self) -> None:
        """'nil' string is converted to None."""
        text = "a\nnil\n1\n"
        records = load_csv(text)
        assert records[0]["a"] is None


# ---------------------------------------------------------------------------
# Tests: load_json
# ---------------------------------------------------------------------------


class TestLoadJson:
    def test_array_of_objects(self, simple_json_text: str) -> None:
        """Basic JSON array-of-objects is parsed correctly."""
        records = load_json(simple_json_text)
        assert len(records) == 3
        assert records[0]["name"] == "Alice"

    def test_none_values_preserved(self, simple_json_text: str) -> None:
        """JSON null values are preserved as Python None."""
        records = load_json(simple_json_text)
        assert records[2]["age"] is None

    def test_numeric_values_preserved(self, simple_json_text: str) -> None:
        """Numeric values from JSON are preserved as int or float."""
        records = load_json(simple_json_text)
        assert records[0]["age"] == 30
        assert records[0]["score"] == 9.5

    def test_boolean_values_preserved(self) -> None:
        """JSON boolean values are preserved as Python booleans."""
        data = json.dumps([{"active": True}, {"active": False}])
        records = load_json(data)
        assert records[0]["active"] is True
        assert records[1]["active"] is False

    def test_wrapped_in_data_key(self) -> None:
        """JSON object with a 'data' key wrapping the array is unwrapped."""
        data = json.dumps({"data": [{"x": 1}, {"x": 2}]})
        records = load_json(data)
        assert len(records) == 2

    def test_wrapped_in_records_key(self) -> None:
        """JSON object with a 'records' key wrapping the array is unwrapped."""
        data = json.dumps({"records": [{"a": 1}]})
        records = load_json(data)
        assert records[0]["a"] == 1

    def test_single_list_key_unwrapped(self) -> None:
        """JSON object with a single list-valued key is automatically unwrapped."""
        data = json.dumps({"items": [{"v": 10}, {"v": 20}]})
        records = load_json(data)
        assert len(records) == 2
        assert records[0]["v"] == 10

    def test_empty_array(self) -> None:
        """Empty JSON array returns an empty list."""
        records = load_json("[]")
        assert records == []

    def test_invalid_json_raises(self) -> None:
        """Invalid JSON raises DataLoadError."""
        with pytest.raises(DataLoadError, match="Failed to parse JSON"):
            load_json("{not valid json")

    def test_non_list_non_dict_raises(self) -> None:
        """A bare JSON string raises DataLoadError."""
        with pytest.raises(DataLoadError, match="Expected a JSON array"):
            load_json('"just a string"')

    def test_array_of_non_dicts_raises(self) -> None:
        """A JSON array of plain numbers raises DataLoadError."""
        with pytest.raises(DataLoadError):
            load_json(json.dumps([1, 2, 3]))

    def test_array_of_arrays(self) -> None:
        """JSON array-of-arrays uses the first row as header."""
        data = json.dumps([["name", "score"], ["Alice", 90], ["Bob", 85]])
        records = load_json(data)
        assert len(records) == 2
        assert records[0]["name"] == "Alice"
        assert records[0]["score"] == 90

    def test_array_of_arrays_missing_values(self) -> None:
        """Rows shorter than the header get None for missing fields."""
        data = json.dumps([["a", "b", "c"], [1, 2], [3, 4, 5]])
        records = load_json(data)
        assert records[0]["c"] is None
        assert records[1]["c"] == 5

    def test_uniform_keys_across_rows(self) -> None:
        """All rows in parsed JSON have the same set of keys."""
        data = json.dumps([{"a": 1, "b": 2}, {"a": 3}, {"b": 4, "c": 5}])
        records = load_json(data)
        key_sets = {frozenset(r.keys()) for r in records}
        assert len(key_sets) == 1

    def test_missing_keys_filled_with_none(self) -> None:
        """Missing keys in some rows are filled with None."""
        data = json.dumps([{"a": 1, "b": 2}, {"a": 3}])
        records = load_json(data)
        assert records[1]["b"] is None

    def test_wrapped_in_results_key(self) -> None:
        """JSON object with a 'results' key wrapping the array is unwrapped."""
        data = json.dumps({"results": [{"z": 99}]})
        records = load_json(data)
        assert records[0]["z"] == 99

    def test_wrapped_in_rows_key(self) -> None:
        """JSON object with a 'rows' key wrapping the array is unwrapped."""
        data = json.dumps({"rows": [{"k": 42}]})
        records = load_json(data)
        assert records[0]["k"] == 42

    def test_array_of_arrays_only_header_raises(self) -> None:
        """Array-of-arrays with only a header row raises DataLoadError."""
        data = json.dumps([["a", "b"]])
        with pytest.raises(DataLoadError):
            load_json(data)


# ---------------------------------------------------------------------------
# Tests: load_data with file paths
# ---------------------------------------------------------------------------


class TestLoadDataFiles:
    def test_loads_csv_fixture(self, sales_csv_path: Path) -> None:
        """The sales CSV fixture loads all 12 records."""
        records = load_data(sales_csv_path)
        assert len(records) == 12
        assert records[0]["month"] == "January"

    def test_loads_json_fixture(self, temperatures_json_path: Path) -> None:
        """The temperatures JSON fixture loads all 24 records (12 months x 2 cities)."""
        records = load_data(temperatures_json_path)
        assert len(records) == 24
        assert records[0]["city"] == "New York"

    def test_csv_sales_numeric_fields(self, sales_csv_path: Path) -> None:
        """Numeric fields in the CSV fixture are parsed as int."""
        records = load_data(sales_csv_path)
        assert isinstance(records[0]["sales"], int)
        assert isinstance(records[0]["returns"], int)

    def test_csv_sales_boolean_field(self, sales_csv_path: Path) -> None:
        """Boolean fields in the CSV fixture are parsed as bool."""
        records = load_data(sales_csv_path)
        assert records[0]["active"] is True
        assert records[6]["active"] is False

    def test_json_temperatures_numeric_fields(self, temperatures_json_path: Path) -> None:
        """Numeric fields in the JSON fixture are parsed as float."""
        records = load_data(temperatures_json_path)
        assert isinstance(records[0]["high_c"], float)
        assert isinstance(records[0]["precipitation_mm"], float)

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        """Files with unsupported extensions raise UnsupportedFormatError."""
        f = tmp_path / "data.xlsx"
        f.write_text("dummy")
        with pytest.raises(UnsupportedFormatError, match="Unsupported file extension"):
            load_data(f)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """A non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_data(tmp_path / "nonexistent.csv")

    def test_string_path_accepted(self, sales_csv_path: Path) -> None:
        """load_data accepts a string path as well as a Path object."""
        records = load_data(str(sales_csv_path))
        assert len(records) == 12

    def test_tmp_csv_file(self, tmp_path: Path) -> None:
        """A temporary CSV file is loaded and parsed correctly."""
        f = tmp_path / "test.csv"
        f.write_text("x,y\n1,2\n3,4\n")
        records = load_data(f)
        assert len(records) == 2
        assert records[0] == {"x": 1, "y": 2}

    def test_tmp_json_file(self, tmp_path: Path) -> None:
        """A temporary JSON file is loaded and parsed correctly."""
        f = tmp_path / "test.json"
        f.write_text(json.dumps([{"a": 10}, {"a": 20}]))
        records = load_data(f)
        assert len(records) == 2
        assert records[0]["a"] == 10

    def test_csv_sales_all_months_present(self, sales_csv_path: Path) -> None:
        """All 12 months appear in the sales CSV fixture."""
        records = load_data(sales_csv_path)
        months = [r["month"] for r in records]
        assert "January" in months
        assert "December" in months
        assert len(months) == 12

    def test_json_temperatures_both_cities(self, temperatures_json_path: Path) -> None:
        """Both New York and Los Angeles appear in the temperature fixture."""
        records = load_data(temperatures_json_path)
        cities = {r["city"] for r in records}
        assert "New York" in cities
        assert "Los Angeles" in cities

    def test_tmp_csv_with_utf8_bom(self, tmp_path: Path) -> None:
        """CSV files with UTF-8 BOM are loaded correctly."""
        f = tmp_path / "bom.csv"
        # Write with UTF-8 BOM
        f.write_bytes(b"\xef\xbb\xbfname,value\nAlice,1\n")
        records = load_data(f)
        assert len(records) == 1
        assert "name" in records[0]
        assert records[0]["name"] == "Alice"


# ---------------------------------------------------------------------------
# Tests: infer_column_types
# ---------------------------------------------------------------------------


class TestInferColumnTypes:
    def test_number_columns(self) -> None:
        """Numeric columns are inferred as 'number'."""
        records = load_csv("a,b\n1,2.5\n3,4.0\n")
        types = infer_column_types(records)
        assert types["a"] == "number"
        assert types["b"] == "number"

    def test_string_column(self) -> None:
        """String columns are inferred as 'string'."""
        records = load_csv("name\nAlice\nBob\n")
        types = infer_column_types(records)
        assert types["name"] == "string"

    def test_boolean_column(self) -> None:
        """Boolean columns are inferred as 'boolean'."""
        records = load_csv("flag\ntrue\nfalse\ntrue\n")
        types = infer_column_types(records)
        assert types["flag"] == "boolean"

    def test_all_null_column_inferred_as_string(self) -> None:
        """Columns with only null values are reported as 'string'."""
        records = load_csv("x,y\n1,\n2,\n")
        types = infer_column_types(records)
        assert types["y"] == "string"

    def test_mixed_type_column_inferred_as_string(self) -> None:
        """Columns with mixed numeric and string values are 'string'."""
        records = [{"col": 1}, {"col": "text"}, {"col": 3}]
        types = infer_column_types(records)
        assert types["col"] == "string"

    def test_empty_records_returns_empty(self) -> None:
        """infer_column_types returns an empty dict for empty input."""
        assert infer_column_types([]) == {}

    def test_sales_fixture_types(self, sales_csv_path: Path) -> None:
        """Sales CSV fixture has the expected column types."""
        records = load_data(sales_csv_path)
        types = infer_column_types(records)
        assert types["month"] == "string"
        assert types["sales"] == "number"
        assert types["returns"] == "number"
        assert types["active"] == "boolean"

    def test_temperature_fixture_types(self, temperatures_json_path: Path) -> None:
        """Temperature JSON fixture has the expected column types."""
        records = load_data(temperatures_json_path)
        types = infer_column_types(records)
        assert types["city"] == "string"
        assert types["month"] == "string"
        assert types["high_c"] == "number"
        assert types["low_c"] == "number"
        assert types["precipitation_mm"] == "number"

    def test_integer_column_is_number(self) -> None:
        """Pure integer columns are reported as 'number'."""
        records = [{"n": 1}, {"n": 2}, {"n": 3}]
        types = infer_column_types(records)
        assert types["n"] == "number"

    def test_mixed_int_float_is_number(self) -> None:
        """Columns mixing ints and floats are reported as 'number'."""
        records = [{"v": 1}, {"v": 2.5}, {"v": 3}]
        types = infer_column_types(records)
        assert types["v"] == "number"

    def test_boolean_with_nulls_still_boolean(self) -> None:
        """Boolean columns with some None values are still reported as 'boolean'."""
        records = [{"f": True}, {"f": None}, {"f": False}]
        types = infer_column_types(records)
        assert types["f"] == "boolean"

    def test_number_with_nulls_still_number(self) -> None:
        """Numeric columns with some None values are still reported as 'number'."""
        records = [{"x": 1.0}, {"x": None}, {"x": 3.0}]
        types = infer_column_types(records)
        assert types["x"] == "number"


# ---------------------------------------------------------------------------
# Tests: get_column_names
# ---------------------------------------------------------------------------


class TestGetColumnNames:
    def test_returns_ordered_list(self) -> None:
        """Column names are returned in header order."""
        records = load_csv("a,b,c\n1,2,3\n")
        assert get_column_names(records) == ["a", "b", "c"]

    def test_empty_records(self) -> None:
        """Empty input returns an empty list."""
        assert get_column_names([]) == []

    def test_union_of_all_row_keys(self) -> None:
        """Column names are the union of all keys across all rows."""
        records = [{"a": 1}, {"b": 2}, {"a": 3, "c": 4}]
        cols = get_column_names(records)
        assert set(cols) == {"a", "b", "c"}

    def test_no_duplicate_keys(self) -> None:
        """No duplicate column names are returned."""
        records = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
        cols = get_column_names(records)
        assert len(cols) == len(set(cols))

    def test_order_by_first_encounter(self) -> None:
        """Column names are ordered by first encounter."""
        records = [{"z": 1, "a": 2}, {"m": 3}]
        cols = get_column_names(records)
        # z and a come from first row, m from second
        assert cols.index("z") < cols.index("m")
        assert cols.index("a") < cols.index("m")


# ---------------------------------------------------------------------------
# Tests: get_column_values
# ---------------------------------------------------------------------------


class TestGetColumnValues:
    def test_basic_extraction(self) -> None:
        """Values for a given column are extracted in record order."""
        records = load_csv("x,y\n1,4\n2,5\n3,6\n")
        assert get_column_values(records, "x") == [1, 2, 3]

    def test_none_for_missing_in_row(self) -> None:
        """Missing values in a row are returned as None."""
        normalized = [{"a": 1, "b": 2}, {"a": 3, "b": None}]
        vals = get_column_values(normalized, "b")
        assert vals == [2, None]

    def test_missing_column_raises(self) -> None:
        """Requesting a non-existent column raises DataLoadError."""
        records = load_csv("x\n1\n2\n")
        with pytest.raises(DataLoadError, match="not found"):
            get_column_values(records, "nonexistent")

    def test_empty_records_raises(self) -> None:
        """Requesting a column from empty records raises DataLoadError."""
        with pytest.raises(DataLoadError):
            get_column_values([], "col")

    def test_all_values_returned_in_order(self) -> None:
        """All column values are returned in row order."""
        records = load_csv("n\n10\n20\n30\n40\n")
        vals = get_column_values(records, "n")
        assert vals == [10, 20, 30, 40]

    def test_string_column_values(self) -> None:
        """String column values are returned correctly."""
        records = load_csv("name\nAlice\nBob\nCarol\n")
        vals = get_column_values(records, "name")
        assert vals == ["Alice", "Bob", "Carol"]


# ---------------------------------------------------------------------------
# Tests: _coerce_value internals
# ---------------------------------------------------------------------------


class TestCoerceValue:
    @pytest.mark.parametrize("val", [None])
    def test_none_passthrough(self, val: None) -> None:
        """None is returned as-is."""
        assert _coerce_value(val) is None

    @pytest.mark.parametrize("val,expected", [
        (True, True),
        (False, False),
        (42, 42),
        (3.14, 3.14),
    ])
    def test_native_types_passthrough(self, val: object, expected: object) -> None:
        """Native Python types are returned unchanged."""
        assert _coerce_value(val) == expected

    @pytest.mark.parametrize("s", ["null", "None", "NULL", "n/a", "N/A", "nan", "", "NaN", "nil"])
    def test_null_strings(self, s: str) -> None:
        """Common null-like strings are coerced to None."""
        assert _coerce_value(s) is None

    @pytest.mark.parametrize("s", ["true", "True", "TRUE", "yes", "YES", "1", "on", "ON"])
    def test_true_strings(self, s: str) -> None:
        """Common truthy strings are coerced to True."""
        assert _coerce_value(s) is True

    @pytest.mark.parametrize("s", ["false", "False", "FALSE", "no", "NO", "0", "off", "OFF"])
    def test_false_strings(self, s: str) -> None:
        """Common falsy strings are coerced to False."""
        assert _coerce_value(s) is False

    @pytest.mark.parametrize("s,expected", [
        ("42", 42),
        ("-10", -10),
        ("+5", 5),
        ("1000", 1000),
    ])
    def test_integer_strings(self, s: str, expected: int) -> None:
        """Integer-like strings are coerced to int."""
        result = _coerce_value(s)
        assert result == expected
        assert isinstance(result, int)

    @pytest.mark.parametrize("s,expected", [
        ("3.14", 3.14),
        ("-0.5", -0.5),
        ("1.0e3", 1000.0),
    ])
    def test_float_strings(self, s: str, expected: float) -> None:
        """Float-like strings are coerced to float."""
        result = _coerce_value(s)
        assert result == pytest.approx(expected)
        assert isinstance(result, float)

    def test_plain_string_returned_stripped(self) -> None:
        """Plain strings are returned with leading/trailing whitespace stripped."""
        assert _coerce_value("  hello world  ") == "hello world"

    def test_non_numeric_string_returned(self) -> None:
        """Non-numeric strings are returned as-is (stripped)."""
        assert _coerce_value("January") == "January"

    def test_bool_true_not_treated_as_int(self) -> None:
        """Python True is returned as bool, not converted to 1."""
        result = _coerce_value(True)
        assert result is True
        assert isinstance(result, bool)

    def test_bool_false_not_treated_as_int(self) -> None:
        """Python False is returned as bool, not converted to 0."""
        result = _coerce_value(False)
        assert result is False
        assert isinstance(result, bool)

    def test_zero_integer_passthrough(self) -> None:
        """Integer 0 is returned as int 0."""
        result = _coerce_value(0)
        assert result == 0
        assert isinstance(result, int)

    def test_float_zero_passthrough(self) -> None:
        """Float 0.0 is returned as float 0.0."""
        result = _coerce_value(0.0)
        assert result == 0.0
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Tests: _is_integer_string
# ---------------------------------------------------------------------------


class TestIsIntegerString:
    @pytest.mark.parametrize("s", ["0", "42", "-10", "+5", "1_000", "1,000"])
    def test_valid(self, s: str) -> None:
        """Valid integer strings return True."""
        assert _is_integer_string(s) is True

    @pytest.mark.parametrize("s", ["3.14", "1.0", "abc", "", "1e5"])
    def test_invalid(self, s: str) -> None:
        """Non-integer strings return False."""
        assert _is_integer_string(s) is False

    def test_plain_zero(self) -> None:
        """'0' is a valid integer string."""
        assert _is_integer_string("0") is True

    def test_large_integer(self) -> None:
        """Large integer strings are valid."""
        assert _is_integer_string("999999999999") is True

    def test_decimal_point_is_invalid(self) -> None:
        """A decimal point makes the string non-integer."""
        assert _is_integer_string("1.") is False

    def test_whitespace_only_is_invalid(self) -> None:
        """A whitespace-only string is not a valid integer."""
        assert _is_integer_string("   ") is False
