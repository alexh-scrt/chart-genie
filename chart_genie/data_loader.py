"""Data loader for chart_genie: parses CSV and JSON into a normalized list-of-dicts.

This module handles reading raw data from CSV files, JSON files, or stdin,
normalizing it into a consistent list-of-dicts structure, and performing
automatic column type inference (numeric, boolean, or string) with null
handling for missing or empty values.

Typical usage::

    from chart_genie.data_loader import load_data, infer_column_types

    # Load from a file path
    records = load_data("sales.csv")

    # Load from stdin (pass None or "-" as path)
    records = load_data(None)   # reads sys.stdin

    # Inspect inferred column types
    types = infer_column_types(records)
    # e.g. {"month": "string", "sales": "number", "active": "boolean"}

The returned records list always has a uniform set of keys across all rows.
Missing values are represented as None.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------

#: A single data record — a mapping of column name to scalar value or None.
Record = dict[str, Any]

#: A mapping of column name to inferred type string.
ColumnTypes = dict[str, str]

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DataLoadError(ValueError):
    """Raised when input data cannot be read or parsed."""


class UnsupportedFormatError(DataLoadError):
    """Raised when the file extension is not recognised."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Values treated as boolean True (case-insensitive).
_TRUE_VALUES: frozenset[str] = frozenset({"true", "yes", "1", "on"})

#: Values treated as boolean False (case-insensitive).
_FALSE_VALUES: frozenset[str] = frozenset({"false", "no", "0", "off"})

#: Values treated as null / missing.
_NULL_VALUES: frozenset[str] = frozenset({"null", "none", "na", "n/a", "nan", "", "nil"})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_data(source: str | Path | None) -> list[Record]:
    """Load data from a CSV or JSON file, or from stdin.

    The source is identified by its file extension (`.csv` or `.json`).  If
    *source* is ``None`` or the string ``"-"``, data is read from stdin.
    Stdin data is detected as JSON if the first non-whitespace character is
    ``[`` or ``{``; otherwise it is treated as CSV.

    Args:
        source: Path to a CSV or JSON file, ``None``, or ``"-"`` to read
            from stdin.

    Returns:
        A list of records (list of dicts) with uniform keys across all rows.
        Missing values are represented as ``None``.  Numeric strings are
        coerced to ``int`` or ``float``; boolean strings are coerced to
        ``bool``.

    Raises:
        DataLoadError: If the file cannot be read or its content cannot be
            parsed.
        UnsupportedFormatError: If the file extension is not ``.csv`` or
            ``.json``.
        FileNotFoundError: If the specified file does not exist.
    """
    if source is None or str(source) == "-":
        return _load_stdin()

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: '{path}'")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv_file(path)
    elif suffix == ".json":
        return _load_json_file(path)
    else:
        raise UnsupportedFormatError(
            f"Unsupported file extension '{suffix}'. "
            "Supported formats are .csv and .json."
        )


def load_csv(text: str) -> list[Record]:
    """Parse a CSV string into a normalized list of records.

    The first row is treated as the header.  Empty values and common null
    representations are converted to ``None``.  Numeric and boolean strings
    are coerced to their native Python types.

    Args:
        text: A string containing CSV-formatted data.

    Returns:
        A list of records with uniform keys.  Missing values are ``None``.

    Raises:
        DataLoadError: If the CSV cannot be parsed or has no header row.
    """
    try:
        reader = csv.DictReader(io.StringIO(text))
        records = _consume_csv_reader(reader)
    except csv.Error as exc:
        raise DataLoadError(f"Failed to parse CSV data: {exc}") from exc
    return _normalize_records(records)


def load_json(text: str) -> list[Record]:
    """Parse a JSON string into a normalized list of records.

    Accepts a JSON array of objects (``[{...}, {...}]``) or a dict with a
    single array-valued key (e.g. ``{"data": [{...}]}``).

    Args:
        text: A string containing JSON-formatted data.

    Returns:
        A list of records with uniform keys.  Missing values are ``None``.

    Raises:
        DataLoadError: If the JSON cannot be parsed or is not a list of dicts.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DataLoadError(f"Failed to parse JSON data: {exc}") from exc
    return _normalize_json(parsed)


def infer_column_types(records: list[Record]) -> ColumnTypes:
    """Infer the dominant type for each column across all records.

    For each column the function inspects all non-None values and returns
    the type that can represent the majority of non-null values:

    * ``"number"`` — all non-null values are numeric (int or float).
    * ``"boolean"`` — all non-null values are bool.
    * ``"string"`` — otherwise.

    If a column contains only ``None`` values its type is reported as
    ``"string"``.

    Args:
        records: A list of records as returned by :func:`load_data`.

    Returns:
        A dict mapping column name → type string (``"number"``,
        ``"boolean"``, or ``"string"``).
    """
    if not records:
        return {}

    all_keys: list[str] = _collect_all_keys(records)
    column_types: ColumnTypes = {}

    for key in all_keys:
        values = [row[key] for row in records if key in row and row[key] is not None]
        if not values:
            column_types[key] = "string"
            continue

        if all(isinstance(v, bool) for v in values):
            column_types[key] = "boolean"
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            column_types[key] = "number"
        else:
            column_types[key] = "string"

    return column_types


def get_column_names(records: list[Record]) -> list[str]:
    """Return an ordered list of all column names present in the records.

    The order is determined by the first record that contains each key.

    Args:
        records: A list of records.

    Returns:
        A list of unique column names in encounter order.
    """
    return _collect_all_keys(records)


def get_column_values(records: list[Record], column: str) -> list[Any]:
    """Extract all values for a specific column from the records.

    Missing values (keys absent in a row) are returned as ``None``.

    Args:
        records: A list of records.
        column: The column name to extract.

    Returns:
        A list of values (possibly including ``None``) in record order.

    Raises:
        DataLoadError: If *column* is not present in any record.
    """
    all_keys = _collect_all_keys(records)
    if column not in all_keys:
        raise DataLoadError(
            f"Column '{column}' not found in data. "
            f"Available columns: {all_keys}."
        )
    return [row.get(column) for row in records]


# ---------------------------------------------------------------------------
# Internal helpers — loading
# ---------------------------------------------------------------------------


def _load_csv_file(path: Path) -> list[Record]:
    """Read a CSV file from disk and return normalized records."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise DataLoadError(f"Cannot read CSV file '{path}': {exc}") from exc
    return load_csv(text)


def _load_json_file(path: Path) -> list[Record]:
    """Read a JSON file from disk and return normalized records."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DataLoadError(f"Cannot read JSON file '{path}': {exc}") from exc
    return load_json(text)


def _load_stdin() -> list[Record]:
    """Read data from stdin, auto-detecting CSV vs JSON."""
    try:
        text = sys.stdin.read()
    except OSError as exc:
        raise DataLoadError(f"Cannot read from stdin: {exc}") from exc

    if not text.strip():
        raise DataLoadError("stdin is empty — no data to load.")

    stripped = text.lstrip()
    if stripped.startswith(("[", "{")):
        return load_json(text)
    return load_csv(text)


# ---------------------------------------------------------------------------
# Internal helpers — CSV
# ---------------------------------------------------------------------------


def _consume_csv_reader(reader: csv.DictReader) -> list[dict[str, str | None]]:
    """Consume a DictReader and return a list of raw string dicts.

    Handles the case where DictReader uses ``None`` as the key for extra
    columns (more fields than header columns) by discarding those fields.

    Args:
        reader: An initialised csv.DictReader.

    Returns:
        A list of raw row dicts with string values.

    Raises:
        DataLoadError: If the CSV has no header or is empty.
    """
    rows: list[dict[str, str | None]] = []
    try:
        fieldnames = reader.fieldnames
    except csv.Error as exc:
        raise DataLoadError(f"CSV header could not be read: {exc}") from exc

    if not fieldnames:
        raise DataLoadError("CSV data has no header row or is empty.")

    for row in reader:
        # Drop the sentinel key csv.DictReader uses for extra columns.
        clean: dict[str, str | None] = {
            k: v for k, v in row.items() if k is not None
        }
        rows.append(clean)
    return rows


# ---------------------------------------------------------------------------
# Internal helpers — JSON
# ---------------------------------------------------------------------------


def _normalize_json(parsed: Any) -> list[Record]:
    """Normalize parsed JSON into a list of records.

    Accepts:
    * A JSON array of objects: ``[{"a": 1}, {"a": 2}]``
    * A JSON object with a single list-valued key (e.g. ``{"data": [...]}``).
    * A JSON array of arrays (first row treated as header).

    Args:
        parsed: The parsed JSON value.

    Returns:
        A normalized list of records.

    Raises:
        DataLoadError: If the structure cannot be interpreted as tabular data.
    """
    if isinstance(parsed, list):
        return _normalize_json_array(parsed)
    if isinstance(parsed, dict):
        # Find a single list-valued key.
        list_keys = [k for k, v in parsed.items() if isinstance(v, list)]
        if len(list_keys) == 1:
            return _normalize_json_array(parsed[list_keys[0]])
        # If there are multiple list keys, try a common wrapper pattern.
        for candidate in ("data", "records", "rows", "items", "results"):
            if candidate in parsed and isinstance(parsed[candidate], list):
                return _normalize_json_array(parsed[candidate])
        raise DataLoadError(
            "JSON object has multiple list-valued keys and no recognised wrapper "
            "key ('data', 'records', 'rows', 'items', 'results'). "
            "Please provide a JSON array of objects."
        )
    raise DataLoadError(
        f"Expected a JSON array or object, got {type(parsed).__name__}."
    )


def _normalize_json_array(arr: list[Any]) -> list[Record]:
    """Normalize a JSON array into a list of records.

    Handles arrays of dicts or arrays of arrays (with first row as header).

    Args:
        arr: A list parsed from JSON.

    Returns:
        A list of normalized records.

    Raises:
        DataLoadError: If the array is empty or contains unsupported types.
    """
    if not arr:
        return []

    first = arr[0]

    if isinstance(first, dict):
        # Standard: array of objects.
        raw_records: list[dict[str, Any]] = []
        for i, item in enumerate(arr):
            if not isinstance(item, dict):
                raise DataLoadError(
                    f"JSON array item {i} is {type(item).__name__}, "
                    "expected a dict (object)."
                )
            raw_records.append(item)
        return _normalize_records(raw_records)

    if isinstance(first, (list, tuple)):
        # Array of arrays: first row is headers.
        if len(arr) < 2:
            raise DataLoadError(
                "JSON array-of-arrays must have at least a header row and one data row."
            )
        headers = [str(h) for h in first]
        raw_records = []
        for i, row in enumerate(arr[1:], start=1):
            if not isinstance(row, (list, tuple)):
                raise DataLoadError(
                    f"JSON array-of-arrays: row {i} is {type(row).__name__}, "
                    "expected a list."
                )
            record: dict[str, Any] = {}
            for j, header in enumerate(headers):
                record[header] = row[j] if j < len(row) else None
            raw_records.append(record)
        return _normalize_records(raw_records)

    raise DataLoadError(
        f"JSON array elements are {type(first).__name__}; "
        "expected dicts (objects) or lists (array-of-arrays)."
    )


# ---------------------------------------------------------------------------
# Internal helpers — normalization and type coercion
# ---------------------------------------------------------------------------


def _normalize_records(raw: list[dict[str, Any]]) -> list[Record]:
    """Normalize a list of raw row dicts into typed, uniform records.

    Ensures every record has the same set of keys (the union of all keys
    across all rows), fills in ``None`` for missing keys, and coerces
    string values to appropriate Python types.

    Args:
        raw: A list of raw row dicts (values may be strings or already typed).

    Returns:
        A list of records with uniform keys and coerced values.
    """
    if not raw:
        return []

    all_keys = _collect_all_keys(raw)
    normalized: list[Record] = []

    for row in raw:
        record: Record = {}
        for key in all_keys:
            raw_value = row.get(key)  # None if key absent
            record[key] = _coerce_value(raw_value)
        normalized.append(record)

    return normalized


def _coerce_value(value: Any) -> Any:
    """Coerce a raw cell value to its most natural Python type.

    Conversion rules (applied in order):

    1. ``None`` → ``None``
    2. Non-string values that are already ``bool``, ``int``, or ``float``
       are returned as-is.
    3. String values matching :data:`_NULL_VALUES` → ``None``
    4. String values matching :data:`_TRUE_VALUES` → ``True``
    5. String values matching :data:`_FALSE_VALUES` → ``False``
    6. String values parseable as ``int`` → ``int``
    7. String values parseable as ``float`` → ``float``
    8. All other strings are returned stripped.

    Args:
        value: The raw cell value.

    Returns:
        The coerced value.
    """
    if value is None:
        return None

    # Already a native type (from JSON or in-memory data)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value

    # From here on, value must be treated as a string.
    str_value = str(value).strip()
    lower = str_value.lower()

    if lower in _NULL_VALUES:
        return None

    if lower in _TRUE_VALUES:
        return True

    if lower in _FALSE_VALUES:
        return False

    # Try integer first (no decimal point)
    if _is_integer_string(str_value):
        try:
            return int(str_value.replace(",", "").replace("_", ""))
        except ValueError:
            pass

    # Try float
    try:
        float_val = float(str_value.replace(",", "").replace("_", ""))
        return float_val
    except ValueError:
        pass

    return str_value


def _is_integer_string(s: str) -> bool:
    """Return True if *s* looks like an integer (optional leading sign, digits only).

    Allows underscore/comma separators that Python's int() also handles.

    Args:
        s: The string to test.

    Returns:
        True if the string represents a whole number.
    """
    cleaned = s.replace(",", "").replace("_", "")
    if cleaned.startswith(("-", "+")):
        cleaned = cleaned[1:]
    return cleaned.isdigit()


# ---------------------------------------------------------------------------
# Internal helpers — key collection
# ---------------------------------------------------------------------------


def _collect_all_keys(records: list[dict[str, Any]]) -> list[str]:
    """Return an ordered list of all unique keys across all records.

    Order is determined by first encounter; subsequent records may add new
    keys at the end.

    Args:
        records: A list of dicts.

    Returns:
        An ordered list of unique keys.
    """
    seen: dict[str, None] = {}  # Use dict as ordered set (Python 3.7+)
    for row in records:
        for key in row:
            seen[key] = None
    return list(seen)
