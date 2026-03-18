"""LLM client for chart_genie: wraps OpenAI and Ollama backends.

This module provides a unified interface for requesting chart configuration
JSON from an LLM backend. It supports:

* **OpenAI** (default): Uses the ``openai`` Python client with GPT-4o by
  default. Requires the ``OPENAI_API_KEY`` environment variable.
* **Ollama** (local): Communicates with a locally-running Ollama server via
  its OpenAI-compatible ``/v1`` API endpoint using ``httpx``. No API key
  required.

Configuration is controlled via environment variables or constructor
arguments (constructor arguments take precedence):

* ``CHART_GENIE_LLM_BACKEND`` — ``"openai"`` or ``"ollama"`` (default: ``"openai"``)
* ``CHART_GENIE_MODEL`` — model name (default: ``"gpt-4o"`` for OpenAI, ``"llama3"`` for Ollama)
* ``CHART_GENIE_OLLAMA_URL`` — Ollama base URL (default: ``"http://localhost:11434"``)
* ``OPENAI_API_KEY`` — OpenAI API key (required for OpenAI backend)

Typical usage::

    from chart_genie.llm_client import LLMClient
    from chart_genie.chart_config import ChartConfig

    client = LLMClient()  # reads config from environment
    config = client.get_chart_config(
        user_description="Show monthly sales as a bar chart",
        records=records,
    )
    # config is a ChartConfig instance ready for rendering
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx
import openai

from chart_genie.chart_config import ChartConfig, ChartConfigError
from chart_genie.data_loader import (
    get_column_names,
    infer_column_types,
)
from chart_genie.prompts import format_messages

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants and defaults
# ---------------------------------------------------------------------------

#: Default backend when ``CHART_GENIE_LLM_BACKEND`` is not set.
_DEFAULT_BACKEND = "openai"

#: Default OpenAI model.
_DEFAULT_OPENAI_MODEL = "gpt-4o"

#: Default Ollama model.
_DEFAULT_OLLAMA_MODEL = "llama3"

#: Default Ollama server base URL.
_DEFAULT_OLLAMA_URL = "http://localhost:11434"

#: Number of sample records to include in the prompt preview section.
_SAMPLE_SIZE = 5

#: Maximum number of records to send in full to the LLM.
#: Above this threshold only the sample is sent to avoid token limits.
_MAX_FULL_RECORDS = 200

#: HTTP timeout in seconds for Ollama requests.
_OLLAMA_TIMEOUT = 120.0

#: Supported backend identifiers.
_SUPPORTED_BACKENDS = frozenset({"openai", "ollama"})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LLMClientError(RuntimeError):
    """Raised when the LLM backend returns an unexpected or unparseable response."""


class LLMBackendError(LLMClientError):
    """Raised when the LLM backend returns an API-level error."""


class LLMResponseParseError(LLMClientError):
    """Raised when the LLM response cannot be parsed into a ChartConfig."""


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------


class LLMClient:
    """Unified LLM client supporting OpenAI and Ollama backends.

    Sends chart description and dataset context to an LLM and parses the
    response into a :class:`~chart_genie.chart_config.ChartConfig` instance.

    Attributes:
        backend: Active backend identifier (``"openai"`` or ``"ollama"``).
        model: Model name used for completions.
        ollama_url: Base URL for the Ollama server (relevant only when
            ``backend == "ollama"``).
        temperature: Sampling temperature passed to the model.
        max_tokens: Maximum tokens the model may generate in its response.
    """

    def __init__(
        self,
        backend: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        ollama_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> None:
        """Initialise the LLM client.

        Args:
            backend: LLM backend to use (``"openai"`` or ``"ollama"``).
                Falls back to the ``CHART_GENIE_LLM_BACKEND`` environment
                variable, then ``"openai"``.
            model: Model name to use for completions. Falls back to
                ``CHART_GENIE_MODEL``, then the backend-specific default.
            api_key: OpenAI API key. Falls back to the ``OPENAI_API_KEY``
                environment variable. Only required for the OpenAI backend.
            ollama_url: Base URL of the Ollama server. Falls back to
                ``CHART_GENIE_OLLAMA_URL``, then ``"http://localhost:11434"``.
            temperature: Sampling temperature (lower = more deterministic).
                Defaults to ``0.2`` for consistent JSON output.
            max_tokens: Maximum tokens for the LLM response.

        Raises:
            ValueError: If *backend* is not one of the supported identifiers.
        """
        # Resolve backend
        resolved_backend = (
            backend
            or os.environ.get("CHART_GENIE_LLM_BACKEND", _DEFAULT_BACKEND)
        ).strip().lower()
        if resolved_backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported LLM backend '{resolved_backend}'. "
                f"Supported backends: {sorted(_SUPPORTED_BACKENDS)}."
            )
        self.backend = resolved_backend

        # Resolve model
        default_model = (
            _DEFAULT_OPENAI_MODEL if self.backend == "openai" else _DEFAULT_OLLAMA_MODEL
        )
        self.model = (
            model
            or os.environ.get("CHART_GENIE_MODEL", default_model)
        ).strip()

        # Resolve Ollama URL
        self.ollama_url = (
            ollama_url
            or os.environ.get("CHART_GENIE_OLLAMA_URL", _DEFAULT_OLLAMA_URL)
        ).rstrip("/")

        # Resolve API key (OpenAI)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.debug(
            "LLMClient initialised: backend=%s, model=%s",
            self.backend,
            self.model,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_chart_config(
        self,
        user_description: str,
        records: list[dict[str, Any]],
    ) -> ChartConfig:
        """Request a chart configuration from the LLM and parse the response.

        Builds the prompt from the user description and dataset metadata,
        calls the configured LLM backend, extracts JSON from the response,
        and validates it into a :class:`~chart_genie.chart_config.ChartConfig`.

        Args:
            user_description: Plain-English description of the desired chart
                (e.g. ``"Show monthly sales as a bar chart with tooltips"``).
            records: The full dataset as a list of dicts (as returned by
                :func:`~chart_genie.data_loader.load_data`).

        Returns:
            A validated :class:`~chart_genie.chart_config.ChartConfig` instance.

        Raises:
            LLMBackendError: If the LLM API returns an error.
            LLMResponseParseError: If the response cannot be parsed as JSON or
                is not a valid chart configuration.
            ValueError: If *user_description* is empty.
        """
        if not user_description.strip():
            raise ValueError("user_description must not be empty.")
        if not records:
            raise ValueError("records must not be empty.")

        messages = self._build_messages(user_description, records)
        logger.debug(
            "Sending request to %s backend (model=%s)",
            self.backend,
            self.model,
        )

        if self.backend == "openai":
            raw_response = self._call_openai(messages)
        else:
            raw_response = self._call_ollama(messages)

        logger.debug("Raw LLM response (first 500 chars): %s", raw_response[:500])
        return self._parse_response(raw_response)

    def get_chart_config_raw(
        self,
        user_description: str,
        records: list[dict[str, Any]],
    ) -> str:
        """Like :meth:`get_chart_config` but returns the raw LLM response string.

        Useful for debugging prompt engineering or inspecting LLM output before
        parsing.

        Args:
            user_description: Plain-English chart description.
            records: The dataset as a list of dicts.

        Returns:
            The raw text response from the LLM.

        Raises:
            LLMBackendError: If the LLM API returns an error.
            ValueError: If inputs are invalid.
        """
        if not user_description.strip():
            raise ValueError("user_description must not be empty.")
        if not records:
            raise ValueError("records must not be empty.")

        messages = self._build_messages(user_description, records)
        if self.backend == "openai":
            return self._call_openai(messages)
        return self._call_ollama(messages)

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        user_description: str,
        records: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Build the chat messages list from the description and dataset.

        Args:
            user_description: The user's chart intent string.
            records: Full dataset records.

        Returns:
            A list of ``{role, content}`` dicts for the chat API.
        """
        column_names = get_column_names(records)
        column_types = infer_column_types(records)
        sample_records = records[:_SAMPLE_SIZE]
        num_records = len(records)

        # Only send the full dataset when it is within the token budget.
        all_records: list[dict[str, Any]] | None
        if num_records <= _MAX_FULL_RECORDS:
            all_records = records
        else:
            logger.warning(
                "Dataset has %d records (> %d). Only a sample will be included "
                "in the LLM prompt to avoid token limits.",
                num_records,
                _MAX_FULL_RECORDS,
            )
            all_records = records[:_MAX_FULL_RECORDS]

        return format_messages(
            user_description=user_description,
            column_names=column_names,
            column_types=column_types,
            sample_records=sample_records,
            num_records=num_records,
            all_records=all_records,
        )

    # ------------------------------------------------------------------
    # Backend: OpenAI
    # ------------------------------------------------------------------

    def _call_openai(self, messages: list[dict[str, str]]) -> str:
        """Send a chat completion request to the OpenAI API.

        Args:
            messages: List of ``{role, content}`` dicts.

        Returns:
            The content string of the first choice message.

        Raises:
            LLMBackendError: On API errors or authentication failures.
        """
        if not self._api_key:
            raise LLMBackendError(
                "OpenAI API key not set. "
                "Set the OPENAI_API_KEY environment variable or pass api_key to LLMClient."
            )

        try:
            client = openai.OpenAI(api_key=self._api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
        except openai.AuthenticationError as exc:
            raise LLMBackendError(
                f"OpenAI authentication failed: {exc}. "
                "Check your OPENAI_API_KEY."
            ) from exc
        except openai.RateLimitError as exc:
            raise LLMBackendError(
                f"OpenAI rate limit exceeded: {exc}. "
                "Wait before retrying or upgrade your plan."
            ) from exc
        except openai.APIConnectionError as exc:
            raise LLMBackendError(
                f"Could not connect to OpenAI API: {exc}. "
                "Check your network connection."
            ) from exc
        except openai.APIStatusError as exc:
            raise LLMBackendError(
                f"OpenAI API returned error status {exc.status_code}: {exc.message}"
            ) from exc
        except openai.OpenAIError as exc:
            raise LLMBackendError(f"OpenAI error: {exc}") from exc

        content = response.choices[0].message.content
        if content is None:
            raise LLMBackendError(
                "OpenAI returned an empty response. "
                "The model may have refused to answer."
            )
        return content

    # ------------------------------------------------------------------
    # Backend: Ollama
    # ------------------------------------------------------------------

    def _call_ollama(self, messages: list[dict[str, str]]) -> str:
        """Send a chat completion request to a local Ollama server.

        Uses Ollama's OpenAI-compatible ``/v1/chat/completions`` endpoint.

        Args:
            messages: List of ``{role, content}`` dicts.

        Returns:
            The content string of the first choice message.

        Raises:
            LLMBackendError: On connection errors, timeouts, or API errors.
        """
        url = f"{self.ollama_url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        try:
            with httpx.Client(timeout=_OLLAMA_TIMEOUT) as http_client:
                response = http_client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
        except httpx.ConnectError as exc:
            raise LLMBackendError(
                f"Cannot connect to Ollama server at '{self.ollama_url}'. "
                "Is Ollama running? Start it with 'ollama serve'."
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMBackendError(
                f"Ollama request timed out after {_OLLAMA_TIMEOUT}s. "
                "The model may be too large or the server is overloaded."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMBackendError(
                f"Ollama server returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:300]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMBackendError(f"HTTP error communicating with Ollama: {exc}") from exc

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise LLMBackendError(
                f"Ollama returned non-JSON response: {response.text[:300]}"
            ) from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMBackendError(
                f"Unexpected Ollama response structure: {str(data)[:300]}"
            ) from exc

        if not content:
            raise LLMBackendError(
                "Ollama returned an empty response content. "
                "The model may not support the requested format."
            )
        return content

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> ChartConfig:
        """Parse a raw LLM response string into a ChartConfig.

        Attempts to extract a JSON object from the response, handling common
        LLM quirks such as leading/trailing prose or markdown code fences.

        Args:
            raw: The raw text content returned by the LLM.

        Returns:
            A validated :class:`~chart_genie.chart_config.ChartConfig`.

        Raises:
            LLMResponseParseError: If JSON cannot be extracted or the parsed
                dict is not a valid chart configuration.
        """
        json_str = _extract_json(raw)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise LLMResponseParseError(
                f"LLM response is not valid JSON: {exc}\n"
                f"Raw response (first 500 chars): {raw[:500]}"
            ) from exc

        if not isinstance(data, dict):
            raise LLMResponseParseError(
                f"Expected a JSON object from the LLM, got {type(data).__name__}. "
                f"Raw response: {raw[:500]}"
            )

        try:
            config = ChartConfig.from_dict(data)
        except ChartConfigError as exc:
            raise LLMResponseParseError(
                f"LLM response is not a valid chart configuration: {exc}\n"
                f"Parsed JSON keys: {list(data.keys())}"
            ) from exc

        logger.debug(
            "Successfully parsed chart config: type=%s, title=%r, datasets=%d",
            config.chart_type,
            config.title,
            len(config.datasets),
        )
        return config


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def get_chart_config(
    user_description: str,
    records: list[dict[str, Any]],
    backend: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    ollama_url: str | None = None,
) -> ChartConfig:
    """Module-level convenience wrapper around :class:`LLMClient`.

    Creates an :class:`LLMClient` with the given parameters and immediately
    calls :meth:`~LLMClient.get_chart_config`.

    Args:
        user_description: Plain-English chart description.
        records: Dataset as a list of dicts.
        backend: LLM backend (``"openai"`` or ``"ollama"``).
        model: Model name override.
        api_key: OpenAI API key override.
        ollama_url: Ollama server URL override.

    Returns:
        A validated :class:`~chart_genie.chart_config.ChartConfig`.

    Raises:
        LLMBackendError: On API-level errors.
        LLMResponseParseError: If the response cannot be parsed.
        ValueError: If inputs are invalid.
    """
    client = LLMClient(
        backend=backend,
        model=model,
        api_key=api_key,
        ollama_url=ollama_url,
    )
    return client.get_chart_config(
        user_description=user_description,
        records=records,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> str:
    """Extract the first JSON object from a string that may contain prose.

    Handles common LLM response patterns:

    1. Pure JSON (no surrounding text).
    2. JSON wrapped in markdown code fences (```json ... ```).
    3. JSON embedded within prose text (scanned for the first ``{`` ... ``}``).

    Args:
        text: The raw LLM response string.

    Returns:
        The extracted JSON string (still needs to be parsed with json.loads).

    Raises:
        LLMResponseParseError: If no JSON object can be found in the text.
    """
    stripped = text.strip()

    # Fast path: the whole response is already a JSON object.
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    # Try to extract from markdown code fences.
    fence_match = re.search(
        r"```(?:json)?\s*\n?({.*?})\s*\n?```",
        stripped,
        flags=re.DOTALL,
    )
    if fence_match:
        return fence_match.group(1).strip()

    # Scan for the outermost JSON object using brace matching.
    start = stripped.find("{")
    if start == -1:
        raise LLMResponseParseError(
            "No JSON object found in LLM response. "
            f"Response (first 500 chars): {text[:500]}"
        )

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(stripped[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : i + 1]

    raise LLMResponseParseError(
        "Could not find a complete JSON object in LLM response (mismatched braces). "
        f"Response (first 500 chars): {text[:500]}"
    )
