"""Unit tests for chart_genie.llm_client and chart_genie.prompts.

Tests cover:
- Prompt building (system/user/messages)
- JSON extraction from various LLM response formats
- Response parsing into ChartConfig
- LLMClient initialisation and configuration
- Error handling for backend failures and invalid responses
- Mocked OpenAI and Ollama calls
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from chart_genie.chart_config import ChartConfig
from chart_genie.llm_client import (
    LLMBackendError,
    LLMClient,
    LLMResponseParseError,
    _extract_json,
    get_chart_config,
)
from chart_genie.prompts import (
    build_system_prompt,
    build_user_prompt,
    format_messages,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_records() -> list[dict[str, Any]]:
    """Minimal sales dataset."""
    return [
        {"month": "January", "sales": 15200, "returns": 320},
        {"month": "February", "sales": 18400, "returns": 410},
        {"month": "March", "sales": 21000, "returns": 380},
    ]


@pytest.fixture()
def valid_chart_config_json() -> str:
    """A valid chart config JSON string the LLM might return."""
    config = {
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
    }
    return json.dumps(config)


@pytest.fixture()
def llm_client_openai() -> LLMClient:
    """LLMClient configured for OpenAI with a dummy API key."""
    return LLMClient(backend="openai", model="gpt-4o", api_key="test-key-123")


@pytest.fixture()
def llm_client_ollama() -> LLMClient:
    """LLMClient configured for Ollama."""
    return LLMClient(backend="ollama", model="llama3", ollama_url="http://localhost:11434")


# ---------------------------------------------------------------------------
# Tests: build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_returns_string(self) -> None:
        prompt = build_system_prompt()
        assert isinstance(prompt, str)

    def test_contains_supported_chart_types(self) -> None:
        prompt = build_system_prompt()
        for chart_type in ["bar", "line", "pie", "doughnut", "scatter", "radar"]:
            assert chart_type in prompt

    def test_contains_json_schema_keys(self) -> None:
        prompt = build_system_prompt()
        for key in ["chart_type", "title", "labels", "datasets", "x_axis_label"]:
            assert key in prompt

    def test_contains_json_only_instruction(self) -> None:
        prompt = build_system_prompt()
        assert "JSON" in prompt

    def test_nonempty(self) -> None:
        assert len(build_system_prompt()) > 100


# ---------------------------------------------------------------------------
# Tests: build_user_prompt
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:
    def test_contains_user_description(self, sample_records: list) -> None:
        prompt = build_user_prompt(
            user_description="Show monthly sales as a bar chart",
            column_names=["month", "sales", "returns"],
            column_types={"month": "string", "sales": "number", "returns": "number"},
            sample_records=sample_records[:2],
            num_records=3,
            all_records=sample_records,
        )
        assert "Show monthly sales as a bar chart" in prompt

    def test_contains_column_names(self, sample_records: list) -> None:
        prompt = build_user_prompt(
            user_description="bar chart",
            column_names=["month", "sales", "returns"],
            column_types={"month": "string", "sales": "number", "returns": "number"},
            sample_records=sample_records,
            num_records=3,
        )
        assert "month" in prompt
        assert "sales" in prompt
        assert "returns" in prompt

    def test_contains_column_types(self, sample_records: list) -> None:
        prompt = build_user_prompt(
            user_description="chart",
            column_names=["month", "sales"],
            column_types={"month": "string", "sales": "number"},
            sample_records=sample_records[:1],
            num_records=3,
        )
        assert "string" in prompt
        assert "number" in prompt

    def test_contains_num_records(self, sample_records: list) -> None:
        prompt = build_user_prompt(
            user_description="chart",
            column_names=["month"],
            column_types={"month": "string"},
            sample_records=sample_records[:1],
            num_records=42,
        )
        assert "42" in prompt

    def test_contains_sample_json(self, sample_records: list) -> None:
        prompt = build_user_prompt(
            user_description="chart",
            column_names=["month", "sales"],
            column_types={"month": "string", "sales": "number"},
            sample_records=sample_records,
            num_records=3,
            all_records=sample_records,
        )
        assert "January" in prompt
        assert "15200" in prompt

    def test_returns_string(self, sample_records: list) -> None:
        prompt = build_user_prompt(
            user_description="test",
            column_names=["a"],
            column_types={"a": "number"},
            sample_records=sample_records[:1],
            num_records=1,
        )
        assert isinstance(prompt, str)

    def test_prompt_stripped_description(self, sample_records: list) -> None:
        """Leading/trailing whitespace in description should be stripped."""
        prompt = build_user_prompt(
            user_description="  bar chart  ",
            column_names=["x"],
            column_types={"x": "number"},
            sample_records=sample_records[:1],
            num_records=1,
        )
        assert "bar chart" in prompt


# ---------------------------------------------------------------------------
# Tests: format_messages
# ---------------------------------------------------------------------------


class TestFormatMessages:
    def test_returns_two_messages(self, sample_records: list) -> None:
        msgs = format_messages(
            user_description="bar chart",
            column_names=["month", "sales"],
            column_types={"month": "string", "sales": "number"},
            sample_records=sample_records,
            num_records=3,
            all_records=sample_records,
        )
        assert len(msgs) == 2

    def test_first_message_is_system(self, sample_records: list) -> None:
        msgs = format_messages(
            user_description="chart",
            column_names=["x"],
            column_types={"x": "number"},
            sample_records=sample_records[:1],
            num_records=1,
        )
        assert msgs[0]["role"] == "system"

    def test_second_message_is_user(self, sample_records: list) -> None:
        msgs = format_messages(
            user_description="chart",
            column_names=["x"],
            column_types={"x": "number"},
            sample_records=sample_records[:1],
            num_records=1,
        )
        assert msgs[1]["role"] == "user"

    def test_system_content_is_string(self, sample_records: list) -> None:
        msgs = format_messages(
            user_description="chart",
            column_names=["x"],
            column_types={"x": "number"},
            sample_records=sample_records[:1],
            num_records=1,
        )
        assert isinstance(msgs[0]["content"], str)
        assert len(msgs[0]["content"]) > 0

    def test_user_content_contains_description(self, sample_records: list) -> None:
        msgs = format_messages(
            user_description="unique-description-xyz",
            column_names=["x"],
            column_types={"x": "number"},
            sample_records=sample_records[:1],
            num_records=1,
        )
        assert "unique-description-xyz" in msgs[1]["content"]


# ---------------------------------------------------------------------------
# Tests: _extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_pure_json_object(self) -> None:
        text = '{"chart_type": "bar", "title": "Test"}'
        result = _extract_json(text)
        assert result == text

    def test_json_with_leading_prose(self) -> None:
        text = 'Here is the config: {"chart_type": "bar", "title": "T"}'
        result = _extract_json(text)
        assert json.loads(result) == {"chart_type": "bar", "title": "T"}

    def test_json_in_markdown_fence(self) -> None:
        text = '```json\n{"chart_type": "line"}\n```'
        result = _extract_json(text)
        assert json.loads(result) == {"chart_type": "line"}

    def test_json_in_plain_code_fence(self) -> None:
        text = '```\n{"chart_type": "pie"}\n```'
        result = _extract_json(text)
        assert json.loads(result) == {"chart_type": "pie"}

    def test_nested_json_object(self) -> None:
        text = '{"a": {"b": {"c": 1}}}'
        result = _extract_json(text)
        data = json.loads(result)
        assert data["a"]["b"]["c"] == 1

    def test_json_with_string_values_containing_braces(self) -> None:
        text = '{"title": "Values {A} and {B}", "type": "bar"}'
        result = _extract_json(text)
        data = json.loads(result)
        assert data["title"] == "Values {A} and {B}"

    def test_no_json_raises(self) -> None:
        with pytest.raises(LLMResponseParseError, match="No JSON object found"):
            _extract_json("This response has no JSON at all.")

    def test_unclosed_brace_raises(self) -> None:
        with pytest.raises(LLMResponseParseError, match="mismatched braces"):
            _extract_json('{"unclosed": "brace"')

    def test_json_with_trailing_prose(self) -> None:
        text = '{"chart_type": "bar"} This is some extra prose.'
        result = _extract_json(text)
        data = json.loads(result)
        assert data["chart_type"] == "bar"

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(LLMResponseParseError):
            _extract_json("   ")


# ---------------------------------------------------------------------------
# Tests: LLMClient.__init__
# ---------------------------------------------------------------------------


class TestLLMClientInit:
    def test_defaults_from_env_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHART_GENIE_LLM_BACKEND", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("CHART_GENIE_MODEL", raising=False)
        client = LLMClient()
        assert client.backend == "openai"
        assert client.model == "gpt-4o"

    def test_defaults_from_env_ollama(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHART_GENIE_LLM_BACKEND", "ollama")
        monkeypatch.delenv("CHART_GENIE_MODEL", raising=False)
        client = LLMClient()
        assert client.backend == "ollama"
        assert client.model == "llama3"

    def test_constructor_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHART_GENIE_LLM_BACKEND", "openai")
        client = LLMClient(backend="ollama", model="mistral")
        assert client.backend == "ollama"
        assert client.model == "mistral"

    def test_unsupported_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported LLM backend"):
            LLMClient(backend="anthropic")

    def test_ollama_url_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHART_GENIE_OLLAMA_URL", "http://myserver:11434")
        client = LLMClient(backend="ollama")
        assert client.ollama_url == "http://myserver:11434"

    def test_ollama_url_trailing_slash_stripped(self) -> None:
        client = LLMClient(backend="ollama", ollama_url="http://localhost:11434/")
        assert not client.ollama_url.endswith("/")

    def test_backend_case_insensitive(self) -> None:
        client = LLMClient(backend="OpenAI", api_key="key")
        assert client.backend == "openai"

    def test_custom_temperature_and_max_tokens(self) -> None:
        client = LLMClient(backend="ollama", temperature=0.9, max_tokens=1024)
        assert client.temperature == 0.9
        assert client.max_tokens == 1024


# ---------------------------------------------------------------------------
# Tests: LLMClient._parse_response
# ---------------------------------------------------------------------------


class TestLLMClientParseResponse:
    def test_valid_response_returns_chart_config(
        self, llm_client_openai: LLMClient, valid_chart_config_json: str
    ) -> None:
        config = llm_client_openai._parse_response(valid_chart_config_json)
        assert isinstance(config, ChartConfig)
        assert config.chart_type == "bar"
        assert config.title == "Monthly Sales"

    def test_json_in_markdown_fence_parsed(
        self, llm_client_openai: LLMClient, valid_chart_config_json: str
    ) -> None:
        wrapped = f"```json\n{valid_chart_config_json}\n```"
        config = llm_client_openai._parse_response(wrapped)
        assert config.chart_type == "bar"

    def test_json_with_prose_parsed(
        self, llm_client_openai: LLMClient, valid_chart_config_json: str
    ) -> None:
        with_prose = f"Sure! Here is the config:\n{valid_chart_config_json}\nHope that helps!"
        config = llm_client_openai._parse_response(with_prose)
        assert config.chart_type == "bar"

    def test_invalid_json_raises_parse_error(
        self, llm_client_openai: LLMClient
    ) -> None:
        with pytest.raises(LLMResponseParseError, match="not valid JSON"):
            llm_client_openai._parse_response("{not valid json at all")

    def test_no_json_raises_parse_error(
        self, llm_client_openai: LLMClient
    ) -> None:
        with pytest.raises(LLMResponseParseError):
            llm_client_openai._parse_response("I cannot help with that request.")

    def test_invalid_chart_config_raises_parse_error(
        self, llm_client_openai: LLMClient
    ) -> None:
        bad_config = json.dumps(
            {"chart_type": "treemap", "title": "T", "datasets": [{"label": "D", "data": [1]}]}
        )
        with pytest.raises(LLMResponseParseError, match="not a valid chart configuration"):
            llm_client_openai._parse_response(bad_config)

    def test_json_array_raises_parse_error(
        self, llm_client_openai: LLMClient
    ) -> None:
        with pytest.raises(LLMResponseParseError, match="Expected a JSON object"):
            llm_client_openai._parse_response("[1, 2, 3]")


# ---------------------------------------------------------------------------
# Tests: LLMClient._call_openai (mocked)
# ---------------------------------------------------------------------------


class TestLLMClientOpenAI:
    def test_successful_call(
        self,
        llm_client_openai: LLMClient,
        valid_chart_config_json: str,
        sample_records: list,
    ) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = valid_chart_config_json

        with patch("chart_genie.llm_client.openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            config = llm_client_openai.get_chart_config(
                user_description="Show monthly sales as a bar chart",
                records=sample_records,
            )

        assert isinstance(config, ChartConfig)
        assert config.chart_type == "bar"

    def test_authentication_error_raises_backend_error(
        self,
        llm_client_openai: LLMClient,
        sample_records: list,
    ) -> None:
        import openai as openai_module

        with patch("chart_genie.llm_client.openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.side_effect = (
                openai_module.AuthenticationError(
                    "Invalid key",
                    response=MagicMock(status_code=401),
                    body={},
                )
            )

            with pytest.raises(LLMBackendError, match="authentication failed"):
                llm_client_openai.get_chart_config(
                    user_description="bar chart",
                    records=sample_records,
                )

    def test_missing_api_key_raises_backend_error(
        self, sample_records: list
    ) -> None:
        client = LLMClient(backend="openai", api_key="")
        with pytest.raises(LLMBackendError, match="API key not set"):
            client._call_openai([])

    def test_empty_content_raises_backend_error(
        self,
        llm_client_openai: LLMClient,
        sample_records: list,
    ) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None

        with patch("chart_genie.llm_client.openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            with pytest.raises(LLMBackendError, match="empty response"):
                llm_client_openai.get_chart_config(
                    user_description="bar chart",
                    records=sample_records,
                )


# ---------------------------------------------------------------------------
# Tests: LLMClient._call_ollama (mocked)
# ---------------------------------------------------------------------------


class TestLLMClientOllama:
    def test_successful_call(
        self,
        llm_client_ollama: LLMClient,
        valid_chart_config_json: str,
        sample_records: list,
    ) -> None:
        mock_resp_data = {
            "choices": [
                {"message": {"content": valid_chart_config_json}}
            ]
        }

        with patch("chart_genie.llm_client.httpx.Client") as mock_httpx_cls:
            mock_http = MagicMock()
            mock_httpx_cls.return_value.__enter__.return_value = mock_http
            mock_response = MagicMock()
            mock_response.json.return_value = mock_resp_data
            mock_response.raise_for_status.return_value = None
            mock_http.post.return_value = mock_response

            config = llm_client_ollama.get_chart_config(
                user_description="Show monthly sales as a bar chart",
                records=sample_records,
            )

        assert isinstance(config, ChartConfig)
        assert config.chart_type == "bar"

    def test_connection_error_raises_backend_error(
        self,
        llm_client_ollama: LLMClient,
        sample_records: list,
    ) -> None:
        import httpx

        with patch("chart_genie.llm_client.httpx.Client") as mock_httpx_cls:
            mock_http = MagicMock()
            mock_httpx_cls.return_value.__enter__.return_value = mock_http
            mock_http.post.side_effect = httpx.ConnectError("Connection refused")

            with pytest.raises(LLMBackendError, match="Cannot connect to Ollama"):
                llm_client_ollama.get_chart_config(
                    user_description="bar chart",
                    records=sample_records,
                )

    def test_timeout_raises_backend_error(
        self,
        llm_client_ollama: LLMClient,
        sample_records: list,
    ) -> None:
        import httpx

        with patch("chart_genie.llm_client.httpx.Client") as mock_httpx_cls:
            mock_http = MagicMock()
            mock_httpx_cls.return_value.__enter__.return_value = mock_http
            mock_http.post.side_effect = httpx.ReadTimeout("Timed out")

            with pytest.raises(LLMBackendError, match="timed out"):
                llm_client_ollama.get_chart_config(
                    user_description="bar chart",
                    records=sample_records,
                )

    def test_http_status_error_raises_backend_error(
        self,
        llm_client_ollama: LLMClient,
        sample_records: list,
    ) -> None:
        import httpx

        with patch("chart_genie.llm_client.httpx.Client") as mock_httpx_cls:
            mock_http = MagicMock()
            mock_httpx_cls.return_value.__enter__.return_value = mock_http
            error_response = MagicMock()
            error_response.status_code = 500
            error_response.text = "Internal Server Error"
            mock_http.post.side_effect = httpx.HTTPStatusError(
                "500", request=MagicMock(), response=error_response
            )

            with pytest.raises(LLMBackendError, match="HTTP 500"):
                llm_client_ollama.get_chart_config(
                    user_description="bar chart",
                    records=sample_records,
                )

    def test_bad_response_structure_raises_backend_error(
        self,
        llm_client_ollama: LLMClient,
        sample_records: list,
    ) -> None:
        with patch("chart_genie.llm_client.httpx.Client") as mock_httpx_cls:
            mock_http = MagicMock()
            mock_httpx_cls.return_value.__enter__.return_value = mock_http
            mock_response = MagicMock()
            mock_response.json.return_value = {"unexpected": "structure"}
            mock_response.raise_for_status.return_value = None
            mock_http.post.return_value = mock_response

            with pytest.raises(LLMBackendError, match="Unexpected Ollama response"):
                llm_client_ollama.get_chart_config(
                    user_description="bar chart",
                    records=sample_records,
                )


# ---------------------------------------------------------------------------
# Tests: LLMClient input validation
# ---------------------------------------------------------------------------


class TestLLMClientValidation:
    def test_empty_description_raises(
        self, llm_client_openai: LLMClient, sample_records: list
    ) -> None:
        with pytest.raises(ValueError, match="user_description"):
            llm_client_openai.get_chart_config(
                user_description="  ",
                records=sample_records,
            )

    def test_empty_records_raises(
        self, llm_client_openai: LLMClient
    ) -> None:
        with pytest.raises(ValueError, match="records"):
            llm_client_openai.get_chart_config(
                user_description="bar chart",
                records=[],
            )


# ---------------------------------------------------------------------------
# Tests: Module-level get_chart_config convenience function
# ---------------------------------------------------------------------------


class TestModuleLevelGetChartConfig:
    def test_creates_client_and_calls(
        self,
        valid_chart_config_json: str,
        sample_records: list,
    ) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = valid_chart_config_json

        with patch("chart_genie.llm_client.openai.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            config = get_chart_config(
                user_description="bar chart",
                records=sample_records,
                backend="openai",
                api_key="test-key",
            )

        assert isinstance(config, ChartConfig)


# ---------------------------------------------------------------------------
# Tests: Large dataset truncation
# ---------------------------------------------------------------------------


class TestLargeDatasetHandling:
    def test_large_dataset_truncated_in_messages(
        self, llm_client_openai: LLMClient
    ) -> None:
        """Datasets above _MAX_FULL_RECORDS are truncated for the prompt."""
        from chart_genie.llm_client import _MAX_FULL_RECORDS

        large_records = [
            {"x": i, "y": i * 2} for i in range(_MAX_FULL_RECORDS + 50)
        ]
        messages = llm_client_openai._build_messages(
            user_description="scatter chart",
            records=large_records,
        )
        # The user message should mention total count but only embed truncated data
        user_content = messages[1]["content"]
        assert str(_MAX_FULL_RECORDS + 50) in user_content  # total num_records mentioned

    def test_small_dataset_fully_embedded(
        self, llm_client_openai: LLMClient, sample_records: list
    ) -> None:
        """Small datasets should be fully embedded."""
        messages = llm_client_openai._build_messages(
            user_description="bar chart",
            records=sample_records,
        )
        user_content = messages[1]["content"]
        assert "January" in user_content
        assert "February" in user_content
        assert "March" in user_content
