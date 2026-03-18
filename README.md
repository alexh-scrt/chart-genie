# Chart Genie 🪄

**Turn plain English and raw data into interactive, embeddable HTML charts — in seconds.**

Chart Genie is a CLI tool that accepts a natural language description and a CSV or JSON data file, then uses an LLM (OpenAI GPT-4o or a local Ollama model) to interpret your intent and generate a self-contained, interactive HTML file powered by [Chart.js](https://www.chartjs.org/). No coding, no config files, no build steps — just describe what you want and get a shareable chart.

---

## Quick Start

**Install from PyPI:**

```bash
pip install chart_genie
```

**Set your OpenAI API key:**

```bash
export OPENAI_API_KEY="sk-..."
```

**Generate your first chart:**

```bash
chart-genie --data sales.csv --prompt "Show monthly sales as a bar chart with tooltips" --output chart.html
```

Open `chart.html` in any browser — done.

---

## Features

- **Natural language intent parsing** — describe any chart in plain English; the LLM automatically extracts chart type, axes, labels, colors, and title.
- **Multi-format data ingestion** — accepts CSV files, JSON arrays, or piped stdin with automatic column type detection and null handling.
- **Self-contained HTML output** — generates a single portable HTML file with Chart.js from CDN, interactive tooltips, legends, and responsive sizing. No build step required.
- **Six chart types supported** — bar, line, pie, doughnut, scatter, and radar charts with sensible defaults and LLM-driven customization.
- **Pluggable LLM backend** — defaults to OpenAI GPT-4o; switch to a local Ollama model via environment variable for offline or cost-free usage.

---

## Usage Examples

### Basic bar chart from CSV

```bash
chart-genie \
  --data sales.csv \
  --prompt "Show monthly sales as a bar chart with a blue color scheme" \
  --output monthly_sales.html
```

### Line chart from JSON

```bash
chart-genie \
  --data temperatures.json \
  --prompt "Compare high and low temperatures for New York across months as a line chart" \
  --output temps.html
```

### Pipe data from stdin

```bash
cat sales.csv | chart-genie \
  --prompt "Show returns as a pie chart" \
  --output returns_pie.html
```

### Use a local Ollama model (no API key needed)

```bash
export CHART_GENIE_LLM_BACKEND=ollama
export CHART_GENIE_MODEL=llama3

chart-genie --data sales.csv --prompt "Bar chart of monthly sales" --output chart.html
```

### Programmatic API

```python
from chart_genie import generate_chart

html = generate_chart(
    data_path="sales.csv",
    prompt="Show monthly sales as a bar chart with tooltips",
    output_path="sales_chart.html",
)
print(f"Chart written to sales_chart.html")
```

---

## Project Structure

```
chart_genie/
├── pyproject.toml                  # Project metadata, dependencies, CLI entry point
├── README.md
├── chart_genie/
│   ├── __init__.py                 # Package init, version, top-level generate_chart API
│   ├── cli.py                      # CLI entry point (argparse + Rich formatting)
│   ├── data_loader.py              # CSV/JSON parser with type inference and null handling
│   ├── llm_client.py               # OpenAI / Ollama backend wrapper
│   ├── chart_config.py             # Validated dataclass for LLM-returned chart config
│   ├── renderer.py                 # Jinja2 HTML renderer producing self-contained output
│   ├── prompts.py                  # System and user prompt templates for the LLM
│   └── templates/
│       └── chart.html.j2           # Chart.js HTML template with dynamic data injection
└── tests/
    ├── test_data_loader.py         # CSV/JSON parsing, type inference, edge cases
    ├── test_chart_config.py        # Config validation, defaults, invalid input rejection
    ├── test_renderer.py            # HTML output, Chart.js markup, data injection
    └── fixtures/
        ├── sales.csv               # Monthly sales sample data
        └── temperatures.json       # Multi-series temperature data
```

---

## Configuration

Chart Genie is configured via environment variables. Constructor arguments to the Python API take precedence over environment variables.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required for OpenAI)* | Your OpenAI API key |
| `CHART_GENIE_LLM_BACKEND` | `openai` | LLM backend to use: `openai` or `ollama` |
| `CHART_GENIE_MODEL` | `gpt-4o` | Model name (`llama3` when using Ollama) |
| `CHART_GENIE_OLLAMA_URL` | `http://localhost:11434` | Base URL for a running Ollama server |

### Example: switching backends

```bash
# Use OpenAI (default)
export OPENAI_API_KEY="sk-..."
export CHART_GENIE_LLM_BACKEND=openai
export CHART_GENIE_MODEL=gpt-4o

# Use local Ollama
export CHART_GENIE_LLM_BACKEND=ollama
export CHART_GENIE_MODEL=llama3
export CHART_GENIE_OLLAMA_URL=http://localhost:11434
```

### Supported chart types

`bar` · `line` · `pie` · `doughnut` · `scatter` · `radar`

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Built with [Jitter](https://github.com/jitter-ai) — an AI agent that ships code daily.*
