# Chart Genie

**Transform natural language descriptions and raw data into interactive, embeddable HTML charts — in seconds.**

Chart Genie is a CLI tool that accepts a plain English description and a CSV or JSON data file, then uses an LLM (OpenAI GPT-4o or a local Ollama model) to interpret your intent and generate a self-contained, interactive HTML file powered by [Chart.js](https://www.chartjs.org/).

---

## Features

- **Natural language chart intent parsing** — describe any chart in plain English; the LLM extracts chart type, axes, labels, colors, and title automatically.
- **Multi-format data ingestion** — accepts CSV files, JSON arrays, or piped stdin with automatic column type detection and null handling.
- **Self-contained HTML output** — generates a single portable HTML file with Chart.js loaded from CDN, tooltips, legends, and responsive sizing — no build step required.
- **Multiple chart types** — bar, line, pie, doughnut, scatter, and radar charts with sensible defaults and LLM-driven customization.
- **Pluggable LLM backend** — works with OpenAI GPT-4o by default; switch to a local Ollama model via an environment variable for offline or cost-free usage.

---

## Installation

### From PyPI (once published)

```bash
pip install chart-genie
```

### From source

```bash
git clone https://github.com/example/chart_genie.git
cd chart_genie
pip install -e .
```

---

## Quick Start

### Basic usage

```bash
chart-genie --data sales.csv --prompt "Show monthly sales as a bar chart with tooltips" --output sales_chart.html
```

This will generate `sales_chart.html` — open it in any browser for a fully interactive chart.

### Read data from stdin

```bash
cat data.json | chart-genie --prompt "Plot temperature trends as a line chart" --output temps.html
```

### Specify an output directory

```bash
chart-genie --data sales.csv --prompt "Pie chart of sales by region" --output ./charts/regions.html
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required for OpenAI)* | Your OpenAI API key |
| `CHART_GENIE_LLM_BACKEND` | `openai` | LLM backend: `openai` or `ollama` |
| `CHART_GENIE_MODEL` | `gpt-4o` | Model name (e.g. `gpt-4o`, `llama3`) |
| `CHART_GENIE_OLLAMA_URL` | `http://localhost:11434` | Ollama server base URL |

### Using Ollama (local, offline)

1. Install and start [Ollama](https://ollama.com/).
2. Pull a model: `ollama pull llama3`
3. Set the environment variables:

```bash
export CHART_GENIE_LLM_BACKEND=ollama
export CHART_GENIE_MODEL=llama3
chart-genie --data data.csv --prompt "Bar chart of values" --output chart.html
```

---

## CLI Reference

```
usage: chart-genie [-h] [--data DATA] [--prompt PROMPT] [--output OUTPUT]
                   [--backend {openai,ollama}] [--model MODEL] [--verbose]

Generate interactive HTML charts from data and a natural language prompt.

options:
  -h, --help            show this help message and exit
  --data DATA, -d DATA  Path to input data file (CSV or JSON). Reads from stdin if omitted.
  --prompt PROMPT, -p PROMPT
                        Natural language description of the desired chart.
  --output OUTPUT, -o OUTPUT
                        Path for the output HTML file. (default: chart.html)
  --backend {openai,ollama}
                        LLM backend to use. Overrides CHART_GENIE_LLM_BACKEND env var.
  --model MODEL, -m MODEL
                        Model name to use. Overrides CHART_GENIE_MODEL env var.
  --verbose, -v         Enable verbose output.
```

---

## Supported Chart Types

| Chart Type | Description |
|---|---|
| `bar` | Vertical or horizontal bar chart |
| `line` | Line chart with optional fill |
| `pie` | Pie / donut chart |
| `doughnut` | Doughnut chart |
| `scatter` | X/Y scatter plot |
| `radar` | Spider/radar chart |

---

## Example: Monthly Sales

**Input CSV (`sales.csv`):**

```csv
month,sales,returns
January,15200,320
February,18400,410
March,21000,380
April,19500,290
```

**Command:**

```bash
chart-genie --data sales.csv --prompt "Show monthly sales and returns as grouped bars, blue and red" --output sales.html
```

**Output:** A self-contained `sales.html` file with:
- A grouped bar chart with blue and red bars
- Tooltips on hover
- A legend
- Responsive sizing

---

## Development

### Setup

```bash
git clone https://github.com/example/chart_genie.git
cd chart_genie
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

### Project Structure

```
chart_genie/
├── __init__.py          # Package init, version
├── cli.py               # CLI entry point (argparse + Rich)
├── data_loader.py       # CSV/JSON parser and normalizer
├── llm_client.py        # OpenAI / Ollama API wrapper
├── chart_config.py      # Chart configuration dataclass
├── renderer.py          # HTML renderer via Jinja2
├── prompts.py           # LLM prompt templates
└── templates/
    └── chart.html.j2    # Jinja2 HTML template
tests/
├── fixtures/
│   ├── sales.csv
│   └── temperatures.json
├── test_data_loader.py
├── test_chart_config.py
└── test_renderer.py
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
