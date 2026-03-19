# MiroFish Numerai Multi-Agent Prediction Pipeline

Uses MiroFish's multi-agent simulation approach for all three Numerai tournaments:

| Script | Tournament | ID | Description |
|--------|------------|-----|-------------|
| `train_numerai_model.py` | Classic | 8 | Stock market predictions using Numerai's v5.0 feature set (~1000+ features) |
| `train_signals_model.py` | Signals | 11 | Global equity predictions using momentum, value, technical indicators (23 features) |
| `train_crypto_model.py` | Crypto | 12 | Crypto asset predictions using Bollinger, RSI, volume, momentum (22 features) |

## How It Works

The pipeline mirrors MiroFish's core architecture:

1. **Agent Profile Generation** (like `OasisProfileGenerator`): The LLM generates diverse model configurations for each analyst agent persona
2. **Parallel Simulation** (like `SimulationRunner`): Each agent independently trains a LightGBM model with unique hyperparameters, feature subset, and data sampling
3. **Report Analysis** (like `ReportAgent`): The LLM analyzes agent performance and assigns ensemble weights
4. **Prediction Aggregation**: Weighted ensemble of all agent predictions

Each tournament uses specialized agent archetypes (stock-oriented for Classic/Signals, crypto-oriented for Crypto).

## Setup

### 1. Install dependencies

```bash
cd numerai
uv sync
```

### 2. Configure MiroFish LLM (optional, for LLM-guided agent generation)

```bash
export LLM_API_KEY="your_api_key"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL_NAME="gpt-4o-mini"
```

### 3. Configure Numerai API (optional, for submission)

```bash
export NUMERAI_PUBLIC_ID="your_public_key"
export NUMERAI_SECRET_KEY="your_secret_key"
```

## Usage

### Classic (Tournament 8)

```bash
uv run python train_numerai_model.py
uv run python train_numerai_model.py --num-agents 8
uv run python train_numerai_model.py --submit --model-name YOUR_MODEL
uv run python train_numerai_model.py --no-llm
```

### Signals (Tournament 11)

```bash
uv run python train_signals_model.py
uv run python train_signals_model.py --num-agents 8
uv run python train_signals_model.py --target target_factor_neutral_20
uv run python train_signals_model.py --submit --model-name YOUR_MODEL
```

Available Signals targets: `target`, `target_factor_neutral_20`, `target_factor_neutral_60`, `target_camille_20`, `target_sydney_20`, and more.

### Crypto (Tournament 12)

```bash
uv run python train_crypto_model.py
uv run python train_crypto_model.py --num-agents 8
uv run python train_crypto_model.py --target target_binned_return_60
uv run python train_crypto_model.py --submit --model-name YOUR_MODEL
```

Available Crypto targets: `target_binned_return_20`, `target_binned_return_60`.

### Common Options

| Flag | Description |
|------|-------------|
| `--num-agents N` | Number of strategy agents (default: 5) |
| `--no-llm` | Skip LLM, use default agent configs |
| `--submit` | Submit predictions after training |
| `--model-name NAME` | Numerai model name for submission |
| `--target COL` | Target column (Signals/Crypto only) |

## Agent Archetypes

### Stock Agents (Classic & Signals)

| Agent | Strategy | Model Style |
|-------|----------|-------------|
| Momentum Trader | Recent trends continue | Aggressive, high learning rate |
| Value Analyst | Fundamental undervaluation | Conservative, slow learning |
| Quant Researcher | Statistical patterns | Complex, many leaves |
| Risk Manager | Downside protection | Defensive, heavy regularization |
| Contrarian Trader | Mean reversion | Moderate, anti-consensus |
| Macro Strategist | Market regimes | Simple, robust |
| Adaptive Learner | Recent data emphasis | Fast, aggressive early stopping |
| Feature Engineer | Curated feature sets | Deep, narrow |

### Crypto Agents

| Agent | Strategy | Model Style |
|-------|----------|-------------|
| Momentum Surfer | Ride strong crypto trends | Aggressive, fast-reacting |
| Mean Reversion Trader | Bollinger/RSI extremes | Contrarian, mean-reversion |
| Market Cap Analyst | Cap flow analysis | Conservative, fundamental |
| Volatility Trader | Risk-adjusted signals | Defensive, regularized |
| Volume Flow Analyst | Volume precedes price | Macro, volume-focused |
| Multi-Timeframe Quant | 20d/60d interactions | Complex, multi-signal |
| Adaptive Crypto Learner | Fast regime shifts | Adaptive, recent data |
| Technical Minimalist | Curated indicators | Selective, reliable |

## Architecture

```
numerai/
├── agent_ensemble.py          # Shared multi-agent engine (archetypes, training, ensemble)
├── train_numerai_model.py     # Classic tournament pipeline
├── train_signals_model.py     # Signals tournament pipeline
├── train_crypto_model.py      # Crypto tournament pipeline
├── pyproject.toml             # Dependencies (uv)
├── data/                      # Downloaded datasets (gitignored)
│   ├── classic/
│   ├── signals/
│   └── crypto/
├── models/                    # Saved models + metadata (gitignored)
└── predictions/               # Generated CSVs (gitignored)
```

## Numerai MCP (for Claude Code)

```bash
export NUMERAI_MCP_AUTH="Token PUBLIC_KEY\$PRIVATE_KEY"
claude mcp add --transport http numerai https://api-tournament.numer.ai/mcp --header "Authorization: Token ${NUMERAI_MCP_AUTH}"
```
