# MiroFish Numerai Multi-Agent Prediction Pipeline

Uses MiroFish's multi-agent simulation approach for Numerai tournament predictions. Instead of a single model, multiple "analyst agents" with distinct trading personas independently train models, and their predictions are ensembled using LLM-guided weighting.

## How It Works

The pipeline mirrors MiroFish's core architecture:

1. **Agent Profile Generation** (like `OasisProfileGenerator`): The LLM generates diverse model configurations for each analyst agent persona (Momentum Trader, Value Analyst, Risk Manager, etc.)
2. **Parallel Simulation** (like `SimulationRunner`): Each agent independently trains a LightGBM model with its own hyperparameters, feature subset, and data sampling
3. **Report Analysis** (like `ReportAgent`): The LLM analyzes agent performance and assigns ensemble weights
4. **Prediction Aggregation**: Weighted ensemble of all agent predictions

## Setup

### 1. Install dependencies

```bash
cd numerai
uv sync
```

### 2. Configure MiroFish LLM (required for LLM-guided agent generation)

Set the same LLM env vars used by MiroFish backend (in `.env` or shell):
```bash
export LLM_API_KEY="your_api_key"
export LLM_BASE_URL="https://api.openai.com/v1"  # or any OpenAI-compatible endpoint
export LLM_MODEL_NAME="gpt-4o-mini"
```

### 3. Configure Numerai API (optional, for submission)

```bash
export NUMERAI_PUBLIC_ID="your_public_key"
export NUMERAI_SECRET_KEY="your_secret_key"
```

## Usage

```bash
# Train with 5 agents (default)
uv run python train_numerai_model.py

# Train with more agents for better ensemble
uv run python train_numerai_model.py --num-agents 8

# Skip LLM, use default agent configs
uv run python train_numerai_model.py --no-llm

# Train and submit predictions
uv run python train_numerai_model.py --submit --model-name YOUR_MODEL_NAME
```

## Agent Archetypes

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

## Output

- `data/` - Downloaded Numerai datasets (parquet)
- `models/` - Agent models + ensemble metadata JSON
- `predictions/` - Generated prediction CSVs
