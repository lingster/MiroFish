# MiroFish Numerai Prediction Pipeline

Download Numerai tournament data, train a LightGBM prediction model, and submit predictions.

## Setup

### 1. Install dependencies

```bash
cd numerai
pip install -r requirements.txt
```

### 2. Configure Numerai API credentials

For downloading data (no auth required):
```bash
python train_numerai_model.py
```

For submitting predictions, set your API keys:
```bash
export NUMERAI_PUBLIC_ID="your_public_key"
export NUMERAI_SECRET_KEY="your_secret_key"
```

### 3. Configure Numerai MCP (for Claude Code)

```bash
export NUMERAI_MCP_AUTH="Token PUBLIC_KEY\$PRIVATE_KEY"
claude mcp add --transport http numerai https://api-tournament.numer.ai/mcp --header "Authorization: Token ${NUMERAI_MCP_AUTH}"
```

## Usage

```bash
# Train model (downloads data automatically on first run)
python train_numerai_model.py

# Train with custom hyperparameters
python train_numerai_model.py --n-estimators 5000 --learning-rate 0.005

# Train and submit predictions
python train_numerai_model.py --submit --model-name YOUR_MODEL_NAME
```

## Output

- `data/` - Downloaded Numerai datasets (parquet)
- `models/` - Saved LightGBM models
- `predictions/` - Generated prediction CSVs

## Model Details

- **Algorithm**: LightGBM (gradient boosting)
- **Feature set**: Numerai v5.0 "medium" feature set
- **Validation**: Era-aware train/val split (80/20 by era)
- **Metrics**: Numerai correlation (Spearman rank correlation)
