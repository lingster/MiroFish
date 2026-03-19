#!/usr/bin/env python3
"""
MiroFish Numerai Signals Prediction Pipeline (Tournament 11)

Uses MiroFish's multi-agent simulation approach for Numerai Signals predictions.
Signals provides its own feature set (v2.1) including momentum, value, volatility,
and technical indicators for global equities.

Usage:
    uv run python train_signals_model.py
    uv run python train_signals_model.py --num-agents 8
    uv run python train_signals_model.py --submit --model-name YOUR_MODEL_NAME
    uv run python train_signals_model.py --no-llm
    uv run python train_signals_model.py --target target_factor_neutral_20
"""

import argparse
import json
import os
import sys
from pathlib import Path

from numerapi import SignalsAPI
import pandas as pd

from agent_ensemble import (
    STOCK_AGENT_ARCHETYPES,
    create_agent_configs,
    generate_ensemble_predictions,
    generate_ensemble_weights,
    init_llm,
    save_agent_models,
    train_agent_ensemble,
)

DATA_DIR = Path(__file__).parent / "data" / "signals"
MODELS_DIR = Path(__file__).parent / "models"
PREDICTIONS_DIR = Path(__file__).parent / "predictions"

# Signals has multiple targets; default is the main one
DEFAULT_TARGET = "target"

# Signals v2.1 dataset paths
TRAIN_DATASET = "signals/v2.1/train.parquet"
LIVE_DATASET = "signals/v2.1/live.parquet"


def download_dataset(sapi: SignalsAPI) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Download the latest Numerai Signals tournament dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    current_round = sapi.get_current_round()
    print(f"Current Numerai Signals round: {current_round}")

    train_path = DATA_DIR / "train.parquet"
    if not train_path.exists():
        print("Downloading Signals training data...")
        sapi.download_dataset(TRAIN_DATASET, dest_path=str(train_path))
    else:
        print(f"Training data already exists at {train_path}")

    live_path = DATA_DIR / f"live_{current_round}.parquet"
    print(f"Downloading Signals live data for round {current_round}...")
    sapi.download_dataset(LIVE_DATASET, dest_path=str(live_path))

    print("Loading datasets...")
    train_df = pd.read_parquet(train_path)
    live_df = pd.read_parquet(live_path)
    print(f"Training data: {train_df.shape}, Live data: {live_df.shape}")

    return train_df, live_df, current_round


def get_features(df: pd.DataFrame) -> list[str]:
    """Get numeric feature columns from the Signals dataset.

    Signals features include momentum, value, volatility, RSI, PPO, TRIX factors.
    We exclude 'feature_country' as it's categorical.
    """
    features = [
        c for c in df.columns
        if c.startswith("feature_") and df[c].dtype in ("float64", "float32", "int64")
    ]
    print(f"Using {len(features)} numeric features")
    return features


def list_available_targets(df: pd.DataFrame) -> list[str]:
    """List all available target columns."""
    return [c for c in df.columns if c.startswith("target")]


def main():
    parser = argparse.ArgumentParser(description="MiroFish Numerai Signals Pipeline")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument(
        "--target", type=str, default=DEFAULT_TARGET,
        help=f"Target column to predict (default: {DEFAULT_TARGET})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MiroFish - Numerai Signals (Tournament 11)")
    print(f"Strategy: {args.num_agents} independent analyst agents")
    print(f"Target: {args.target}")
    print("=" * 60)

    # Signals API
    public_id = os.environ.get("NUMERAI_PUBLIC_ID")
    secret_key = os.environ.get("NUMERAI_SECRET_KEY")
    if public_id and secret_key:
        sapi = SignalsAPI(public_id=public_id, secret_key=secret_key)
        print("Authenticated with Numerai Signals API")
    else:
        sapi = SignalsAPI()
        print("Using unauthenticated Signals API (download only)")
        if args.submit:
            print("ERROR: Set NUMERAI_PUBLIC_ID and NUMERAI_SECRET_KEY for submission.")
            sys.exit(1)

    llm = init_llm(args.no_llm)

    # Download & prepare
    train_df, live_df, current_round = download_dataset(sapi)
    all_features = get_features(train_df)

    # Show available targets
    targets = list_available_targets(train_df)
    print(f"Available targets: {targets}")

    if args.target not in train_df.columns:
        print(f"ERROR: Target '{args.target}' not found. Available: {targets}")
        sys.exit(1)

    # Signals uses 'date' as the era column for time-based splitting
    era_col = "date"

    # Create agents
    agent_configs = create_agent_configs(
        llm, all_features, args.num_agents,
        STOCK_AGENT_ARCHETYPES,
        tournament_context=(
            "Numerai Signals stock prediction tournament. "
            "Features include momentum (12w/26w/52w), value, volatility, beta, "
            "book-to-price, dividend yield, earnings yield, growth, RSI, PPO, "
            "TRIX technical indicators for global equities."
        ),
    )

    # Train
    agent_results = train_agent_ensemble(
        agent_configs, train_df, all_features,
        target_col=args.target, era_col=era_col,
        tournament_name="Signals",
    )

    # Ensemble
    weights = generate_ensemble_weights(llm, agent_results, "Signals")
    save_agent_models(agent_results, MODELS_DIR, current_round, prefix="signals")

    # Predict - Signals expects 'numerai_ticker' as the index for submission
    # Ensure live_df has the right ID column
    if "numerai_ticker" in live_df.columns and live_df.index.name != "numerai_ticker":
        live_pred_df = live_df.set_index("numerai_ticker")
    else:
        live_pred_df = live_df

    pred_df = generate_ensemble_predictions(
        agent_results, weights, live_pred_df,
        PREDICTIONS_DIR, current_round, prefix="signals",
    )

    # Submit
    if args.submit:
        if not args.model_name:
            print("ERROR: --model-name required for submission.")
            sys.exit(1)
        submission_path = PREDICTIONS_DIR / f"signals_submission_round{current_round}.csv"
        pred_df.to_csv(submission_path)
        sid = sapi.upload_predictions(str(submission_path), model_id=args.model_name)
        print(f"Submission successful! ID: {sid}")

    print("\n" + "=" * 60)
    print("Signals pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
