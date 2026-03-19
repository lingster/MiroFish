#!/usr/bin/env python3
"""
MiroFish Numerai Crypto Prediction Pipeline (Tournament 12)

Uses MiroFish's multi-agent simulation approach for Numerai CryptoSignals predictions.
The crypto dataset includes Bollinger bands, momentum, RSI, Sharpe ratio, volatility,
and volume indicators for ~300 crypto assets.

Usage:
    uv run python train_crypto_model.py
    uv run python train_crypto_model.py --num-agents 8
    uv run python train_crypto_model.py --submit --model-name YOUR_MODEL_NAME
    uv run python train_crypto_model.py --no-llm
    uv run python train_crypto_model.py --target target_binned_return_60
"""

import argparse
import json
import os
import sys
from pathlib import Path

from numerapi import CryptoAPI
import pandas as pd

from agent_ensemble import (
    CRYPTO_AGENT_ARCHETYPES,
    create_agent_configs,
    generate_ensemble_predictions,
    generate_ensemble_weights,
    init_llm,
    save_agent_models,
    train_agent_ensemble,
)

DATA_DIR = Path(__file__).parent / "data" / "crypto"
MODELS_DIR = Path(__file__).parent / "models"
PREDICTIONS_DIR = Path(__file__).parent / "predictions"

DEFAULT_TARGET = "target_binned_return_20"

TRAIN_DATASET = "crypto/v2.0/train.parquet"
LIVE_DATASET = "crypto/v2.0/live.parquet"


def download_dataset(capi: CryptoAPI) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Download the latest Numerai Crypto tournament dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    current_round = capi.get_current_round()
    print(f"Current Numerai Crypto round: {current_round}")

    train_path = DATA_DIR / "train.parquet"
    if not train_path.exists():
        print("Downloading Crypto training data...")
        capi.download_dataset(TRAIN_DATASET, dest_path=str(train_path))
    else:
        print(f"Training data already exists at {train_path}")

    live_path = DATA_DIR / f"live_{current_round}.parquet"
    print(f"Downloading Crypto live data for round {current_round}...")
    capi.download_dataset(LIVE_DATASET, dest_path=str(live_path))

    print("Loading datasets...")
    train_df = pd.read_parquet(train_path)
    live_df = pd.read_parquet(live_path)
    print(f"Training data: {train_df.shape}, Live data: {live_df.shape}")

    return train_df, live_df, current_round


def get_features(df: pd.DataFrame) -> list[str]:
    """Get numeric feature columns from the Crypto dataset.

    Crypto features: bollinger bands, close averages/EWA, market cap,
    momentum, RSI, Sharpe ratio, volatility, volume (all in 20d/60d variants).
    """
    features = [
        c for c in df.columns
        if c.startswith("feature_") and df[c].dtype in ("float64", "float32", "int64")
    ]
    print(f"Using {len(features)} crypto features")
    return features


def list_available_targets(df: pd.DataFrame) -> list[str]:
    """List all available target columns."""
    return [c for c in df.columns if c.startswith("target")]


def main():
    parser = argparse.ArgumentParser(description="MiroFish Numerai Crypto Pipeline")
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
    print("MiroFish - Numerai Crypto (Tournament 12)")
    print(f"Strategy: {args.num_agents} independent analyst agents")
    print(f"Target: {args.target}")
    print("=" * 60)

    # Crypto API
    public_id = os.environ.get("NUMERAI_PUBLIC_ID")
    secret_key = os.environ.get("NUMERAI_SECRET_KEY")
    if public_id and secret_key:
        capi = CryptoAPI(public_id=public_id, secret_key=secret_key)
        print("Authenticated with Numerai Crypto API")
    else:
        capi = CryptoAPI()
        print("Using unauthenticated Crypto API (download only)")
        if args.submit:
            print("ERROR: Set NUMERAI_PUBLIC_ID and NUMERAI_SECRET_KEY for submission.")
            sys.exit(1)

    llm = init_llm(args.no_llm)

    # Download & prepare
    train_df, live_df, current_round = download_dataset(capi)
    all_features = get_features(train_df)

    # Show available targets
    targets = list_available_targets(train_df)
    print(f"Available targets: {targets}")

    if args.target not in train_df.columns:
        print(f"ERROR: Target '{args.target}' not found. Available: {targets}")
        sys.exit(1)

    # Crypto uses 'date' as the era column for time-based splitting
    era_col = "date"

    # Crypto has fewer features (22) than Classic (~1000+), so agents should
    # use a higher feature ratio to maintain enough signal
    agent_configs = create_agent_configs(
        llm, all_features, args.num_agents,
        CRYPTO_AGENT_ARCHETYPES,
        tournament_context=(
            "Numerai Crypto prediction tournament. ~300 crypto assets with 22 features "
            "including Bollinger bands (20d/60d), close price averages and EWA, "
            "market cap averages, momentum (20d/60d), RSI (20d/60d), Sharpe ratio, "
            "volatility, and volume indicators. Since there are only 22 features, "
            "feature_sample_ratio should be high (0.7-1.0)."
        ),
    )

    # Override low feature ratios for crypto (too few features to subsample aggressively)
    for config in agent_configs:
        config["feature_sample_ratio"] = max(config.get("feature_sample_ratio", 0.8), 0.6)

    # Train
    agent_results = train_agent_ensemble(
        agent_configs, train_df, all_features,
        target_col=args.target, era_col=era_col,
        tournament_name="Crypto",
    )

    # Ensemble
    weights = generate_ensemble_weights(llm, agent_results, "Crypto")
    save_agent_models(agent_results, MODELS_DIR, current_round, prefix="crypto")

    # Predict - Crypto uses 'symbol' as the index
    pred_df = generate_ensemble_predictions(
        agent_results, weights, live_df,
        PREDICTIONS_DIR, current_round, prefix="crypto",
    )

    # Submit
    if args.submit:
        if not args.model_name:
            print("ERROR: --model-name required for submission.")
            sys.exit(1)
        submission_path = PREDICTIONS_DIR / f"crypto_submission_round{current_round}.csv"
        pred_df.to_csv(submission_path)
        sid = capi.upload_predictions(str(submission_path), model_id=args.model_name)
        print(f"Submission successful! ID: {sid}")

    print("\n" + "=" * 60)
    print("Crypto pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
