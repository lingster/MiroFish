#!/usr/bin/env python3
"""
MiroFish Numerai Classic Prediction Pipeline (Tournament 8)

Uses MiroFish's multi-agent simulation approach for Numerai Classic predictions.
Multiple "analyst agents" with different strategies independently train models,
and their predictions are ensembled using LLM-guided weighting.

Usage:
    uv run python train_numerai_model.py
    uv run python train_numerai_model.py --num-agents 8
    uv run python train_numerai_model.py --submit --model-name YOUR_MODEL_NAME
    uv run python train_numerai_model.py --no-llm
"""

import argparse
import json
import os
import sys
from pathlib import Path

from numerapi import NumerAPI
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

DATA_DIR = Path(__file__).parent / "data" / "classic"
MODELS_DIR = Path(__file__).parent / "models"
PREDICTIONS_DIR = Path(__file__).parent / "predictions"

TARGET_COL = "target"


def download_dataset(napi: NumerAPI) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Download the latest Numerai Classic tournament dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    current_round = napi.get_current_round()
    print(f"Current Numerai Classic round: {current_round}")

    train_path = DATA_DIR / "train.parquet"
    if not train_path.exists():
        print("Downloading Classic training data (this may take a while)...")
        napi.download_dataset("v5.0/train.parquet", dest_path=str(train_path))
    else:
        print(f"Training data already exists at {train_path}")

    live_path = DATA_DIR / f"live_{current_round}.parquet"
    print(f"Downloading live data for round {current_round}...")
    napi.download_dataset("v5.0/live.parquet", dest_path=str(live_path))

    features_path = DATA_DIR / "features.json"
    if not features_path.exists():
        print("Downloading feature metadata...")
        napi.download_dataset("v5.0/features.json", dest_path=str(features_path))

    print("Loading datasets...")
    train_df = pd.read_parquet(train_path)
    live_df = pd.read_parquet(live_path)
    print(f"Training data: {train_df.shape}, Live data: {live_df.shape}")

    return train_df, live_df, current_round


def get_features(df: pd.DataFrame) -> list[str]:
    """Get feature columns, preferring the 'medium' feature set."""
    features_path = DATA_DIR / "features.json"
    if features_path.exists():
        with open(features_path) as f:
            meta = json.load(f)
        feature_set = meta.get("feature_sets", {}).get("medium")
        if feature_set:
            available = [f for f in feature_set if f in df.columns]
            print(f"Using 'medium' feature set: {len(available)} features")
            return available

    features = [c for c in df.columns if c.startswith("feature_")]
    print(f"Using all available features: {len(features)} features")
    return features


def main():
    parser = argparse.ArgumentParser(description="MiroFish Numerai Classic Pipeline")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("MiroFish - Numerai Classic (Tournament 8)")
    print(f"Strategy: {args.num_agents} independent analyst agents")
    print("=" * 60)

    # Numerai API
    public_id = os.environ.get("NUMERAI_PUBLIC_ID")
    secret_key = os.environ.get("NUMERAI_SECRET_KEY")
    if public_id and secret_key:
        napi = NumerAPI(public_id=public_id, secret_key=secret_key)
        print("Authenticated with Numerai API")
    else:
        napi = NumerAPI()
        print("Using unauthenticated Numerai API (download only)")
        if args.submit:
            print("ERROR: Set NUMERAI_PUBLIC_ID and NUMERAI_SECRET_KEY for submission.")
            sys.exit(1)

    llm = init_llm(args.no_llm)

    # Download & prepare
    train_df, live_df, current_round = download_dataset(napi)
    all_features = get_features(train_df)

    # Create agents
    agent_configs = create_agent_configs(
        llm, all_features, args.num_agents,
        STOCK_AGENT_ARCHETYPES,
        tournament_context="Numerai Classic stock prediction tournament (v5.0 features)",
    )

    # Train
    agent_results = train_agent_ensemble(
        agent_configs, train_df, all_features,
        target_col=TARGET_COL, era_col="era",
        tournament_name="Classic",
    )

    # Ensemble
    weights = generate_ensemble_weights(llm, agent_results, "Classic")
    save_agent_models(agent_results, MODELS_DIR, current_round, prefix="classic")

    # Predict
    pred_df = generate_ensemble_predictions(
        agent_results, weights, live_df,
        PREDICTIONS_DIR, current_round, prefix="classic",
    )

    # Submit
    if args.submit:
        if not args.model_name:
            print("ERROR: --model-name required for submission.")
            sys.exit(1)
        submission_path = PREDICTIONS_DIR / f"classic_submission_round{current_round}.csv"
        pred_df.to_csv(submission_path)
        sid = napi.upload_predictions(str(submission_path), model_id=args.model_name)
        print(f"Submission successful! ID: {sid}")

    print("\n" + "=" * 60)
    print("Classic pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
