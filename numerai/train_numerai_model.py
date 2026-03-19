#!/usr/bin/env python3
"""
MiroFish Numerai Prediction Pipeline

Downloads the latest Numerai tournament dataset, trains a LightGBM model,
generates predictions, and optionally submits them via the Numerai API.

Usage:
    # Train and generate predictions (no submission):
    uv run python train_numerai_model.py

    # Train and submit predictions:
    uv run python train_numerai_model.py --submit

    # Train with custom parameters:
    uv run python train_numerai_model.py --n-estimators 5000 --learning-rate 0.005

    # Use a specific model name for submission:
    uv run python train_numerai_model.py --submit --model-name YOUR_MODEL_NAME
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from numerapi import NumerAPI
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm


# ==============================================================================
# Configuration
# ==============================================================================

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
PREDICTIONS_DIR = Path(__file__).parent / "predictions"

TARGET_COL = "target"

DEFAULT_LGB_PARAMS = {
    "objective": "regression",
    "metric": "mse",
    "boosting_type": "gbdt",
    "num_leaves": 64,
    "learning_rate": 0.01,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_estimators": 2000,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}


# ==============================================================================
# Data Download
# ==============================================================================


def download_latest_dataset(napi: NumerAPI) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download the latest Numerai tournament dataset (train + live)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    current_round = napi.get_current_round()
    print(f"Current Numerai round: {current_round}")

    # Download training data
    train_path = DATA_DIR / "train.parquet"
    if not train_path.exists():
        print("Downloading training data (this may take a while)...")
        napi.download_dataset("v5.0/train.parquet", dest_path=str(train_path))
    else:
        print(f"Training data already exists at {train_path}")

    # Download live data for the current round
    live_path = DATA_DIR / f"live_{current_round}.parquet"
    print(f"Downloading live data for round {current_round}...")
    napi.download_dataset("v5.0/live.parquet", dest_path=str(live_path))

    # Download feature metadata
    features_path = DATA_DIR / "features.json"
    if not features_path.exists():
        print("Downloading feature metadata...")
        napi.download_dataset(
            "v5.0/features.json", dest_path=str(features_path)
        )

    print("Loading datasets...")
    train_df = pd.read_parquet(train_path)
    live_df = pd.read_parquet(live_path)

    print(f"Training data shape: {train_df.shape}")
    print(f"Live data shape: {live_df.shape}")

    return train_df, live_df, current_round


def load_feature_metadata() -> dict:
    """Load feature metadata including feature sets."""
    features_path = DATA_DIR / "features.json"
    if features_path.exists():
        with open(features_path) as f:
            return json.load(f)
    return {}


# ==============================================================================
# Feature Engineering
# ==============================================================================


def get_feature_columns(df: pd.DataFrame, feature_metadata: dict) -> list[str]:
    """Get the feature columns to use for training.

    Uses the 'medium' feature set if available from metadata,
    otherwise falls back to all columns starting with 'feature_'.
    """
    if feature_metadata and "feature_sets" in feature_metadata:
        # Use medium feature set for a balance of performance and speed
        feature_set = feature_metadata["feature_sets"].get("medium")
        if feature_set:
            available = [f for f in feature_set if f in df.columns]
            print(f"Using 'medium' feature set: {len(available)} features")
            return available

    features = [c for c in df.columns if c.startswith("feature_")]
    print(f"Using all available features: {len(features)} features")
    return features


# ==============================================================================
# Model Training
# ==============================================================================


def numerai_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Numerai's correlation metric (rank correlation)."""
    ranked_preds = pd.Series(y_pred).rank(pct=True, method="first")
    corr, _ = spearmanr(y_true, ranked_preds)
    return corr


def train_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    lgb_params: dict,
) -> lgb.LGBMRegressor:
    """Train a LightGBM model on the Numerai training data."""
    print(f"\nTraining LightGBM model with {len(feature_cols)} features...")

    # Get eras for time-series aware validation
    if "era" in train_df.columns:
        eras = train_df["era"].unique()
        n_eras = len(eras)
        print(f"Total eras: {n_eras}")

        # Use the last 20% of eras as validation
        split_idx = int(n_eras * 0.8)
        val_eras = set(eras[split_idx:])
        train_mask = ~train_df["era"].isin(val_eras)
        val_mask = train_df["era"].isin(val_eras)

        X_train = train_df.loc[train_mask, feature_cols]
        y_train = train_df.loc[train_mask, TARGET_COL]
        X_val = train_df.loc[val_mask, feature_cols]
        y_val = train_df.loc[val_mask, TARGET_COL]

        print(f"Train eras: {split_idx}, Val eras: {n_eras - split_idx}")
        print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    else:
        # Fallback: simple 80/20 split
        split_idx = int(len(train_df) * 0.8)
        X_train = train_df.iloc[:split_idx][feature_cols]
        y_train = train_df.iloc[:split_idx][TARGET_COL]
        X_val = train_df.iloc[split_idx:][feature_cols]
        y_val = train_df.iloc[split_idx:][TARGET_COL]

        print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    # Train model
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=200),
        ],
    )

    # Evaluate on validation set
    val_preds = model.predict(X_val)
    val_corr = numerai_corr(y_val.values, val_preds)
    print(f"\nValidation Numerai Correlation: {val_corr:.6f}")

    # Per-era correlation analysis
    if "era" in train_df.columns:
        val_df = train_df.loc[val_mask].copy()
        val_df["prediction"] = val_preds
        era_corrs = val_df.groupby("era").apply(
            lambda x: numerai_corr(x[TARGET_COL].values, x["prediction"].values)
        )
        print(f"Mean per-era correlation: {era_corrs.mean():.6f}")
        print(f"Std per-era correlation:  {era_corrs.std():.6f}")
        print(f"Sharpe (corr):            {era_corrs.mean() / era_corrs.std():.4f}")

    return model


# ==============================================================================
# Prediction & Submission
# ==============================================================================


def generate_predictions(
    model: lgb.LGBMRegressor,
    live_df: pd.DataFrame,
    feature_cols: list[str],
    current_round: int,
) -> pd.DataFrame:
    """Generate predictions for the live tournament data."""
    print("\nGenerating predictions on live data...")

    predictions = model.predict(live_df[feature_cols])

    # Rank predictions between 0 and 1 (Numerai expects this)
    ranked_predictions = pd.Series(predictions, index=live_df.index).rank(
        pct=True, method="first"
    )

    pred_df = pd.DataFrame(
        {"prediction": ranked_predictions.values}, index=live_df.index
    )

    # Save predictions
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = PREDICTIONS_DIR / f"predictions_round{current_round}_{timestamp}.csv"
    pred_df.to_csv(pred_path)
    print(f"Predictions saved to {pred_path}")
    print(f"Prediction stats: min={pred_df['prediction'].min():.4f}, "
          f"max={pred_df['prediction'].max():.4f}, "
          f"mean={pred_df['prediction'].mean():.4f}")

    return pred_df


def submit_predictions(
    napi: NumerAPI,
    pred_df: pd.DataFrame,
    model_name: str,
    current_round: int,
) -> None:
    """Submit predictions to Numerai tournament."""
    print(f"\nSubmitting predictions for model '{model_name}', round {current_round}...")

    # Save to temp file for submission
    submission_path = PREDICTIONS_DIR / f"submission_round{current_round}.csv"
    pred_df.to_csv(submission_path)

    submission_id = napi.upload_predictions(
        str(submission_path), model_id=model_name
    )
    print(f"Submission successful! Submission ID: {submission_id}")


def save_model(model: lgb.LGBMRegressor, current_round: int) -> Path:
    """Save the trained model to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"lgbm_round{current_round}_{timestamp}.txt"
    model.booster_.save_model(str(model_path))
    print(f"Model saved to {model_path}")
    return model_path


# ==============================================================================
# Main Pipeline
# ==============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MiroFish Numerai Prediction Pipeline"
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit predictions to Numerai after training",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Numerai model name/ID for submission",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_LGB_PARAMS["n_estimators"],
        help="Number of boosting rounds (default: 2000)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LGB_PARAMS["learning_rate"],
        help="Learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=DEFAULT_LGB_PARAMS["num_leaves"],
        help="Number of leaves per tree (default: 64)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("MiroFish - Numerai Prediction Pipeline")
    print("=" * 60)

    # Initialize Numerai API client
    # NumerAPI reads NUMERAI_PUBLIC_ID and NUMERAI_SECRET_KEY from env
    public_id = os.environ.get("NUMERAI_PUBLIC_ID")
    secret_key = os.environ.get("NUMERAI_SECRET_KEY")

    if public_id and secret_key:
        napi = NumerAPI(public_id=public_id, secret_key=secret_key)
        print("Authenticated with Numerai API")
    else:
        napi = NumerAPI()
        print("Using unauthenticated Numerai API (download only, no submissions)")
        if args.submit:
            print("ERROR: Cannot submit without API credentials.")
            print("Set NUMERAI_PUBLIC_ID and NUMERAI_SECRET_KEY environment variables.")
            sys.exit(1)

    # Step 1: Download latest dataset
    train_df, live_df, current_round = download_latest_dataset(napi)

    # Step 2: Get feature columns
    feature_metadata = load_feature_metadata()
    feature_cols = get_feature_columns(train_df, feature_metadata)

    # Step 3: Train model
    lgb_params = DEFAULT_LGB_PARAMS.copy()
    lgb_params["n_estimators"] = args.n_estimators
    lgb_params["learning_rate"] = args.learning_rate
    lgb_params["num_leaves"] = args.num_leaves

    model = train_model(train_df, feature_cols, lgb_params)

    # Step 4: Save model
    save_model(model, current_round)

    # Step 5: Generate predictions
    pred_df = generate_predictions(model, live_df, feature_cols, current_round)

    # Step 6: Optionally submit
    if args.submit:
        model_name = args.model_name
        if not model_name:
            print("ERROR: --model-name is required for submission.")
            print("Usage: python train_numerai_model.py --submit --model-name YOUR_MODEL")
            sys.exit(1)
        submit_predictions(napi, pred_df, model_name, current_round)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
