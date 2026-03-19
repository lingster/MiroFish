#!/usr/bin/env python3
"""
MiroFish Numerai Prediction Pipeline

Uses MiroFish's multi-agent simulation approach for Numerai predictions.
Multiple "analyst agents" with different strategies independently analyze
the data and their predictions are ensembled for the final submission.

The pipeline leverages MiroFish's LLM client to:
  1. Analyze feature importance and select optimal feature subsets per agent
  2. Generate diverse hyperparameter configurations for each strategy agent
  3. Ensemble predictions using LLM-guided weighting

Usage:
    # Run with default 5 agents:
    uv run python train_numerai_model.py

    # Run with more agents for better ensemble:
    uv run python train_numerai_model.py --num-agents 10

    # Submit predictions:
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
from tqdm import tqdm

# Add MiroFish backend to path so we can import its services
BACKEND_DIR = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.utils.llm_client import LLMClient


# ==============================================================================
# Configuration
# ==============================================================================

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
PREDICTIONS_DIR = Path(__file__).parent / "predictions"

TARGET_COL = "target"


# ==============================================================================
# Agent Definitions - MiroFish Style
# ==============================================================================

# Each agent represents a distinct investment strategy persona,
# similar to how MiroFish creates social media agent profiles.
# The LLM generates tailored hyperparameters for each.

AGENT_ARCHETYPES = [
    {
        "name": "Momentum Trader",
        "persona": (
            "You are an aggressive momentum trader who believes recent trends "
            "continue. You prefer models that react quickly to recent data and "
            "use features related to recent price movements and volume."
        ),
        "style": "aggressive",
        "preferred_feature_groups": ["momentum", "volume", "short_term"],
    },
    {
        "name": "Value Analyst",
        "persona": (
            "You are a conservative value analyst who looks for undervalued "
            "assets. You prefer stable, slow-learning models and features "
            "related to fundamental valuation metrics."
        ),
        "style": "conservative",
        "preferred_feature_groups": ["value", "fundamental", "long_term"],
    },
    {
        "name": "Quant Researcher",
        "persona": (
            "You are a quantitative researcher who uses statistical signals. "
            "You focus on feature interactions and non-linear patterns, "
            "preferring complex models with many leaves."
        ),
        "style": "complex",
        "preferred_feature_groups": ["statistical", "interaction", "volatility"],
    },
    {
        "name": "Risk Manager",
        "persona": (
            "You are a risk-averse portfolio manager. You prioritize stability "
            "over returns, using heavily regularized models. You prefer features "
            "that capture downside risk and correlation."
        ),
        "style": "defensive",
        "preferred_feature_groups": ["risk", "correlation", "stability"],
    },
    {
        "name": "Contrarian Trader",
        "persona": (
            "You are a contrarian who bets against the crowd. You look for "
            "mean-reversion signals and use features that capture overreaction "
            "and sentiment extremes."
        ),
        "style": "contrarian",
        "preferred_feature_groups": ["sentiment", "mean_reversion", "extreme"],
    },
    {
        "name": "Macro Strategist",
        "persona": (
            "You are a macro strategist who focuses on broad market regimes. "
            "You use features related to market-wide factors and economic "
            "cycles, preferring simple but robust models."
        ),
        "style": "macro",
        "preferred_feature_groups": ["macro", "sector", "cycle"],
    },
    {
        "name": "Adaptive Learner",
        "persona": (
            "You are an adaptive ML practitioner who emphasizes recent data. "
            "You use high learning rates with aggressive early stopping and "
            "prefer a diverse mix of features."
        ),
        "style": "adaptive",
        "preferred_feature_groups": ["diverse", "recent", "mixed"],
    },
    {
        "name": "Feature Engineer",
        "persona": (
            "You are a feature engineering specialist. You use a small, "
            "carefully selected set of the most predictive features with "
            "a deep, narrow model architecture."
        ),
        "style": "selective",
        "preferred_feature_groups": ["top_features", "curated", "signal"],
    },
]


# ==============================================================================
# Data Download
# ==============================================================================


def download_latest_dataset(napi: NumerAPI) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Download the latest Numerai tournament dataset (train + live)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    current_round = napi.get_current_round()
    print(f"Current Numerai round: {current_round}")

    train_path = DATA_DIR / "train.parquet"
    if not train_path.exists():
        print("Downloading training data (this may take a while)...")
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


def get_all_feature_columns(df: pd.DataFrame, feature_metadata: dict) -> list[str]:
    """Get all available feature columns."""
    if feature_metadata and "feature_sets" in feature_metadata:
        feature_set = feature_metadata["feature_sets"].get("medium")
        if feature_set:
            available = [f for f in feature_set if f in df.columns]
            print(f"Using 'medium' feature set: {len(available)} features")
            return available

    features = [c for c in df.columns if c.startswith("feature_")]
    print(f"Using all available features: {len(features)} features")
    return features


# ==============================================================================
# MiroFish Agent System
# ==============================================================================


def create_agent_configs(
    llm: LLMClient,
    all_features: list[str],
    num_agents: int,
) -> list[dict]:
    """
    Use MiroFish's LLM to generate diverse agent configurations.

    Each agent gets a unique combination of:
    - Feature subset (selected from all available features)
    - LightGBM hyperparameters
    - Training data sampling strategy

    This mirrors MiroFish's OasisProfileGenerator which creates
    diverse agent profiles for simulation.
    """
    archetypes = AGENT_ARCHETYPES[:num_agents]
    if num_agents > len(AGENT_ARCHETYPES):
        # Cycle through archetypes for extra agents
        for i in range(num_agents - len(AGENT_ARCHETYPES)):
            archetypes.append(AGENT_ARCHETYPES[i % len(AGENT_ARCHETYPES)])

    # Sample feature names for the LLM prompt (too many to list all)
    sample_features = all_features[:50]
    feature_sample_str = ", ".join(sample_features[:20])

    print(f"\nGenerating {num_agents} agent configurations via MiroFish LLM...")

    agent_configs = []
    for i, archetype in enumerate(archetypes):
        print(f"  Agent {i}: {archetype['name']}...", end=" ", flush=True)

        prompt = f"""You are configuring a LightGBM model for the Numerai stock prediction tournament.

Agent persona: {archetype['persona']}
Strategy style: {archetype['style']}

Total available features: {len(all_features)}
Sample feature names: {feature_sample_str}...

Generate a LightGBM configuration that matches this agent's trading style.
Return JSON with these exact keys:

{{
    "num_leaves": <int, 16-256>,
    "learning_rate": <float, 0.001-0.1>,
    "n_estimators": <int, 500-5000>,
    "feature_fraction": <float, 0.1-1.0>,
    "bagging_fraction": <float, 0.3-1.0>,
    "bagging_freq": <int, 1-10>,
    "min_child_samples": <int, 5-500>,
    "lambda_l1": <float, 0.0-10.0>,
    "lambda_l2": <float, 0.0-10.0>,
    "feature_sample_ratio": <float, 0.3-1.0, what fraction of features to use>,
    "era_subsample_ratio": <float, 0.5-1.0, what fraction of training eras to use>,
    "reasoning": "<brief explanation of choices>"
}}"""

        try:
            result = llm.chat_json(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a quantitative finance expert configuring "
                            "ML models. Return valid JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1024,
            )

            config = {
                "agent_id": i,
                "name": archetype["name"],
                "persona": archetype["persona"],
                "lgb_params": {
                    "objective": "regression",
                    "metric": "mse",
                    "boosting_type": "gbdt",
                    "num_leaves": int(result.get("num_leaves", 64)),
                    "learning_rate": float(result.get("learning_rate", 0.01)),
                    "n_estimators": int(result.get("n_estimators", 2000)),
                    "feature_fraction": float(result.get("feature_fraction", 0.5)),
                    "bagging_fraction": float(result.get("bagging_fraction", 0.8)),
                    "bagging_freq": int(result.get("bagging_freq", 1)),
                    "min_child_samples": int(result.get("min_child_samples", 50)),
                    "lambda_l1": float(result.get("lambda_l1", 0.0)),
                    "lambda_l2": float(result.get("lambda_l2", 0.0)),
                    "verbose": -1,
                    "n_jobs": -1,
                    "seed": 42 + i,
                },
                "feature_sample_ratio": float(
                    result.get("feature_sample_ratio", 0.7)
                ),
                "era_subsample_ratio": float(
                    result.get("era_subsample_ratio", 1.0)
                ),
                "reasoning": result.get("reasoning", ""),
            }
            agent_configs.append(config)
            print("OK")

        except Exception as e:
            print(f"LLM failed ({e}), using defaults")
            config = _default_agent_config(i, archetype)
            agent_configs.append(config)

    return agent_configs


def _default_agent_config(agent_id: int, archetype: dict) -> dict:
    """Generate a default config when LLM is unavailable."""
    # Vary params based on style to maintain diversity
    style_params = {
        "aggressive": {"num_leaves": 128, "learning_rate": 0.05, "n_estimators": 1000},
        "conservative": {"num_leaves": 32, "learning_rate": 0.005, "n_estimators": 3000},
        "complex": {"num_leaves": 256, "learning_rate": 0.01, "n_estimators": 2000},
        "defensive": {"num_leaves": 48, "learning_rate": 0.008, "n_estimators": 2500},
        "contrarian": {"num_leaves": 96, "learning_rate": 0.02, "n_estimators": 1500},
        "macro": {"num_leaves": 24, "learning_rate": 0.01, "n_estimators": 3000},
        "adaptive": {"num_leaves": 64, "learning_rate": 0.05, "n_estimators": 800},
        "selective": {"num_leaves": 64, "learning_rate": 0.01, "n_estimators": 2000},
    }
    style = archetype.get("style", "conservative")
    params = style_params.get(style, style_params["conservative"])

    return {
        "agent_id": agent_id,
        "name": archetype["name"],
        "persona": archetype["persona"],
        "lgb_params": {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42 + agent_id,
            "feature_fraction": 0.5,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            **params,
        },
        "feature_sample_ratio": 0.7,
        "era_subsample_ratio": 1.0,
        "reasoning": f"Default {style} configuration",
    }


# ==============================================================================
# Multi-Agent Training & Ensemble
# ==============================================================================


def numerai_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Numerai's correlation metric (rank correlation)."""
    ranked_preds = pd.Series(y_pred).rank(pct=True, method="first")
    corr, _ = spearmanr(y_true, ranked_preds)
    return corr


def train_single_agent(
    agent_config: dict,
    train_df: pd.DataFrame,
    all_features: list[str],
) -> tuple[lgb.LGBMRegressor, list[str], float]:
    """
    Train a single agent's model.

    Returns the trained model, feature list used, and validation correlation.
    """
    agent_name = agent_config["name"]
    lgb_params = agent_config["lgb_params"]

    # Select feature subset
    rng = np.random.RandomState(lgb_params["seed"])
    feature_ratio = agent_config.get("feature_sample_ratio", 0.7)
    n_features = max(10, int(len(all_features) * feature_ratio))
    selected_features = list(rng.choice(all_features, size=n_features, replace=False))

    # Subsample eras if configured
    era_ratio = agent_config.get("era_subsample_ratio", 1.0)
    if "era" in train_df.columns and era_ratio < 1.0:
        eras = train_df["era"].unique()
        n_keep = max(10, int(len(eras) * era_ratio))
        selected_eras = set(rng.choice(eras, size=n_keep, replace=False))
        agent_train_df = train_df[train_df["era"].isin(selected_eras)]
    else:
        agent_train_df = train_df

    # Era-based train/val split
    if "era" in agent_train_df.columns:
        eras = agent_train_df["era"].unique()
        n_eras = len(eras)
        split_idx = int(n_eras * 0.8)
        val_eras = set(eras[split_idx:])
        train_mask = ~agent_train_df["era"].isin(val_eras)
        val_mask = agent_train_df["era"].isin(val_eras)
    else:
        n = len(agent_train_df)
        split_idx = int(n * 0.8)
        train_mask = pd.Series([True] * split_idx + [False] * (n - split_idx), index=agent_train_df.index)
        val_mask = ~train_mask

    X_train = agent_train_df.loc[train_mask, selected_features]
    y_train = agent_train_df.loc[train_mask, TARGET_COL]
    X_val = agent_train_df.loc[val_mask, selected_features]
    y_val = agent_train_df.loc[val_mask, TARGET_COL]

    # Train
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),  # silent
        ],
    )

    # Evaluate
    val_preds = model.predict(X_val)
    val_corr = numerai_corr(y_val.values, val_preds)

    return model, selected_features, val_corr


def train_agent_ensemble(
    agent_configs: list[dict],
    train_df: pd.DataFrame,
    all_features: list[str],
) -> list[dict]:
    """
    Train all agents and return their models + metadata.

    This mirrors MiroFish's simulation execution where multiple agents
    independently act and produce outputs.
    """
    print(f"\n{'='*60}")
    print("Multi-Agent Training (MiroFish Simulation)")
    print(f"{'='*60}")

    agent_results = []

    for config in tqdm(agent_configs, desc="Training agents"):
        agent_name = config["name"]
        agent_id = config["agent_id"]

        model, features, val_corr = train_single_agent(
            config, train_df, all_features
        )

        agent_results.append({
            "agent_id": agent_id,
            "name": agent_name,
            "model": model,
            "features": features,
            "val_corr": val_corr,
            "config": config,
        })

        print(
            f"  Agent {agent_id} ({agent_name}): "
            f"corr={val_corr:.6f}, "
            f"features={len(features)}, "
            f"trees={model.n_estimators_}"
        )

    # Summary
    corrs = [r["val_corr"] for r in agent_results]
    print(f"\nAgent ensemble summary:")
    print(f"  Mean correlation: {np.mean(corrs):.6f}")
    print(f"  Best agent: {agent_results[np.argmax(corrs)]['name']} ({max(corrs):.6f})")
    print(f"  Worst agent: {agent_results[np.argmin(corrs)]['name']} ({min(corrs):.6f})")

    return agent_results


def generate_ensemble_weights(
    llm: LLMClient,
    agent_results: list[dict],
) -> np.ndarray:
    """
    Use MiroFish's LLM to determine optimal ensemble weights.

    This mirrors MiroFish's ReportAgent which analyzes simulation results
    to generate insights.
    """
    print("\nGenerating ensemble weights via MiroFish LLM...")

    agent_summary = []
    for r in agent_results:
        agent_summary.append({
            "name": r["name"],
            "val_corr": round(r["val_corr"], 6),
            "n_features": len(r["features"]),
            "n_trees": r["model"].n_estimators_,
            "reasoning": r["config"].get("reasoning", ""),
        })

    prompt = f"""You are analyzing the results of a multi-agent ensemble for the Numerai tournament.
Each agent trained an independent model with different hyperparameters and feature subsets.

Agent results:
{json.dumps(agent_summary, indent=2)}

Assign a weight (0.0 to 1.0) to each agent for the final ensemble.
Consider:
- Higher correlation should generally get more weight
- But diversity is valuable - don't zero out agents completely
- Agents with very different strategies complement each other

Return JSON:
{{
    "weights": [<weight for agent 0>, <weight for agent 1>, ...],
    "reasoning": "<brief explanation>"
}}"""

    try:
        result = llm.chat_json(
            messages=[
                {
                    "role": "system",
                    "content": "You are a portfolio construction expert. Return valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )

        weights = np.array(result.get("weights", []))
        if len(weights) != len(agent_results):
            raise ValueError("Weight count mismatch")
        weights = np.clip(weights, 0.01, 10.0)
        weights = weights / weights.sum()
        print(f"  LLM weights: {weights.round(3)}")
        print(f"  Reasoning: {result.get('reasoning', 'N/A')}")
        return weights

    except Exception as e:
        print(f"  LLM weighting failed ({e}), using correlation-based weights")
        return _correlation_based_weights(agent_results)


def _correlation_based_weights(agent_results: list[dict]) -> np.ndarray:
    """Fallback: weight by validation correlation."""
    corrs = np.array([r["val_corr"] for r in agent_results])
    # Softmax-like weighting
    corrs = np.clip(corrs, 0, None)
    weights = np.exp(corrs * 10)
    weights = weights / weights.sum()
    return weights


def generate_predictions(
    agent_results: list[dict],
    weights: np.ndarray,
    live_df: pd.DataFrame,
    current_round: int,
) -> pd.DataFrame:
    """Generate weighted ensemble predictions from all agents."""
    print("\nGenerating ensemble predictions on live data...")

    all_preds = []
    for r in agent_results:
        preds = r["model"].predict(live_df[r["features"]])
        # Rank each agent's predictions
        ranked = pd.Series(preds, index=live_df.index).rank(pct=True, method="first")
        all_preds.append(ranked.values)

    # Weighted average of ranked predictions
    all_preds = np.array(all_preds)
    ensemble_preds = np.average(all_preds, axis=0, weights=weights)

    # Final ranking
    final_ranked = pd.Series(ensemble_preds, index=live_df.index).rank(
        pct=True, method="first"
    )

    pred_df = pd.DataFrame({"prediction": final_ranked.values}, index=live_df.index)

    # Save
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = PREDICTIONS_DIR / f"predictions_round{current_round}_{timestamp}.csv"
    pred_df.to_csv(pred_path)
    print(f"Predictions saved to {pred_path}")
    print(
        f"Prediction stats: min={pred_df['prediction'].min():.4f}, "
        f"max={pred_df['prediction'].max():.4f}, "
        f"mean={pred_df['prediction'].mean():.4f}"
    )

    return pred_df


def save_agent_models(agent_results: list[dict], current_round: int):
    """Save all agent models and metadata."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for r in agent_results:
        model_path = MODELS_DIR / f"agent{r['agent_id']}_{r['name'].replace(' ', '_')}_round{current_round}_{timestamp}.txt"
        r["model"].booster_.save_model(str(model_path))

    # Save ensemble metadata
    meta = {
        "round": current_round,
        "timestamp": timestamp,
        "agents": [
            {
                "agent_id": r["agent_id"],
                "name": r["name"],
                "val_corr": r["val_corr"],
                "n_features": len(r["features"]),
                "n_trees": r["model"].n_estimators_,
                "config": {
                    k: v
                    for k, v in r["config"].items()
                    if k != "persona"
                },
            }
            for r in agent_results
        ],
    }
    meta_path = MODELS_DIR / f"ensemble_meta_round{current_round}_{timestamp}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Agent models and metadata saved to {MODELS_DIR}")


def submit_predictions(
    napi: NumerAPI,
    pred_df: pd.DataFrame,
    model_name: str,
    current_round: int,
) -> None:
    """Submit predictions to Numerai tournament."""
    print(f"\nSubmitting predictions for model '{model_name}', round {current_round}...")
    submission_path = PREDICTIONS_DIR / f"submission_round{current_round}.csv"
    pred_df.to_csv(submission_path)
    submission_id = napi.upload_predictions(str(submission_path), model_id=model_name)
    print(f"Submission successful! Submission ID: {submission_id}")


# ==============================================================================
# Main Pipeline
# ==============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MiroFish Numerai Multi-Agent Prediction Pipeline"
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
        "--num-agents",
        type=int,
        default=5,
        help="Number of strategy agents in ensemble (default: 5)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM and use default agent configurations",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("MiroFish - Numerai Multi-Agent Prediction Pipeline")
    print("=" * 60)
    print(f"Strategy: {args.num_agents} independent analyst agents")
    print()

    # Initialize Numerai API client
    public_id = os.environ.get("NUMERAI_PUBLIC_ID")
    secret_key = os.environ.get("NUMERAI_SECRET_KEY")

    if public_id and secret_key:
        napi = NumerAPI(public_id=public_id, secret_key=secret_key)
        print("Authenticated with Numerai API")
    else:
        napi = NumerAPI()
        print("Using unauthenticated Numerai API (download only)")
        if args.submit:
            print("ERROR: Cannot submit without API credentials.")
            print("Set NUMERAI_PUBLIC_ID and NUMERAI_SECRET_KEY env vars.")
            sys.exit(1)

    # Initialize MiroFish LLM client
    llm = None
    if not args.no_llm:
        try:
            llm = LLMClient()
            print(f"MiroFish LLM connected: {llm.model} @ {llm.base_url}")
        except Exception as e:
            print(f"LLM unavailable ({e}), using default configs")
            llm = None

    # Step 1: Download data
    train_df, live_df, current_round = download_latest_dataset(napi)

    # Step 2: Get feature columns
    feature_metadata = load_feature_metadata()
    all_features = get_all_feature_columns(train_df, feature_metadata)

    # Step 3: Create agent configurations (MiroFish-style profile generation)
    if llm:
        agent_configs = create_agent_configs(llm, all_features, args.num_agents)
    else:
        archetypes = AGENT_ARCHETYPES[: args.num_agents]
        agent_configs = [
            _default_agent_config(i, arch) for i, arch in enumerate(archetypes)
        ]

    # Save agent configs for reproducibility
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config_path = MODELS_DIR / f"agent_configs_round{current_round}.json"
    with open(config_path, "w") as f:
        json.dump(
            [{k: v for k, v in c.items() if k != "persona"} for c in agent_configs],
            f,
            indent=2,
        )

    # Step 4: Train all agents (MiroFish-style parallel simulation)
    agent_results = train_agent_ensemble(agent_configs, train_df, all_features)

    # Step 5: Generate ensemble weights (MiroFish-style report analysis)
    if llm:
        weights = generate_ensemble_weights(llm, agent_results)
    else:
        weights = _correlation_based_weights(agent_results)
        print(f"Correlation-based weights: {weights.round(3)}")

    # Step 6: Save models
    save_agent_models(agent_results, current_round)

    # Step 7: Generate ensemble predictions
    pred_df = generate_predictions(agent_results, weights, live_df, current_round)

    # Step 8: Optionally submit
    if args.submit:
        if not args.model_name:
            print("ERROR: --model-name required for submission.")
            sys.exit(1)
        submit_predictions(napi, pred_df, args.model_name, current_round)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
