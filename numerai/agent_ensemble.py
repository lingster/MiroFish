"""
MiroFish Multi-Agent Ensemble Engine

Shared agent system used by all Numerai tournament pipelines (Classic, Signals, Crypto).
Provides LLM-powered agent configuration, training, ensemble weighting, and prediction.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

# Add MiroFish backend to path so we can import its services
BACKEND_DIR = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.utils.llm_client import LLMClient


# ==============================================================================
# Metrics
# ==============================================================================


def numerai_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Numerai's correlation metric (Spearman rank correlation)."""
    ranked_preds = pd.Series(y_pred).rank(pct=True, method="first")
    corr, _ = spearmanr(y_true, ranked_preds)
    return corr


# ==============================================================================
# Agent Archetypes
# ==============================================================================

# Classic/Signals archetypes: stock market-oriented strategies
STOCK_AGENT_ARCHETYPES = [
    {
        "name": "Momentum Trader",
        "persona": (
            "You are an aggressive momentum trader who believes recent trends "
            "continue. You prefer models that react quickly to recent data and "
            "use features related to recent price movements and volume."
        ),
        "style": "aggressive",
    },
    {
        "name": "Value Analyst",
        "persona": (
            "You are a conservative value analyst who looks for undervalued "
            "assets. You prefer stable, slow-learning models and features "
            "related to fundamental valuation metrics."
        ),
        "style": "conservative",
    },
    {
        "name": "Quant Researcher",
        "persona": (
            "You are a quantitative researcher who uses statistical signals. "
            "You focus on feature interactions and non-linear patterns, "
            "preferring complex models with many leaves."
        ),
        "style": "complex",
    },
    {
        "name": "Risk Manager",
        "persona": (
            "You are a risk-averse portfolio manager. You prioritize stability "
            "over returns, using heavily regularized models. You prefer features "
            "that capture downside risk and correlation."
        ),
        "style": "defensive",
    },
    {
        "name": "Contrarian Trader",
        "persona": (
            "You are a contrarian who bets against the crowd. You look for "
            "mean-reversion signals and use features that capture overreaction "
            "and sentiment extremes."
        ),
        "style": "contrarian",
    },
    {
        "name": "Macro Strategist",
        "persona": (
            "You are a macro strategist who focuses on broad market regimes. "
            "You use features related to market-wide factors and economic "
            "cycles, preferring simple but robust models."
        ),
        "style": "macro",
    },
    {
        "name": "Adaptive Learner",
        "persona": (
            "You are an adaptive ML practitioner who emphasizes recent data. "
            "You use high learning rates with aggressive early stopping and "
            "prefer a diverse mix of features."
        ),
        "style": "adaptive",
    },
    {
        "name": "Feature Engineer",
        "persona": (
            "You are a feature engineering specialist. You use a small, "
            "carefully selected set of the most predictive features with "
            "a deep, narrow model architecture."
        ),
        "style": "selective",
    },
]

# Crypto archetypes: crypto market-oriented strategies
CRYPTO_AGENT_ARCHETYPES = [
    {
        "name": "Momentum Surfer",
        "persona": (
            "You are a crypto momentum trader who rides strong trends. "
            "Crypto markets trend harder than equities. You prefer fast-reacting "
            "models using momentum, RSI, and volume features."
        ),
        "style": "aggressive",
    },
    {
        "name": "Mean Reversion Trader",
        "persona": (
            "You are a crypto mean-reversion specialist. You look for "
            "overextended moves using Bollinger bands, RSI extremes, and "
            "volatility spikes, betting on reversion to the mean."
        ),
        "style": "contrarian",
    },
    {
        "name": "Market Cap Analyst",
        "persona": (
            "You are a fundamental crypto analyst focused on market cap flows. "
            "You use market cap features and EWA signals to identify assets "
            "gaining relative strength. Conservative model settings."
        ),
        "style": "conservative",
    },
    {
        "name": "Volatility Trader",
        "persona": (
            "You are a crypto volatility specialist. You use volatility and "
            "Sharpe ratio features to identify risk-adjusted opportunities. "
            "Heavily regularized models to avoid overfitting to noise."
        ),
        "style": "defensive",
    },
    {
        "name": "Volume Flow Analyst",
        "persona": (
            "You are a volume flow analyst. You believe volume precedes price "
            "in crypto. You focus on volume features and their moving averages "
            "to predict price direction."
        ),
        "style": "macro",
    },
    {
        "name": "Multi-Timeframe Quant",
        "persona": (
            "You are a multi-timeframe quant who combines 20d and 60d signals. "
            "You use complex models to capture interactions between short and "
            "long timeframes."
        ),
        "style": "complex",
    },
    {
        "name": "Adaptive Crypto Learner",
        "persona": (
            "You are an adaptive ML practitioner specializing in crypto markets. "
            "High learning rate with aggressive early stopping. Crypto regimes "
            "shift fast, so you emphasize recent data."
        ),
        "style": "adaptive",
    },
    {
        "name": "Technical Minimalist",
        "persona": (
            "You are a technical analyst who uses only the most reliable "
            "indicators. Small curated feature set with RSI, momentum, and "
            "Bollinger bands."
        ),
        "style": "selective",
    },
]

# Default LGB params per style
_STYLE_PARAMS = {
    "aggressive": {"num_leaves": 128, "learning_rate": 0.05, "n_estimators": 1000},
    "conservative": {"num_leaves": 32, "learning_rate": 0.005, "n_estimators": 3000},
    "complex": {"num_leaves": 256, "learning_rate": 0.01, "n_estimators": 2000},
    "defensive": {"num_leaves": 48, "learning_rate": 0.008, "n_estimators": 2500},
    "contrarian": {"num_leaves": 96, "learning_rate": 0.02, "n_estimators": 1500},
    "macro": {"num_leaves": 24, "learning_rate": 0.01, "n_estimators": 3000},
    "adaptive": {"num_leaves": 64, "learning_rate": 0.05, "n_estimators": 800},
    "selective": {"num_leaves": 64, "learning_rate": 0.01, "n_estimators": 2000},
}


# ==============================================================================
# Agent Configuration
# ==============================================================================


def create_agent_configs(
    llm: Optional[LLMClient],
    all_features: list[str],
    num_agents: int,
    archetypes: list[dict],
    tournament_context: str = "Numerai stock prediction tournament",
) -> list[dict]:
    """
    Generate diverse agent configurations, optionally using MiroFish LLM.

    Args:
        llm: MiroFish LLM client (None for default configs)
        all_features: list of available feature column names
        num_agents: number of agents to create
        archetypes: list of agent archetype dicts (name, persona, style)
        tournament_context: description of the tournament for LLM prompt
    """
    # Cycle archetypes if we need more agents than archetypes
    selected = []
    for i in range(num_agents):
        selected.append(archetypes[i % len(archetypes)])

    if llm is None:
        return [
            _default_agent_config(i, arch) for i, arch in enumerate(selected)
        ]

    feature_sample_str = ", ".join(all_features[:20])

    print(f"\nGenerating {num_agents} agent configurations via MiroFish LLM...")

    agent_configs = []
    for i, archetype in enumerate(selected):
        print(f"  Agent {i}: {archetype['name']}...", end=" ", flush=True)

        prompt = f"""You are configuring a LightGBM model for the {tournament_context}.

Agent persona: {archetype['persona']}
Strategy style: {archetype['style']}

Total available features: {len(all_features)}
Feature names: {feature_sample_str}{"..." if len(all_features) > 20 else ""}

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
            agent_configs.append(_default_agent_config(i, archetype))

    return agent_configs


def _default_agent_config(agent_id: int, archetype: dict) -> dict:
    """Generate a default config when LLM is unavailable."""
    style = archetype.get("style", "conservative")
    params = _STYLE_PARAMS.get(style, _STYLE_PARAMS["conservative"])

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
# Training
# ==============================================================================


def train_single_agent(
    agent_config: dict,
    train_df: pd.DataFrame,
    all_features: list[str],
    target_col: str = "target",
    era_col: str = "era",
) -> tuple[lgb.LGBMRegressor, list[str], float]:
    """
    Train a single agent's model.

    Args:
        agent_config: agent configuration dict
        train_df: training dataframe
        all_features: all available feature columns
        target_col: name of the target column
        era_col: name of the era/date column for time-based splitting

    Returns:
        (trained model, feature list used, validation correlation)
    """
    lgb_params = agent_config["lgb_params"]

    # Select feature subset
    rng = np.random.RandomState(lgb_params["seed"])
    feature_ratio = agent_config.get("feature_sample_ratio", 0.7)
    n_features = max(3, int(len(all_features) * feature_ratio))
    n_features = min(n_features, len(all_features))
    selected_features = list(rng.choice(all_features, size=n_features, replace=False))

    # Subsample eras if configured
    era_ratio = agent_config.get("era_subsample_ratio", 1.0)
    if era_col in train_df.columns and era_ratio < 1.0:
        eras = train_df[era_col].unique()
        n_keep = max(5, int(len(eras) * era_ratio))
        selected_eras = set(rng.choice(eras, size=n_keep, replace=False))
        agent_train_df = train_df[train_df[era_col].isin(selected_eras)]
    else:
        agent_train_df = train_df

    # Era-based or time-based train/val split
    if era_col in agent_train_df.columns:
        eras = sorted(agent_train_df[era_col].unique())
        n_eras = len(eras)
        split_idx = int(n_eras * 0.8)
        val_eras = set(eras[split_idx:])
        train_mask = ~agent_train_df[era_col].isin(val_eras)
        val_mask = agent_train_df[era_col].isin(val_eras)
    else:
        n = len(agent_train_df)
        split_idx = int(n * 0.8)
        train_mask = pd.Series(
            [True] * split_idx + [False] * (n - split_idx),
            index=agent_train_df.index,
        )
        val_mask = ~train_mask

    X_train = agent_train_df.loc[train_mask, selected_features]
    y_train = agent_train_df.loc[train_mask, target_col]
    X_val = agent_train_df.loc[val_mask, selected_features]
    y_val = agent_train_df.loc[val_mask, target_col]

    # Drop NaN targets
    train_valid = y_train.notna()
    val_valid = y_val.notna()
    X_train, y_train = X_train[train_valid], y_train[train_valid]
    X_val, y_val = X_val[val_valid], y_val[val_valid]

    if len(X_val) == 0 or len(X_train) == 0:
        raise ValueError("Empty train or validation set after NaN removal")

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_preds = model.predict(X_val)
    val_corr = numerai_corr(y_val.values, val_preds)

    return model, selected_features, val_corr


def train_agent_ensemble(
    agent_configs: list[dict],
    train_df: pd.DataFrame,
    all_features: list[str],
    target_col: str = "target",
    era_col: str = "era",
    tournament_name: str = "Classic",
) -> list[dict]:
    """Train all agents and return their models + metadata."""
    print(f"\n{'='*60}")
    print(f"Multi-Agent Training - {tournament_name} (MiroFish Simulation)")
    print(f"{'='*60}")

    agent_results = []

    for config in tqdm(agent_configs, desc="Training agents"):
        try:
            model, features, val_corr = train_single_agent(
                config, train_df, all_features, target_col, era_col
            )
            agent_results.append({
                "agent_id": config["agent_id"],
                "name": config["name"],
                "model": model,
                "features": features,
                "val_corr": val_corr,
                "config": config,
            })
            print(
                f"  Agent {config['agent_id']} ({config['name']}): "
                f"corr={val_corr:.6f}, "
                f"features={len(features)}, "
                f"trees={model.n_estimators_}"
            )
        except Exception as e:
            print(f"  Agent {config['agent_id']} ({config['name']}): FAILED - {e}")

    if not agent_results:
        raise RuntimeError("All agents failed to train")

    corrs = [r["val_corr"] for r in agent_results]
    print(f"\nAgent ensemble summary:")
    print(f"  Successful agents: {len(agent_results)}/{len(agent_configs)}")
    print(f"  Mean correlation: {np.mean(corrs):.6f}")
    print(f"  Best agent: {agent_results[np.argmax(corrs)]['name']} ({max(corrs):.6f})")
    print(f"  Worst agent: {agent_results[np.argmin(corrs)]['name']} ({min(corrs):.6f})")

    return agent_results


# ==============================================================================
# Ensemble Weighting
# ==============================================================================


def generate_ensemble_weights(
    llm: Optional[LLMClient],
    agent_results: list[dict],
    tournament_name: str = "Classic",
) -> np.ndarray:
    """Generate ensemble weights, optionally via LLM."""
    if llm is None:
        weights = _correlation_based_weights(agent_results)
        print(f"Correlation-based weights: {weights.round(3)}")
        return weights

    print("\nGenerating ensemble weights via MiroFish LLM...")

    agent_summary = [
        {
            "name": r["name"],
            "val_corr": round(r["val_corr"], 6),
            "n_features": len(r["features"]),
            "n_trees": r["model"].n_estimators_,
            "reasoning": r["config"].get("reasoning", ""),
        }
        for r in agent_results
    ]

    prompt = f"""You are analyzing the results of a multi-agent ensemble for the Numerai {tournament_name} tournament.
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
    corrs = np.clip(corrs, 0, None)
    weights = np.exp(corrs * 10)
    weights = weights / weights.sum()
    return weights


# ==============================================================================
# Predictions
# ==============================================================================


def generate_ensemble_predictions(
    agent_results: list[dict],
    weights: np.ndarray,
    live_df: pd.DataFrame,
    output_dir: Path,
    current_round: int,
    prefix: str = "predictions",
) -> pd.DataFrame:
    """Generate weighted ensemble predictions from all agents."""
    print("\nGenerating ensemble predictions on live data...")

    all_preds = []
    for r in agent_results:
        preds = r["model"].predict(live_df[r["features"]])
        ranked = pd.Series(preds, index=live_df.index).rank(pct=True, method="first")
        all_preds.append(ranked.values)

    all_preds = np.array(all_preds)
    ensemble_preds = np.average(all_preds, axis=0, weights=weights)

    final_ranked = pd.Series(ensemble_preds, index=live_df.index).rank(
        pct=True, method="first"
    )

    pred_df = pd.DataFrame({"prediction": final_ranked.values}, index=live_df.index)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path = output_dir / f"{prefix}_round{current_round}_{timestamp}.csv"
    pred_df.to_csv(pred_path)
    print(f"Predictions saved to {pred_path}")
    print(
        f"Prediction stats: min={pred_df['prediction'].min():.4f}, "
        f"max={pred_df['prediction'].max():.4f}, "
        f"mean={pred_df['prediction'].mean():.4f}"
    )

    return pred_df


def save_agent_models(
    agent_results: list[dict],
    models_dir: Path,
    current_round: int,
    prefix: str = "",
):
    """Save all agent models and ensemble metadata."""
    models_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{prefix}_" if prefix else ""

    for r in agent_results:
        safe_name = r["name"].replace(" ", "_")
        model_path = models_dir / f"{tag}agent{r['agent_id']}_{safe_name}_round{current_round}_{timestamp}.txt"
        r["model"].booster_.save_model(str(model_path))

    meta = {
        "round": current_round,
        "timestamp": timestamp,
        "prefix": prefix,
        "agents": [
            {
                "agent_id": r["agent_id"],
                "name": r["name"],
                "val_corr": r["val_corr"],
                "n_features": len(r["features"]),
                "n_trees": r["model"].n_estimators_,
                "config": {
                    k: v for k, v in r["config"].items() if k != "persona"
                },
            }
            for r in agent_results
        ],
    }
    meta_path = models_dir / f"{tag}ensemble_meta_round{current_round}_{timestamp}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Agent models and metadata saved to {models_dir}")


# ==============================================================================
# LLM Initialization
# ==============================================================================


def init_llm(no_llm: bool = False) -> Optional[LLMClient]:
    """Initialize MiroFish LLM client."""
    if no_llm:
        return None
    try:
        llm = LLMClient()
        print(f"MiroFish LLM connected: {llm.model} @ {llm.base_url}")
        return llm
    except Exception as e:
        print(f"LLM unavailable ({e}), using default configs")
        return None
