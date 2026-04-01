"""
Unified Goal Outcome Model Trainer (minute-based, all states)

This version trains on *all* logged match states (not only spikes / high-pressure)
from a minute-level log CSV. Labels are computed here, not in the live bot.

Data source (minute states):
    minute_states_log.csv  (or env MINUTE_LOG_PATH)

Required columns:
    timestamp, match_id, league, home, away,
    minute, half,
    goals_home, goals_away, goals_total,
    total_pressure, pressure_diff,
    shots_on_target, total_shots,
    corners_total, possession_ratio

We derive, per (match_id, minute):

    final_total_goals = max(goals_total over match)
    label_goal_next_10m:
        1 if there is a later row within +10 minutes where goals_total > current
    label_goal_before_ht:
        1 if half == 1 and a later row with minute <= 45 has goals_total > current
    label_final_over_2_5:
        1 if final_total_goals >= 3

Targets:
    GOAL_TARGET_TYPE env var:
        "next_10m"        -> label_goal_next_10m
        "before_ht"       -> label_goal_before_ht
        "final_over_2_5"  -> label_final_over_2_5

Output:
    goal_model_<target_type>.pkl
"""

import os
import pickle
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

import warnings
warnings.filterwarnings("ignore")

# Map logical target types to CSV label columns
TARGET_COL_MAP = {
    "next_10m": "label_goal_next_10m",
    "before_ht": "label_goal_before_ht",
    "final_over_2_5": "label_final_over_2_5",
}


class GoalOutcomeModel:
    """
    Minute-level unified model for goal-related outcomes.

    Feature set (designed to match live bot usage via GoalOutcomeMultiPredictor):
        - league_goal_rate
        - minute
        - half
        - goals_total
        - total_pressure
        - pressure_diff
        - shots_on_target
        - total_shots
        - corners_total
        - corners_in_window   (computed from corners_total over last 10 minutes)
        - possession_ratio
    """

    def __init__(self, model_type: str = "gradient_boosting", target_type: str = "next_10m"):
        if target_type not in TARGET_COL_MAP:
            raise ValueError(f"Unknown target_type: {target_type}")
        self.model_type = model_type
        self.target_type = target_type
        self.target_col = TARGET_COL_MAP[target_type]

        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = [
            "league_goal_rate",
            "minute",
            "half",
            "goals_total",
            "total_pressure",
            "pressure_diff",
            "shots_on_target",
            "total_shots",
            "corners_total",
            "corners_in_window",
            "possession_ratio",
        ]
        self.is_trained = False
        self.feature_importance: Dict[str, float] = {}

    def _create_model(self):
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                min_samples_split=4,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42,
            )
        else:  # logistic_regression
            self.model = LogisticRegression(
                max_iter=2000,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )

    # ------------------------------------------------------------------
    # DATA LOADING & LABELING
    # ------------------------------------------------------------------
    def load_minute_data_with_labels(
        self,
        csv_path: str = "minute_states_log.csv",
        min_rows: int = 1000,
        goal_window_minutes: int = 10,
    ) -> pd.DataFrame:
        """
        Load minute-level states and compute labels for all targets.
        Returns a dataframe with features + target column.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Minute states CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        required_cols = [
            "match_id",
            "league",
            "minute",
            "half",
            "goals_home",
            "goals_away",
            "goals_total",
            "total_pressure",
            "pressure_diff",
            "shots_on_target",
            "total_shots",
            "corners_total",
            "possession_ratio",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in minute_states_log.csv: {missing}")

        # Ensure numeric types
        numeric_cols = [
            "minute",
            "half",
            "goals_home",
            "goals_away",
            "goals_total",
            "total_pressure",
            "pressure_diff",
            "shots_on_target",
            "total_shots",
            "corners_total",
            "possession_ratio",
        ]
        for col in numeric_cols:
            df[col] = df[col].fillna(0).astype(float)

        # Sort for stable labeling
        df = df.sort_values(["match_id", "minute"]).reset_index(drop=True)

        # final_total_goals per match
        final_goals = df.groupby("match_id")["goals_total"].max().to_dict()
        df["final_total_goals"] = df["match_id"].map(final_goals).astype(float)

        # Simple over 2.5 label
        df["label_final_over_2_5"] = (df["final_total_goals"] >= 3).astype(int)

        # Pre-allocate labels
        df["label_goal_next_10m"] = 0
        df["label_goal_before_ht"] = 0

        grouped = df.groupby("match_id", sort=False)
        next_10 = np.zeros(len(df), dtype=int)
        before_ht = np.zeros(len(df), dtype=int)

        window = goal_window_minutes

        for match_id, idx in grouped.indices.items():
            idx_arr = np.array(sorted(idx))
            minutes = df.loc[idx_arr, "minute"].values.astype(int)
            goals = df.loc[idx_arr, "goals_total"].values.astype(int)
            halves = df.loc[idx_arr, "half"].values.astype(int)

            n = len(idx_arr)
            for pos in range(n):
                m_i = minutes[pos]
                g_i = goals[pos]
                cutoff_n10 = m_i + window

                label_n10 = 0
                label_bht = 0

                for pos2 in range(pos + 1, n):
                    m_j = minutes[pos2]
                    g_j = goals[pos2]
                    if g_j > g_i:
                        if m_j <= cutoff_n10:
                            label_n10 = 1
                        if halves[pos] == 1 and m_j <= 45:
                            label_bht = 1
                    if m_j > cutoff_n10 and m_j > 45:
                        break

                next_10[idx_arr[pos]] = label_n10
                before_ht[idx_arr[pos]] = label_bht

        df["label_goal_next_10m"] = next_10
        df["label_goal_before_ht"] = before_ht

        # corners_in_window: extra info about recent attacking
        df["corners_in_window"] = 0.0
        for match_id, idx in grouped.indices.items():
            idx_arr = np.array(sorted(idx))
            minutes = df.loc[idx_arr, "minute"].values.astype(int)
            corners = df.loc[idx_arr, "corners_total"].values.astype(float)
            in_window = np.zeros(len(idx_arr), dtype=float)

            for i in range(len(idx_arr)):
                m_i = minutes[i]
                cutoff = m_i - window
                baseline = corners[i]
                for j in range(i - 1, -1, -1):
                    if minutes[j] < cutoff:
                        break
                    baseline = corners[j]
                in_window[i] = max(0.0, corners[i] - baseline)

            df.loc[idx_arr, "corners_in_window"] = in_window

        # league_goal_rate for THIS target
        target_col = self.target_col
        if target_col not in df.columns:
            raise ValueError(f"Internal error: missing target column {target_col}")

        league_rates = df.groupby("league")[target_col].mean().to_dict()
        df["league_goal_rate"] = df["league"].map(league_rates).astype(float)
        self.league_goal_rates = league_rates

        if len(df) < min_rows:
            print(
                f"WARNING: Only {len(df)} rows available (< {min_rows}). "
                "Training will still run but may be noisy."
            )

        cols_needed = set(self.feature_names + [target_col])
        missing_final = cols_needed - set(df.columns)
        if missing_final:
            raise ValueError(f"Missing columns after preparation: {missing_final}")

        df = df[list(self.feature_names) + [target_col]]
        df = df.dropna(subset=[target_col])
        df["target"] = df[target_col].astype(int)
        return df

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------
    def train(self, df_train: pd.DataFrame, test_size: float = 0.2, verbose: bool = True):
        for col in self.feature_names + ["target"]:
            if col not in df_train.columns:
                raise ValueError(f"Training data missing required column: {col}")

        if len(df_train) < 50:
            raise ValueError("Not enough samples to train (need at least 50).")

        X = df_train[self.feature_names].fillna(0.0)
        y = df_train["target"].astype(int)

        stratify = y if len(y.unique()) > 1 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=stratify,
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self._create_model()
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test_scaled)
        if hasattr(self.model, "predict_proba"):
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred_proba = np.zeros_like(y_pred, dtype=float)

        if verbose:
            print("=" * 60)
            print("UNIFIED GOAL OUTCOME MODEL - TRAINING RESULTS")
            print("=" * 60)
            print(f"Target type: {self.target_type} ({self.target_col})")
            print(f"Model Type:  {self.model_type}")
            print(f"Total events: {len(X)}")
            print(f"Train events: {len(X_train)}")
            print(f"Test events:  {len(X_test)}")
            print(f"Positive rate: {y.mean():.2%}")
            print()
            print("TEST SET PERFORMANCE:")
            acc = (y_pred == y_test).mean()
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
            print(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
            print(f"  F1-Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
            if len(np.unique(y_test)) > 1 and hasattr(self.model, "predict_proba"):
                try:
                    auc = roc_auc_score(y_test, y_pred_proba)
                    print(f"  ROC-AUC:   {auc:.4f}")
                except Exception:
                    pass

            print()
            print("CLASSIFICATION REPORT:")
            print(
                classification_report(
                    y_test,
                    y_pred,
                    target_names=["No", "Yes"],
                    zero_division=0,
                )
            )
            print()
            print("CONFUSION MATRIX:")
            print(confusion_matrix(y_test, y_pred))
            print()

            if hasattr(self.model, "feature_importances_"):
                imp = self.model.feature_importances_
                self.feature_importance = dict(zip(self.feature_names, imp))
                print("FEATURE IMPORTANCE:")
                for fname, score in sorted(
                    self.feature_importance.items(), key=lambda x: -x[1]
                ):
                    print(f"  {fname}: {score:.4f}")
            print("=" * 60)

    # ------------------------------------------------------------------
    # PREDICTION (for offline tests)
    # ------------------------------------------------------------------
    def predict_proba(self, features: Dict[str, Any]) -> float:
        if not self.is_trained:
            raise ValueError("Model not trained.")

        x = [float(features.get(fname, 0.0) or 0.0) for fname in self.feature_names]
        X_scaled = self.scaler.transform([x])

        if hasattr(self.model, "predict_proba"):
            proba = float(self.model.predict_proba(X_scaled)[0][1])
        else:
            proba = float(self.model.predict(X_scaled)[0])

        return proba

    # ------------------------------------------------------------------
    # SAVE / LOAD
    # ------------------------------------------------------------------
    def save_model(self, filepath: str):
        if not self.is_trained:
            raise ValueError("Model not trained. Cannot save.")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
            "target_type": self.target_type,
            "target_col": self.target_col,
            "feature_importance": self.feature_importance,
            "league_goal_rates": getattr(self, "league_goal_rates", {}),
            "trained_at": datetime.now().isoformat(),
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Saved model to {filepath}")
        

if __name__ == "__main__":
    MINUTE_LOG_PATH = os.getenv("MINUTE_LOG_PATH", "minute_states_log.csv")
    MODEL_TYPE = os.getenv("GOAL_MODEL_TYPE", "gradient_boosting")
    TARGET_TYPE = os.getenv("GOAL_TARGET_TYPE", "next_10m")  # "next_10m" | "before_ht" | "final_over_2_5"

    target_col = TARGET_COL_MAP.get(TARGET_TYPE, "label_goal_next_10m")
    MODEL_PATH = f"goal_model_{TARGET_TYPE}.pkl"

    print("============================================================")
    print("Unified Goal Outcome Model Trainer (minute-based)")
    print("============================================================")
    print(f"Minute log CSV: {MINUTE_LOG_PATH}")
    print(f"Model type:     {MODEL_TYPE}")
    print(f"Target type:    {TARGET_TYPE} ({target_col})")
    print(f"Output:         {MODEL_PATH}")
    print("============================================================")

    model = GoalOutcomeModel(model_type=MODEL_TYPE, target_type=TARGET_TYPE)

    print("Loading minute-level data and computing labels...")
    df_train = model.load_minute_data_with_labels(
        csv_path=MINUTE_LOG_PATH,
        min_rows=1000,
        goal_window_minutes=int(os.getenv("GOAL_WINDOW_MINUTES", "10")),
    )
    print(f"Loaded {len(df_train)} rows.")
    print(f"Positive rate: {df_train['target'].mean():.2%}")
    print()

    print("Training model...")
    model.train(df_train, test_size=0.2, verbose=True)

    print("\nSaving model...")
    model.save_model(MODEL_PATH)

    print("\nDone. Use this model in your live bot to estimate P(goal | current state).")
