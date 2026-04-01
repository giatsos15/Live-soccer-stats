"""
Goals Regression Trainer – Expected Remaining Goals (minute-based)

We train a regression model to estimate:
    remaining_goals = final_total_goals - goals_total_at_minute

Data source:
    minute_states_log.csv (or env MINUTE_LOG_PATH)

Required columns:
    timestamp, match_id, league, home, away,
    minute, half,
    goals_home, goals_away, goals_total,
    total_pressure, pressure_diff,
    shots_on_target, total_shots,
    corners_total, possession_ratio

We derive, per match_id:
    final_total_goals = max(goals_total)
    remaining_goals = final_total_goals - goals_total (clipped to [0, max_remaining_clip])

Features (aligned with classification models):
    - league_goal_rate   (mean final goals per match in that league)
    - minute
    - half
    - goals_total
    - total_pressure
    - pressure_diff
    - shots_on_target
    - total_shots
    - corners_total
    - corners_in_window (corners gained in last 10 minutes)
    - possession_ratio

Output:
    goals_remaining_regressor.pkl
"""

import os
import pickle
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

import warnings
warnings.filterwarnings("ignore")


class GoalsRemainingRegressor:
    def __init__(self, model_type: str = "gradient_boosting"):
        self.model_type = model_type
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
            self.model = RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                min_samples_split=4,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42,
            )
        else:
            self.model = LinearRegression(n_jobs=-1)

    # ------------------------------------------------------------------
    # DATA LOADING
    # ------------------------------------------------------------------
    def load_minute_data(
        self,
        csv_path: str = "minute_states_log.csv",
        min_rows: int = 1000,
        goal_window_minutes: int = 10,
        max_remaining_clip: float = 6.0,
    ) -> pd.DataFrame:
        """
        Load minute-level states and build remaining_goals target.
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

        df = df.sort_values(["match_id", "minute"]).reset_index(drop=True)

        # final_total_goals per match
        final_goals = df.groupby("match_id")["goals_total"].max().to_dict()
        df["final_total_goals"] = df["match_id"].map(final_goals).astype(float)

        # remaining_goals target
        df["remaining_goals"] = df["final_total_goals"] - df["goals_total"]
        df["remaining_goals"] = df["remaining_goals"].clip(lower=0.0, upper=max_remaining_clip)

        # corners_in_window (same as in classifier)
        df["corners_in_window"] = 0.0
        grouped = df.groupby("match_id", sort=False)
        window = goal_window_minutes

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

        # league_goal_rate = average final_total_goals per match in league
        df_matches = df[["league", "match_id", "final_total_goals"]].drop_duplicates("match_id")
        league_rates = df_matches.groupby("league")["final_total_goals"].mean().to_dict()
        df["league_goal_rate"] = df["league"].map(league_rates).astype(float)

        if len(df) < min_rows:
            print(
                f"WARNING: Only {len(df)} rows available (< {min_rows}). "
                "Training will still run but may be noisy."
            )

        cols_needed = set(self.feature_names + ["remaining_goals"])
        missing_final = cols_needed - set(df.columns)
        if missing_final:
            raise ValueError(f"Missing columns after preparation: {missing_final}")

        return df[list(self.feature_names) + ["remaining_goals"]]

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------
    def train(self, df: pd.DataFrame, test_size: float = 0.2, verbose: bool = True):
        if len(df) < 50:
            raise ValueError("Not enough rows to train regression model (need at least 50).")

        for col in self.feature_names + ["remaining_goals"]:
            if col not in df.columns:
                raise ValueError(f"Training data missing column: {col}")

        X = df[self.feature_names].fillna(0.0)
        y = df["remaining_goals"].astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        self.scaler.fit(X_train.values)
        X_train_scaled = self.scaler.transform(X_train.values)
        X_test_scaled = self.scaler.transform(X_test.values)

        self._create_model()
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test_scaled)

        if verbose:
            print("=" * 60)
            print("GOALS REMAINING REGRESSION MODEL - TRAINING RESULTS")
            print("=" * 60)
            print(f"Model Type:  {self.model_type}")
            print(f"Total events: {len(df)}")
            print(f"Train events: {len(X_train)}")
            print(f"Test events:  {len(X_test)}")
            print()
            mse = mean_squared_error(y_test, y_pred)
            rmse = mse ** 0.5
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  R^2:  {r2:.4f}")
            print()
            print("Sample predictions (y_true -> y_pred):")
            for yt, yp in list(zip(y_test[:10], y_pred[:10])):
                print(f"  {yt:.2f} -> {yp:.2f}")
            print()

            if hasattr(self.model, "feature_importances_"):
                imp = self.model.feature_importances_
                self.feature_importance = dict(zip(self.feature_names, imp))
                print("FEATURE IMPORTANCE:")
                for fname, score in sorted(self.feature_importance.items(), key=lambda x: -x[1]):
                    print(f"  {fname}: {score:.4f}")
            print("=" * 60)

    # ------------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------------
    def predict_remaining_goals(self, features: Dict[str, Any]) -> float:
        if not self.is_trained:
            raise ValueError("Model not trained.")

        x = [float(features.get(fname, 0.0) or 0.0) for fname in self.feature_names]
        X_scaled = self.scaler.transform([x])
        pred = float(self.model.predict(X_scaled)[0])
        return max(0.0, pred)

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
            "feature_importance": self.feature_importance,
            "trained_at": datetime.now().isoformat(),
        }
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> "GoalsRemainingRegressor":
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        obj = cls(model_type=model_data.get("model_type", "gradient_boosting"))
        obj.model = model_data["model"]
        obj.scaler = model_data["scaler"]
        obj.feature_names = model_data["feature_names"]
        obj.feature_importance = model_data.get("feature_importance", {})
        obj.is_trained = True
        return obj


if __name__ == "__main__":
    MINUTE_LOG_PATH = os.getenv("MINUTE_LOG_PATH", "minute_states_log.csv")
    MODEL_TYPE = os.getenv("GOALS_REG_MODEL_TYPE", "gradient_boosting")
    MODEL_PATH = os.getenv("GOALS_REG_MODEL_PATH", "goals_remaining_regressor.pkl")

    print("============================================================")
    print("Goals Remaining Regression Trainer (minute-based)")
    print("============================================================")
    print(f"Minute log CSV: {MINUTE_LOG_PATH}")
    print(f"Model type:     {MODEL_TYPE}")
    print(f"Output:         {MODEL_PATH}")
    print("============================================================")

    model = GoalsRemainingRegressor(model_type=MODEL_TYPE)

    print("Loading minute-level data...")
    df_train = model.load_minute_data(
        csv_path=MINUTE_LOG_PATH,
        min_rows=1000,
        goal_window_minutes=int(os.getenv("GOAL_WINDOW_MINUTES", "10")),
        max_remaining_clip=float(os.getenv("MAX_REMAINING_CLIP", "6.0")),
    )
    print(f"Loaded {len(df_train)} rows.")
    print(f"Mean remaining_goals: {df_train['remaining_goals'].mean():.3f}")
    print()

    print("Training model...")
    model.train(df_train, test_size=0.2, verbose=True)

    print("\nSaving model...")
    model.save_model(MODEL_PATH)

    print("\nDone. Use this model in your live bot to estimate expected remaining goals.")
