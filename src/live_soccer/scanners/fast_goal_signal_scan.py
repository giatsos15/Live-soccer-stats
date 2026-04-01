"""
fast_goal_signal_scan.py

Purpose
-------
Fast scanner for your real minute_states_log.csv that evaluates only the most
promising live states for your goals classifier.

Why this is faster
------------------
1. Reads only the last N rows from the minute log
2. Keeps only regulation live-play rows (1H/2H)
3. Scores only rows that look like alert candidates
4. Requires minimum history before scoring
5. Prints progress during execution
6. Optionally skips support models for even more speed

Outputs
-------
- Console summaries
- fast_goal_signal_scan_results.csv
"""

import os
import pickle
from collections import defaultdict
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

# ============================================================
# CONFIG
# ============================================================
MINUTE_LOG_PATH = os.getenv("MINUTE_LOG_PATH", "minute_states_log.csv")
GOALS_CLS_MODEL_PATH = os.getenv("GOALS_CLS_MODEL_PATH", "goals_remaining_classifier_v2.pkl")
GOAL_MODEL_NEXT10_PATH = os.getenv("GOAL_MODEL_NEXT10_PATH", "goal_model_next_10m.pkl")
GOAL_MODEL_BEFORE_HT_PATH = os.getenv("GOAL_MODEL_BEFORE_HT_PATH", "goal_model_before_ht.pkl")

LAST_N_ROWS = int(os.getenv("SCAN_LAST_N_ROWS", "15000"))
TOP_N = int(os.getenv("SCAN_TOP_N", "30"))
MIN_HISTORY_ROWS = int(os.getenv("SCAN_MIN_HISTORY_ROWS", "6"))
LOAD_SUPPORT_MODELS = os.getenv("SCAN_LOAD_SUPPORT_MODELS", "1") == "1"
PRINT_EVERY = int(os.getenv("SCAN_PRINT_EVERY", "1000"))

# Candidate filter: keep rows likely to matter
MIN_MINUTE = int(os.getenv("SCAN_MIN_MINUTE", "10"))
MAX_MINUTE = int(os.getenv("SCAN_MAX_MINUTE", "88"))

MIN_TOTAL_PRESSURE = float(os.getenv("SCAN_MIN_TOTAL_PRESSURE", "120"))
MIN_SHOTS_ON_TARGET = float(os.getenv("SCAN_MIN_SOT", "3"))
MIN_CORNERS_TOTAL = float(os.getenv("SCAN_MIN_CORNERS", "2"))

MIN_PRESSURE_LAST_5 = float(os.getenv("SCAN_MIN_PRESSURE_LAST5", "18"))
MIN_PRESSURE_LAST_10 = float(os.getenv("SCAN_MIN_PRESSURE_LAST10", "35"))
MIN_SOT_LAST_10 = float(os.getenv("SCAN_MIN_SOT_LAST10", "1"))
MIN_CORNERS_LAST_10 = float(os.getenv("SCAN_MIN_CORNERS_LAST10", "1"))


# ============================================================
# MODEL RUNTIMES
# ============================================================
class GoalOutcomeRuntime:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.target_type: str = "next_10m"
        self.league_goal_rates: Dict[str, float] = {}
        self.is_trained = False

    @classmethod
    def load_model(cls, filepath: str) -> "GoalOutcomeRuntime":
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        obj = cls()
        obj.model = data["model"]
        obj.scaler = data["scaler"]
        obj.feature_names = data["feature_names"]
        obj.target_type = data.get("target_type", "next_10m")
        obj.league_goal_rates = data.get("league_goal_rates", {})
        obj.is_trained = True
        return obj

    def _league_rate(self, league_name: str) -> float:
        if not self.league_goal_rates:
            return 2.5
        if league_name in self.league_goal_rates:
            return float(self.league_goal_rates[league_name])
        return float(sum(self.league_goal_rates.values()) / len(self.league_goal_rates))

    def predict_proba(self, features: Dict[str, Any]) -> float:
        row = [float(features.get(fname, 0.0) or 0.0) for fname in self.feature_names]
        X_df = pd.DataFrame([row], columns=self.feature_names)
        X_scaled = self.scaler.transform(X_df)

        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(X_scaled)[0][1])
        return float(self.model.predict(X_scaled)[0])


class GoalsRemainingClassifierRuntime:
    def __init__(self):
        self.feature_names: List[str] = []
        self.first_half: Dict[int, Any] = {}
        self.second_half: Dict[int, Any] = {}
        self.is_trained = False

    @classmethod
    def load_model(cls, filepath: str) -> "GoalsRemainingClassifierRuntime":
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        obj = cls()
        obj.feature_names = list(data["features"])
        obj.first_half = data["first_half"]
        obj.second_half = data["second_half"]
        obj.is_trained = True
        return obj

    def predict_probs(self, half: int, features: Dict[str, float]) -> Dict[str, float]:
        row = [float(features.get(fname, 0.0) or 0.0) for fname in self.feature_names]
        X = pd.DataFrame([row], columns=self.feature_names)

        bundle = self.first_half if half == 1 else self.second_half
        out = {}

        for k in (1, 2, 3):
            entry = bundle.get(k)
            if not entry or entry.get("model") is None:
                out[f"ge{k}"] = 0.0
                continue
            model = entry["model"]
            out[f"ge{k}"] = float(model.predict_proba(X)[0][1])

        return out


# ============================================================
# HELPERS
# ============================================================
def map_half(raw_half: int, minute: int) -> Optional[int]:
    if raw_half == 10:
        return 1
    if raw_half == 12:
        return 2
    if raw_half == 1:
        return 1
    if raw_half == 2:
        return 2
    if minute <= 45:
        return 1
    if minute <= 95:
        return 2
    return None


def live_league_goal_rate(
    league_name: str,
    next10_model: Optional[GoalOutcomeRuntime],
    before_ht_model: Optional[GoalOutcomeRuntime],
) -> float:
    if next10_model is not None and next10_model.is_trained:
        return float(next10_model._league_rate(league_name))
    if before_ht_model is not None and before_ht_model.is_trained:
        return float(before_ht_model._league_rate(league_name))
    return 2.5


def support_features_from_row(row: Dict[str, Any]) -> Dict[str, float]:
    return {
        "minute": float(row["minute"]),
        "half": float(row["half"]),
        "goals_total": float(row["goals_total"]),
        "total_pressure": float(row["total_pressure"]),
        "pressure_diff": float(row["pressure_diff"]),
        "shots_on_target": float(row["shots_on_target"]),
        "total_shots": float(row["total_shots"]),
        "corners_total": float(row["corners_total"]),
        "possession_ratio": float(row["possession_ratio"]),
    }


def evaluate_signal_label(
    half: int,
    p_ge1: float,
    p_ge2: float,
    p_ge3: float,
    p_next10: Optional[float],
    p_before_ht: Optional[float],
) -> Optional[str]:
    GOAL_PROB_THRESHOLD_NEXT10 = 0.35
    GOAL_PROB_THRESHOLD_BEFORE_HT = 0.40

    FIRST_HALF_GE1_INFO_THRESHOLD = 0.78
    FIRST_HALF_GE2_STRONG_THRESHOLD = 0.62
    FIRST_HALF_GE3_EXPLOSIVE_THRESHOLD = 0.30

    SECOND_HALF_GE1_STRONG_THRESHOLD = 0.70
    SECOND_HALF_GE2_STRONG_THRESHOLD = 0.45
    SECOND_HALF_GE2_VERY_STRONG_THRESHOLD = 0.55
    SECOND_HALF_GE3_EXPLOSIVE_THRESHOLD = 0.22

    if half == 1:
        explosive = p_ge3 >= FIRST_HALF_GE3_EXPLOSIVE_THRESHOLD
        strong = (
            p_ge2 >= FIRST_HALF_GE2_STRONG_THRESHOLD and (
                (p_next10 is not None and p_next10 >= GOAL_PROB_THRESHOLD_NEXT10) or
                (p_before_ht is not None and p_before_ht >= GOAL_PROB_THRESHOLD_BEFORE_HT)
            )
        )
        watch = p_ge1 >= FIRST_HALF_GE1_INFO_THRESHOLD

        if explosive:
            return "EXPLOSIVE OVER (1H)"
        if strong:
            return "STRONG OVER (1H)"
        if watch:
            return "LIVE OVER WATCH (1H)"
        return None

    explosive = p_ge3 >= SECOND_HALF_GE3_EXPLOSIVE_THRESHOLD
    very_strong = p_ge2 >= SECOND_HALF_GE2_VERY_STRONG_THRESHOLD
    strong = (p_ge1 >= SECOND_HALF_GE1_STRONG_THRESHOLD and p_ge2 >= SECOND_HALF_GE2_STRONG_THRESHOLD)

    if explosive:
        return "EXPLOSIVE OVER (2H)"
    if very_strong:
        return "VERY STRONG OVER (2H)"
    if strong:
        return "STRONG OVER (2H)"
    return None


# ============================================================
# FAST HISTORY FEATURES
# ============================================================
def compute_window_features(hist: List[Dict[str, Any]], current: Dict[str, Any]) -> Dict[str, float]:
    """
    hist includes current row as the last element.
    Uses only the last 20 rows of match history for speed.
    """
    recent = hist[-20:]
    minute = float(current["minute"])

    total_pressure = float(current["total_pressure"])
    total_shots = float(current["total_shots"])
    shots_on_target = float(current["shots_on_target"])
    corners_total = float(current["corners_total"])
    possession_ratio = float(current["possession_ratio"])

    def delta_over(field: str, current_value: float, window: int) -> float:
        cutoff = minute - window
        baseline = None
        for snap in recent:
            if float(snap["minute"]) >= cutoff:
                baseline = float(snap[field])
                break
        if baseline is None:
            baseline = current_value
        return max(0.0, current_value - baseline)

    def prev_delta_over(field: str, current_value: float, window: int) -> float:
        end_cutoff = minute - window
        start_cutoff = minute - 2 * window

        val_end = None
        for snap in recent:
            if float(snap["minute"]) >= end_cutoff:
                val_end = float(snap[field])
                break
        if val_end is None:
            val_end = current_value

        val_start = None
        for snap in recent:
            if float(snap["minute"]) >= start_cutoff:
                val_start = float(snap[field])
                break
        if val_start is None:
            val_start = val_end

        return max(0.0, val_end - val_start)

    def mean_over(field: str, start_exclusive: float, end_inclusive: float, default_value: float) -> float:
        vals = [
            float(snap[field])
            for snap in recent
            if float(snap["minute"]) > start_exclusive and float(snap["minute"]) <= end_inclusive
        ]
        if not vals:
            return default_value
        return float(sum(vals) / len(vals))

    pressure_last_5 = delta_over("total_pressure", total_pressure, 5)
    pressure_last_10 = delta_over("total_pressure", total_pressure, 10)
    shots_last_5 = delta_over("total_shots", total_shots, 5)
    shots_last_10 = delta_over("total_shots", total_shots, 10)
    sot_last_5 = delta_over("shots_on_target", shots_on_target, 5)
    sot_last_10 = delta_over("shots_on_target", shots_on_target, 10)
    corners_last_5 = delta_over("corners_total", corners_total, 5)
    corners_last_10 = delta_over("corners_total", corners_total, 10)

    pressure_prev_5 = prev_delta_over("total_pressure", total_pressure, 5)
    pressure_prev_10 = prev_delta_over("total_pressure", total_pressure, 10)
    shots_prev_5 = prev_delta_over("total_shots", total_shots, 5)
    shots_prev_10 = prev_delta_over("total_shots", total_shots, 10)
    sot_prev_5 = prev_delta_over("shots_on_target", shots_on_target, 5)
    sot_prev_10 = prev_delta_over("shots_on_target", shots_on_target, 10)
    corners_prev_5 = prev_delta_over("corners_total", corners_total, 5)
    corners_prev_10 = prev_delta_over("corners_total", corners_total, 10)

    possession_last_5_mean = mean_over("possession_ratio", minute - 5.0, minute, possession_ratio)
    possession_last_10_mean = mean_over("possession_ratio", minute - 10.0, minute, possession_ratio)
    possession_prev_5_mean = mean_over("possession_ratio", minute - 10.0, minute - 5.0, possession_last_5_mean)
    possession_prev_10_mean = mean_over("possession_ratio", minute - 20.0, minute - 10.0, possession_last_10_mean)

    return {
        "pressure_last_5": pressure_last_5,
        "pressure_last_10": pressure_last_10,
        "shots_last_5": shots_last_5,
        "shots_last_10": shots_last_10,
        "sot_last_5": sot_last_5,
        "sot_last_10": sot_last_10,
        "corners_last_5": corners_last_5,
        "corners_last_10": corners_last_10,
        "pressure_prev_5": pressure_prev_5,
        "pressure_prev_10": pressure_prev_10,
        "shots_prev_5": shots_prev_5,
        "shots_prev_10": shots_prev_10,
        "sot_prev_5": sot_prev_5,
        "sot_prev_10": sot_prev_10,
        "corners_prev_5": corners_prev_5,
        "corners_prev_10": corners_prev_10,
        "possession_last_5_mean": possession_last_5_mean,
        "possession_last_10_mean": possession_last_10_mean,
        "possession_prev_5_mean": possession_prev_5_mean,
        "possession_prev_10_mean": possession_prev_10_mean,
    }


def build_remaining_goals_features(
    hist: List[Dict[str, Any]],
    current: Dict[str, Any],
    next10_model: Optional[GoalOutcomeRuntime],
    before_ht_model: Optional[GoalOutcomeRuntime],
) -> Tuple[int, Dict[str, float], Dict[str, float]]:
    minute = float(current["minute"])
    half = int(current["half_mapped"])
    league_name = str(current["league"])

    goals_home = float(current["goals_home"])
    goals_away = float(current["goals_away"])
    goals_total = float(current["goals_total"])

    total_pressure = float(current["total_pressure"])
    pressure_diff = float(current["pressure_diff"])
    shots_on_target = float(current["shots_on_target"])
    total_shots = float(current["total_shots"])
    corners_total = float(current["corners_total"])
    possession_ratio = float(current["possession_ratio"])

    league_goal_rate = live_league_goal_rate(league_name, next10_model, before_ht_model)
    minutes_remaining = max(0.0, 95.0 - minute)
    goal_diff = abs(goals_home - goals_away)

    is_draw = float(goals_home == goals_away)
    is_home_leading = float(goals_home > goals_away)
    is_away_leading = float(goals_away > goals_home)
    is_00 = float(goals_home == 0 and goals_away == 0)
    is_11 = float(goals_home == 1 and goals_away == 1)
    is_level_high_scoring = float(is_draw == 1.0 and goals_total >= 2.0)
    is_one_goal_game = float(goal_diff == 1.0)
    is_two_plus_goal_margin = float(goal_diff >= 2.0)

    w = compute_window_features(hist, current)

    pressure_spike_5 = w["pressure_last_5"] - w["pressure_prev_5"]
    pressure_spike_10 = w["pressure_last_10"] - w["pressure_prev_10"]
    shots_spike_5 = w["shots_last_5"] - w["shots_prev_5"]
    shots_spike_10 = w["shots_last_10"] - w["shots_prev_10"]
    sot_spike_5 = w["sot_last_5"] - w["sot_prev_5"]
    sot_spike_10 = w["sot_last_10"] - w["sot_prev_10"]
    corners_spike_5 = w["corners_last_5"] - w["corners_prev_5"]
    corners_spike_10 = w["corners_last_10"] - w["corners_prev_10"]
    possession_spike_5 = w["possession_last_5_mean"] - w["possession_prev_5_mean"]
    possession_spike_10 = w["possession_last_10_mean"] - w["possession_prev_10_mean"]

    played_minutes = max(1.0, minute)
    pressure_vs_match_avg = w["pressure_last_10"] / ((total_pressure / played_minutes) + 1e-6)
    shots_vs_match_avg = w["shots_last_10"] / ((total_shots / played_minutes) + 1e-6)
    sot_vs_match_avg = w["sot_last_10"] / ((shots_on_target / played_minutes) + 1e-6)
    corners_vs_match_avg = w["corners_last_10"] / ((corners_total / played_minutes) + 1e-6)

    sot_share_total_shots = (shots_on_target / total_shots) if total_shots > 0 else 0.0
    pressure_per_shot = (total_pressure / total_shots) if total_shots > 0 else 0.0

    draw_pressure_interaction = is_draw * w["pressure_last_10"]
    draw_shots_interaction = is_draw * w["shots_last_10"]
    close_game_pressure_interaction = is_one_goal_game * w["pressure_last_10"]
    close_game_shots_interaction = is_one_goal_game * w["shots_last_10"]

    features = {
        "league_goal_rate": league_goal_rate,
        "minutes_remaining": minutes_remaining,
        "goals_home": goals_home,
        "goals_away": goals_away,
        "goals_total": goals_total,
        "goal_diff": goal_diff,
        "is_draw": is_draw,
        "is_home_leading": is_home_leading,
        "is_away_leading": is_away_leading,
        "is_00": is_00,
        "is_11": is_11,
        "is_level_high_scoring": is_level_high_scoring,
        "is_one_goal_game": is_one_goal_game,
        "is_two_plus_goal_margin": is_two_plus_goal_margin,
        "total_pressure": total_pressure,
        "pressure_diff": pressure_diff,
        "shots_on_target": shots_on_target,
        "total_shots": total_shots,
        "corners_total": corners_total,
        "possession_ratio": possession_ratio,
        "pressure_last_5": w["pressure_last_5"],
        "pressure_last_10": w["pressure_last_10"],
        "shots_last_5": w["shots_last_5"],
        "shots_last_10": w["shots_last_10"],
        "sot_last_5": w["sot_last_5"],
        "sot_last_10": w["sot_last_10"],
        "corners_last_5": w["corners_last_5"],
        "corners_last_10": w["corners_last_10"],
        "possession_last_5_mean": w["possession_last_5_mean"],
        "possession_last_10_mean": w["possession_last_10_mean"],
        "pressure_prev_5": w["pressure_prev_5"],
        "pressure_prev_10": w["pressure_prev_10"],
        "shots_prev_5": w["shots_prev_5"],
        "shots_prev_10": w["shots_prev_10"],
        "sot_prev_5": w["sot_prev_5"],
        "sot_prev_10": w["sot_prev_10"],
        "corners_prev_5": w["corners_prev_5"],
        "corners_prev_10": w["corners_prev_10"],
        "possession_prev_5_mean": w["possession_prev_5_mean"],
        "possession_prev_10_mean": w["possession_prev_10_mean"],
        "pressure_spike_5": pressure_spike_5,
        "pressure_spike_10": pressure_spike_10,
        "shots_spike_5": shots_spike_5,
        "shots_spike_10": shots_spike_10,
        "sot_spike_5": sot_spike_5,
        "sot_spike_10": sot_spike_10,
        "corners_spike_5": corners_spike_5,
        "corners_spike_10": corners_spike_10,
        "possession_spike_5": possession_spike_5,
        "possession_spike_10": possession_spike_10,
        "pressure_vs_match_avg": pressure_vs_match_avg,
        "shots_vs_match_avg": shots_vs_match_avg,
        "sot_vs_match_avg": sot_vs_match_avg,
        "corners_vs_match_avg": corners_vs_match_avg,
        "sot_share_total_shots": sot_share_total_shots,
        "pressure_per_shot": pressure_per_shot,
        "draw_pressure_interaction": draw_pressure_interaction,
        "draw_shots_interaction": draw_shots_interaction,
        "close_game_pressure_interaction": close_game_pressure_interaction,
        "close_game_shots_interaction": close_game_shots_interaction,
    }

    return half, features, w


def is_candidate_row(current: Dict[str, Any], w: Dict[str, float]) -> bool:
    if current["minute"] < MIN_MINUTE or current["minute"] > MAX_MINUTE:
        return False

    if current["total_pressure"] < MIN_TOTAL_PRESSURE:
        return False

    if (
        current["shots_on_target"] < MIN_SHOTS_ON_TARGET
        and current["corners_total"] < MIN_CORNERS_TOTAL
        and w["pressure_last_10"] < MIN_PRESSURE_LAST_10
    ):
        return False

    if (
        w["pressure_last_5"] < MIN_PRESSURE_LAST_5
        and w["pressure_last_10"] < MIN_PRESSURE_LAST_10
        and w["sot_last_10"] < MIN_SOT_LAST_10
        and w["corners_last_10"] < MIN_CORNERS_LAST_10
    ):
        return False

    return True


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    if not os.path.exists(MINUTE_LOG_PATH):
        raise FileNotFoundError(f"Minute log not found: {MINUTE_LOG_PATH}")
    if not os.path.exists(GOALS_CLS_MODEL_PATH):
        raise FileNotFoundError(f"Classifier bundle not found: {GOALS_CLS_MODEL_PATH}")

    print(f"Loading minute log: {MINUTE_LOG_PATH}")
    df = pd.read_csv(MINUTE_LOG_PATH)

    if LAST_N_ROWS > 0 and len(df) > LAST_N_ROWS:
        df = df.tail(LAST_N_ROWS).copy()

    df = df.sort_values(["match_id", "minute"]).reset_index(drop=True)
    df["half_mapped"] = df.apply(lambda r: map_half(int(r["half"]), int(r["minute"])), axis=1)
    df = df[df["half_mapped"].isin([1, 2])].copy()
    df = df[(df["minute"] >= MIN_MINUTE) & (df["minute"] <= MAX_MINUTE)].copy()

    print(f"Rows kept for scan: {len(df)}")

    goals_cls = GoalsRemainingClassifierRuntime.load_model(GOALS_CLS_MODEL_PATH)
    print(f"Loaded goals classifier: {GOALS_CLS_MODEL_PATH}")

    next10_model = None
    before_ht_model = None

    if LOAD_SUPPORT_MODELS and os.path.exists(GOAL_MODEL_NEXT10_PATH):
        next10_model = GoalOutcomeRuntime.load_model(GOAL_MODEL_NEXT10_PATH)
        print(f"Loaded support model: {GOAL_MODEL_NEXT10_PATH}")

    if LOAD_SUPPORT_MODELS and os.path.exists(GOAL_MODEL_BEFORE_HT_PATH):
        before_ht_model = GoalOutcomeRuntime.load_model(GOAL_MODEL_BEFORE_HT_PATH)
        print(f"Loaded support model: {GOAL_MODEL_BEFORE_HT_PATH}")

    histories = defaultdict(list)
    results = []

    total_rows = len(df)
    scored_rows = 0
    candidate_rows = 0

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        rec = row.to_dict()
        mid = str(rec["match_id"])

        histories[mid].append(rec)
        hist = histories[mid]

        if len(hist) < MIN_HISTORY_ROWS:
            continue

        try:
            half, features, w = build_remaining_goals_features(hist, rec, next10_model, before_ht_model)
        except Exception:
            continue

        if not is_candidate_row(rec, w):
            continue

        candidate_rows += 1

        try:
            probs = goals_cls.predict_probs(half, features)
        except Exception:
            continue

        p_ge1 = float(probs.get("ge1", 0.0))
        p_ge2 = float(probs.get("ge2", 0.0))
        p_ge3 = float(probs.get("ge3", 0.0))

        p_next10 = None
        p_before_ht = None

        if next10_model is not None:
            try:
                sf = support_features_from_row(rec)
                sf["league_goal_rate"] = next10_model._league_rate(str(rec["league"]))
                p_next10 = float(next10_model.predict_proba(sf))
            except Exception:
                pass

        if half == 1 and before_ht_model is not None:
            try:
                sf = support_features_from_row(rec)
                sf["league_goal_rate"] = before_ht_model._league_rate(str(rec["league"]))
                p_before_ht = float(before_ht_model.predict_proba(sf))
            except Exception:
                pass

        label = evaluate_signal_label(
            half=half,
            p_ge1=p_ge1,
            p_ge2=p_ge2,
            p_ge3=p_ge3,
            p_next10=p_next10,
            p_before_ht=p_before_ht,
        )

        results.append({
            "match_id": mid,
            "league": rec["league"],
            "home": rec["home"],
            "away": rec["away"],
            "minute": int(rec["minute"]),
            "half": half,
            "score": f"{int(rec['goals_home'])}-{int(rec['goals_away'])}",
            "p_ge1": p_ge1,
            "p_ge2": p_ge2,
            "p_ge3": p_ge3,
            "p_next10": p_next10,
            "p_before_ht": p_before_ht,
            "label": label,
            "total_pressure": float(rec["total_pressure"]),
            "shots_on_target": float(rec["shots_on_target"]),
            "corners_total": float(rec["corners_total"]),
            "pressure_last_5": float(w["pressure_last_5"]),
            "pressure_last_10": float(w["pressure_last_10"]),
            "sot_last_10": float(w["sot_last_10"]),
            "corners_last_10": float(w["corners_last_10"]),
        })
        scored_rows += 1

        if i % PRINT_EVERY == 0:
            print(
                f"Processed {i}/{total_rows} rows | "
                f"candidates={candidate_rows} | scored={scored_rows}"
            )

    out = pd.DataFrame(results)

    if out.empty:
        print("No candidate rows scored.")
        return

    out = out.sort_values(["p_ge3", "p_ge2", "p_ge1"], ascending=False).reset_index(drop=True)
    out.to_csv("fast_goal_signal_scan_results.csv", index=False)

    print("\nSaved: fast_goal_signal_scan_results.csv")
    print(f"Total scored rows: {len(out)}")
    print(f"Rows with non-null label: {out['label'].notna().sum()}")

    print("\nTOP BY p_ge1")
    print(out.sort_values("p_ge1", ascending=False).head(TOP_N).to_string(index=False))

    print("\nTOP BY p_ge2")
    print(out.sort_values("p_ge2", ascending=False).head(TOP_N).to_string(index=False))

    print("\nTOP BY p_ge3")
    print(out.sort_values("p_ge3", ascending=False).head(TOP_N).to_string(index=False))

    labeled = out[out["label"].notna()].copy()
    if not labeled.empty:
        print("\nROWS THAT WOULD ACTUALLY TRIGGER A LABEL")
        print(labeled.head(TOP_N).to_string(index=False))
    else:
        print("\nNo rows met your current live Telegram label thresholds.")


if __name__ == "__main__":
    main()