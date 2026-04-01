"""
Overlyzer Live Bot + Threshold Goal Classifiers

What this version adds:
- Keeps your pressure spike / high-pressure alerts.
- Loads a goals-remaining classifier bundle (goals_remaining_classifier_v2.pkl).
- Uses live rolling per-match history to build momentum / spike features.
- Predicts:
    P(>=1 more goal)
    P(>=2 more goals)
    P(>=3 more goals)
- Keeps short-horizon support models:
    goal_model_next_10m.pkl
    goal_model_before_ht.pkl
- Sends Telegram messages oriented toward live totals / value-bet workflow.

IMPORTANT:
- Telegram + Overlyzer tokens are read from env vars by default.
  Set:
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID
    OVERLYZER_TOKEN
"""

import json
import os
import signal
import sys
import time
import pickle
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple

import pandas as pd
import requests

# ----------------------------
# Force UTF-8 on Windows console
# ----------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ============================================================
# CONFIG
# ============================================================
API_URL = "https://connect.overlyzer.ws/api/v2/live"
HOMEPAGE_URL = "https://www.overlyzer.com/"

# Polling
KEEPALIVE_MODE = True
KEEPALIVE_INTERVAL = 30
ERROR_BACKOFF = 15
REQUEST_TIMEOUT = 20

# Output
REPORT_ONLY_ALERTS = True
REPORT_KICKOFFS = False

# Minutes filter
SPIKES_MAX_MINUTE = 92

# Pressure spike (TOTAL)
SPIKE_DELTA_TOTAL = 27
SPIKE_PER_MIN_TOTAL = 40
ALERT_COOLDOWN_SECONDS_TOTAL = 45

# Pressure spike (TEAM)
SPIKE_DELTA_TEAM = 30
SPIKE_PER_MIN_TEAM = 45
ALERT_COOLDOWN_SECONDS_TEAM = 150

# High pressure stats: CORNER BURST + SOT-vs-goals ratio
STATS_MIN_MINUTE = 10
SPC_ALERT_COOLDOWN = 120

CORNERS_BURST_WINDOW_MINUTES = 10
CORNERS_BURST_REQUIRED = 3

# SOT ratio settings
SOT_MIN_IF_0_GOALS = 3
SOT_RATIO_PER_GOAL = 3
SOT_GENERAL_MIN = 5
SOT_MUST_BEAT_GOALS_BY = 1

# Telegram
TELEGRAM_ENABLED = True
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Overlyzer token
OVERLYZER_TOKEN = os.getenv("OVERLYZER_TOKEN", "").strip()

# Based on your browser behavior
USE_AUTH_HEADER = os.getenv("OVERLYZER_AUTH_MODE", "authorization").strip().lower()
SET_X_WEB_CLIENTTOKEN = True

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/143.0.0.0 Safari/537.36"
)

STATE_JSON_PATH = "overlyzer_state.json"
SAVE_STATE_TO_JSON = True

# ============================================================
# OPTIONAL PRESSURE SPIKE VALIDATOR
# ============================================================
ML_ENABLED = False
ML_MODEL_PATH = os.getenv("PRESSURE_SPIKE_MODEL_PATH", "pressure_spike_model.pkl").strip()
ML_CONFIDENCE_THRESHOLD = float(os.getenv("PRESSURE_SPIKE_ML_THRESHOLD", "0.70"))

# ============================================================
# MINUTE STATE LOGGING
# ============================================================
MINUTE_LOG_ENABLED = os.getenv("MINUTE_LOG_ENABLED", "1") == "1"
MINUTE_LOG_PATH = os.getenv("MINUTE_LOG_PATH", "minute_states_log.csv")
_minute_log_header_written = False

# ============================================================
# GOAL / TOTALS MODELS
# ============================================================
GOAL_ML_ENABLED = os.getenv("GOAL_ML_ENABLED", "1") == "1"

# Support models from your older setup
GOAL_MODEL_NEXT10_PATH = os.getenv("GOAL_MODEL_NEXT10_PATH", "goal_model_next_10m.pkl")
GOAL_MODEL_BEFORE_HT_PATH = os.getenv("GOAL_MODEL_BEFORE_HT_PATH", "goal_model_before_ht.pkl")

# Main new classifier bundle
GOALS_CLS_ENABLED = os.getenv("GOALS_CLS_ENABLED", "1") == "1"
GOALS_CLS_MODEL_PATH = os.getenv("GOALS_CLS_MODEL_PATH", "goals_remaining_classifier_v2.pkl").strip()

# Support model thresholds
GOAL_PROB_THRESHOLD_NEXT10 = float(os.getenv("GOAL_PROB_THRESHOLD_NEXT10", "0.35"))
GOAL_PROB_THRESHOLD_BEFORE_HT = float(os.getenv("GOAL_PROB_THRESHOLD_BEFORE_HT", "0.40"))

# Thresholds for classifier-based signals
FIRST_HALF_GE1_INFO_THRESHOLD = float(os.getenv("FIRST_HALF_GE1_INFO_THRESHOLD", "0.78"))
FIRST_HALF_GE2_STRONG_THRESHOLD = float(os.getenv("FIRST_HALF_GE2_STRONG_THRESHOLD", "0.62"))
FIRST_HALF_GE3_EXPLOSIVE_THRESHOLD = float(os.getenv("FIRST_HALF_GE3_EXPLOSIVE_THRESHOLD", "0.30"))

SECOND_HALF_GE1_STRONG_THRESHOLD = float(os.getenv("SECOND_HALF_GE1_STRONG_THRESHOLD", "0.70"))
SECOND_HALF_GE2_STRONG_THRESHOLD = float(os.getenv("SECOND_HALF_GE2_STRONG_THRESHOLD", "0.45"))
SECOND_HALF_GE2_VERY_STRONG_THRESHOLD = float(os.getenv("SECOND_HALF_GE2_VERY_STRONG_THRESHOLD", "0.55"))
SECOND_HALF_GE3_EXPLOSIVE_THRESHOLD = float(os.getenv("SECOND_HALF_GE3_EXPLOSIVE_THRESHOLD", "0.22"))

GOAL_MIN_MINUTE = int(os.getenv("GOAL_MIN_MINUTE", "10"))
GOAL_MAX_MINUTE = int(os.getenv("GOAL_MAX_MINUTE", "88"))
GOAL_ALERT_COOLDOWN = int(os.getenv("GOAL_ALERT_COOLDOWN", "180"))

# Rolling feature history
FEATURE_HISTORY_MAXLEN = int(os.getenv("FEATURE_HISTORY_MAXLEN", "40"))


# ============================================================
# STATE
# ============================================================
_stop = False

# total pressure history
last_total_pressure: Dict[str, Dict[str, float]] = {}
last_total_alert_ts: Dict[str, float] = {}

# team pressure history
last_team_pressure: Dict[Tuple[str, str], Dict[str, float]] = {}
last_team_alert_ts: Dict[Tuple[str, str], float] = {}

# high-stats cooldown
last_stats_alert_ts: Dict[str, float] = {}

# goal model cooldown
last_goal_ml_alert_ts: Dict[str, float] = {}

# corners history per match
corner_history: Dict[str, Deque[Tuple[int, int]]] = {}

# rolling feature history per match
match_feature_history: Dict[str, Deque[Dict[str, float]]] = {}

# state/debug
seen_matches: Dict[str, Dict[str, Any]] = {}
match_last: Dict[str, Dict[str, Any]] = {}
match_updates: Dict[str, int] = {}
match_stat_accum: Dict[str, Dict[str, int]] = {}

global_stats = {
    "polls_total": 0,
    "matches_seen_unique": 0,
    "matches_in_feed_sum": 0,
    "new_matches_sum": 0,
    "updated_matches_sum": 0,
}

snapshots: List[Dict[str, Any]] = []

# sessions
session = requests.Session()
telegram_session = requests.Session()


# ============================================================
# SIGNAL HANDLING
# ============================================================
def _handle_sigint(sig, frame):
    global _stop
    _stop = True
    print("\nStopping gracefully...")


signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)


# ============================================================
# UTIL
# ============================================================
def safe_print(s: str) -> None:
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("ascii", "replace").decode("ascii"))


def g(d: Dict[str, Any], *path: str, default: Any = 0) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur if cur is not None else default


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def map_raw_period_to_half(raw_period: Optional[int], minute: int) -> int:
    if raw_period == 10:
        return 1
    if raw_period == 12:
        return 2
    return 1 if minute <= 45 else 2


def get_raw_period(m: Dict[str, Any], minute: int) -> Optional[int]:
    p = g(m, "time", "p", default=None)
    if isinstance(p, int):
        return p
    return 10 if minute <= 45 else 12


def telegram_notify(message: str) -> bool:
    if not TELEGRAM_ENABLED:
        return False

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        safe_print("Telegram NOT configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars, then restart.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "disable_web_page_preview": True,
    }

    try:
        r = telegram_session.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            safe_print(f"Telegram send failed: {r.status_code} {r.text}")
            return False
        return True
    except Exception as e:
        safe_print(f"Telegram error: {e}")
        return False


def telegram_startup_test() -> None:
    ok = telegram_notify("✅ Overlyzer bot started. Telegram is working.")
    if ok:
        safe_print("Telegram: startup test sent OK.")
    else:
        safe_print("Telegram: startup test NOT sent (see error above).")


# ============================================================
# MINUTE LOGGING
# ============================================================
def _ensure_minute_log_header() -> None:
    global _minute_log_header_written

    if _minute_log_header_written or not MINUTE_LOG_ENABLED:
        return

    need_header = (not os.path.exists(MINUTE_LOG_PATH)) or (os.path.getsize(MINUTE_LOG_PATH) == 0)

    if need_header:
        header = (
            "timestamp,match_id,league,home,away,"
            "minute,half,"
            "goals_home,goals_away,goals_total,"
            "total_pressure,pressure_diff,"
            "shots_on_target,total_shots,"
            "corners_total,"
            "possession_ratio\n"
        )
        with open(MINUTE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(header)

    _minute_log_header_written = True


def log_minute_state(m: Dict[str, Any]) -> None:
    """
    Logs raw period code into the CSV 'half' column, matching your existing data style:
      10 = first half
      11 = HT break
      12 = second half
      etc.
    """
    if not MINUTE_LOG_ENABLED:
        return

    _ensure_minute_log_header()

    mid = m.get("id")
    if not mid:
        return

    minute = g(m, "time", "m", default=None)
    if not isinstance(minute, int):
        return

    raw_period = get_raw_period(m, minute)
    half_for_log = raw_period if isinstance(raw_period, int) else map_raw_period_to_half(None, minute)

    league = g(m, "tournament", "name", default="")
    home = g(m, "home", "name", default="")
    away = g(m, "away", "name", default="")

    gh = int(g(m, "scores", "current", "h", default=0) or 0)
    ga = int(g(m, "scores", "current", "a", default=0) or 0)
    goals_total = gh + ga

    pressure = match_pressure(m)
    total_pressure = int(pressure["total_pressure"])
    pressure_diff = int(pressure["pressure_diff"])

    sot, corners_total = sot_and_corners(m)

    hs = g(m, "stats", "home", default={}) or {}
    as_ = g(m, "stats", "away", default={}) or {}
    soff_home = int(hs.get("shotofftarget", 0) or 0)
    soff_away = int(as_.get("shotofftarget", 0) or 0)
    total_shots = sot + soff_home + soff_away

    possession_ratio = _possession_ratio_from_stats(hs, as_)

    ts = datetime.utcnow().isoformat()

    row = (
        f"{ts},{mid},{league},{home},{away},"
        f"{minute},{half_for_log},"
        f"{gh},{ga},{goals_total},"
        f"{total_pressure},{pressure_diff},"
        f"{sot},{total_shots},"
        f"{corners_total},"
        f"{possession_ratio:.3f}\n"
    )

    try:
        with open(MINUTE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(row)
    except Exception as e:
        safe_print(f"Minute log error: {e}")


# ============================================================
# PRESSURE / STATS
# ============================================================
def team_pressure(stats: Dict[str, Any]) -> int:
    attacks = int(stats.get("attacks", 0) or 0)
    dang = int(stats.get("dangerousattacks", 0) or 0)
    on = int(stats.get("shotontarget", 0) or 0)
    off = int(stats.get("shotofftarget", 0) or 0)
    corners = int(stats.get("corners", 0) or 0)
    return (attacks * 1) + (dang * 3) + (on * 8) + (off * 2) + (corners * 4)


def match_pressure(m: Dict[str, Any]) -> Dict[str, int]:
    hs = g(m, "stats", "home", default={}) or {}
    as_ = g(m, "stats", "away", default={}) or {}
    hp = team_pressure(hs)
    ap = team_pressure(as_)
    return {
        "home_pressure": hp,
        "away_pressure": ap,
        "total_pressure": hp + ap,
        "pressure_diff": abs(hp - ap),
    }


def sot_and_corners(m: Dict[str, Any]) -> Tuple[int, int]:
    hs = g(m, "stats", "home", default={}) or {}
    as_ = g(m, "stats", "away", default={}) or {}
    sot = int(hs.get("shotontarget", 0) or 0) + int(as_.get("shotontarget", 0) or 0)
    corners = int(hs.get("corners", 0) or 0) + int(as_.get("corners", 0) or 0)
    return sot, corners


def _possession_ratio_from_stats(hs: Dict[str, Any], as_: Dict[str, Any]) -> float:
    ph = hs.get("possession", None)
    pa = as_.get("possession", None)
    if ph is None and pa is None:
        return 0.5

    try:
        phf = float(ph or 0.0)
        paf = float(pa or 0.0)
        dom = max(phf, paf)
        if dom > 1.5:
            return max(0.0, min(1.0, dom / 100.0))
        return max(0.0, min(1.0, dom))
    except Exception:
        return 0.5


def is_match_updated(old: Dict[str, Any], new: Dict[str, Any]) -> bool:
    old_score = (g(old, "scores", "current", "h", default=0), g(old, "scores", "current", "a", default=0))
    new_score = (g(new, "scores", "current", "h", default=0), g(new, "scores", "current", "a", default=0))
    if new_score != old_score:
        return True

    old_min = g(old, "time", "m", default=None)
    new_min = g(new, "time", "m", default=None)
    if new_min != old_min:
        return True

    if match_pressure(old)["total_pressure"] != match_pressure(new)["total_pressure"]:
        return True

    old_sot, old_c = sot_and_corners(old)
    new_sot, new_c = sot_and_corners(new)
    return (old_sot, old_c) != (new_sot, new_c)


# ============================================================
# LIVE FEATURE HISTORY
# ============================================================
def extract_live_snapshot(m: Dict[str, Any]) -> Optional[Dict[str, float]]:
    mid = m.get("id")
    if not mid:
        return None

    minute = g(m, "time", "m", default=None)
    if not isinstance(minute, int):
        return None

    raw_period = get_raw_period(m, minute)
    half = map_raw_period_to_half(raw_period, minute)

    gh = int(g(m, "scores", "current", "h", default=0) or 0)
    ga = int(g(m, "scores", "current", "a", default=0) or 0)
    goals_total = gh + ga

    pressure = match_pressure(m)
    total_pressure = float(pressure["total_pressure"])
    pressure_diff = float(pressure["pressure_diff"])

    sot, corners_total = sot_and_corners(m)

    hs = g(m, "stats", "home", default={}) or {}
    as_ = g(m, "stats", "away", default={}) or {}
    soff_home = int(hs.get("shotofftarget", 0) or 0)
    soff_away = int(as_.get("shotofftarget", 0) or 0)
    total_shots = sot + soff_home + soff_away

    possession_ratio = _possession_ratio_from_stats(hs, as_)

    return {
        "minute": float(minute),
        "raw_period": float(raw_period if raw_period is not None else 0),
        "half": float(half),
        "goals_home": float(gh),
        "goals_away": float(ga),
        "goals_total": float(goals_total),
        "total_pressure": float(total_pressure),
        "pressure_diff": float(pressure_diff),
        "shots_on_target": float(sot),
        "total_shots": float(total_shots),
        "corners_total": float(corners_total),
        "possession_ratio": float(possession_ratio),
    }


def update_match_feature_history(m: Dict[str, Any]) -> None:
    snap = extract_live_snapshot(m)
    if snap is None:
        return

    mid = m.get("id")
    if not mid:
        return

    hist = match_feature_history.get(mid)
    if hist is None:
        hist = deque(maxlen=FEATURE_HISTORY_MAXLEN)
        match_feature_history[mid] = hist

    # If same minute, replace last snapshot with fresher data
    if hist and int(hist[-1]["minute"]) == int(snap["minute"]):
        hist[-1] = snap
        return

    # If somehow minute jumps backward strongly, reset history
    if hist and snap["minute"] < hist[-1]["minute"] - 3:
        hist.clear()

    hist.append(snap)


def _baseline_value_at_or_after_cutoff(
    hist: List[Dict[str, float]],
    field: str,
    cutoff: float,
    current_value: float,
) -> float:
    for snap in hist:
        if snap["minute"] >= cutoff:
            return float(snap.get(field, current_value))
    return float(current_value)


def _window_delta_from_history(
    hist: List[Dict[str, float]],
    current_minute: float,
    field: str,
    current_value: float,
    window: int,
) -> float:
    cutoff = current_minute - window
    baseline = _baseline_value_at_or_after_cutoff(hist, field, cutoff, current_value)
    return max(0.0, float(current_value) - float(baseline))


def _previous_window_delta_from_history(
    hist: List[Dict[str, float]],
    current_minute: float,
    field: str,
    current_value: float,
    window: int,
) -> float:
    end_cutoff = current_minute - window
    start_cutoff = current_minute - (2 * window)

    val_end = _baseline_value_at_or_after_cutoff(hist, field, end_cutoff, current_value)
    val_start = _baseline_value_at_or_after_cutoff(hist, field, start_cutoff, val_end)

    return max(0.0, float(val_end) - float(val_start))


def _window_mean_from_history(
    hist: List[Dict[str, float]],
    current_minute: float,
    field: str,
    start_exclusive: float,
    end_inclusive: float,
    default_value: float,
) -> float:
    vals = [
        float(snap.get(field, default_value))
        for snap in hist
        if (snap["minute"] > start_exclusive and snap["minute"] <= end_inclusive)
    ]
    if not vals:
        return float(default_value)
    return float(sum(vals) / len(vals))


def get_current_goal_probabilities(m: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Returns current live probabilities for:
      - next_goal_10m
      - before_ht
      - ge1 / ge2 / ge3 (>=1, >=2, >=3 more goals)
    """
    out = {
        "next_goal_10m": None,
        "before_ht": None,
        "ge1": None,
        "ge2": None,
        "ge3": None,
    }

    mid = m.get("id")
    if not mid:
        return out

    minute = g(m, "time", "m", default=None)
    if not isinstance(minute, int):
        return out

    raw_period = g(m, "time", "p", default=None)
    if raw_period not in [10, 12]:
        return out

    half = map_raw_period_to_half(raw_period, minute)
    league_name = g(m, "tournament", "name", default="Unknown League")

    # --- Support model: next goal in next 10 minutes
    try:
        if goal_model_next10 is not None and goal_model_next10.is_trained:
            f = build_goal_support_features_from_match(m)
            f["league_goal_rate"] = goal_model_next10._league_rate(league_name)
            out["next_goal_10m"] = float(goal_model_next10.predict_proba(f))
    except Exception as e:
        safe_print(f"get_current_goal_probabilities next10 error: {e}")

    # --- Support model: goal before HT
    try:
        if half == 1 and goal_model_before_ht is not None and goal_model_before_ht.is_trained:
            f = build_goal_support_features_from_match(m)
            f["league_goal_rate"] = goal_model_before_ht._league_rate(league_name)
            out["before_ht"] = float(goal_model_before_ht.predict_proba(f))
    except Exception as e:
        safe_print(f"get_current_goal_probabilities beforeHT error: {e}")

    # --- Main classifier: >=1 / >=2 / >=3 more goals
    try:
        if goals_remaining_classifier is not None and goals_remaining_classifier.is_trained:
            built = build_remaining_goals_features_from_match(m)
            if built is not None:
                built_half, features = built
                probs = goals_remaining_classifier.predict_probs(built_half, features)

                p_ge1 = float(probs.get("ge1", 0.0))
                p_ge2 = float(probs.get("ge2", 0.0))
                p_ge3 = float(probs.get("ge3", 0.0))

                # enforce monotonicity
                p_ge2 = min(p_ge2, p_ge1)
                p_ge3 = min(p_ge3, p_ge2)

                out["ge1"] = p_ge1
                out["ge2"] = p_ge2
                out["ge3"] = p_ge3
    except Exception as e:
        safe_print(f"get_current_goal_probabilities remaining-goals error: {e}")

    return out


def format_goal_prob_summary(probs: Dict[str, Optional[float]]) -> str:
    parts = []

    if probs.get("next_goal_10m") is not None:
        parts.append(f"P(next goal 10m): {probs['next_goal_10m']:.0%}")

    if probs.get("before_ht") is not None:
        parts.append(f"P(goal before HT): {probs['before_ht']:.0%}")

    if probs.get("ge1") is not None:
        parts.append(f"P(>=1 more): {probs['ge1']:.0%}")

    if probs.get("ge2") is not None:
        parts.append(f"P(>=2 more): {probs['ge2']:.0%}")

    if probs.get("ge3") is not None:
        parts.append(f"P(>=3 more): {probs['ge3']:.0%}")

    return " | ".join(parts)



# ============================================================
# ML RUNTIMES
# ============================================================
class PressureSpikeValidator:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.is_trained = False

    @classmethod
    def load_model(cls, filepath: str) -> "PressureSpikeValidator":
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        v = cls()
        v.model = model_data["model"]
        v.scaler = model_data["scaler"]
        v.feature_names = model_data["feature_names"]
        v.is_trained = True
        return v

    def predict(self, event: Dict[str, Any]) -> Tuple[bool, float]:
        if not self.is_trained:
            raise ValueError("Model not trained/loaded.")

        x = [float(event.get(fname, 0.0) or 0.0) for fname in self.feature_names]
        X_scaled = self.scaler.transform([x])
        pred = int(self.model.predict(X_scaled)[0])
        proba = float(self.model.predict_proba(X_scaled)[0][1])
        return (pred == 1), proba


class GoalOutcomeRuntime:
    """
    Loader for the existing short-horizon support models.
    """

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
        if not self.is_trained:
            raise ValueError("Goal outcome model not loaded.")

        row = [float(features.get(fname, 0.0) or 0.0) for fname in self.feature_names]
        X_df = pd.DataFrame([row], columns=self.feature_names)
        X_scaled = self.scaler.transform(X_df)

        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(X_scaled)[0][1])

        return float(self.model.predict(X_scaled)[0])


class GoalsRemainingClassifierRuntime:
    """
    Loader for goals_remaining_classifier_v2.pkl bundle.

    Expected bundle structure:
    {
        "features": [...],
        "first_half": {1: {..."model": clf...}, 2: {...}, 3: {...}},
        "second_half": {1: {...}, 2: {...}, 3: {...}},
        ...
    }
    """

    def __init__(self):
        self.feature_names: List[str] = []
        self.models_first: Dict[int, Any] = {}
        self.models_second: Dict[int, Any] = {}
        self.is_trained = False

    @classmethod
    def load_model(cls, filepath: str) -> "GoalsRemainingClassifierRuntime":
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        obj = cls()
        obj.feature_names = list(data.get("features", []))

        first_half = data.get("first_half", {}) or {}
        second_half = data.get("second_half", {}) or {}

        for k, res in first_half.items():
            if isinstance(res, dict) and res.get("model") is not None:
                obj.models_first[int(k)] = res["model"]

        for k, res in second_half.items():
            if isinstance(res, dict) and res.get("model") is not None:
                obj.models_second[int(k)] = res["model"]

        obj.is_trained = bool(obj.feature_names and (obj.models_first or obj.models_second))
        return obj

    def predict_probs(self, features: Dict[str, Any], half: int) -> Dict[str, float]:
        if not self.is_trained:
            raise ValueError("Goals remaining classifier bundle not loaded.")

        row = [float(features.get(fname, 0.0) or 0.0) for fname in self.feature_names]
        X_df = pd.DataFrame([row], columns=self.feature_names)

        models = self.models_first if half == 1 else self.models_second
        out: Dict[str, float] = {}

        for k in (1, 2, 3):
            model = models.get(k)
            if model is None:
                continue
            out[f"ge{k}"] = float(model.predict_proba(X_df)[0][1])

        return out


spike_validator: Optional[PressureSpikeValidator] = None
goal_model_next10: Optional[GoalOutcomeRuntime] = None
goal_model_before_ht: Optional[GoalOutcomeRuntime] = None
goals_remaining_classifier: Optional[GoalsRemainingClassifierRuntime] = None


def load_spike_validator() -> None:
    global spike_validator
    if not ML_ENABLED:
        spike_validator = None
        safe_print("ML validator: disabled (ML_ENABLED=False).")
        return

    if not ML_MODEL_PATH or not os.path.exists(ML_MODEL_PATH):
        spike_validator = None
        safe_print(f"ML validator: model not found at '{ML_MODEL_PATH}'. Running without ML filtering.")
        return

    try:
        spike_validator = PressureSpikeValidator.load_model(ML_MODEL_PATH)
        safe_print(f"ML validator: loaded model from '{ML_MODEL_PATH}' (threshold={ML_CONFIDENCE_THRESHOLD:.2f}).")
    except Exception as e:
        spike_validator = None
        safe_print(f"ML validator: failed to load '{ML_MODEL_PATH}': {e}. Running without ML filtering.")


def load_goal_models() -> None:
    global goal_model_next10, goal_model_before_ht, goals_remaining_classifier

    if not GOAL_ML_ENABLED:
        safe_print("Goal ML: disabled (GOAL_ML_ENABLED=False).")
        goal_model_next10 = None
        goal_model_before_ht = None
        goals_remaining_classifier = None
        return

    def _load_support(path: str, desc: str) -> Optional[GoalOutcomeRuntime]:
        if not path or not os.path.exists(path):
            safe_print(f"Goal ML: support model not found for {desc}: {path}")
            return None
        try:
            m = GoalOutcomeRuntime.load_model(path)
            safe_print(f"Goal ML: loaded support model '{desc}' from {path} (target={m.target_type}).")
            return m
        except Exception as e:
            safe_print(f"Goal ML: failed to load support model '{desc}' from {path}: {e}")
            return None

    goal_model_next10 = _load_support(GOAL_MODEL_NEXT10_PATH, "next_10m")
    goal_model_before_ht = _load_support(GOAL_MODEL_BEFORE_HT_PATH, "before_ht")

    if not GOALS_CLS_ENABLED:
        goals_remaining_classifier = None
        safe_print("Goal ML: goals remaining classifier disabled (GOALS_CLS_ENABLED=False).")
    elif not GOALS_CLS_MODEL_PATH or not os.path.exists(GOALS_CLS_MODEL_PATH):
        goals_remaining_classifier = None
        safe_print(f"Goal ML: goals remaining classifier bundle not found: {GOALS_CLS_MODEL_PATH}")
    else:
        try:
            goals_remaining_classifier = GoalsRemainingClassifierRuntime.load_model(GOALS_CLS_MODEL_PATH)
            safe_print(f"Goal ML: loaded goals remaining classifier bundle from {GOALS_CLS_MODEL_PATH}")
        except Exception as e:
            goals_remaining_classifier = None
            safe_print(f"Goal ML: failed to load classifier bundle '{GOALS_CLS_MODEL_PATH}': {e}")



# ============================================================
# FEATURE BUILDERS FOR MODELS
# ============================================================
def live_league_goal_rate(league_name: str) -> float:
    if goal_model_next10 and goal_model_next10.is_trained:
        return float(goal_model_next10._league_rate(league_name))
    if goal_model_before_ht and goal_model_before_ht.is_trained:
        return float(goal_model_before_ht._league_rate(league_name))
    return 2.5


def build_goal_features_from_match(m: Dict[str, Any]) -> Dict[str, float]:
    """
    Support-model feature builder for the older next10 / beforeHT models.
    Keeps the 'half' field behavior compatible with your old setup.
    """
    minute = int(g(m, "time", "m", default=0) or 0)
    raw_period = get_raw_period(m, minute)

    # IMPORTANT:
    # Keep old behavior for support models to avoid breaking older trained files.
    half_feature = float(raw_period if isinstance(raw_period, int) else map_raw_period_to_half(None, minute))

    gh = int(g(m, "scores", "current", "h", default=0) or 0)
    ga = int(g(m, "scores", "current", "a", default=0) or 0)
    goals_total = gh + ga

    pressure = match_pressure(m)
    total_pressure = float(pressure["total_pressure"])
    pressure_diff = float(pressure["pressure_diff"])

    sot, corners_total = sot_and_corners(m)

    hs = g(m, "stats", "home", default={}) or {}
    as_ = g(m, "stats", "away", default={}) or {}
    soff_home = int(hs.get("shotofftarget", 0) or 0)
    soff_away = int(as_.get("shotofftarget", 0) or 0)
    total_shots = sot + soff_home + soff_away

    possession_ratio = _possession_ratio_from_stats(hs, as_)

    return {
        "minute": float(minute),
        "half": float(half_feature),
        "goals_total": float(goals_total),
        "total_pressure": float(total_pressure),
        "pressure_diff": float(pressure_diff),
        "shots_on_target": float(sot),
        "total_shots": float(total_shots),
        "corners_total": float(corners_total),
        "possession_ratio": float(possession_ratio),
    }


def build_remaining_goals_features_from_match(m: Dict[str, Any]) -> Optional[Tuple[int, Dict[str, float]]]:
    """
    Build full feature set for the goals_remaining_classifier_v2 bundle.
    Returns (half, features) or None if not available / not in regulation live-play.
    """
    if goals_remaining_classifier is None or not goals_remaining_classifier.is_trained:
        return None

    mid = m.get("id")
    if not mid:
        return None

    hist_deque = match_feature_history.get(mid)
    if not hist_deque:
        return None

    hist = list(hist_deque)
    current = hist[-1]

    minute = float(current["minute"])
    raw_period = int(current["raw_period"])
    if raw_period not in [10, 12]:
        return None

    half = map_raw_period_to_half(raw_period, int(minute))

    league_name = g(m, "tournament", "name", default="Unknown League")
    league_goal_rate = live_league_goal_rate(league_name)

    goals_home = float(current["goals_home"])
    goals_away = float(current["goals_away"])
    goals_total = float(current["goals_total"])

    total_pressure = float(current["total_pressure"])
    pressure_diff = float(current["pressure_diff"])
    shots_on_target = float(current["shots_on_target"])
    total_shots = float(current["total_shots"])
    corners_total = float(current["corners_total"])
    possession_ratio = float(current["possession_ratio"])

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

    pressure_last_5 = _window_delta_from_history(hist, minute, "total_pressure", total_pressure, 5)
    pressure_last_10 = _window_delta_from_history(hist, minute, "total_pressure", total_pressure, 10)
    shots_last_5 = _window_delta_from_history(hist, minute, "total_shots", total_shots, 5)
    shots_last_10 = _window_delta_from_history(hist, minute, "total_shots", total_shots, 10)
    sot_last_5 = _window_delta_from_history(hist, minute, "shots_on_target", shots_on_target, 5)
    sot_last_10 = _window_delta_from_history(hist, minute, "shots_on_target", shots_on_target, 10)
    corners_last_5 = _window_delta_from_history(hist, minute, "corners_total", corners_total, 5)
    corners_last_10 = _window_delta_from_history(hist, minute, "corners_total", corners_total, 10)

    pressure_prev_5 = _previous_window_delta_from_history(hist, minute, "total_pressure", total_pressure, 5)
    pressure_prev_10 = _previous_window_delta_from_history(hist, minute, "total_pressure", total_pressure, 10)
    shots_prev_5 = _previous_window_delta_from_history(hist, minute, "total_shots", total_shots, 5)
    shots_prev_10 = _previous_window_delta_from_history(hist, minute, "total_shots", total_shots, 10)
    sot_prev_5 = _previous_window_delta_from_history(hist, minute, "shots_on_target", shots_on_target, 5)
    sot_prev_10 = _previous_window_delta_from_history(hist, minute, "shots_on_target", shots_on_target, 10)
    corners_prev_5 = _previous_window_delta_from_history(hist, minute, "corners_total", corners_total, 5)
    corners_prev_10 = _previous_window_delta_from_history(hist, minute, "corners_total", corners_total, 10)

    possession_last_5_mean = _window_mean_from_history(
        hist, minute, "possession_ratio", minute - 5.0, minute, possession_ratio
    )
    possession_last_10_mean = _window_mean_from_history(
        hist, minute, "possession_ratio", minute - 10.0, minute, possession_ratio
    )
    possession_prev_5_mean = _window_mean_from_history(
        hist, minute, "possession_ratio", minute - 10.0, minute - 5.0, possession_last_5_mean
    )
    possession_prev_10_mean = _window_mean_from_history(
        hist, minute, "possession_ratio", minute - 20.0, minute - 10.0, possession_last_10_mean
    )

    pressure_spike_5 = pressure_last_5 - pressure_prev_5
    pressure_spike_10 = pressure_last_10 - pressure_prev_10
    shots_spike_5 = shots_last_5 - shots_prev_5
    shots_spike_10 = shots_last_10 - shots_prev_10
    sot_spike_5 = sot_last_5 - sot_prev_5
    sot_spike_10 = sot_last_10 - sot_prev_10
    corners_spike_5 = corners_last_5 - corners_prev_5
    corners_spike_10 = corners_last_10 - corners_prev_10
    possession_spike_5 = possession_last_5_mean - possession_prev_5_mean
    possession_spike_10 = possession_last_10_mean - possession_prev_10_mean

    played_minutes = max(1.0, minute)

    pressure_vs_match_avg = pressure_last_10 / ((total_pressure / played_minutes) + 1e-6)
    shots_vs_match_avg = shots_last_10 / ((total_shots / played_minutes) + 1e-6)
    sot_vs_match_avg = sot_last_10 / ((shots_on_target / played_minutes) + 1e-6)
    corners_vs_match_avg = corners_last_10 / ((corners_total / played_minutes) + 1e-6)

    sot_share_total_shots = (shots_on_target / total_shots) if total_shots > 0 else 0.0
    pressure_per_shot = (total_pressure / total_shots) if total_shots > 0 else 0.0

    draw_pressure_interaction = is_draw * pressure_last_10
    draw_shots_interaction = is_draw * shots_last_10
    close_game_pressure_interaction = is_one_goal_game * pressure_last_10
    close_game_shots_interaction = is_one_goal_game * shots_last_10

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

        "pressure_last_5": pressure_last_5,
        "pressure_last_10": pressure_last_10,
        "shots_last_5": shots_last_5,
        "shots_last_10": shots_last_10,
        "sot_last_5": sot_last_5,
        "sot_last_10": sot_last_10,
        "corners_last_5": corners_last_5,
        "corners_last_10": corners_last_10,
        "possession_last_5_mean": possession_last_5_mean,
        "possession_last_10_mean": possession_last_10_mean,

        "pressure_prev_5": pressure_prev_5,
        "pressure_prev_10": pressure_prev_10,
        "shots_prev_5": shots_prev_5,
        "shots_prev_10": shots_prev_10,
        "sot_prev_5": sot_prev_5,
        "sot_prev_10": sot_prev_10,
        "corners_prev_5": corners_prev_5,
        "corners_prev_10": corners_prev_10,
        "possession_prev_5_mean": possession_prev_5_mean,
        "possession_prev_10_mean": possession_prev_10_mean,

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

    return half, features


# ============================================================
# PRESSURE SPIKE VALIDATOR EVENT BUILDER
# ============================================================
def build_ml_event_from_match(m: Dict[str, Any], dt: float, dp: float, per_min: float) -> Dict[str, float]:
    minute = int(g(m, "time", "m", default=0) or 0)

    hs = g(m, "stats", "home", default={}) or {}
    as_ = g(m, "stats", "away", default={}) or {}

    sot_home = int(hs.get("shotontarget", hs.get("shots_on_target", 0)) or 0)
    sot_away = int(as_.get("shotontarget", as_.get("shots_on_target", 0)) or 0)
    sot_total = sot_home + sot_away

    soff_home = int(hs.get("shotofftarget", hs.get("shots_off_target", 0)) or 0)
    soff_away = int(as_.get("shotofftarget", as_.get("shots_off_target", 0)) or 0)
    total_shots = sot_total + soff_home + soff_away

    corners_home = int(hs.get("corners", 0) or 0)
    corners_away = int(as_.get("corners", 0) or 0)
    corners_total = corners_home + corners_away

    total_pressure = float(match_pressure(m)["total_pressure"])

    sot_ratio = (sot_total / max(total_shots, 1)) if total_shots >= 0 else 0.0
    possession_ratio = _possession_ratio_from_stats(hs, as_)

    corner_ratio = corners_total / max(10.0, float(minute if minute > 0 else 1))
    pressure_trend = 0.0

    return {
        "minute": float(minute),
        "time_delta_sec": float(dt),
        "pressure_delta": float(dp),
        "pressure_per_min": float(per_min),
        "total_pressure": float(total_pressure),
        "sot_ratio": float(sot_ratio),
        "possession_ratio": float(possession_ratio),
        "corner_ratio": float(corner_ratio),
        "pressure_trend": float(pressure_trend),
    }


# ============================================================
# COMPAT / DEFAULTS FOR NEW GOALS CLASSIFIER LOGIC
# ============================================================
if "GOALS_CLS_ENABLED" not in globals():
    GOALS_CLS_ENABLED = os.getenv("GOALS_CLS_ENABLED", "1") == "1"
if "GOALS_CLS_MODEL_PATH" not in globals():
    GOALS_CLS_MODEL_PATH = os.getenv("GOALS_CLS_MODEL_PATH", "goals_remaining_classifier_v2.pkl").strip()

if "GOALS_CLS_MIN_MINUTE" not in globals():
    GOALS_CLS_MIN_MINUTE = int(os.getenv("GOALS_CLS_MIN_MINUTE", "10"))
if "GOALS_CLS_MAX_MINUTE" not in globals():
    GOALS_CLS_MAX_MINUTE = int(os.getenv("GOALS_CLS_MAX_MINUTE", "88"))
if "GOALS_CLS_ALERT_COOLDOWN" not in globals():
    GOALS_CLS_ALERT_COOLDOWN = int(os.getenv("GOALS_CLS_ALERT_COOLDOWN", "180"))

if "FIRST_HALF_GE1_INFO_THRESHOLD" not in globals():
    FIRST_HALF_GE1_INFO_THRESHOLD = float(os.getenv("FIRST_HALF_GE1_INFO_THRESHOLD", "0.78"))
if "FIRST_HALF_GE2_STRONG_THRESHOLD" not in globals():
    FIRST_HALF_GE2_STRONG_THRESHOLD = float(os.getenv("FIRST_HALF_GE2_STRONG_THRESHOLD", "0.62"))
if "FIRST_HALF_GE3_EXPLOSIVE_THRESHOLD" not in globals():
    FIRST_HALF_GE3_EXPLOSIVE_THRESHOLD = float(os.getenv("FIRST_HALF_GE3_EXPLOSIVE_THRESHOLD", "0.30"))

if "SECOND_HALF_GE1_STRONG_THRESHOLD" not in globals():
    SECOND_HALF_GE1_STRONG_THRESHOLD = float(os.getenv("SECOND_HALF_GE1_STRONG_THRESHOLD", "0.70"))
if "SECOND_HALF_GE2_STRONG_THRESHOLD" not in globals():
    SECOND_HALF_GE2_STRONG_THRESHOLD = float(os.getenv("SECOND_HALF_GE2_STRONG_THRESHOLD", "0.45"))
if "SECOND_HALF_GE2_VERY_STRONG_THRESHOLD" not in globals():
    SECOND_HALF_GE2_VERY_STRONG_THRESHOLD = float(os.getenv("SECOND_HALF_GE2_VERY_STRONG_THRESHOLD", "0.55"))
if "SECOND_HALF_GE3_EXPLOSIVE_THRESHOLD" not in globals():
    SECOND_HALF_GE3_EXPLOSIVE_THRESHOLD = float(os.getenv("SECOND_HALF_GE3_EXPLOSIVE_THRESHOLD", "0.22"))

if "SEND_RAW_SPIKE_ALERTS" not in globals():
    SEND_RAW_SPIKE_ALERTS = True

if "GOAL_SUPPORT_ENABLED" not in globals():
    GOAL_SUPPORT_ENABLED = GOAL_ML_ENABLED

if "last_goal_signal_alert_ts" not in globals():
    last_goal_signal_alert_ts: Dict[str, float] = {}

if "match_feature_history" not in globals():
    match_feature_history: Dict[str, Deque[Dict[str, Any]]] = {}

if "goals_remaining_classifier" not in globals():
    goals_remaining_classifier = None


# ============================================================
# GOALS REMAINING CLASSIFIER RUNTIME
# ============================================================
class GoalsRemainingClassifierRuntime:
    """
    Runtime loader for goals_remaining_classifier_v2.pkl

    Expected structure:
      {
        "features": [...],
        "first_half": {
            1: {"model": ...},
            2: {"model": ...},
            3: {"model": ...},
        },
        "second_half": {
            1: {"model": ...},
            2: {"model": ...},
            3: {"model": ...},
        }
      }
    """

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
        obj.feature_names = data["features"]
        obj.first_half = data["first_half"]
        obj.second_half = data["second_half"]
        obj.is_trained = True
        return obj

    def predict_probs(self, half: int, features: Dict[str, float]) -> Dict[str, float]:
        if not self.is_trained:
            raise ValueError("Goals remaining classifier not loaded.")

        if half not in (1, 2):
            raise ValueError(f"Invalid half={half}")

        row = [float(features.get(fname, 0.0) or 0.0) for fname in self.feature_names]
        X = pd.DataFrame([row], columns=self.feature_names)

        bundle = self.first_half if half == 1 else self.second_half
        out: Dict[str, float] = {}

        for k in (1, 2, 3):
            entry = bundle.get(k)
            if not entry or entry.get("model") is None:
                out[f"ge{k}"] = 0.0
                continue

            model = entry["model"]
            p = float(model.predict_proba(X)[0][1])
            out[f"ge{k}"] = p

        return out


# ============================================================
# SUPPORT FEATURE BUILDER (old next10 / beforeHT models)
# ============================================================
def build_goal_support_features_from_match(m: Dict[str, Any]) -> Dict[str, float]:
    minute = int(g(m, "time", "m", default=0) or 0)
    raw_period = g(m, "time", "p", default=None)
    half = map_raw_period_to_half(raw_period, minute)

    gh = int(g(m, "scores", "current", "h", default=0) or 0)
    ga = int(g(m, "scores", "current", "a", default=0) or 0)
    goals_total = gh + ga

    pressure = match_pressure(m)
    total_pressure = float(pressure["total_pressure"])
    pressure_diff = float(pressure["pressure_diff"])

    sot, corners_total = sot_and_corners(m)

    hs = g(m, "stats", "home", default={}) or {}
    as_ = g(m, "stats", "away", default={}) or {}
    soff_home = int(hs.get("shotofftarget", hs.get("shots_off_target", 0)) or 0)
    soff_away = int(as_.get("shotofftarget", as_.get("shots_off_target", 0)) or 0)
    total_shots = sot + soff_home + soff_away

    possession_ratio = _possession_ratio_from_stats(hs, as_)

    return {
        "minute": float(minute),
        "half": float(half),
        "goals_total": float(goals_total),
        "total_pressure": float(total_pressure),
        "pressure_diff": float(pressure_diff),
        "shots_on_target": float(sot),
        "total_shots": float(total_shots),
        "corners_total": float(corners_total),
        "possession_ratio": float(possession_ratio),
    }


# ============================================================
# LIVE MATCH FEATURE HISTORY
# ============================================================
def build_match_feature_snapshot(m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    mid = m.get("id")
    if not mid:
        return None

    minute = g(m, "time", "m", default=None)
    if not isinstance(minute, int):
        return None

    raw_period = g(m, "time", "p", default=None)
    if raw_period not in [10, 12]:
        return None

    half = map_raw_period_to_half(raw_period, minute)

    gh = int(g(m, "scores", "current", "h", default=0) or 0)
    ga = int(g(m, "scores", "current", "a", default=0) or 0)
    goals_total = gh + ga

    pressure = match_pressure(m)
    total_pressure = float(pressure["total_pressure"])
    pressure_diff = float(pressure["pressure_diff"])

    sot, corners_total = sot_and_corners(m)

    hs = g(m, "stats", "home", default={}) or {}
    as_ = g(m, "stats", "away", default={}) or {}
    soff_home = int(hs.get("shotofftarget", hs.get("shots_off_target", 0)) or 0)
    soff_away = int(as_.get("shotofftarget", as_.get("shots_off_target", 0)) or 0)
    total_shots = sot + soff_home + soff_away

    possession_ratio = _possession_ratio_from_stats(hs, as_)

    return {
        "ts": time.time(),
        "match_id": str(mid),
        "minute": int(minute),
        "raw_period": int(raw_period),
        "half": int(half),
        "goals_home": gh,
        "goals_away": ga,
        "goals_total": goals_total,
        "total_pressure": float(total_pressure),
        "pressure_diff": float(pressure_diff),
        "shots_on_target": float(sot),
        "total_shots": float(total_shots),
        "corners_total": float(corners_total),
        "possession_ratio": float(possession_ratio),
    }


def update_match_feature_history(m: Dict[str, Any]) -> None:
    snap = build_match_feature_snapshot(m)
    if snap is None:
        return

    mid = str(snap["match_id"])
    hist = match_feature_history.get(mid)
    if hist is None:
        hist = deque(maxlen=120)
        match_feature_history[mid] = hist

    if hist and int(hist[-1]["minute"]) == int(snap["minute"]):
        hist[-1] = snap
    else:
        hist.append(snap)


# ============================================================
# LIVE LEAGUE RATE
# ============================================================
def live_league_goal_rate(league_name: str) -> float:
    # prefer support-model league stats if available
    try:
        if goal_model_next10 is not None and goal_model_next10.is_trained:
            return float(goal_model_next10._league_rate(league_name))
    except Exception:
        pass

    try:
        if goal_model_before_ht is not None and goal_model_before_ht.is_trained:
            return float(goal_model_before_ht._league_rate(league_name))
    except Exception:
        pass

    return 2.5


# ============================================================
# GOALS CLASSIFIER TELEGRAM FORMATTER
# ============================================================
def format_goals_classifier_message(
    m: Dict[str, Any],
    spike_reason: str,
    label: str,
    p_ge1: float,
    p_ge2: float,
    p_ge3: float,
    p_next10: Optional[float] = None,
    p_before_ht: Optional[float] = None,
) -> str:
    minute = int(g(m, "time", "m", default=0) or 0)
    league_name = g(m, "tournament", "name", default="Unknown League")
    home = g(m, "home", "name", default="Home")
    away = g(m, "away", "name", default="Away")
    gh = int(g(m, "scores", "current", "h", default=0) or 0)
    ga = int(g(m, "scores", "current", "a", default=0) or 0)
    goals_total = gh + ga

    pressure = match_pressure(m)
    total_pressure = int(pressure["total_pressure"])
    sot, corners_total = sot_and_corners(m)

    hs = g(m, "stats", "home", default={}) or {}
    as_ = g(m, "stats", "away", default={}) or {}
    possession = _possession_ratio_from_stats(hs, as_) * 100.0

    lines = [
        "⚽ LIVE GOALS MODEL",
        league_name,
        f"{minute}' {home} {gh}-{ga} {away}",
        f"Signal: {label}",
        f"Reason: {spike_reason}",
        "",
        f"Over {goals_total + 0.5:.1f}: {p_ge1:.0%}",
        f"Over {goals_total + 1.5:.1f}: {p_ge2:.0%}",
        f"Over {goals_total + 2.5:.1f}: {p_ge3:.0%}",
    ]

    if p_next10 is not None:
        lines.append(f"P(goal next 10′): {p_next10:.0%}")
    if p_before_ht is not None:
        lines.append(f"P(goal before HT): {p_before_ht:.0%}")

    lines.extend([
        "",
        f"Pressure: {total_pressure} | SOT: {sot} | Corners: {corners_total}",
        f"Dominant possession: {possession:.0f}%",
    ])

    return "\n".join(lines)


# ============================================================
# LOAD GOAL MODELS (OVERRIDES OLD VERSION)
# ============================================================
def load_goal_models() -> None:
    global goal_model_next10, goal_model_before_ht, goals_remaining_classifier

    if not GOAL_SUPPORT_ENABLED:
        goal_model_next10 = None
        goal_model_before_ht = None
        safe_print("Goal support models: disabled.")
    else:
        def _load_one(path: str, desc: str) -> Optional[GoalOutcomeRuntime]:
            if not path or not os.path.exists(path):
                safe_print(f"Goal support: {desc} model file not found: {path}")
                return None
            try:
                m = GoalOutcomeRuntime.load_model(path)
                safe_print(f"Goal support: loaded {desc} model from {path} (target={m.target_type}).")
                return m
            except Exception as e:
                safe_print(f"Goal support: failed to load {desc} model '{path}': {e}")
                return None

        goal_model_next10 = _load_one(GOAL_MODEL_NEXT10_PATH, "next_10m")
        goal_model_before_ht = _load_one(GOAL_MODEL_BEFORE_HT_PATH, "before_ht")

    if not GOALS_CLS_ENABLED:
        goals_remaining_classifier = None
        safe_print("Goals remaining classifier: disabled.")
    elif not GOALS_CLS_MODEL_PATH or not os.path.exists(GOALS_CLS_MODEL_PATH):
        goals_remaining_classifier = None
        safe_print(f"Goals remaining classifier: file not found: {GOALS_CLS_MODEL_PATH}")
    else:
        try:
            goals_remaining_classifier = GoalsRemainingClassifierRuntime.load_model(GOALS_CLS_MODEL_PATH)
            safe_print(f"Goals remaining classifier: loaded bundle from {GOALS_CLS_MODEL_PATH}")
        except Exception as e:
            goals_remaining_classifier = None
            safe_print(f"Goals remaining classifier: failed to load '{GOALS_CLS_MODEL_PATH}': {e}")


# ============================================================
# MAIN GOALS MODEL SIGNAL ENGINE
# ============================================================
def maybe_run_goal_models(m: Dict[str, Any], spike_reason: str) -> None:
    if not GOALS_CLS_ENABLED:
        return

    if goals_remaining_classifier is None or not goals_remaining_classifier.is_trained:
        return

    mid = m.get("id")
    if not mid:
        return

    minute = g(m, "time", "m", default=None)
    if not isinstance(minute, int):
        return

    if minute < GOALS_CLS_MIN_MINUTE or minute > GOALS_CLS_MAX_MINUTE:
        return

    raw_period = g(m, "time", "p", default=None)
    if raw_period not in [10, 12]:
        return

    now = time.time()
    last = last_goal_signal_alert_ts.get(str(mid), 0.0)
    if now - last < GOALS_CLS_ALERT_COOLDOWN:
        return

    built = build_remaining_goals_features_from_match(m)
    if built is None:
        return

    half, features = built

    try:
        probs = goals_remaining_classifier.predict_probs(half, features)
    except Exception as e:
        safe_print(f"Goals remaining classifier prediction failed for match {mid}: {e}")
        return

    p_ge1 = float(probs.get("ge1", 0.0))
    p_ge2 = float(probs.get("ge2", 0.0))
    p_ge3 = float(probs.get("ge3", 0.0))

    # Enforce logical monotonicity: P(>=1) >= P(>=2) >= P(>=3)
    p_ge2 = min(p_ge2, p_ge1)
    p_ge3 = min(p_ge3, p_ge2)

    # support models
    league_name = g(m, "tournament", "name", default="Unknown League")
    p_next10: Optional[float] = None
    p_before_ht: Optional[float] = None

    try:
        if goal_model_next10 is not None and goal_model_next10.is_trained:
            f = build_goal_support_features_from_match(m)
            f["league_goal_rate"] = goal_model_next10._league_rate(league_name)
            p_next10 = float(goal_model_next10.predict_proba(f))
    except Exception as e:
        safe_print(f"Goal support next10 failed for match {mid}: {e}")

    try:
        if half == 1 and goal_model_before_ht is not None and goal_model_before_ht.is_trained:
            f = build_goal_support_features_from_match(m)
            f["league_goal_rate"] = goal_model_before_ht._league_rate(league_name)
            p_before_ht = float(goal_model_before_ht.predict_proba(f))
    except Exception as e:
        safe_print(f"Goal support beforeHT failed for match {mid}: {e}")

    label: Optional[str] = None

    pressure = match_pressure(m)
    total_pressure = float(pressure["total_pressure"])
    sot, corners_total = sot_and_corners(m)

    if half == 1:
        if (
            p_ge3 >= 0.30
            and p_ge2 >= 0.45
            and p_ge1 >= 0.60
            and total_pressure >= 140
            and (sot >= 2 or corners_total >= 3)
        ):
            label = "EXPLOSIVE OVER (1H)"
        elif (
            p_ge2 >= 0.50
            and p_ge1 >= 0.70
            and (p_next10 is not None and p_next10 >= 0.30)
        ):
            label = "STRONG OVER (1H)"
        elif p_ge1 >= 0.80:
            label = "LIVE OVER WATCH (1H)"

    else:
        if (
            p_ge3 >= 0.22
            and p_ge2 >= 0.35
            and p_ge1 >= 0.55
            and total_pressure >= 160
            and (sot >= 3 or corners_total >= 4)
        ):
            label = "EXPLOSIVE OVER (2H)"
        elif (
            p_ge2 >= 0.45
            and p_ge1 >= 0.65
        ):
            label = "VERY STRONG OVER (2H)"
        elif p_ge1 >= 0.72:
            label = "STRONG OVER (2H)"

    if label is None:
        return

    last_goal_signal_alert_ts[str(mid)] = now

    msg = format_goals_classifier_message(
        m=m,
        spike_reason=spike_reason,
        label=label,
        p_ge1=p_ge1,
        p_ge2=p_ge2,
        p_ge3=p_ge3,
        p_next10=p_next10,
        p_before_ht=p_before_ht,
    )
    safe_print(msg)
    telegram_notify(msg)


# backward-compatible alias
def maybe_run_goal_ml(m: Dict[str, Any], spike_reason: str) -> None:
    maybe_run_goal_models(m, spike_reason)


# ============================================================
# ALERT LOGIC
# ============================================================
def maybe_alert_total_pressure_spike(m: Dict[str, Any]) -> None:
    mid = m.get("id")
    if not mid:
        return

    minute = g(m, "time", "m", default=None)
    if isinstance(minute, int) and minute > SPIKES_MAX_MINUTE:
        return

    now = time.time()
    p_now = float(match_pressure(m)["total_pressure"])

    prev = last_total_pressure.get(str(mid))
    if prev is None:
        last_total_pressure[str(mid)] = {"p": p_now, "ts": now}
        return

    dt = now - float(prev.get("ts", now))
    if dt <= 0:
        last_total_pressure[str(mid)] = {"p": p_now, "ts": now}
        return

    p_prev = float(prev.get("p", p_now))
    dp = p_now - p_prev
    per_min = dp / (dt / 60.0) if dt > 0 else 0.0

    last_total_pressure[str(mid)] = {"p": p_now, "ts": now}

    if dp < SPIKE_DELTA_TOTAL or per_min < SPIKE_PER_MIN_TOTAL:
        return

    last_ts = last_total_alert_ts.get(str(mid), 0.0)
    if now - last_ts < ALERT_COOLDOWN_SECONDS_TOTAL:
        return

    ml_conf: Optional[float] = None
    if spike_validator is not None and ML_ENABLED:
        try:
            event = build_ml_event_from_match(m, dt=dt, dp=dp, per_min=per_min)
            legit, ml_conf = spike_validator.predict(event)
            if not legit or (ml_conf is not None and ml_conf < ML_CONFIDENCE_THRESHOLD):
                return
        except Exception as e:
            safe_print(f"ML validator error (fallback to raw): {e}")

    last_total_alert_ts[str(mid)] = now

    goal_probs = get_current_goal_probabilities(m)
    goal_prob_text = format_goal_prob_summary(goal_probs)

    try:
        maybe_run_goal_models(m, spike_reason="TOTAL_PRESSURE_SPIKE")
    except Exception as e:
        safe_print(f"Goals model error on TOTAL_PRESSURE_SPIKE for match {mid}: {e}")

    if not SEND_RAW_SPIKE_ALERTS:
        return

    tour = g(m, "tournament", "name", default="(unknown tournament)")
    home = g(m, "home", "name", default="Home")
    away = g(m, "away", "name", default="Away")
    sh = g(m, "scores", "current", "h", default=0)
    sa = g(m, "scores", "current", "a", default=0)

    msg = (
        f"🚨TOTAL PRESSURE SPIKE\n"
        f"{tour}\n"
        f"{minute}' {home} {sh}-{sa} {away}\n"
        f"DeltaP=+{int(dp)} (~{per_min:.0f}/min) | TotalP={int(p_now)}"
    )
    if ml_conf is not None:
        msg += f"\nML confidence: {ml_conf:.0%} (thr={ML_CONFIDENCE_THRESHOLD:.0%})"

    if goal_prob_text:
        msg += f"\n{goal_prob_text}"

    safe_print(msg)
    telegram_notify(msg)


def maybe_alert_team_pressure_spike(m: Dict[str, Any]) -> None:
    mid = m.get("id")
    if not mid:
        return

    minute = g(m, "time", "m", default=None)
    if isinstance(minute, int) and minute > SPIKES_MAX_MINUTE:
        return

    pressure = match_pressure(m)
    p_home_now = float(pressure.get("home_pressure", 0.0))
    p_away_now = float(pressure.get("away_pressure", 0.0))
    now = time.time()

    def _check_side(side_key: str, side_label: str, p_now: float) -> None:
        key = (str(mid), side_key)

        prev = last_team_pressure.get(key)
        if prev is None:
            last_team_pressure[key] = {"p": p_now, "ts": now}
            return

        dt = now - float(prev.get("ts", now))
        if dt <= 0:
            last_team_pressure[key] = {"p": p_now, "ts": now}
            return

        p_prev = float(prev.get("p", p_now))
        dp = p_now - p_prev
        per_min = dp / (dt / 60.0) if dt > 0 else 0.0

        last_team_pressure[key] = {"p": p_now, "ts": now}

        if dp < SPIKE_DELTA_TEAM or per_min < SPIKE_PER_MIN_TEAM:
            return

        last_ts = last_team_alert_ts.get(key, 0.0)
        if now - last_ts < ALERT_COOLDOWN_SECONDS_TEAM:
            return

        last_team_alert_ts[key] = now

        goal_probs = get_current_goal_probabilities(m)
        goal_prob_text = format_goal_prob_summary(goal_probs)

        try:
            maybe_run_goal_models(m, spike_reason=f"{side_label}_TEAM_PRESSURE_SPIKE")
        except Exception as e:
            safe_print(f"Goals model error on {side_label}_TEAM_PRESSURE_SPIKE for match {mid}: {e}")

        if not SEND_RAW_SPIKE_ALERTS:
            return

        tour = g(m, "tournament", "name", default="(unknown tournament)")
        home = g(m, "home", "name", default="Home")
        away = g(m, "away", "name", default="Away")
        sh = g(m, "scores", "current", "h", default=0)
        sa = g(m, "scores", "current", "a", default=0)

        msg = (
            f"💥TEAM PRESSURE SPIKE ({side_label})\n"
            f"{tour}\n"
            f"{minute}' {home} {sh}-{sa} {away}\n"
            f"ΔP=+{int(dp)} (~{per_min:.0f}/min) | P_now={int(p_now)}"
        )

        if goal_prob_text:
            msg += f"\n{goal_prob_text}"
        safe_print(msg)
        telegram_notify(msg)

    _check_side("H", "HOME", p_home_now)
    _check_side("A", "AWAY", p_away_now)


def _update_and_check_corner_burst(mid: str, minute: int, corners_total: int) -> Tuple[bool, int]:
    hist = corner_history.get(mid)
    if hist is None:
        hist = deque()
        corner_history[mid] = hist

    hist.append((minute, corners_total))

    min_allowed = minute - CORNERS_BURST_WINDOW_MINUTES
    while hist and hist[0][0] < min_allowed:
        hist.popleft()

    if len(hist) < 2:
        return False, 0

    _, earliest_c = hist[0]
    _, latest_c = hist[-1]
    corners_in_window = max(0, latest_c - earliest_c)

    return corners_in_window >= CORNERS_BURST_REQUIRED, corners_in_window


def _sot_vs_goals_ok(sot: int, goals_total: int) -> Tuple[bool, int]:
    ratio_req = SOT_MIN_IF_0_GOALS if goals_total == 0 else goals_total * SOT_RATIO_PER_GOAL
    required_sot = max(ratio_req, SOT_GENERAL_MIN)
    ok = (sot >= required_sot) and (sot >= goals_total + SOT_MUST_BEAT_GOALS_BY)
    return ok, required_sot


def maybe_alert_high_pressure_stats(m: Dict[str, Any]) -> None:
    mid = m.get("id")
    if not mid:
        return

    minute = g(m, "time", "m", default=None)
    if not isinstance(minute, int):
        return

    if minute < STATS_MIN_MINUTE or minute > SPIKES_MAX_MINUTE:
        return

    sot, corners = sot_and_corners(m)
    gh = int(g(m, "scores", "current", "h", default=0) or 0)
    ga = int(g(m, "scores", "current", "a", default=0) or 0)
    goals_total = gh + ga

    burst, corners_in_window = _update_and_check_corner_burst(str(mid), minute, corners)
    if not burst:
        return

    sot_ok, required_sot = _sot_vs_goals_ok(sot, goals_total)
    if not sot_ok:
        return

    now = time.time()
    last = last_stats_alert_ts.get(str(mid), 0.0)
    if now - last < SPC_ALERT_COOLDOWN:
        return
    last_stats_alert_ts[str(mid)] = now

    goal_probs = get_current_goal_probabilities(m)
    goal_prob_text = format_goal_prob_summary(goal_probs)

    prev_best = match_stat_accum.get(str(mid), {}).get("best_spc", 0)
    match_stat_accum[str(mid)] = {"best_spc": max(prev_best, sot + corners)}

    try:
        maybe_run_goal_models(m, spike_reason="STATS_BURST_CORNER_SOT")
    except Exception as e:
        safe_print(f"Goals model error on STATS_BURST_CORNER_SOT for match {mid}: {e}")

    if not SEND_RAW_SPIKE_ALERTS:
        return

    tour = g(m, "tournament", "name", default="(unknown tournament)")
    home = g(m, "home", "name", default="Home")
    away = g(m, "away", "name", default="Away")
    sh = g(m, "scores", "current", "h", default=0)
    sa = g(m, "scores", "current", "a", default=0)

    msg = (
        f"🔥HIGH PRESSURE STATS (corner burst)\n"
        f"{tour}\n"
        f"{minute}' {home} {sh}-{sa} {away}\n"
        f"SOT: {sot} (req>={required_sot}, goals={goals_total})\n"
        f"Corners: {corners} (+{corners_in_window} in last {CORNERS_BURST_WINDOW_MINUTES}m)"
    )

    if goal_prob_text:
        msg += f"\n{goal_prob_text}"

    safe_print(msg)
    telegram_notify(msg)


# ============================================================
# FETCHING (subscription token)
# ============================================================
def fetch_live_data(timeout: int = REQUEST_TIMEOUT) -> Dict[str, Any]:
    # warm cookies
    session.get(HOMEPAGE_URL, timeout=timeout)

    headers = {
        "content-type": "application/json",
        "origin": "https://www.overlyzer.com",
        "referer": "https://www.overlyzer.com/",
        "accept": "*/*",
        "user-agent": USER_AGENT,
        "x-app-mode": "0",
    }

    if OVERLYZER_TOKEN:
        if USE_AUTH_HEADER == "x-app-authtoken":
            headers["x-app-authtoken"] = OVERLYZER_TOKEN
        else:
            headers["authorization"] = OVERLYZER_TOKEN

        if SET_X_WEB_CLIENTTOKEN:
            headers["x-web-clienttoken"] = OVERLYZER_TOKEN

    payload = {"clientToken": ""}

    r = session.post(API_URL, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ============================================================
# POLL ONCE
# ============================================================
def poll_once() -> Tuple[int, int, int, int]:
    global_stats["polls_total"] += 1

    d = fetch_live_data()
    live = d.get("liveMatches", {}) or {}
    ms: List[Dict[str, Any]] = live.get("matches", []) or []
    total_available = int(live.get("totalCount", len(ms)) or len(ms))

    visible_count = len(ms)
    global_stats["matches_in_feed_sum"] += visible_count

    new_count = 0
    updated_count = 0
    ids: List[str] = []

    for m in ms:
        # minute logging for future training
        try:
            log_minute_state(m)
        except Exception as e:
            safe_print(f"Minute log error during poll: {e}")

        # update rolling feature history BEFORE alerts/models
        try:
            update_match_feature_history(m)
        except Exception as e:
            safe_print(f"Match feature history update error: {e}")

        # alerts
        maybe_alert_total_pressure_spike(m)
        maybe_alert_team_pressure_spike(m)
        maybe_alert_high_pressure_stats(m)

        # track per-match state / updates
        mid = m.get("id")
        if not mid:
            continue

        mid = str(mid)
        ids.append(mid)

        if mid not in seen_matches:
            seen_matches[mid] = m
            new_count += 1
            global_stats["matches_seen_unique"] += 1

        if mid not in match_last:
            match_last[mid] = m
        else:
            if is_match_updated(match_last[mid], m):
                match_last[mid] = m
                updated_count += 1
                match_updates[mid] = match_updates.get(mid, 0) + 1

    global_stats["new_matches_sum"] += new_count
    global_stats["updated_matches_sum"] += updated_count

    snapshots.append({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "visible_count": visible_count,
        "total_available": total_available,
        "match_ids": ids,
    })

    if not REPORT_ONLY_ALERTS:
        now_str = datetime.now().strftime("%H:%M:%S")
        safe_print(
            f"[{now_str}] visible={visible_count} | total={total_available} | "
            f"new={new_count} | updated={updated_count} | total_seen={global_stats['matches_seen_unique']}"
        )

    return visible_count, new_count, updated_count, total_available


# ============================================================
# SAVE STATE
# ============================================================
def save_state():
    if not SAVE_STATE_TO_JSON:
        return

    state = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "global_stats": global_stats,
        "seen_matches_count": len(seen_matches),
        "match_last_count": len(match_last),
        "match_updates_count": len(match_updates),
        "match_feature_history_count": len(match_feature_history),
        "snapshots": snapshots,
        "match_updates": match_updates,
        "match_last": match_last,
        "match_stat_accum": match_stat_accum,
    }

    try:
        with open(STATE_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ============================================================
# RUNNER
# ============================================================
def main():
    safe_print(f"Keepalive ON — polling every {KEEPALIVE_INTERVAL}s (REST)")
    if OVERLYZER_TOKEN:
        safe_print(f"Overlyzer auth: enabled ({USE_AUTH_HEADER} + x-web-clienttoken={SET_X_WEB_CLIENTTOKEN})")
    else:
        safe_print("Overlyzer auth: NOT SET (set env var OVERLYZER_TOKEN)")

    load_spike_validator()
    load_goal_models()
    telegram_startup_test()

    while not _stop:
        try:
            visible, newc, updc, total_avail = poll_once()
            safe_print(f"[FEED DEBUG] visible={visible} | total_available={total_avail}")
        except Exception as e:
            safe_print(f"Error during poll: {e}")
            backoff_left = ERROR_BACKOFF
            while backoff_left > 0 and not _stop:
                time.sleep(1)
                backoff_left -= 1
            continue

        sleep_left = KEEPALIVE_INTERVAL
        while sleep_left > 0 and not _stop:
            time.sleep(1)
            sleep_left -= 1

    save_state()
    safe_print("Exited cleanly.")


if __name__ == "__main__":
    if KEEPALIVE_MODE:
        main()