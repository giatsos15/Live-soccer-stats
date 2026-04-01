import os

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
OVERLYZER_TOKEN = os.getenv("OVERLYZER_TOKEN", "").strip()

MINUTE_LOG_PATH = os.getenv("MINUTE_LOG_PATH", "data/logs/minute_states_log.csv")

GOAL_MODEL_NEXT10_PATH = os.getenv("GOAL_MODEL_NEXT10_PATH", "models/goal_model_next_10m.pkl")
GOAL_MODEL_BEFORE_HT_PATH = os.getenv("GOAL_MODEL_BEFORE_HT_PATH", "models/goal_model_before_ht.pkl")
GOALS_CLS_MODEL_PATH = os.getenv("GOALS_CLS_MODEL_PATH", "models/goals_remaining_classifier_v2.pkl")
GOALS_REG_MODEL_PATH = os.getenv("GOALS_REG_MODEL_PATH", "models/goals_remaining_regressor.pkl")