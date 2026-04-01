
# Live Soccer Stats

A real-time soccer monitoring and machine learning system built on top of Overlyzer live match data. The project focuses on detecting high-probability goal scenarios using minute-level match state features and predictive models.

---

## Overview

This project combines:

1. **Live match ingestion** from the Overlyzer API  
2. **Feature engineering** from minute-level match states  
3. **Machine learning models** for short-term goal prediction  
4. **Automated alerting** via Telegram  

The system is designed to identify high-value in-play betting opportunities based on match dynamics such as pressure, shots, corners, and momentum.

---

## How the System Works

### 1. Live Data Ingestion

The bot continuously polls the Overlyzer API:
https://connect.overlyzer.ws/api/v2/live



For each live match, it retrieves:

- attacks and dangerous attacks  
- shots on target / off target  
- corners  
- possession  
- match time and score  

---

### 2. Match State Construction

Each poll is transformed into a structured **minute-level state**.

For every match and minute, the system builds features such as:

- `minute`, `half`  
- `goals_home`, `goals_away`, `goals_total`  
- `total_pressure` (derived from attack metrics)  
- `pressure_diff`  
- `shots_on_target`, `total_shots`  
- `corners_total`  
- `possession_ratio`  

These states are appended to a CSV log:
data/logs/minute_states_log.csv


This dataset is the foundation for all modeling.

---

### 3. Feature Engineering & Momentum

The system tracks short-term dynamics such as:

- pressure spikes  
- shot intensity  
- corner bursts  
- pressure vs goals mismatch  

These signals are used both:

- in real-time alert logic  
- as features for machine learning models  

---

### 4. Machine Learning Models

Two types of models are used:

#### A. Goal Outcome Classification

Predicts probabilities for:

- goal in the next 10 minutes  
- goal before half-time  
- final match over 2.5 goals  

Training is performed using historical minute states.

---

#### B. Remaining Goals Regression

Predicts:
remaining_goals = final_total_goals - current_goals


This estimates how many goals are expected until the end of the match.

---

### 5. Real-Time Decision Logic

During live execution, the bot:

1. Maintains a rolling history per match  
2. Detects high-pressure or high-activity states  
3. Applies trained models (if enabled)  
4. Combines statistical signals + model probabilities  
5. Generates alerts when thresholds are exceeded  

---

### 6. Telegram Alerting

When a strong signal is detected, the bot sends a message to a Telegram channel including:

- match information  
- current score and minute  
- pressure and shot metrics  
- model probabilities (if available)  

---

### 7. Offline Signal Scanner

A separate script analyzes recent logged data:

- reads latest rows from the minute log  
- filters valid in-play states  
- scores them using trained models  
- outputs ranked opportunities  

Output file:
fast_goal_signal_scan_results.csv

---

## Repository Structure
```text
live-soccer-stats/
├── src/
│   └── live_soccer/
│       ├── bot.py
│       ├── config.py
│       ├── models/
│       │   ├── goal_outcome_trainer.py
│       │   └── goals_regression_trainer.py
│       └── scanners/
│           └── fast_goal_signal_scan.py
├── scripts/
│   ├── run_bot.py
│   ├── train_goal_outcome.py
│   ├── train_goals_regression.py
│   └── scan_signals.py
├── data/
│   ├── sample/
│   └── logs/
├── models/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Environment Setup

Create a `.env` file based on `.env.example`:

```env
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
OVERLYZER_TOKEN=

MINUTE_LOG_PATH=data/logs/minute_states_log.csv

GOAL_MODEL_NEXT10_PATH=models/goal_model_next_10m.pkl
GOALS_REG_MODEL_PATH=models/goals_remaining_regressor.pkl
```

## Installation

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

### Linux / macOS

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Run the live bot

```bash
python scripts/run_bot.py
```

### Train regression model

```bash
python scripts/train_goals_regression.py
```

### Run signal scanner

```bash
python scripts/scan_signals.py
```

## Data & Model Notes

This repository does not include:

- raw minute logs
- trained model binaries
- runtime state files

These are generated locally during execution.

## Design Philosophy

The project focuses on:

- real-time feature extraction from live sports data
- combining statistical signals with machine learning
- identifying short-horizon opportunities rather than long-term forecasting
- building a modular pipeline for experimentation and improvement

## Future Improvements

- model calibration and probability reliability
- automated backtesting and ROI tracking
- feature importance analysis
- deployment with Docker or on a VPS
- integration with betting APIs

## License

MIT License
