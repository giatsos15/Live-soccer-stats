# Live Soccer Stats

A live soccer monitoring and modeling project built around Overlyzer live match data, minute-level state logging, and machine learning models for short-horizon goal probability and expected remaining goals.

---

## ?? Key Features

- Live match monitoring via Overlyzer API  
- Minute-level feature engineering and logging  
- Machine learning models for:
  - Goal in next 10 minutes  
  - Goal before half-time  
  - Final over 2.5  
  - Expected remaining goals  
- Real-time Telegram alerting  
- Offline signal scanning and validation  

---

## Project Overview

The codebase combines three core workflows:

1. **Live ingestion and alerting** through an Overlyzer polling bot with Telegram notifications  
2. **Model training** from minute-by-minute match state logs  
3. **Offline scanning / validation** of promising live states  

The system:
- polls live match data  
- logs minute states  
- computes pressure and momentum features  
- runs ML models  
- sends betting-oriented alerts  

---

## What this repository contains

### ? Included

- Source code  
- Configuration templates  
- Dependency definitions  
- Documentation  
- Lightweight sample data  

### ? Excluded

- API tokens / credentials  
- Large raw logs  
- Runtime state files  
- Trained model binaries  
- Generated outputs  

---

## Repository Structure

```text
live-soccer-stats/
??? .gitignore
??? .env.example
??? LICENSE
??? README.md
??? requirements.txt
??? config/
?   ??? settings.example.yaml
??? data/
?   ??? raw/
?   ?   ??? .gitkeep
?   ??? processed/
?   ?   ??? .gitkeep
?   ??? models/
?   ?   ??? .gitkeep
?   ??? reports/
?       ??? .gitkeep
??? docs/
?   ??? architecture.md
?   ??? data_dictionary.md
?   ??? experiments.md
??? notebooks/
?   ??? .gitkeep
??? scripts/
?   ??? train_goal_outcome.py
?   ??? train_goals_regression.py
?   ??? run_live_bot.py
?   ??? scan_recent_states.py
??? src/
?   ??? live_soccer/
?       ??? __init__.py
?       ??? bot.py
?       ??? config.py
?       ??? logging_utils.py
?       ??? models/
?       ?   ??? __init__.py
?       ?   ??? goal_outcome_trainer.py
?       ?   ??? goals_regression_trainer.py
?       ??? scanners/
?       ?   ??? __init__.py
?       ?   ??? fast_goal_signal_scan.py
?       ??? utils/
?           ??? __init__.py
?           ??? features.py
??? tests/
    ??? test_smoke.py