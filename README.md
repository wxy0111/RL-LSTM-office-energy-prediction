# RL-LSTM Energy Prediction for Office Buildings

An energy consumption prediction system integrating **Bidirectional LSTM** and **Deep Deterministic Policy Gradient (DDPG)** reinforcement learning, designed for non-stationary building energy data.

This work is associated with the IEEE conference paper:
> *RL-LSTM and Heuristic Optimization for Energy-Efficient Office Management*  
> ISPDS 2025 (IEEE) | DOI: [10.1109/ISPDS67367.2025.11391185](https://doi.org/10.1109/ISPDS67367.2025.11391185)

---

## Overview

Traditional LSTM models struggle with the non-stationary nature of office energy data caused by variable occupant behavior. This system addresses that by using a **DDPG reinforcement learning agent** to dynamically adjust the LSTM model's learning rate at inference time — effectively allowing the model to adapt to real-time prediction errors.

**Key results:**
- CVRMSE improved by **23.3%** (0.30 → 0.23) compared to baseline LSTM
- MAPE improved from **18.36% → 13.73%**
- Electricity consumption reduced by **12.87%** in real-world office deployment

---

## System Architecture

```
Raw Sensor Data (CSV)
        │
        ▼
┌─────────────────┐
│  DataProcessor  │  ← Outlier detection, PMV calculation,
│                 │    time feature engineering, RobustScaler
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
LSTM Train  RL Train   (60% / 20% / 20% split)
    │         │
    ▼         ▼
┌──────────────────────┐
│   LSTMRLSystem       │
│  ┌────────────────┐  │
│  │  BiLSTM Model  │  │  ← Bidirectional LSTM + BatchNorm + Dropout
│  └────────────────┘  │
│  ┌────────────────┐  │
│  │   DDPG Agent   │  │  ← Actor-Critic, Replay Buffer, soft target update
│  └────────────────┘  │
└──────────────────────┘
         │
         ▼
  Prediction + Evaluation
  (RMSE, CVRMSE, MAPE)
```

---

## Results

### Training Convergence

![Training Curves](training_curves.png)

The Bidirectional LSTM base model converges stably within 20 epochs.  
Final training loss: **0.0200** | Validation loss: **0.0291**  
Final training MAE: **0.1680** | Validation MAE: **0.2112**

---

### Prediction Performance

![Prediction Results](prediction_results.png)

Panel (a) shows full test period predictions (Nov 13–23, 2024).  
Panel (b) shows a 9-hour detailed view on Nov 19, demonstrating that **RL-LSTM tracks actual values more closely** than the baseline LSTM, particularly during peak usage periods.

---

### Quantitative Evaluation

| Metric | LSTM (Baseline) | RL-LSTM (Proposed) | Improvement |
|---|---|---|---|
| CVRMSE | 0.30 | **0.23** | ↓ 23.3% |
| MAPE | 18.36% | **13.73%** | ↓ 4.63 pp |

The DDPG agent dynamically adjusts the LSTM model's learning factor at each inference step based on the current prediction error, enabling continuous adaptation to behavioral drift without full retraining.

---

## Features

- **Bidirectional LSTM** with BatchNorm and Dropout for robust base prediction
- **DDPG reinforcement learning** agent that generates dynamic learning factors to update LSTM weights online
- **PMV (Predicted Mean Vote)** thermal comfort index calculation integrated as input feature
- **Japan holiday-aware** time feature engineering (`jpholiday`)
- **RobustScaler** for outlier-resilient feature normalization
- **IQR-based outlier detection** for energy data cleaning

---

## Dataset

Real-world sensor data collected from a small-to-medium office in Japan (2024).

| Feature | Description |
|---|---|
| `indoor_temperature` | Indoor air temperature (°C) |
| `indoor_humidity` | Indoor relative humidity (%) |
| `indoor_globe_temperature` | Globe temperature for MRT calculation (°C) |
| `indoor_co2` | CO₂ concentration (ppm) |
| `indoor_lux` | Illuminance (lux) |
| `outdoor_temperature` | Outdoor air temperature (°C) |
| `outdoor_relativehumidity` | Outdoor relative humidity (%) |
| `total_electric[Wh]` | **Target**: Total electricity consumption (Wh) |

Sampling interval: 1 minute → resampled to 10 minutes  
Total records: ~73,000 data points

---

## Requirements

```bash
pip install torch pandas numpy scikit-learn matplotlib pythermalcomfort jpholiday
```

Python 3.9+ recommended.

---

## Usage

```bash
# Place merged_data.csv in the project root
python main.py
```

The script will:
1. Load and preprocess sensor data
2. Calculate PMV thermal comfort index
3. Engineer time features (cyclic encoding, Japanese holidays)
4. Train Bidirectional LSTM base model (20 epochs)
5. Train DDPG RL agent (100 episodes)
6. Evaluate and compare LSTM vs RL-LSTM predictions
7. Save models to `models/` directory
8. Export IEEE-quality figures (PNG + EPS)

---

## File Structure

```
├── main.py                      # Main training and evaluation pipeline
├── lstm_rl_model.py             # Model definitions (LSTM, DDPG, DataProcessor)
├── merged_data.csv              # Sensor dataset
├── training_curves.png          # LSTM training loss & MAE curves
├── prediction_results.png       # LSTM vs RL-LSTM prediction comparison
├── models/                      # Saved model weights (generated on run)
│   ├── rl_lstm_model.pth
│   ├── rl_actor.pth
│   ├── rl_critic.pth
│   └── best_lstm_params.pkl
└── README.md
```

---

## Citation

```bibtex
@inproceedings{wang2025rl,
  title={RL-LSTM and Heuristic Optimization for Energy-Efficient Office Management},
  author={Wang, Xiangyu and Chen, Yutong and Ishibashi, Soichiro and Oh, Jewon and Ueno, Takahiro and Sumiyoshi, Daisuke},
  booktitle={Proceedings of the 6th International Conference on Information Science, Parallel and Distributed Systems (ISPDS 2025)},
  year={2025},
  doi={10.1109/ISPDS67367.2025.11391185}
}
```

---

## Author

**Wang Xiangyu (王 翔宇)**  
Doctoral Student, Graduate School of Human-Environment Studies  
Kyushu University, Japan
