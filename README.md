# Predictive Waste Forecasting for Textile Dyeing Operations

## Abstract

Daily industrial textile dyeing produces solid, yarn, chemical sludge, and liquid effluent waste streams driven by nonlinear interactions between batch scheduling, bath chemistry, and shift utilization. This repository implements a two-stage decision pipeline coupling per-target gradient-boosted regression (`PerTargetXGB` over 67 causal features) with a Mixed-Integer Linear Program (MILP) that optimizes multi-center waste routing and in-factory Effluent Treatment Plant (ETP) water recovery. On an unseen 2024 chronological hold-out set (359 days), the pipeline achieves R² = 0.8026 (MAPE = 9.15%) on wastewater and R² = 0.7912 (MAPE = 9.29%) on chemical sludge while solving exact daily routing plans in 9.56 ms.

---

## 1. System Architecture

The system separates **waste forecasting** from **reverse logistics allocation**:

```
Raw Logs (731 × 39)
   │  Drop 12 target/leakage/metadata cols; keep 27 process inputs
   ▼
Causal Feature Engineering (`model/features.py`) ──► (717, 67)
   │  Train: 2023 (358, 67) | Hold-out Test: 2024 (359, 67)
   ▼
PerTargetXGB (`model/train.py`)
   │  4 independent XGBRegressor heads (Solid, Yarn, Chemical, Wastewater)
   ▼  Predicted Streams (B, 4)
   ├──► In-Factory ETP Loop: W_ETP = max(0, Wastewater - 0.8 × Chemical_Sludge)
   │    75% reused in dyeing (+0.002 BDT/L), 25% drained (-0.0005 BDT/L)
   │
   └──► MILP Router (`optimizer/milp.py` — CPLEX / SciPy HiGHS fallback)
        Routes Solid, Yarn, Chemical across 3 Collection Centers (CC_1..CC_3)
        Outputs optimal hub activation y[j] and mass flows x[w,j]
```

### Core Formulation (Abstracted)

- **Stage 1 — Per-Target Gradient Boosting (`PerTargetXGB`):** Fits four independent L2-regularized `XGBRegressor` ensembles—one per waste stream—so noisy mechanical streams (`Solid_Waste_kg`) and tight chemical/hydraulic streams (`Chemical_Waste_kg`, `Wastewater_L`) receive distinct tree depths and regularization without cross-target compromise. Hyperparameters are selected strictly on 2023 walk-forward `TimeSeriesSplit(gap=7)` optimizing negative MAE.
- **Stage 2 — Reverse Logistics MILP:** Given predicted daily volumes `D_w` and net ETP feed `W_ETP`, the MILP selects active Collection Centers `y_j ∈ {0, 1}` and continuous shipments `x_(w,j) ≥ 0` to maximize net daily value `Z` (BDT):

  `Z = Recycling Revenue (solid, yarn) + Circularity Bonus + ETP Reuse Value - Chemical Disposal Fees - Transport - Holding - Working Capital - CO₂ Penalty - Fixed Hub Contract Costs`

  subject to full daily waste clearance (`∑_j x_(w,j) = D_w`) and capacity-activation bounds (`x_(w,j) ≤ CAP_(w,j) × y_j`).

---

## 2. Dataset & Features

- **Source:** `Textile_Dyeing_Dataset_Causal.csv` (`731` daily records × `39` columns, `2023-01-01` to `2024-12-31`, `0` missing values).
- **Chronological Split:** Trained on **2023** (`358` rows) and evaluated on **2024** (`359` rows) after dropping the first 7 days of each year for lag-7 initialization. Lag/rolling features are grouped by calendar year with `.shift(1)` to prevent look-ahead leakage.
- **Target Distribution:** All four targets exhibit skewness within `[-0.5, +0.5]` (`0.250` to `0.474` on training data), so models train in raw physical units (kg / L) without log transformation or feature scaling.

| Feature Group | Count | Abstracted Inputs |
|---|---:|---|
| **Production & Labor** | 5 | `Monthly_Demand_Factor`, `Production_Volume_{kg,ton}`, `Machine_Utilization_%`, `Number_of_Workers` |
| **Utilities & Effluent** | 7 | `Total_Energy_kWh`, `Energy_kWh_per_kg`, `Total_Water_Consumption_L`, `Water_L_per_kg`, `BOD`, `COD`, `pH` |
| **Dye-Bath & Chemistry** | 15 | `Dye_Fixation_Ratio_%`, `Liquor_Ratio_1x`, `Temperature_C`, `Cycle_Time_hr`, `Dye_Batch_Count`, `{Dye, Salt, Alkali, Acid, Auxiliary}_{kg, g_per_kg}` |
| **Cyclical Calendar** | 5 | `sin/cos` encodings of `Month` and `Day_of_Week`, plus `Quarter` |
| **Domain Interactions & Ratios** | 10 | `Chem_inputs_total`, `Fixation_loss_kg`, `Prod × {Batch, Acid, Salt, Alkali, Aux}`, `Chem_per_litre`, `{Energy, Prod}_per_worker` |
| **Causal Lag & Rolling** | 25 | Per-year `lag1`, `lag7`, `roll7`, `roll30`, `std7` across the 4 targets and `Production_Volume_kg` |
| **Total** | **67** | **27 raw process features + 40 engineered features** (12 leakage/metadata cols excluded) |

---

## 3. Empirical Results & Footprint

### 3.1 2024 Chronological Hold-Out Benchmarks (`n = 359`, 67 Features)

| Target Stream | Model | MAE | RMSE | R² | MAPE |
|---|---|---:|---:|---:|---:|
| **`Solid_Waste_kg`** | Baseline XGBoost | 249.85 | 320.08 | 0.5988 | 12.18% |
| | Tuned Random Forest | **237.45** | **305.79** | **0.6339** | **11.58%** |
| | Tuned `PerTargetXGB` | 252.73 | 323.46 | 0.5903 | 12.25% |
| **`Yarn_Waste_kg`** | Baseline XGBoost | 99.75 | 124.05 | 0.6789 | 11.53% |
| | Tuned Random Forest | **99.07** | **121.16** | **0.6937** | **11.40%** |
| | Tuned `PerTargetXGB` | 99.22 | 122.68 | 0.6859 | **11.40%** |
| **`Chemical_Waste_kg`** | Baseline XGBoost | 318.85 | 407.16 | 0.7714 | 9.50% |
| | Tuned Random Forest | 324.54 | 411.37 | 0.7667 | 9.69% |
| | Tuned `PerTargetXGB` | **312.27** | **389.10** | **0.7912** | **9.29%** |
| **`Wastewater_L`** | Baseline XGBoost | 48,350.88 | 61,269.89 | 0.8165 | 8.80% |
| | Tuned Random Forest | **46,119.43** | **57,983.84** | **0.8356** | **8.34%** |
| | Tuned `PerTargetXGB` | 50,574.96 | 63,548.70 | 0.8026 | 9.15% |

- **Primary Drivers (XGBoost Split Gain):** `Solid_Waste_kg` and `Yarn_Waste_kg` are governed by production scale (`Production_Volume_{ton,kg}`: 35.0% and 49.1% combined); `Chemical_Waste_kg` is driven by engineered chemistry (`Chem_inputs_total`: 29.2%, `Prod_x_Salt`: 27.3%); `Wastewater_L` is governed by `Total_Water_Consumption_L` (44.7%) and `Production_Volume_ton` (22.2%).
- **Reverse Logistics Impact (2024):** `207.02M L` recoverable ETP water out of `208.03M L` predicted effluent (`99.5%` post-sludge recovery); reference shift MILP (`1665.87 / 722.36 / 3202.32 kg`) dispatches optimally to `CC_2` and `CC_3` at `-31,057.36 BDT` net cost.

### 3.2 Compute & Latency Profile

- **Execution Environment:** CPU-only (8-core x86-64, Windows 11, Python `3.13.15`, `xgboost 3.4.1`, `scikit-learn 1.9.1`, `scipy 1.18.1`); FP64/FP32 precision, no GPU/CUDA required.
- **Artifact Size:** `3.41 MB` (`output_figures_xgb/xgboost_waste_model_improved.pkl`, 4 × 700 trees), `1.12 KB` (`model/artifacts/feature_cols.json`).
- **Measured Latency:** **36.51 ms** (`B = 1`) and **41.25 ms** (`B = 32`) for 4-stream inference; **9.56 ms** per MILP solve (`scipy.optimize.milp`).

---

## 4. Setup & Usage

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1  |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
# Optional IBM CPLEX solver (falls back to scipy.optimize.milp automatically): pip install docplex cplex
```

```bash
# 1. Verify feature pipeline (outputs: (717, 79) 67)
python -c "import pandas as pd; from model.features import engineer; df = pd.read_csv('Textile_Dyeing_Dataset_Causal.csv', parse_dates=['Date']); X, cols = engineer(df); print(X.shape, len(cols))"

# 2. Train PerTargetXGB with walk-forward GridSearchCV
python -m model.train --data Textile_Dyeing_Dataset_Causal.xlsx

# 3. Inspect 2024 hold-out metrics & run standalone MILP optimizer
python -c "import pandas as pd; print(pd.read_csv('output_figures_xgb/model_metrics_xgb.csv').to_string(index=False))"
python -c "from optimizer.milp import optimize; import json; print(json.dumps(optimize({'solid': 1665.87, 'yarn': 722.36, 'chemical': 3202.32}, 449024.70), indent=2))"

# 4. Launch interactive CLI optimizer, REST API, or open static frontend/index.html directly
python reverse_logistics_cplex.py
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---
## 5. Citation

```bibtex
@misc{valmiki2026textilewaste,
  author       = {Karthik Valmiki},
  title        = {Predictive Multi-Stream Waste Forecasting and MILP Reverse Logistics for Textile Dyeing Operations},
  year         = {2026},
  howpublished = {\url{https://github.com/Karthik-Valmiki/Smart_Mega_Factories}},
  note         = {Version 1.0.0}
}
```
