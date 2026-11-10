# Predictive Multi-Stream Waste Forecasting Textile Dyeing Operations

## Abstract

Daily industrial textile dyeing generates heterogeneous solid, yarn, chemical sludge, and liquid effluent waste streams whose nonlinear dependence on batch scheduling, dye-bath chemistry, and shift utilization undermines reactive disposal planning. We formulate a leakage-free, two-stage decision pipeline coupling per-target gradient-boosted regression (`PerTargetXGB`) over 67 domain-engineered temporal features with a Mixed-Integer Linear Program (MILP) that jointly optimizes multi-center waste routing, carbon penalties, working capital cost, and in-factory Effluent Treatment Plant (ETP) water recovery. Evaluated on an unseen 359-day chronological hold-out set (2024), the framework achieves $R^2 = 0.8026$ ($\text{MAPE} = 9.15\%$) on effluent wastewater and $R^2 = 0.7912$ ($\text{MAPE} = 9.29\%$) on chemical sludge—outperforming a tuned Random Forest baseline on hazardous chemical residue—while solving exact global reverse-logistics allocations in $9.56\text{ ms}$.

---

## 1. Core Architecture & Theoretical Formulation

### 1.1 Stage 1: Leakage-Free Temporal Feature Operator and Per-Target XGBoost

Let $\mathcal{D} = \{(\mathbf{u}_t, \mathbf{y}_t)\}_{t=1}^{T}$ denote the daily operational log over $T = 731$ consecutive days (`2023-01-01` to `2024-12-31`), where $\mathbf{u}_t \in \mathbb{R}^{27}$ represents the raw shift, utility, dye-bath, and chemical-recipe inputs, and $\mathbf{y}_t = [y_t^{(1)}, y_t^{(2)}, y_t^{(3)}, y_t^{(4)}]^\top \in \mathbb{R}_{>0}^4$ represents the four target waste streams:
- $y_t^{(1)}$: `Solid_Waste_kg` (fabric cuttings, cone remnants, packaging)
- $y_t^{(2)}$: `Yarn_Waste_kg` (defective filament and winding loss)
- $y_t^{(3)}$: `Chemical_Waste_kg` (spent dye-bath chemical sludge cake)
- $y_t^{(4)}$: `Wastewater_L` (raw liquid effluent volume)

To model temporal inertia and nonlinear chemical stoichiometry without future look-ahead, we define a causal transformation $\mathbf{x}_t = \Phi(\mathbf{u}_t, \mathcal{H}_{t-1}) \in \mathbb{R}^{67}$ using strictly past observations $\mathcal{H}_{t-1} = \{(\mathbf{u}_\tau, \mathbf{y}_\tau)\}_{\tau < t}$ restricted to the same calendar year:

1. **Cyclical Calendar Encoding (5 features):**
   $$\text{Month\_Sin}_t = \sin\!\left(\frac{2\pi m_t}{12}\right),\quad \text{Month\_Cos}_t = \cos\!\left(\frac{2\pi m_t}{12}\right),\quad \text{Day\_Sin}_t = \sin\!\left(\frac{2\pi d_t}{7}\right),\quad \text{Day\_Cos}_t = \cos\!\left(\frac{2\pi d_t}{7}\right),\quad \text{Quarter}_t \in \{1,2,3,4\}$$
2. **Stoichiometric & Operational Interactions (6 features):**
   $$\text{Chem\_inputs\_total}_t = \text{Acid}_t + \text{Salt}_t + \text{Alkali}_t + \text{Auxiliary}_t + \text{Dye\_Used}_t$$
   together with multiplicative cross-terms $P_t \cdot z_t$ for $P_t = \text{Production\_Volume\_kg}_t$ and $z_t \in \{\text{Dye\_Batch\_Count}_t, \text{Acid}_t, \text{Salt}_t, \text{Alkali}_t, \text{Auxiliary}_t\}$.
3. **Physical Process & Intensity Ratios (4 features):**
   $$\text{Fixation\_loss\_kg}_t = \left(1 - \frac{\text{Dye\_Fixation\_Ratio}_t}{100}\right) \cdot \text{Dye\_Used}_t, \qquad \text{Chem\_per\_litre}_t = \frac{\text{Chem\_inputs\_total}_t}{\text{Total\_Water\_Consumption\_L}_t + 1}$$
   $$\text{Energy\_per\_worker}_t = \frac{\text{Total\_Energy\_kWh}_t}{\text{Number\_of\_Workers}_t + 1}, \qquad \text{Prod\_per\_worker}_t = \frac{P_t}{\text{Number\_of\_Workers}_t + 1}$$
4. **Causal Within-Year Lag & Rolling Statistics (25 features):** For each series $s_t \in \{y_t^{(1)}, y_t^{(2)}, y_t^{(3)}, y_t^{(4)}, P_t\}$, we compute five strictly backward-looking statistics using `.shift(1)` and `.shift(7)`:
   $$s_{t,\text{lag1}} = s_{t-1}, \quad s_{t,\text{lag7}} = s_{t-7}, \quad s_{t,\text{roll7}} = \frac{1}{7}\sum_{i=1}^{7} s_{t-i}, \quad s_{t,\text{roll30}} = \frac{1}{30}\sum_{i=1}^{30} s_{t-i}, \quad s_{t,\text{std7}} = \sqrt{\frac{1}{6}\sum_{i=1}^{7} (s_{t-i} - s_{t,\text{roll7}})^2}$$

Because solid mechanical waste, spinning loss, chemical precipitation, and hydraulic discharge follow distinct physical regimes, `PerTargetXGB` trains $K = 4$ independent gradient-boosted ensembles $\hat{y}_t^{(k)} = f_k(\mathbf{x}_t) = \sum_{m=1}^{M_k} \eta_k \, g_{k,m}(\mathbf{x}_t)$ where $g_{k,m} \in \mathcal{F}$ are CART regression trees. Each estimator minimizes the L2-regularized additive objective:

$$\mathcal{L}^{(k)} = \sum_{t \in \mathcal{T}_{\text{train}}} \frac{1}{2}\left(y_t^{(k)} - \hat{y}_t^{(k)}\right)^2 + \sum_{m=1}^{M_k} \left[\gamma T_{k,m} + \frac{1}{2}\lambda_k \sum_{l=1}^{T_{k,m}} w_{k,m,l}^2\right]$$

where $T_{k,m}$ is the leaf count of tree $m$, $w_{k,m,l}$ are leaf weights, $\lambda_k$ is the L2 leaf regularization penalty, and hyperparameters $(\text{max\_depth}_k, \eta_k, M_k, \text{min\_child\_weight}_k, \text{subsample}_k, \text{colsample\_bytree}_k, \lambda_k)$ are selected per target under walk-forward cross-validation using negative Mean Absolute Error (`neg_mean_absolute_error`) to prevent extreme single-day spikes from dominating model selection.

---

### 1.2 Stage 2: Reverse Logistics MILP & In-Factory ETP Coupling

Given predicted daily quantities $D_w = \hat{y}_t^{(w)}$ for $w \in \mathcal{W} = \{\text{solid}, \text{yarn}, \text{chemical}\}$ and predicted raw effluent $\hat{y}_t^{(4)}$, net ETP-treatable water is first reduced by moisture retained in chemical sludge cake:

$$W_{\text{ETP}} = \max\!\left(0,\; \hat{y}_t^{(4)} - 0.8 \cdot \hat{y}_t^{(3)}\right)$$

Within the factory ETP, a fixed recycling fraction $\rho = 0.75$ is reclaimed for dye-bath reuse (credit $v_{\text{reuse}} = 0.002\text{ BDT/L}$) and $(1 - \rho) = 0.25$ is discharged within municipal limits (cost $c_{\text{drain}} = 0.0005\text{ BDT/L}$), yielding net ETP economic value:

$$\text{ETP}_{\text{net}} = \rho\, W_{\text{ETP}}\, v_{\text{reuse}} - (1 - \rho)\, W_{\text{ETP}}\, c_{\text{drain}}$$

For the three external streams $w \in \mathcal{W}$ routed across candidate Collection Centers $j \in \mathcal{J} = \{1, 2, 3\}$, let $x_{w,j} \in \mathbb{R}_{\ge 0}$ denote the allocated mass (kg) and $y_j \in \{0, 1\}$ denote binary contract activation. The MILP maximizes daily net economic objective $Z$ (BDT):

$$\begin{aligned}
\max_{x, y} \quad Z &= \sum_{w \in \{\text{solid}, \text{yarn}\}} \sum_{j \in \mathcal{J}} R_{w,j}\, x_{w,j}
- \sum_{j \in \mathcal{J}} F_{\text{chem},j}\, x_{\text{chem},j}
- \sum_{w \in \mathcal{W}} \sum_{j \in \mathcal{J}} \left(c^{\text{tr}}_w d_j + h_w + r\, T_{w,j}\, R^{\text{eff}}_{w,j} + P_{\text{CO}_2}\, e_{w,j} - B_{\text{circ}}\, \kappa_{w,j}\right) x_{w,j} \\
&\quad - \sum_{j \in \mathcal{J}} \text{FC}_j\, y_j + \text{ETP}_{\text{net}}
\end{aligned}$$

subject to full waste clearance and capacity-activation coupling constraints:

$$\sum_{j \in \mathcal{J}} x_{w,j} = D_w \quad \forall w \in \mathcal{W}, \qquad\qquad 0 \le x_{w,j} \le \text{CAP}_{w,j}\, y_j \quad \forall w \in \mathcal{W},\; \forall j \in \mathcal{J}, \qquad\qquad y_j \in \{0, 1\} \quad \forall j \in \mathcal{J}$$

where $R_{w,j}$ is recycling revenue per kg, $F_{\text{chem},j}$ is chemical sludge disposal fee, $c^{\text{tr}}_w d_j$ is distance-weighted haulage, $h_w$ is factory holding cost, $r = 0.0003/\text{day}$ is daily working capital cost over processing lead time $T_{w,j}$ ($R^{\text{eff}}_{w,j} = R_{w,j}$ for solid/yarn, $F_{\text{chem},j}$ for chemical), $P_{\text{CO}_2} = 1.5\text{ BDT/kg CO}_2$ is carbon penalty on emission factor $e_{w,j}$, $B_{\text{circ}} = 2.0\text{ BDT/kg}$ is the circularity incentive on recovery rate $\kappa_{w,j}$, and $\text{FC}_j$ is fixed contract activation cost.

---

### 1.3 System Dataflow & Tensor Shape Diagram

```
Raw Daily Log (CSV / XLSX)
  Shape: (731, 39)  [2023: 365 days | 2024: 366 days]
         │
         ├─► Drop 12 leakage/metadata columns (4 targets + 4 target ratios + Date, Day_Type, Year, Month)
         │   Retain 27 raw process variables u_t: Shape (B, 27)
         │
         ▼
Causal Feature Engineering (`model/features.py: engineer()`)
  ├─► Cyclical Calendar (Sin/Cos Month, Day, Quarter)       : (B, 5)
  ├─► Interactions & Chem_inputs_total                     : (B, 6)
  ├─► Physical Fixation Loss & Intensity Ratios            : (B, 4)
  └─► Within-Year Shift(1)/Shift(7) Lags, Roll-7/30, Std-7 : (B, 25)
         │
         │   Drop initial 7 rows/year (NaN from lag-7)
         ▼
  Design Matrix X: Shape (717, 67)
         ├──► Train Split (2023): X_train (358, 67), Y_train (358, 4)
         └──► Hold-out Test (2024): X_test (359, 67), Y_test (359, 4)
                     │
                     ▼
        PerTargetXGB (`model/train.py`)
         ├──► Estimator 0 (Solid_Waste_kg, depth=2, M=700)     ──► (B, 1)
         ├──► Estimator 1 (Yarn_Waste_kg, depth=4, M=700)      ──► (B, 1)
         ├──► Estimator 2 (Chemical_Waste_kg, depth=3, M=700)  ──► (B, 1)
         └──► Estimator 3 (Wastewater_L, depth=4, M=700)       ──► (B, 1)
                     │
                     ▼  np.column_stack
        Predicted Waste Tensor Y_hat: Shape (B, 4)
                     │
         ┌───────────┴─────────────────────────────────────────┐
         ▼                                                     ▼
ETP Water Balance                                 MILP Solver (`optimizer/milp.py`)
  W_ETP = max(0, Y_hat[:,3] - 0.8 * Y_hat[:,2])     Inputs: D = Y_hat[0, :3] (3,)
  Recycled (75%) | Drained (25%)                    Vars:   x[w,j] (3, 3) continuous
                                                            y[j]   (3,)   binary
                                                    Output: Optimal Z (BDT), Active CCs
```

---

## 2. Dataset & Preprocessing Pipeline

### 2.1 Artifact Inventory & Checksums

| Artifact | Path | Records / Size | SHA-256 Prefix (`[:16]`) |
|---|---|---|---|
| Raw Dataset (CSV) | `Textile_Dyeing_Dataset_Causal.csv` | 731 rows × 39 cols (`174,690 B`) | `0d458e677ac98d45` |
| Raw Dataset (Excel) | `Textile_Dyeing_Dataset_Causal.xlsx` | 731 rows × 39 cols (`166,336 B`) | `1095853b6228535f` |
| Serialized Model (`PerTargetXGB`) | `output_figures_xgb/xgboost_waste_model_improved.pkl` | 4 × 700 trees (`3,406,388 B`) | `f80980be1706c990` |
| Ordered Feature Schema | `model/artifacts/feature_cols.json` | 67 ordered feature keys (`1,122 B`) | `88c5af0a3e30b199` |
| Test Metrics Summary | `output_figures_xgb/model_metrics_xgb.csv` | 4 targets × 5 metrics (`258 B`) | `07d8bbcc671971c6` |

### 2.2 Temporal Splitting, Regime Distribution & Skewness Verification

- **Missing Values:** 0 null entries across all 731 × 39 cells.
- **Chronological Partitioning (No Random Shuffling):**
  - **Training Set (2023):** 365 calendar days $\rightarrow$ **358 rows** after dropping `2023-01-01`..`2023-01-07` for lag-7 initialization.
  - **Hold-Out Test Set (2024):** 366 calendar days (leap year) $\rightarrow$ **359 rows** after dropping `2024-01-01`..`2024-01-07`. Grouping by `Date.dt.year` before `_add_lag_features` ensures zero cross-year lag leakage at the `2023-12-31` $\rightarrow$ `2024-01-01` boundary.
- **Operational Shift Distribution (`Day_Type`, metadata-only, excluded from `X`):** `normal`: 426 days (58.3%), `good`: 154 days (21.1%), `bad`: 151 days (20.7%).
- **Target Skewness Verification:** Sample skewness $\gamma_1$ was evaluated on both the full 731-day series and the 358-row training split under the $|\gamma_1| \le 0.5$ normality criterion:
  - `Solid_Waste_kg`: $0.3805$ (full) / $0.461$ (train)
  - `Yarn_Waste_kg`: $0.3918$ (full) / $0.474$ (train)
  - `Chemical_Waste_kg`: $0.3096$ (full) / $0.350$ (train)
  - `Wastewater_L`: $0.3505$ (full) / $0.250$ (train)
  Because all four targets fall within $[-0.5, +0.5]$, log transformation (`log1p`) was explicitly omitted to avoid Jensen's inequality back-transformation bias during exponentiation.
- **Scaling Policy:** No linear feature standardization (`StandardScaler` / `MinMaxScaler`) is applied, as CART decision-tree split boundaries are invariant to monotonic feature scaling and retain direct interpretability in physical engineering units (kg, L, °C, mg/L).

### 2.3 Input Feature Space Breakdown (27 Raw + 40 Engineered = 67 Features)

| Feature Group | Count | Included Columns |
|---|---|---|
| **Demand & Output Scale** | 3 | `Monthly_Demand_Factor`, `Production_Volume_ton`, `Production_Volume_kg` |
| **Machine & Workforce** | 2 | `Machine_Utilization_%`, `Number_of_Workers` |
| **Energy & Water Utilities** | 4 | `Energy_Consumption_kWh_per_kg`, `Total_Energy_kWh`, `Water_Consumption_L_per_kg`, `Total_Water_Consumption_L` |
| **Dye-Bath Process** | 5 | `Dye_Fixation_Ratio_%`, `Liquor_Ratio_1x`, `Temperature_C`, `Cycle_Time_hr`, `Dye_Batch_Count` |
| **Chemical Recipe Dosing** | 10 | `{Dye_Used, Salt, Alkali, Acid, Auxiliary}_{kg, g_per_kg}` |
| **Effluent Quality** | 3 | `BOD_mg_per_L`, `COD_mg_per_L`, `pH` |
| **Engineered Temporal & Physical** | 40 | 5 cyclical calendar + 6 multiplicative/total chemistry + 4 physical/intensity ratios + 25 causal lag/rolling features |

---

## 3. Hardware & Compute Footprint

All training, cross-validation, TreeSHAP attribution, and MILP optimization stages execute on commodity CPU hardware without GPU acceleration:

| Compute Dimension | Verified Specification / Measurement |
|---|---|
| **Reference Host** | 8-core x86-64 CPU, Windows 11 (`10.0.26300`), Python `3.13.15` |
| **Accelerator / CUDA** | None required (`n_jobs=-1` multi-threaded CPU execution; no CUDA/cuDNN dependency) |
| **Numerical Precision** | IEEE-754 **FP64** (`pandas`/`numpy`/`scipy.optimize.milp` HiGHS) and **FP32** internal XGBoost histogram/split evaluations (`float32` predictions serialized from `XGBRegressor`); AMP (FP16/BF16) is not applicable to tabular gradient-boosted trees |
| **Hyperparameter Search Budget** | Per-target `GridSearchCV` over $3 \times 3 \times 2 \times 3 \times 2 \times 2 \times 2 = 432$ configurations $\times 3$ walk-forward folds = $1,296$ fits per target ($5,184$ total `XGBRegressor` fits across 4 targets) |
| **Serialized Footprint** | `xgboost_waste_model_improved.pkl`: `3.41 MB` (2,800 trees total); `feature_cols.json`: `1.12 KB` |
| **Measured Inference Latency (`PerTargetXGB`)** | **$36.51\text{ ms}$** at batch size $B = 1$; **$41.25\text{ ms}$** at batch size $B = 32$ ($1.29\text{ ms/sample}$ amortized) |
| **Measured MILP Latency (`scipy.optimize.milp`)** | **$9.56\text{ ms}$** per daily allocation (9 continuous + 3 binary variables, 7 linear constraints) |

---

## 4. Empirical Evaluation & Baselines

### 4.1 2024 Chronological Hold-Out Comparison ($n = 359$ Days, 67 Features)

We compare three models trained on identical 2023 chronological data (`358` rows) and evaluated on identical 2024 hold-out records (`359` rows):
1. **Baseline XGBoost:** Default `XGBRegressor` hyperparameters without walk-forward tuning.
2. **Tuned Random Forest:** Per-target tuned `RandomForestRegressor` on the exact same 67-feature temporal split (logged in `model_XGBoost.ipynb`, Cell 31).
3. **Tuned `PerTargetXGB` (Proposed):** Per-target `XGBRegressor` tuned via 3-fold walk-forward `TimeSeriesSplit(gap=7)` optimizing `neg_mean_absolute_error`.

| Target Stream | Model | MAE | RMSE | $R^2$ | MAPE (%) |
|---|---|---:|---:|---:|---:|
| **`Solid_Waste_kg`** | Baseline XGBoost | 249.850 | 320.080 | 0.5988 | 12.18% |
| **`Solid_Waste_kg`** | Tuned Random Forest | **237.447** | **305.791** | **0.6339** | **11.58%** |
| **`Solid_Waste_kg`** | Tuned `PerTargetXGB` | 252.734 | 323.462 | 0.5903 | 12.25% |
| **`Yarn_Waste_kg`** | Baseline XGBoost | 99.750 | 124.050 | 0.6789 | 11.53% |
| **`Yarn_Waste_kg`** | Tuned Random Forest | **99.068** | **121.158** | **0.6937** | **11.40%** |
| **`Yarn_Waste_kg`** | Tuned `PerTargetXGB` | 99.215 | 122.681 | 0.6859 | **11.40%** |
| **`Chemical_Waste_kg`** | Baseline XGBoost | 318.850 | 407.160 | 0.7714 | 9.50% |
| **`Chemical_Waste_kg`** | Tuned Random Forest | 324.535 | 411.373 | 0.7667 | 9.69% |
| **`Chemical_Waste_kg`** | Tuned `PerTargetXGB` | **312.274** | **389.104** | **0.7912** | **9.29%** |
| **`Wastewater_L`** | Baseline XGBoost | 48,350.880 | 61,269.890 | 0.8165 | 8.80% |
| **`Wastewater_L`** | Tuned Random Forest | **46,119.431** | **57,983.836** | **0.8356** | **8.34%** |
| **`Wastewater_L`** | Tuned `PerTargetXGB` | 50,574.959 | 63,548.702 | 0.8026 | 9.15% |

**Empirical Observations:**
- On **`Chemical_Waste_kg`** (the sole hazardous stream incurring disposal fees rather than recycling revenue), Tuned `PerTargetXGB` improves over both Baseline XGBoost ($+0.0198\; R^2$, $-6.58\text{ kg}$ MAE) and Tuned Random Forest ($+0.0245\; R^2$, $-12.26\text{ kg}$ MAE, $-22.27\text{ kg}$ RMSE).
- On mechanical and hydraulic streams (`Solid_Waste_kg`, `Yarn_Waste_kg`, `Wastewater_L`), variance-reducing bagging (`RandomForestRegressor`) retains a modest margin under $n = 358$ training rows, whereas `PerTargetXGB` enforces stricter regularization (`max_depth=2, reg_lambda=5.0` on Solid Waste) selected strictly on 2023 CV without peeking at 2024 hold-out metrics.
- **5-Fold Walk-Forward CV Stability (`TimeSeriesSplit(n_splits=5, gap=7)` within 2023):** `Solid_Waste_kg`: $R^2 = 0.3938 \pm 0.1678$; `Yarn_Waste_kg`: $R^2 = 0.3571 \pm 0.2901$; `Chemical_Waste_kg`: $R^2 = 0.5194 \pm 0.3175$; `Wastewater_L`: $R^2 = 0.5512 \pm 0.2310$ (lower early-fold means reflect small initial 2–3 month training windows).

### 4.2 Per-Target Optimal Hyperparameters & Top Split-Gain Attribution

| Target | `max_depth` | `learning_rate` | `n_estimators` | `min_child_weight` | `subsample` | `colsample_bytree` | `reg_lambda` | Top-2 Split-Gain Features (Share) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `Solid_Waste_kg` | 2 | 0.03 | 700 | 3 | 0.8 | 0.9 | 5.0 | `Production_Volume_ton` (21.62%), `Production_Volume_kg` (13.40%) |
| `Yarn_Waste_kg` | 4 | 0.05 | 700 | 5 | 0.8 | 0.7 | 1.0 | `Production_Volume_ton` (34.34%), `Production_Volume_kg` (14.74%) |
| `Chemical_Waste_kg` | 3 | 0.05 | 700 | 10 | 0.8 | 0.9 | 1.0 | `Chem_inputs_total` (29.24%), `Prod_x_Salt` (27.34%) |
| `Wastewater_L` | 4 | 0.05 | 700 | 3 | 1.0 | 0.7 | 1.0 | `Total_Water_Consumption_L` (44.66%), `Production_Volume_ton` (22.17%) |

### 4.3 Design & Methodological Ablation Summary

| Component / Design Choice | Verified Configuration | Empirical Effect & Rationale |
|---|---|---|
| **Domain Interactions (`Prod × Salt`, `Chem_inputs_total`)** | 6 interaction/sum + 4 ratio features added to 27 raw inputs | Account for **56.58%** of total split gain on `Chemical_Waste_kg` (`Chem_inputs_total`: 29.24%, `Prod_x_Salt`: 27.34%), lifting tuned XGBoost to $R^2 = 0.7912$ (exceeding RF $0.7667$). |
| **Target Skewness Guard ($|\gamma_1| \le 0.5$)** | Identity target space (no `log1p`) | All 4 targets satisfy $|\gamma_1| \in [0.250, 0.474]$; avoids exponentiation re-transformation bias on high-output shifts. |
| **Causal Temporal Splitting (`gap=7`)** | Within-year `shift(1)` + `TimeSeriesSplit(gap=7)` | Eliminates future look-ahead and cross-year boundary contamination (`2023-12-31` $\not\to$ `2024-01-01`). |
| **Per-Target Hyperparameter Specialization** | Independent `GridSearchCV` per stream (`neg_MAE`) | Isolates shallow/high-L2 trees (`depth=2, reg_lambda=5.0`) for noisy `Solid_Waste_kg` while permitting deeper splits (`depth=4`) for `Yarn_Waste_kg` and `Wastewater_L`. |
| **MILP + ETP Coupling** | Exact 3-hub MILP + 75% ETP water loop | Yields $207,021,824\text{ L}$ recoverable ETP feed out of $208,029,536\text{ L}$ predicted 2024 effluent ($99.5\%$ post-sludge recovery) and optimal CC dispatch (`CC_2, CC_3` at $-31,057.36\text{ BDT}$ net cost on reference shift `1665.87 / 722.36 / 3202.32 kg`). |

---

## 5. Reproducibility & Environment Setup

### 5.1 Environment Installation

Pinned reference runtime verified on Python `3.13.15` (`requirements.txt` floor constraints + exact tested versions):

```bash
# Option A: Python venv
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS:        source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
# Exact verified versions: xgboost==3.4.1 scikit-learn==1.9.1 pandas==3.1.0 numpy==2.5.3 scipy==1.18.1 joblib==1.6.0
```

```bash
# Option B: Conda
conda create -n textile-waste python=3.13 -y
conda activate textile-waste
pip install -r requirements.txt
```

Optional IBM CPLEX Academic / Community solver (automatically falls back to `scipy.optimize.milp` HiGHS when absent):
```bash
pip install docplex cplex
```

### 5.2 CLI Execution Pipeline

1. **Data Preparation & Feature Schema Verification:**
   ```bash
   python -c "import pandas as pd; from model.features import engineer; df = pd.read_csv('Textile_Dyeing_Dataset_Causal.csv', parse_dates=['Date']); X, cols = engineer(df); print(X.shape, len(cols))"
   ```
2. **Model Training with Walk-Forward Per-Target Grid Search:**
   ```bash
   python -m model.train --data Textile_Dyeing_Dataset_Causal.xlsx
   ```
   Writes `xgb_model.pkl`, `feature_cols.json`, `best_params.json`, and `metrics.json` to `model/artifacts/`.
3. **Checkpoint Evaluation, Single-Shift Prediction & MILP Optimization:**
   ```bash
   # Verify saved 2024 hold-out metrics
   python -c "import pandas as pd; print(pd.read_csv('output_figures_xgb/model_metrics_xgb.csv').to_string(index=False))"

   # Run standalone MILP optimizer (DOcplex / SciPy fallback)
   python -c "from optimizer.milp import optimize; import json; print(json.dumps(optimize({'solid': 1665.87, 'yarn': 722.36, 'chemical': 3202.32}, 449024.70), indent=2))"

   # Interactive CLI optimizer
   python reverse_logistics_cplex.py
   ```
   Alternatively, launch the optional FastAPI service or open the standalone static console (`frontend/index.html`) directly in any browser:
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
4. **Artifact Serialization & Schema Inspection:**
   Tree ensembles are persisted via `joblib` (`PerTargetXGB`) paired with deterministic JSON feature order (`model/artifacts/feature_cols.json`) rather than ONNX/TorchScript, preserving exact floating-point tree split thresholds:
   ```bash
   python -c "import json, joblib; m = joblib.load('model/artifacts/xgb_model.pkl'); fc = json.load(open('model/artifacts/feature_cols.json')); print('Targets:', m.targets, '| Features:', len(fc))"
   ```

---

## 5. Citation

```bibtex
@misc{valmiki2026textilewaste,
  author       = {Karthik Valmiki},
  title        = {Predictive Multi-Stream Waste Forecasting and MILP Reverse Logistics Optimization for Circular Textile Dyeing Operations},
  year         = {2026},
  howpublished = {\url{https://github.com/Karthik-Valmiki/Smart_Mega_Factories}},
  note         = {Version 1.0.0}
}
```
