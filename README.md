# Textile Dyeing Waste Intelligence 

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
