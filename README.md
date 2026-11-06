# Textile Dyeing Waste Intelligence 

### 1. Train the model
```bash
python -m model.train
```

### 2. Run the API
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI: http://localhost:8000/docs

---

## API Endpoints

### Health
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check, model load status |

### ML Prediction
| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Predict 4 waste streams from factory inputs |
| POST | `/predict-and-optimize` | Predict waste then run MILP optimizer |

### MILP Optimization
| Method | Path | Description |
|--------|------|-------------|
| POST | `/optimize` | Run MILP with given waste quantities |
| GET  | `/optimizer/defaults` | Default CC + cost parameters |

### Dashboard (frontend charts)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/dashboard/overview` | Annual KPI summary (2023 + 2024) |
| GET | `/dashboard/wastage-trend` | Daily or monthly waste time-series |
| GET | `/dashboard/production-vs-waste` | Production volume vs total waste |
| GET | `/dashboard/day-type-analysis` | Waste by shift quality (good/normal/bad) |
| GET | `/dashboard/monthly-summary` | Monthly aggregated waste totals |
| GET | `/dashboard/correlation` | Input feature correlations with a waste target |
| GET | `/dashboard/chemical-input-breakdown` | Monthly chemical input totals |
| GET | `/dashboard/wastewater-etp` | Wastewater + ETP recycling estimates |

### Model Info
| Method | Path | Description |
|--------|------|-------------|
| GET | `/model/metrics` | R², MAE, RMSE, MAPE per target |
| GET | `/model/feature-importance` | XGBoost gain importance for a target |
| GET | `/model/targets` | Target names, units, descriptions |

---

## Model Performance (2024 hold-out test set)

| Target | R² | MAE | MAPE |
|---|---|---|---|
| Solid_Waste_kg | 0.60 | 251 kg | 12.2% |
| Yarn_Waste_kg | 0.71 | 95 kg | 11.0% |
| Chemical_Waste_kg | 0.79 | 311 kg | 9.3% |
| Wastewater_L | 0.81 | 50,709 L | 9.1% |

