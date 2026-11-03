"""
Textile Dyeing Factory — Waste Intelligence API
FastAPI application exposing all endpoints needed by the frontend dashboard.

Run:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Base URL: http://localhost:8000
Docs:     http://localhost:8000/docs
"""

import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Resolve paths regardless of where uvicorn is launched from
BASE_DIR  = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from model.features  import TARGETS
from optimizer.milp  import default_cc_params, default_cost_params, optimize

# ---------------------------------------------------------------------------
app = FastAPI(
    title="Textile Dyeing Waste Intelligence API",
    version="1.0.0",
    description="Predictive waste classification + reverse logistics optimization for a thread dyeing factory.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy model loader (only imports joblib/sklearn when the model is available)
# ---------------------------------------------------------------------------

_model       = None
_feature_cols: list[str] = []
_metrics: dict = {}
_df_hist: pd.DataFrame | None = None


def _load_artifacts():
    global _model, _feature_cols, _metrics
    if _model is not None:
        return

    artifacts = BASE_DIR / "model" / "artifacts"
    if not (artifacts / "xgb_model.pkl").exists():
        return  # model not yet trained — endpoints gracefully degrade

    import joblib
    _model = joblib.load(artifacts / "xgb_model.pkl")
    with open(artifacts / "feature_cols.json") as f:
        _feature_cols = json.load(f)
    with open(artifacts / "metrics.json") as f:
        _metrics = json.load(f)


def _load_history() -> pd.DataFrame:
    global _df_hist
    if _df_hist is not None:
        return _df_hist
    csv_path = BASE_DIR / "Textile_Dyeing_Dataset_Causal.csv"
    if csv_path.exists():
        _df_hist = pd.read_csv(csv_path, parse_dates=["Date"])
    else:
        xlsx_path = BASE_DIR / "Textile_Dyeing_Dataset_Causal.xlsx"
        if xlsx_path.exists():
            from model.features import engineer
            df_raw = pd.read_excel(xlsx_path, sheet_name="Sheet1")
            df_raw["Date"] = pd.to_datetime(df_raw["Date"])
            _df_hist = df_raw
    return _df_hist


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class FactoryInputs(BaseModel):
    """Raw operational parameters for one production day."""
    date:                        str   = Field(..., example="2025-07-15", description="YYYY-MM-DD")
    Production_Volume_kg:        float = Field(..., example=15300)
    Machine_Utilization_pct:     float = Field(..., example=85.0)
    Number_of_Workers:           int   = Field(..., example=100)
    Total_Energy_kWh:            float = Field(..., example=25000)
    Total_Water_Consumption_L:   float = Field(..., example=750000)
    Dye_Fixation_Ratio_pct:      float = Field(..., example=65.0)
    Liquor_Ratio:                float = Field(..., example=8.5)
    Temperature_C:               float = Field(..., example=65.0)
    Cycle_Time_hr:               float = Field(..., example=6.0)
    Dye_Batch_Count:             int   = Field(..., example=280)
    Dye_Used_kg:                 float = Field(..., example=50.0)
    Salt_kg:                     float = Field(..., example=320.0)
    Alkali_kg:                   float = Field(..., example=110.0)
    Acid_kg:                     float = Field(..., example=30.0)
    Auxiliary_kg:                float = Field(..., example=80.0)
    BOD_mg_per_L:                float = Field(..., example=200.0)
    COD_mg_per_L:                float = Field(..., example=850.0)
    pH:                          float = Field(..., example=8.5)
    Energy_Consumption_kWh_per_kg: float = Field(..., example=1.67)
    Water_Consumption_L_per_kg:  float = Field(..., example=50.0)
    Production_Volume_ton:       float = Field(..., example=15.3)
    Monthly_Demand_Factor:       float = Field(..., example=1.0)
    Dye_Used_g_per_kg:           float = Field(..., example=27.0)
    Salt_g_per_kg:               float = Field(..., example=280.0)
    Alkali_g_per_kg:             float = Field(..., example=73.0)
    Acid_g_per_kg:               float = Field(..., example=50.0)
    Auxiliary_g_per_kg:          float = Field(..., example=70.0)


class PredictRequest(BaseModel):
    factory_inputs: FactoryInputs


class OptimizeRequest(BaseModel):
    solid_kg:    float = Field(..., example=1665.87)
    yarn_kg:     float = Field(..., example=722.36)
    chemical_kg: float = Field(..., example=3202.32)
    wastewater_L: float = Field(..., example=451586.56)
    ccs: Optional[list[dict]] = None   # override CC params (uses defaults if null)
    cost: Optional[dict]      = None   # override cost params (uses defaults if null)


class PredictThenOptimizeRequest(BaseModel):
    """One-shot: predict waste then immediately run the MILP optimizer."""
    factory_inputs: FactoryInputs
    ccs:  Optional[list[dict]] = None
    cost: Optional[dict]       = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _predict_from_inputs(fi: FactoryInputs) -> dict:
    """Run ML prediction from FactoryInputs. Raises 503 if model not loaded."""
    _load_artifacts()
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run `python -m model.train` first.",
        )

    from model.features import build_single_row

    raw = {
        "Production_Volume_kg":          fi.Production_Volume_kg,
        "Machine_Utilization_%":         fi.Machine_Utilization_pct,
        "Number_of_Workers":             fi.Number_of_Workers,
        "Total_Energy_kWh":              fi.Total_Energy_kWh,
        "Total_Water_Consumption_L":     fi.Total_Water_Consumption_L,
        "Dye_Fixation_Ratio_%":          fi.Dye_Fixation_Ratio_pct,
        "Liquor_Ratio_1x":               fi.Liquor_Ratio,
        "Temperature_C":                 fi.Temperature_C,
        "Cycle_Time_hr":                 fi.Cycle_Time_hr,
        "Dye_Batch_Count":               fi.Dye_Batch_Count,
        "Dye_Used_kg":                   fi.Dye_Used_kg,
        "Salt_kg":                       fi.Salt_kg,
        "Alkali_kg":                     fi.Alkali_kg,
        "Acid_kg":                       fi.Acid_kg,
        "Auxiliary_kg":                  fi.Auxiliary_kg,
        "BOD_mg_per_L":                  fi.BOD_mg_per_L,
        "COD_mg_per_L":                  fi.COD_mg_per_L,
        "pH":                            fi.pH,
        "Energy_Consumption_kWh_per_kg": fi.Energy_Consumption_kWh_per_kg,
        "Water_Consumption_L_per_kg":    fi.Water_Consumption_L_per_kg,
        "Production_Volume_ton":         fi.Production_Volume_ton,
        "Monthly_Demand_Factor":         fi.Monthly_Demand_Factor,
        "Dye_Used_g_per_kg":             fi.Dye_Used_g_per_kg,
        "Salt_g_per_kg":                 fi.Salt_g_per_kg,
        "Alkali_g_per_kg":               fi.Alkali_g_per_kg,
        "Acid_g_per_kg":                 fi.Acid_g_per_kg,
        "Auxiliary_g_per_kg":            fi.Auxiliary_g_per_kg,
    }

    recent = _load_history()
    X = build_single_row(fi.date, raw, recent, _feature_cols)
    preds = _model.predict(X)[0]
    return {t: round(float(v), 2) for t, v in zip(TARGETS, preds)}


# ---------------------------------------------------------------------------
# Routes — Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
def health():
    """Service liveness check."""
    _load_artifacts()
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# Routes — ML Prediction
# ---------------------------------------------------------------------------

@app.post("/predict", tags=["ML Prediction"])
def predict(req: PredictRequest):
    """
    Predict daily waste volumes from factory operational parameters.

    Returns predicted kg/L for all four waste streams:
    - Solid_Waste_kg
    - Yarn_Waste_kg
    - Chemical_Waste_kg
    - Wastewater_L
    """
    predictions = _predict_from_inputs(req.factory_inputs)
    prod_kg = req.factory_inputs.Production_Volume_kg
    return {
        "date": req.factory_inputs.date,
        "predictions": predictions,
        "waste_pct_of_production": {
            "solid_pct":    round(predictions["Solid_Waste_kg"]    / prod_kg * 100, 3),
            "yarn_pct":     round(predictions["Yarn_Waste_kg"]     / prod_kg * 100, 3),
            "chemical_per_1000kg": round(predictions["Chemical_Waste_kg"] / prod_kg * 1000, 3),
            "wastewater_L_per_kg": round(predictions["Wastewater_L"]      / prod_kg, 3),
        },
        "treated_wastewater_L": round(
            max(0, predictions["Wastewater_L"] - predictions["Chemical_Waste_kg"] * 0.8), 2
        ),
    }


# ---------------------------------------------------------------------------
# Routes — MILP Optimization
# ---------------------------------------------------------------------------

@app.post("/optimize", tags=["MILP Optimization"])
def optimize_endpoint(req: OptimizeRequest):
    """
    Run MILP reverse logistics optimizer with given waste quantities.

    Determines which collection centers to activate and how to allocate
    each waste stream to maximize net profit while respecting capacities,
    transportation costs, CO2 penalties, and circularity bonuses.
    """
    wastes = {
        "solid":    req.solid_kg,
        "yarn":     req.yarn_kg,
        "chemical": req.chemical_kg,
    }
    ccs  = req.ccs  or default_cc_params()
    cost = req.cost or default_cost_params()
    try:
        result = optimize(wastes, req.wastewater_L, ccs, cost)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    return result


@app.post("/predict-and-optimize", tags=["ML Prediction", "MILP Optimization"])
def predict_and_optimize(req: PredictThenOptimizeRequest):
    """
    One-shot endpoint: predict waste from factory inputs, then immediately
    run the MILP optimizer.

    Returns both the ML predictions and the optimal logistics plan.
    """
    predictions = _predict_from_inputs(req.factory_inputs)

    wastes = {
        "solid":    predictions["Solid_Waste_kg"],
        "yarn":     predictions["Yarn_Waste_kg"],
        "chemical": predictions["Chemical_Waste_kg"],
    }
    treated_ww = max(0, predictions["Wastewater_L"] - predictions["Chemical_Waste_kg"] * 0.8)
    ccs  = req.ccs  or default_cc_params()
    cost = req.cost or default_cost_params()

    try:
        milp_result = optimize(wastes, treated_ww, ccs, cost)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "date":        req.factory_inputs.date,
        "predictions": predictions,
        "milp":        milp_result,
    }


# ---------------------------------------------------------------------------
# Routes — Dashboard Data (read from historical CSV)
# ---------------------------------------------------------------------------

@app.get("/dashboard/overview", tags=["Dashboard"])
def dashboard_overview():
    """
    High-level KPIs for the dashboard home page.
    Returns totals and averages split by year.
    """
    df = _load_history()
    if df is None:
        raise HTTPException(status_code=503, detail="Historical data not found.")

    result = {}
    for year in [2023, 2024]:
        sub = df[df["Date"].dt.year == year]
        prod = sub["Production_Volume_kg"].sum()
        result[str(year)] = {
            "production_kg":          round(float(prod), 2),
            "production_tons":        round(float(prod / 1000), 2),
            "solid_waste_kg":         round(float(sub["Solid_Waste_kg"].sum()), 2),
            "yarn_waste_kg":          round(float(sub["Yarn_Waste_kg"].sum()), 2),
            "chemical_waste_kg":      round(float(sub["Chemical_Waste_kg"].sum()), 2),
            "wastewater_L":           round(float(sub["Wastewater_L"].sum()), 2),
            "solid_waste_pct":        round(float(sub["Solid_Waste_%_of_Production"].mean()), 3),
            "yarn_waste_pct":         round(float(sub["Yarn_Waste_%_of_Production"].mean()), 3),
            "chemical_per_1000kg":    round(float(sub["Chemical_Waste_kg_per_1000kg"].mean()), 3),
            "wastewater_L_per_kg":    round(float(sub["Wastewater_L_per_kg"].mean()), 3),
            "avg_dye_fixation_pct":   round(float(sub["Dye_Fixation_Ratio_%"].mean()), 2),
            "avg_machine_util_pct":   round(float(sub["Machine_Utilization_%"].mean()), 2),
            "days":                   int(len(sub)),
        }
    return result


@app.get("/dashboard/wastage-trend", tags=["Dashboard"])
def wastage_trend(
    year:       int = Query(2024, description="Year to return (2023 or 2024)"),
    stream:     str = Query("all", description="solid | yarn | chemical | wastewater | all"),
    granularity: str = Query("daily", description="daily | monthly"),
):
    """
    Time-series wastage data for charts.
    Returns date + kg/L for the requested waste stream(s).
    """
    df = _load_history()
    if df is None:
        raise HTTPException(status_code=503, detail="Historical data not found.")

    sub = df[df["Date"].dt.year == year].sort_values("Date").copy()

    stream_map = {
        "solid":     "Solid_Waste_kg",
        "yarn":      "Yarn_Waste_kg",
        "chemical":  "Chemical_Waste_kg",
        "wastewater": "Wastewater_L",
    }

    cols = list(stream_map.values()) if stream == "all" else [stream_map.get(stream)]
    if None in cols:
        raise HTTPException(status_code=400, detail=f"Unknown stream '{stream}'. Use: solid|yarn|chemical|wastewater|all")

    if granularity == "monthly":
        sub["month"] = sub["Date"].dt.to_period("M").astype(str)
        grouped = sub.groupby("month")[cols].sum().reset_index()
        return grouped.rename(columns={"month": "period"}).to_dict(orient="records")
    else:
        sub["date"] = sub["Date"].dt.strftime("%Y-%m-%d")
        return sub[["date"] + cols].to_dict(orient="records")


@app.get("/dashboard/production-vs-waste", tags=["Dashboard"])
def production_vs_waste(year: int = Query(2024)):
    """
    Daily production volume vs total waste for scatter / correlation chart.
    """
    df = _load_history()
    if df is None:
        raise HTTPException(status_code=503, detail="Historical data not found.")

    sub = df[df["Date"].dt.year == year].sort_values("Date").copy()
    sub["total_waste_kg"] = (
        sub["Solid_Waste_kg"] + sub["Yarn_Waste_kg"] + sub["Chemical_Waste_kg"]
    )
    sub["date"] = sub["Date"].dt.strftime("%Y-%m-%d")
    return sub[["date", "Production_Volume_kg", "total_waste_kg",
                "Solid_Waste_kg", "Yarn_Waste_kg", "Chemical_Waste_kg",
                "Wastewater_L"]].to_dict(orient="records")


@app.get("/dashboard/day-type-analysis", tags=["Dashboard"])
def day_type_analysis():
    """
    Average waste metrics grouped by day type (good / normal / bad).
    Useful for bar charts showing how shift quality affects waste output.
    """
    df = _load_history()
    if df is None:
        raise HTTPException(status_code=503, detail="Historical data not found.")

    cols = [
        "Solid_Waste_%_of_Production",
        "Yarn_Waste_%_of_Production",
        "Chemical_Waste_kg_per_1000kg",
        "Wastewater_L_per_kg",
        "Machine_Utilization_%",
        "Dye_Fixation_Ratio_%",
    ]
    grouped = df.groupby("Day_Type")[cols].mean().round(3).reset_index()
    return grouped.to_dict(orient="records")


@app.get("/dashboard/monthly-summary", tags=["Dashboard"])
def monthly_summary(year: int = Query(2024)):
    """
    Monthly aggregated waste totals for KPI cards / bar charts.
    """
    df = _load_history()
    if df is None:
        raise HTTPException(status_code=503, detail="Historical data not found.")

    sub = df[df["Date"].dt.year == year].copy()
    sub["month"]     = sub["Date"].dt.month
    sub["month_name"] = sub["Date"].dt.strftime("%B")

    agg = sub.groupby(["month", "month_name"]).agg(
        production_kg=("Production_Volume_kg", "sum"),
        solid_waste_kg=("Solid_Waste_kg", "sum"),
        yarn_waste_kg=("Yarn_Waste_kg", "sum"),
        chemical_waste_kg=("Chemical_Waste_kg", "sum"),
        wastewater_L=("Wastewater_L", "sum"),
        avg_dye_fixation=("Dye_Fixation_Ratio_%", "mean"),
        avg_machine_util=("Machine_Utilization_%", "mean"),
    ).reset_index().sort_values("month")

    agg = agg.round(2)
    return agg.to_dict(orient="records")


@app.get("/dashboard/correlation", tags=["Dashboard"])
def correlation_matrix(target: str = Query("Solid_Waste_kg")):
    """
    Pearson correlation of operational inputs with a chosen waste target.
    Used to build correlation bar charts.
    """
    df = _load_history()
    if df is None:
        raise HTTPException(status_code=503, detail="Historical data not found.")

    if target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Unknown target '{target}'.")

    numeric = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[numeric].corr()[target].drop(target).sort_values(key=abs, ascending=False)
    return [{"feature": k, "correlation": round(float(v), 4)} for k, v in corr.items()]


@app.get("/dashboard/chemical-input-breakdown", tags=["Dashboard"])
def chemical_input_breakdown(year: int = Query(2024)):
    """
    Monthly totals of each chemical input (Dye, Salt, Alkali, Acid, Auxiliary).
    Useful for stacked bar charts.
    """
    df = _load_history()
    if df is None:
        raise HTTPException(status_code=503, detail="Historical data not found.")

    sub = df[df["Date"].dt.year == year].copy()
    sub["month"]     = sub["Date"].dt.month
    sub["month_name"] = sub["Date"].dt.strftime("%B")

    agg = sub.groupby(["month", "month_name"]).agg(
        dye_kg=("Dye_Used_kg", "sum"),
        salt_kg=("Salt_kg", "sum"),
        alkali_kg=("Alkali_kg", "sum"),
        acid_kg=("Acid_kg", "sum"),
        auxiliary_kg=("Auxiliary_kg", "sum"),
        chemical_waste_kg=("Chemical_Waste_kg", "sum"),
    ).reset_index().sort_values("month").round(2)

    return agg.to_dict(orient="records")


@app.get("/dashboard/wastewater-etp", tags=["Dashboard"])
def wastewater_etp(year: int = Query(2024)):
    """
    Monthly wastewater volumes with ETP recycling estimates.
    recycled_L = wastewater_L * 0.75 (75% ETP recovery rate).
    """
    df = _load_history()
    if df is None:
        raise HTTPException(status_code=503, detail="Historical data not found.")

    sub = df[df["Date"].dt.year == year].copy()
    sub["month"]       = sub["Date"].dt.month
    sub["month_name"]  = sub["Date"].dt.strftime("%B")
    sub["recycled_L"]  = sub["Wastewater_L"] * 0.75
    sub["drained_L"]   = sub["Wastewater_L"] * 0.25

    agg = sub.groupby(["month", "month_name"]).agg(
        wastewater_L=("Wastewater_L", "sum"),
        recycled_L=("recycled_L", "sum"),
        drained_L=("drained_L", "sum"),
        bod_avg=("BOD_mg_per_L", "mean"),
        cod_avg=("COD_mg_per_L", "mean"),
        ph_avg=("pH", "mean"),
    ).reset_index().sort_values("month").round(2)

    return agg.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Routes — Model Metadata
# ---------------------------------------------------------------------------

@app.get("/model/metrics", tags=["Model Info"])
def model_metrics():
    """
    Returns the test-set performance metrics (R², MAE, RMSE, MAPE) for
    all four waste prediction targets.
    """
    _load_artifacts()
    if not _metrics:
        raise HTTPException(status_code=503, detail="Model not trained yet.")
    return _metrics


@app.get("/model/feature-importance", tags=["Model Info"])
def feature_importance(target: str = Query("Solid_Waste_kg")):
    """
    Returns top-N feature importance scores (XGBoost gain) for a target.
    """
    _load_artifacts()
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet.")

    if target not in TARGETS:
        raise HTTPException(status_code=400, detail=f"target must be one of {TARGETS}")

    idx = TARGETS.index(target)
    est = _model.estimators_[idx]
    importances = est.feature_importances_
    result = sorted(
        zip(_feature_cols, importances.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return [{"feature": f, "importance": round(v, 6)} for f, v in result]


@app.get("/model/targets", tags=["Model Info"])
def model_targets():
    """Lists the prediction targets and their units."""
    return [
        {"target": "Solid_Waste_kg",    "unit": "kg",    "description": "Solid fabric/cone/packaging waste per day"},
        {"target": "Yarn_Waste_kg",     "unit": "kg",    "description": "Defective yarn and winding loss per day"},
        {"target": "Chemical_Waste_kg", "unit": "kg",    "description": "Spent dye bath sludge and chemical residue per day"},
        {"target": "Wastewater_L",      "unit": "litres","description": "Effluent wastewater generated per day"},
    ]


@app.get("/optimizer/defaults", tags=["MILP Optimization"])
def optimizer_defaults():
    """Returns the default CC parameters and cost parameters used by the optimizer."""
    return {
        "collection_centers": default_cc_params(),
        "cost_parameters":    default_cost_params(),
    }


# Mount frontend static directory if present
frontend_path = BASE_DIR / "frontend"
if frontend_path.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")


