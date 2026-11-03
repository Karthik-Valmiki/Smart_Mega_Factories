"""
Feature engineering for the textile dyeing waste prediction models.
Train on 2023 (365 days), test on 2024 (366 days).
Targets: Solid_Waste_kg, Yarn_Waste_kg, Chemical_Waste_kg, Wastewater_L
"""

import numpy as np
import pandas as pd

TARGETS = [
    "Solid_Waste_kg",
    "Yarn_Waste_kg",
    "Chemical_Waste_kg",
    "Wastewater_L",
]

# These columns are excluded from features (targets, derived ratios, metadata)
_EXCLUDE = TARGETS + [
    "Solid_Waste_%_of_Production",
    "Yarn_Waste_%_of_Production",
    "Chemical_Waste_kg_per_1000kg",
    "Wastewater_L_per_kg",
    "Date",
    "Day_Type",
    "Year",
    "Month",
]

# Columns that get lag/rolling treatment
_LAG_COLS = [
    ("Chemical_Waste_kg",    "chem"),
    ("Solid_Waste_kg",       "solid"),
    ("Yarn_Waste_kg",        "yarn"),
    ("Wastewater_L",         "water"),
    ("Production_Volume_kg", "prod"),
]


def _add_lag_features(g: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag-1, lag-7, roll-7-mean, roll-30-mean, roll-7-std for each
    target + production column.  Computed per calendar year so the
    2023/2024 boundary never leaks into lag windows.
    shift(1) prevents same-day look-ahead.
    """
    g = g.sort_values("Date").copy()
    for col, prefix in _LAG_COLS:
        s = g[col].shift(1)
        g[f"{prefix}_lag1"]   = s
        g[f"{prefix}_lag7"]   = g[col].shift(7)
        g[f"{prefix}_roll7"]  = s.rolling(7,  min_periods=1).mean()
        g[f"{prefix}_roll30"] = s.rolling(30, min_periods=1).mean()
        g[f"{prefix}_std7"]   = s.rolling(7,  min_periods=1).std().fillna(0)
    return g


def engineer(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply all feature engineering to the raw dataframe.
    Returns (df_engineered, feature_cols).
    df_engineered still contains TARGETS and Date columns.
    """
    df = df_raw.sort_values("Date").copy().reset_index(drop=True)

    # Rename colon column (breaks some sklearn/xgboost ops)
    if "Liquor_Ratio_1:x" in df.columns:
        df = df.rename(columns={"Liquor_Ratio_1:x": "Liquor_Ratio_1x"})

    # Cyclical time encoding
    months      = df["Date"].dt.month
    day_of_week = df["Date"].dt.dayofweek
    df["Month_Sin"] = np.sin(2 * np.pi * months / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * months / 12)
    df["Day_Sin"]   = np.sin(2 * np.pi * day_of_week / 7)
    df["Day_Cos"]   = np.cos(2 * np.pi * day_of_week / 7)
    df["Quarter"]   = df["Date"].dt.quarter

    # Interaction features
    p = df["Production_Volume_kg"]
    df["Prod_x_DyeBatch"]   = p * df["Dye_Batch_Count"]
    df["Prod_x_Acid"]       = p * df["Acid_kg"]
    df["Prod_x_Salt"]       = p * df["Salt_kg"]
    df["Prod_x_Alkali"]     = p * df["Alkali_kg"]
    df["Prod_x_Auxiliary"]  = p * df["Auxiliary_kg"]
    df["Chem_inputs_total"] = (
        df["Acid_kg"] + df["Salt_kg"] + df["Alkali_kg"]
        + df["Auxiliary_kg"] + df["Dye_Used_kg"]
    )

    # Physically motivated: unfixed dye goes to chemical waste / wastewater
    df["Fixation_loss_kg"]  = (1 - df["Dye_Fixation_Ratio_%"] / 100) * df["Dye_Used_kg"]
    df["Chem_per_litre"]    = df["Chem_inputs_total"] / (df["Total_Water_Consumption_L"] + 1)
    df["Energy_per_worker"] = df["Total_Energy_kWh"] / (df["Number_of_Workers"] + 1)
    df["Prod_per_worker"]   = p / (df["Number_of_Workers"] + 1)

    # Lag + rolling — computed within each calendar year to prevent leakage
    df = (
        df.groupby(df["Date"].dt.year, group_keys=False)
          .apply(_add_lag_features)
          .reset_index(drop=True)
    )

    # Drop the first 7 rows of each year (NaN from lag-7)
    df = df.dropna().reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in _EXCLUDE]
    return df, feature_cols


def build_single_row(
    date_str: str,
    factory_values: dict,
    recent_data: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Build one prediction row from a factory_values dict.
    recent_data is a DataFrame of recent historical rows (for lag proxies).
    Returns a single-row DataFrame aligned to feature_cols.
    """
    row = factory_values.copy()
    row["Date"] = pd.to_datetime(date_str)

    df_in = pd.DataFrame([row])
    if "Liquor_Ratio_1:x" in df_in.columns:
        df_in = df_in.rename(columns={"Liquor_Ratio_1:x": "Liquor_Ratio_1x"})

    m  = df_in["Date"].dt.month.values[0]
    dw = df_in["Date"].dt.dayofweek.values[0]
    df_in["Month_Sin"] = np.sin(2 * np.pi * m / 12)
    df_in["Month_Cos"] = np.cos(2 * np.pi * m / 12)
    df_in["Day_Sin"]   = np.sin(2 * np.pi * dw / 7)
    df_in["Day_Cos"]   = np.cos(2 * np.pi * dw / 7)
    df_in["Quarter"]   = pd.to_datetime(date_str).quarter

    p = df_in["Production_Volume_kg"].values[0]
    df_in["Prod_x_DyeBatch"]   = p * df_in["Dye_Batch_Count"]
    df_in["Prod_x_Acid"]       = p * df_in["Acid_kg"]
    df_in["Prod_x_Salt"]       = p * df_in["Salt_kg"]
    df_in["Prod_x_Alkali"]     = p * df_in["Alkali_kg"]
    df_in["Prod_x_Auxiliary"]  = p * df_in["Auxiliary_kg"]
    df_in["Chem_inputs_total"] = (
        df_in["Acid_kg"] + df_in["Salt_kg"] + df_in["Alkali_kg"]
        + df_in["Auxiliary_kg"] + df_in["Dye_Used_kg"]
    )
    df_in["Fixation_loss_kg"]  = (1 - df_in["Dye_Fixation_Ratio_%"] / 100) * df_in["Dye_Used_kg"]
    df_in["Chem_per_litre"]    = df_in["Chem_inputs_total"] / (df_in["Total_Water_Consumption_L"] + 1)
    df_in["Energy_per_worker"] = df_in["Total_Energy_kWh"] / (df_in["Number_of_Workers"] + 1)
    df_in["Prod_per_worker"]   = p / (df_in["Number_of_Workers"] + 1)

    # Lag proxies from recent history (last 30 days rolling mean/std)
    for col, prefix in _LAG_COLS:
        if col in recent_data.columns:
            series = recent_data[col].tail(30)
            mean_  = float(series.mean())
            std_   = float(series.std()) if len(series) > 1 else 0.0
        else:
            mean_, std_ = 0.0, 0.0
        df_in[f"{prefix}_lag1"]   = mean_
        df_in[f"{prefix}_lag7"]   = mean_
        df_in[f"{prefix}_roll7"]  = mean_
        df_in[f"{prefix}_roll30"] = mean_
        df_in[f"{prefix}_std7"]   = std_

    return df_in[feature_cols]

