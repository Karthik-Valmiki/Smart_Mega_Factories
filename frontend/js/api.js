/**
 * Textile Dyeing Waste Intelligence — Static Data & Local Engine
 *
 * Zero-backend mode: all telemetry, model metrics, feature gain rankings,
 * XGBoost-calibrated inference, and exact 3-hub MILP optimization run
 * locally in the browser with no external API calls.
 *
 * Metrics & feature gain: extracted from output_figures_xgb/model_metrics_xgb.csv
 * and output_figures_xgb/xgboost_waste_model_improved.pkl
 * Historical telemetry: aggregated from Textile_Dyeing_Dataset_Causal.csv (731 days)
 */

// ─── Actual trained XGBoost metrics (2024 hold-out test set) ─────────────────
const MODEL_METRICS = {
  Solid_Waste_kg: {
    r2: 0.5903, mae: 252.734, rmse: 323.462, mape_pct: 12.25
  },
  Yarn_Waste_kg: {
    r2: 0.6859, mae: 99.215, rmse: 122.681, mape_pct: 11.40
  },
  Chemical_Waste_kg: {
    r2: 0.7912, mae: 312.274, rmse: 389.104, mape_pct: 9.29
  },
  Wastewater_L: {
    r2: 0.8026, mae: 50574.959, rmse: 63548.702, mape_pct: 9.15
  }
};

// ─── Real XGBoost split-gain feature importance (from trained .pkl) ──────────
const FEATURE_IMPORTANCE = {
  "Solid_Waste_kg": [
    {
      "feature": "Production_Volume_ton",
      "importance": 0.216198
    },
    {
      "feature": "Production_Volume_kg",
      "importance": 0.133964
    },
    {
      "feature": "Total_Water_Consumption_L",
      "importance": 0.069663
    },
    {
      "feature": "Prod_x_Alkali",
      "importance": 0.059224
    },
    {
      "feature": "Chem_inputs_total",
      "importance": 0.038584
    },
    {
      "feature": "Prod_x_DyeBatch",
      "importance": 0.02691
    },
    {
      "feature": "Prod_per_worker",
      "importance": 0.019295
    },
    {
      "feature": "Prod_x_Auxiliary",
      "importance": 0.018521
    },
    {
      "feature": "Prod_x_Acid",
      "importance": 0.016399
    },
    {
      "feature": "Prod_x_Salt",
      "importance": 0.015015
    }
  ],
  "Yarn_Waste_kg": [
    {
      "feature": "Production_Volume_ton",
      "importance": 0.343363
    },
    {
      "feature": "Production_Volume_kg",
      "importance": 0.147357
    },
    {
      "feature": "Total_Water_Consumption_L",
      "importance": 0.076113
    },
    {
      "feature": "Prod_x_Salt",
      "importance": 0.065543
    },
    {
      "feature": "Fixation_loss_kg",
      "importance": 0.028852
    },
    {
      "feature": "Total_Energy_kWh",
      "importance": 0.026068
    },
    {
      "feature": "Dye_Fixation_Ratio_%",
      "importance": 0.024261
    },
    {
      "feature": "Energy_Consumption_kWh_per_kg",
      "importance": 0.021782
    },
    {
      "feature": "Chem_inputs_total",
      "importance": 0.021015
    },
    {
      "feature": "Prod_x_DyeBatch",
      "importance": 0.015085
    }
  ],
  "Chemical_Waste_kg": [
    {
      "feature": "Chem_inputs_total",
      "importance": 0.29244
    },
    {
      "feature": "Prod_x_Salt",
      "importance": 0.273395
    },
    {
      "feature": "Production_Volume_ton",
      "importance": 0.113977
    },
    {
      "feature": "Production_Volume_kg",
      "importance": 0.04567
    },
    {
      "feature": "Prod_x_Acid",
      "importance": 0.030131
    },
    {
      "feature": "COD_mg_per_L",
      "importance": 0.022237
    },
    {
      "feature": "Prod_x_DyeBatch",
      "importance": 0.021958
    },
    {
      "feature": "Total_Water_Consumption_L",
      "importance": 0.021042
    },
    {
      "feature": "Salt_kg",
      "importance": 0.011223
    },
    {
      "feature": "Dye_Fixation_Ratio_%",
      "importance": 0.010618
    }
  ],
  "Wastewater_L": [
    {
      "feature": "Total_Water_Consumption_L",
      "importance": 0.44655
    },
    {
      "feature": "Production_Volume_ton",
      "importance": 0.221709
    },
    {
      "feature": "Production_Volume_kg",
      "importance": 0.043645
    },
    {
      "feature": "Water_Consumption_L_per_kg",
      "importance": 0.029456
    },
    {
      "feature": "Prod_x_DyeBatch",
      "importance": 0.017648
    },
    {
      "feature": "Prod_x_Auxiliary",
      "importance": 0.013884
    },
    {
      "feature": "Chem_per_litre",
      "importance": 0.01352
    },
    {
      "feature": "chem_roll7",
      "importance": 0.013508
    },
    {
      "feature": "Total_Energy_kWh",
      "importance": 0.01083
    },
    {
      "feature": "water_roll7",
      "importance": 0.010357
    }
  ]
};

// ─── Annual KPI aggregates (2023 train / 2024 test, from dataset) ────────────
const OVERVIEW_DATA = {
  "2023": {
    "production_kg": 5111811.0,
    "production_tons": 5111.81,
    "solid_waste_kg": 785169.7,
    "yarn_waste_kg": 319244.3,
    "chemical_waste_kg": 1290209.7,
    "wastewater_L": 216680638.1,
    "solid_waste_pct": 15.333,
    "yarn_waste_pct": 6.253,
    "chemical_per_1000kg": 252.095,
    "wastewater_L_per_kg": 42.391,
    "avg_dye_fixation_pct": 74.73,
    "avg_machine_util_pct": 83.36,
    "days": 365
  },
  "2024": {
    "production_kg": 5051227.0,
    "production_tons": 5051.23,
    "solid_waste_kg": 775431.7,
    "yarn_waste_kg": 321411.4,
    "chemical_waste_kg": 1275463.5,
    "wastewater_L": 212292294.3,
    "solid_waste_pct": 15.375,
    "yarn_waste_pct": 6.358,
    "chemical_per_1000kg": 252.512,
    "wastewater_L_per_kg": 42.032,
    "avg_dye_fixation_pct": 75.11,
    "avg_machine_util_pct": 83.21,
    "days": 366
  }
};

// ─── Monthly chemical input & sludge totals (12 months, from dataset) ────────
const MONTHLY_PROFILES = {
  "2023": [
    {
      "month": 1,
      "month_name": "Jan 2023",
      "dye_kg": 8686.5,
      "salt_kg": 88342.8,
      "alkali_kg": 19275.3,
      "acid_kg": 16728.7,
      "auxiliary_kg": 22615.1,
      "chemical_waste_kg": 86994.1
    },
    {
      "month": 2,
      "month_name": "Feb 2023",
      "dye_kg": 7647.4,
      "salt_kg": 84457.8,
      "alkali_kg": 20296.7,
      "acid_kg": 15416.2,
      "auxiliary_kg": 21109.1,
      "chemical_waste_kg": 87226.0
    },
    {
      "month": 3,
      "month_name": "Mar 2023",
      "dye_kg": 9371.1,
      "salt_kg": 99263.5,
      "alkali_kg": 23372.0,
      "acid_kg": 17973.8,
      "auxiliary_kg": 23725.6,
      "chemical_waste_kg": 105519.4
    },
    {
      "month": 4,
      "month_name": "Apr 2023",
      "dye_kg": 10190.8,
      "salt_kg": 109600.7,
      "alkali_kg": 25830.8,
      "acid_kg": 19597.2,
      "auxiliary_kg": 27191.5,
      "chemical_waste_kg": 117552.7
    },
    {
      "month": 5,
      "month_name": "May 2023",
      "dye_kg": 11712.1,
      "salt_kg": 126916.3,
      "alkali_kg": 28559.3,
      "acid_kg": 23071.7,
      "auxiliary_kg": 29860.9,
      "chemical_waste_kg": 132090.9
    },
    {
      "month": 6,
      "month_name": "Jun 2023",
      "dye_kg": 11651.0,
      "salt_kg": 128058.7,
      "alkali_kg": 27887.9,
      "acid_kg": 24220.3,
      "auxiliary_kg": 30301.1,
      "chemical_waste_kg": 130686.6
    },
    {
      "month": 7,
      "month_name": "Jul 2023",
      "dye_kg": 11776.4,
      "salt_kg": 129093.6,
      "alkali_kg": 27693.1,
      "acid_kg": 23758.3,
      "auxiliary_kg": 32095.2,
      "chemical_waste_kg": 132872.7
    },
    {
      "month": 8,
      "month_name": "Aug 2023",
      "dye_kg": 11031.2,
      "salt_kg": 118813.4,
      "alkali_kg": 26622.4,
      "acid_kg": 23145.0,
      "auxiliary_kg": 29533.7,
      "chemical_waste_kg": 122511.9
    },
    {
      "month": 9,
      "month_name": "Sep 2023",
      "dye_kg": 9050.1,
      "salt_kg": 99375.6,
      "alkali_kg": 24038.0,
      "acid_kg": 18730.9,
      "auxiliary_kg": 24197.2,
      "chemical_waste_kg": 98484.5
    },
    {
      "month": 10,
      "month_name": "Oct 2023",
      "dye_kg": 8567.5,
      "salt_kg": 96822.6,
      "alkali_kg": 22694.9,
      "acid_kg": 17581.3,
      "auxiliary_kg": 22675.4,
      "chemical_waste_kg": 101844.5
    },
    {
      "month": 11,
      "month_name": "Nov 2023",
      "dye_kg": 7413.2,
      "salt_kg": 80920.2,
      "alkali_kg": 17644.4,
      "acid_kg": 15161.5,
      "auxiliary_kg": 18809.2,
      "chemical_waste_kg": 81818.1
    },
    {
      "month": 12,
      "month_name": "Dec 2023",
      "dye_kg": 8657.1,
      "salt_kg": 93538.3,
      "alkali_kg": 20649.0,
      "acid_kg": 15943.2,
      "auxiliary_kg": 20323.2,
      "chemical_waste_kg": 92608.3
    }
  ],
  "2024": [
    {
      "month": 1,
      "month_name": "Jan 2024",
      "dye_kg": 8235.4,
      "salt_kg": 85201.5,
      "alkali_kg": 19068.5,
      "acid_kg": 15065.6,
      "auxiliary_kg": 20225.2,
      "chemical_waste_kg": 85313.7
    },
    {
      "month": 2,
      "month_name": "Feb 2024",
      "dye_kg": 8184.0,
      "salt_kg": 90271.4,
      "alkali_kg": 21483.9,
      "acid_kg": 16346.4,
      "auxiliary_kg": 21127.5,
      "chemical_waste_kg": 88319.3
    },
    {
      "month": 3,
      "month_name": "Mar 2024",
      "dye_kg": 9436.7,
      "salt_kg": 98606.1,
      "alkali_kg": 22469.0,
      "acid_kg": 17240.3,
      "auxiliary_kg": 24818.4,
      "chemical_waste_kg": 103171.9
    },
    {
      "month": 4,
      "month_name": "Apr 2024",
      "dye_kg": 10135.6,
      "salt_kg": 111794.9,
      "alkali_kg": 24349.5,
      "acid_kg": 19753.2,
      "auxiliary_kg": 26578.2,
      "chemical_waste_kg": 113098.3
    },
    {
      "month": 5,
      "month_name": "May 2024",
      "dye_kg": 10223.4,
      "salt_kg": 116648.1,
      "alkali_kg": 30012.9,
      "acid_kg": 23553.6,
      "auxiliary_kg": 31048.2,
      "chemical_waste_kg": 122373.6
    },
    {
      "month": 6,
      "month_name": "Jun 2024",
      "dye_kg": 12545.6,
      "salt_kg": 130459.8,
      "alkali_kg": 28900.9,
      "acid_kg": 24146.5,
      "auxiliary_kg": 31307.1,
      "chemical_waste_kg": 130224.1
    },
    {
      "month": 7,
      "month_name": "Jul 2024",
      "dye_kg": 12537.5,
      "salt_kg": 135107.7,
      "alkali_kg": 28891.0,
      "acid_kg": 23596.6,
      "auxiliary_kg": 30343.8,
      "chemical_waste_kg": 137683.0
    },
    {
      "month": 8,
      "month_name": "Aug 2024",
      "dye_kg": 11244.8,
      "salt_kg": 119120.1,
      "alkali_kg": 25856.9,
      "acid_kg": 21168.4,
      "auxiliary_kg": 30770.0,
      "chemical_waste_kg": 123051.2
    },
    {
      "month": 9,
      "month_name": "Sep 2024",
      "dye_kg": 9582.1,
      "salt_kg": 104439.2,
      "alkali_kg": 25771.2,
      "acid_kg": 18471.4,
      "auxiliary_kg": 24657.8,
      "chemical_waste_kg": 105380.1
    },
    {
      "month": 10,
      "month_name": "Oct 2024",
      "dye_kg": 9024.3,
      "salt_kg": 94661.9,
      "alkali_kg": 21122.6,
      "acid_kg": 18173.2,
      "auxiliary_kg": 22753.1,
      "chemical_waste_kg": 96495.0
    },
    {
      "month": 11,
      "month_name": "Nov 2024",
      "dye_kg": 7323.0,
      "salt_kg": 75472.9,
      "alkali_kg": 17366.8,
      "acid_kg": 14714.5,
      "auxiliary_kg": 18566.2,
      "chemical_waste_kg": 76618.4
    },
    {
      "month": 12,
      "month_name": "Dec 2024",
      "dye_kg": 8191.9,
      "salt_kg": 90055.9,
      "alkali_kg": 19582.4,
      "acid_kg": 16798.0,
      "auxiliary_kg": 21654.4,
      "chemical_waste_kg": 93734.9
    }
  ]
};

// ─── Monthly wastewater & ETP recovery (75% recycled, 25% drained) ───────────
const ETP_PROFILES = {
  "2023": [
    {
      "month": 1,
      "month_name": "Jan 2023",
      "wastewater_L": 15371932.2,
      "recycled_L": 11528949.15,
      "drained_L": 3842983.05,
      "bod_avg": 262.75,
      "cod_avg": 829.12,
      "ph_avg": 9.78
    },
    {
      "month": 2,
      "month_name": "Feb 2023",
      "wastewater_L": 14498134.5,
      "recycled_L": 10873600.87,
      "drained_L": 3624533.62,
      "bod_avg": 278.56,
      "cod_avg": 859.25,
      "ph_avg": 9.78
    },
    {
      "month": 3,
      "month_name": "Mar 2023",
      "wastewater_L": 17641115.4,
      "recycled_L": 13230836.55,
      "drained_L": 4410278.85,
      "bod_avg": 270.77,
      "cod_avg": 857.09,
      "ph_avg": 9.69
    },
    {
      "month": 4,
      "month_name": "Apr 2023",
      "wastewater_L": 20729035.4,
      "recycled_L": 15546776.55,
      "drained_L": 5182258.85,
      "bod_avg": 280.33,
      "cod_avg": 837.98,
      "ph_avg": 9.76
    },
    {
      "month": 5,
      "month_name": "May 2023",
      "wastewater_L": 22201620.8,
      "recycled_L": 16651215.6,
      "drained_L": 5550405.2,
      "bod_avg": 278.7,
      "cod_avg": 864.13,
      "ph_avg": 9.79
    },
    {
      "month": 6,
      "month_name": "Jun 2023",
      "wastewater_L": 21323211.3,
      "recycled_L": 15992408.48,
      "drained_L": 5330802.82,
      "bod_avg": 266.48,
      "cod_avg": 842.85,
      "ph_avg": 9.78
    },
    {
      "month": 7,
      "month_name": "Jul 2023",
      "wastewater_L": 22140694.7,
      "recycled_L": 16605521.02,
      "drained_L": 5535173.67,
      "bod_avg": 267.09,
      "cod_avg": 845.61,
      "ph_avg": 9.75
    },
    {
      "month": 8,
      "month_name": "Aug 2023",
      "wastewater_L": 20072820.6,
      "recycled_L": 15054615.45,
      "drained_L": 5018205.15,
      "bod_avg": 277.93,
      "cod_avg": 862.62,
      "ph_avg": 9.79
    },
    {
      "month": 9,
      "month_name": "Sep 2023",
      "wastewater_L": 16690175.5,
      "recycled_L": 12517631.62,
      "drained_L": 4172543.87,
      "bod_avg": 264.78,
      "cod_avg": 842.59,
      "ph_avg": 9.89
    },
    {
      "month": 10,
      "month_name": "Oct 2023",
      "wastewater_L": 16732350.5,
      "recycled_L": 12549262.87,
      "drained_L": 4183087.62,
      "bod_avg": 263.58,
      "cod_avg": 849.29,
      "ph_avg": 9.8
    },
    {
      "month": 11,
      "month_name": "Nov 2023",
      "wastewater_L": 14216389.0,
      "recycled_L": 10662291.75,
      "drained_L": 3554097.25,
      "bod_avg": 262.71,
      "cod_avg": 854.3,
      "ph_avg": 9.73
    },
    {
      "month": 12,
      "month_name": "Dec 2023",
      "wastewater_L": 15063158.2,
      "recycled_L": 11297368.65,
      "drained_L": 3765789.55,
      "bod_avg": 273.74,
      "cod_avg": 852.07,
      "ph_avg": 9.79
    }
  ],
  "2024": [
    {
      "month": 1,
      "month_name": "Jan 2024",
      "wastewater_L": 14470034.5,
      "recycled_L": 10852525.87,
      "drained_L": 3617508.62,
      "bod_avg": 263.04,
      "cod_avg": 839.33,
      "ph_avg": 9.75
    },
    {
      "month": 2,
      "month_name": "Feb 2024",
      "wastewater_L": 14151569.2,
      "recycled_L": 10613676.9,
      "drained_L": 3537892.3,
      "bod_avg": 267.87,
      "cod_avg": 828.36,
      "ph_avg": 9.76
    },
    {
      "month": 3,
      "month_name": "Mar 2024",
      "wastewater_L": 17307214.4,
      "recycled_L": 12980410.8,
      "drained_L": 4326803.6,
      "bod_avg": 263.76,
      "cod_avg": 844.04,
      "ph_avg": 9.77
    },
    {
      "month": 4,
      "month_name": "Apr 2024",
      "wastewater_L": 18879442.7,
      "recycled_L": 14159582.02,
      "drained_L": 4719860.68,
      "bod_avg": 269.26,
      "cod_avg": 863.33,
      "ph_avg": 9.67
    },
    {
      "month": 5,
      "month_name": "May 2024",
      "wastewater_L": 20445362.3,
      "recycled_L": 15334021.72,
      "drained_L": 5111340.57,
      "bod_avg": 262.58,
      "cod_avg": 813.71,
      "ph_avg": 9.79
    },
    {
      "month": 6,
      "month_name": "Jun 2024",
      "wastewater_L": 22946659.0,
      "recycled_L": 17209994.25,
      "drained_L": 5736664.75,
      "bod_avg": 270.69,
      "cod_avg": 860.02,
      "ph_avg": 9.8
    },
    {
      "month": 7,
      "month_name": "Jul 2024",
      "wastewater_L": 22155012.7,
      "recycled_L": 16616259.52,
      "drained_L": 5538753.18,
      "bod_avg": 275.04,
      "cod_avg": 872.5,
      "ph_avg": 9.77
    },
    {
      "month": 8,
      "month_name": "Aug 2024",
      "wastewater_L": 20060113.4,
      "recycled_L": 15045085.05,
      "drained_L": 5015028.35,
      "bod_avg": 265.39,
      "cod_avg": 854.97,
      "ph_avg": 9.77
    },
    {
      "month": 9,
      "month_name": "Sep 2024",
      "wastewater_L": 17762860.4,
      "recycled_L": 13322145.3,
      "drained_L": 4440715.1,
      "bod_avg": 269.53,
      "cod_avg": 848.65,
      "ph_avg": 9.78
    },
    {
      "month": 10,
      "month_name": "Oct 2024",
      "wastewater_L": 15610148.4,
      "recycled_L": 11707611.3,
      "drained_L": 3902537.1,
      "bod_avg": 273.96,
      "cod_avg": 845.7,
      "ph_avg": 9.72
    },
    {
      "month": 11,
      "month_name": "Nov 2024",
      "wastewater_L": 13021361.8,
      "recycled_L": 9766021.35,
      "drained_L": 3255340.45,
      "bod_avg": 280.05,
      "cod_avg": 840.44,
      "ph_avg": 9.78
    },
    {
      "month": 12,
      "month_name": "Dec 2024",
      "wastewater_L": 15482515.5,
      "recycled_L": 11611886.62,
      "drained_L": 3870628.87,
      "bod_avg": 277.13,
      "cod_avg": 874.1,
      "ph_avg": 9.76
    }
  ]
};

// ─── Monthly aggregated waste trends ─────────────────────────────────────────
const TREND_MONTHLY = {
  "2023": [
    {
      "month": 1,
      "period": "Jan 2023",
      "Solid_Waste_kg": 56451.5,
      "Yarn_Waste_kg": 22691.7,
      "Chemical_Waste_kg": 86994.1,
      "Wastewater_L": 15371932.2
    },
    {
      "month": 2,
      "period": "Feb 2023",
      "Solid_Waste_kg": 52105.7,
      "Yarn_Waste_kg": 21676.3,
      "Chemical_Waste_kg": 87226.0,
      "Wastewater_L": 14498134.5
    },
    {
      "month": 3,
      "period": "Mar 2023",
      "Solid_Waste_kg": 65122.6,
      "Yarn_Waste_kg": 26651.8,
      "Chemical_Waste_kg": 105519.4,
      "Wastewater_L": 17641115.4
    },
    {
      "month": 4,
      "period": "Apr 2023",
      "Solid_Waste_kg": 73629.1,
      "Yarn_Waste_kg": 29753.0,
      "Chemical_Waste_kg": 117552.7,
      "Wastewater_L": 20729035.4
    },
    {
      "month": 5,
      "period": "May 2023",
      "Solid_Waste_kg": 75331.5,
      "Yarn_Waste_kg": 30880.7,
      "Chemical_Waste_kg": 132090.9,
      "Wastewater_L": 22201620.8
    },
    {
      "month": 6,
      "period": "Jun 2023",
      "Solid_Waste_kg": 83275.0,
      "Yarn_Waste_kg": 32822.2,
      "Chemical_Waste_kg": 130686.6,
      "Wastewater_L": 21323211.3
    },
    {
      "month": 7,
      "period": "Jul 2023",
      "Solid_Waste_kg": 79011.0,
      "Yarn_Waste_kg": 32716.4,
      "Chemical_Waste_kg": 132872.7,
      "Wastewater_L": 22140694.7
    },
    {
      "month": 8,
      "period": "Aug 2023",
      "Solid_Waste_kg": 73350.1,
      "Yarn_Waste_kg": 29773.0,
      "Chemical_Waste_kg": 122511.9,
      "Wastewater_L": 20072820.6
    },
    {
      "month": 9,
      "period": "Sep 2023",
      "Solid_Waste_kg": 58681.9,
      "Yarn_Waste_kg": 24435.3,
      "Chemical_Waste_kg": 98484.5,
      "Wastewater_L": 16690175.5
    },
    {
      "month": 10,
      "period": "Oct 2023",
      "Solid_Waste_kg": 59584.6,
      "Yarn_Waste_kg": 23891.0,
      "Chemical_Waste_kg": 101844.5,
      "Wastewater_L": 16732350.5
    },
    {
      "month": 11,
      "period": "Nov 2023",
      "Solid_Waste_kg": 52746.6,
      "Yarn_Waste_kg": 20672.2,
      "Chemical_Waste_kg": 81818.1,
      "Wastewater_L": 14216389.0
    },
    {
      "month": 12,
      "period": "Dec 2023",
      "Solid_Waste_kg": 55880.1,
      "Yarn_Waste_kg": 23280.7,
      "Chemical_Waste_kg": 92608.3,
      "Wastewater_L": 15063158.2
    }
  ],
  "2024": [
    {
      "month": 1,
      "period": "Jan 2024",
      "Solid_Waste_kg": 54318.8,
      "Yarn_Waste_kg": 22008.4,
      "Chemical_Waste_kg": 85313.7,
      "Wastewater_L": 14470034.5
    },
    {
      "month": 2,
      "period": "Feb 2024",
      "Solid_Waste_kg": 53001.1,
      "Yarn_Waste_kg": 22067.5,
      "Chemical_Waste_kg": 88319.3,
      "Wastewater_L": 14151569.2
    },
    {
      "month": 3,
      "period": "Mar 2024",
      "Solid_Waste_kg": 60925.2,
      "Yarn_Waste_kg": 25555.2,
      "Chemical_Waste_kg": 103171.9,
      "Wastewater_L": 17307214.4
    },
    {
      "month": 4,
      "period": "Apr 2024",
      "Solid_Waste_kg": 71373.8,
      "Yarn_Waste_kg": 29190.7,
      "Chemical_Waste_kg": 113098.3,
      "Wastewater_L": 18879442.7
    },
    {
      "month": 5,
      "period": "May 2024",
      "Solid_Waste_kg": 75156.4,
      "Yarn_Waste_kg": 31488.1,
      "Chemical_Waste_kg": 122373.6,
      "Wastewater_L": 20445362.3
    },
    {
      "month": 6,
      "period": "Jun 2024",
      "Solid_Waste_kg": 78023.1,
      "Yarn_Waste_kg": 33533.1,
      "Chemical_Waste_kg": 130224.1,
      "Wastewater_L": 22946659.0
    },
    {
      "month": 7,
      "period": "Jul 2024",
      "Solid_Waste_kg": 79632.6,
      "Yarn_Waste_kg": 33402.3,
      "Chemical_Waste_kg": 137683.0,
      "Wastewater_L": 22155012.7
    },
    {
      "month": 8,
      "period": "Aug 2024",
      "Solid_Waste_kg": 75814.7,
      "Yarn_Waste_kg": 31880.2,
      "Chemical_Waste_kg": 123051.2,
      "Wastewater_L": 20060113.4
    },
    {
      "month": 9,
      "period": "Sep 2024",
      "Solid_Waste_kg": 64875.4,
      "Yarn_Waste_kg": 26086.7,
      "Chemical_Waste_kg": 105380.1,
      "Wastewater_L": 17762860.4
    },
    {
      "month": 10,
      "period": "Oct 2024",
      "Solid_Waste_kg": 59579.1,
      "Yarn_Waste_kg": 24590.1,
      "Chemical_Waste_kg": 96495.0,
      "Wastewater_L": 15610148.4
    },
    {
      "month": 11,
      "period": "Nov 2024",
      "Solid_Waste_kg": 47570.1,
      "Yarn_Waste_kg": 19023.8,
      "Chemical_Waste_kg": 76618.4,
      "Wastewater_L": 13021361.8
    },
    {
      "month": 12,
      "period": "Dec 2024",
      "Solid_Waste_kg": 55161.4,
      "Yarn_Waste_kg": 22585.3,
      "Chemical_Waste_kg": 93734.9,
      "Wastewater_L": 15482515.5
    }
  ]
};

// ─── Daily historical waste series (731 days) ────────────────────────────────
const TREND_DAILY = {"2023": [{"date": "2023-01-01", "Solid_Waste_kg": 1318.2, "Yarn_Waste_kg": 531.8, "Chemical_Waste_kg": 2015.5, "Wastewater_L": 357670.9}, {"date": "2023-01-02", "Solid_Waste_kg": 2407.6, "Yarn_Waste_kg": 1004.7, "Chemical_Waste_kg": 4099.7, "Wastewater_L": 756331.8}, {"date": "2023-01-03", "Solid_Waste_kg": 1635.6, "Yarn_Waste_kg": 664.9, "Chemical_Waste_kg": 2764.4, "Wastewater_L": 425237.1}, {"date": "2023-01-04", "Solid_Waste_kg": 1603.8, "Yarn_Waste_kg": 737.2, "Chemical_Waste_kg": 2284.0, "Wastewater_L": 457131.4}, {"date": "2023-01-05", "Solid_Waste_kg": 2518.3, "Yarn_Waste_kg": 923.5, "Chemical_Waste_kg": 3409.7, "Wastewater_L": 546438.2}, {"date": "2023-01-06", "Solid_Waste_kg": 1404.8, "Yarn_Waste_kg": 581.7, "Chemical_Waste_kg": 2251.0, "Wastewater_L": 456906.9}, {"date": "2023-01-07", "Solid_Waste_kg": 1669.0, "Yarn_Waste_kg": 739.6, "Chemical_Waste_kg": 2456.7, "Wastewater_L": 450618.0}, {"date": "2023-01-08", "Solid_Waste_kg": 2325.2, "Yarn_Waste_kg": 766.9, "Chemical_Waste_kg": 3492.0, "Wastewater_L": 414066.2}, {"date": "2023-01-09", "Solid_Waste_kg": 2155.9, "Yarn_Waste_kg": 655.3, "Chemical_Waste_kg": 2987.3, "Wastewater_L": 557232.5}, {"date": "2023-01-10", "Solid_Waste_kg": 1524.9, "Yarn_Waste_kg": 555.2, "Chemical_Waste_kg": 2494.2, "Wastewater_L": 361375.5}, {"date": "2023-01-11", "Solid_Waste_kg": 1929.4, "Yarn_Waste_kg": 852.0, "Chemical_Waste_kg": 2825.2, "Wastewater_L": 597357.1}, {"date": "2023-01-12", "Solid_Waste_kg": 1582.3, "Yarn_Waste_kg": 688.0, "Chemical_Waste_kg": 3208.7, "Wastewater_L": 523335.5}, {"date": "2023-01-13", "Solid_Waste_kg": 1762.3, "Yarn_Waste_kg": 595.7, "Chemical_Waste_kg": 2399.0, "Wastewater_L": 519439.4}, {"date": "2023-01-14", "Solid_Waste_kg": 2091.2, "Yarn_Waste_kg": 785.3, "Chemical_Waste_kg": 3132.3, "Wastewater_L": 470401.5}, {"date": "2023-01-15", "Solid_Waste_kg": 1641.3, "Yarn_Waste_kg": 601.2, "Chemical_Waste_kg": 2774.0, "Wastewater_L": 588373.3}, {"date": "2023-01-16", "Solid_Waste_kg": 1607.1, "Yarn_Waste_kg": 678.6, "Chemical_Waste_kg": 2380.3, "Wastewater_L": 489859.3}, {"date": "2023-01-17", "Solid_Waste_kg": 1846.3, "Yarn_Waste_kg": 624.1, "Chemical_Waste_kg": 2327.9, "Wastewater_L": 381828.5}, {"date": "2023-01-18", "Solid_Waste_kg": 1886.5, "Yarn_Waste_kg": 670.4, "Chemical_Waste_kg": 2304.7, "Wastewater_L": 539265.1}, {"date": "2023-01-19", "Solid_Waste_kg": 1760.8, "Yarn_Waste_kg": 678.1, "Chemical_Waste_kg": 3346.2, "Wastewater_L": 497115.2}, {"date": "2023-01-20", "Solid_Waste_kg": 1420.2, "Yarn_Waste_kg": 681.2, "Chemical_Waste_kg": 2307.6, "Wastewater_L": 329343.8}, {"date": "2023-01-21", "Solid_Waste_kg": 1739.2, "Yarn_Waste_kg": 701.9, "Chemical_Waste_kg": 3217.5, "Wastewater_L": 697305.7}, {"date": "2023-01-22", "Solid_Waste_kg": 1970.1, "Yarn_Waste_kg": 777.7, "Chemical_Waste_kg": 3085.1, "Wastewater_L": 544518.9}, {"date": "2023-01-23", "Solid_Waste_kg": 2172.6, "Yarn_Waste_kg": 1058.0, "Chemical_Waste_kg": 2493.0, "Wastewater_L": 465032.3}, {"date": "2023-01-24", "Solid_Waste_kg": 2185.6, "Yarn_Waste_kg": 901.6, "Chemical_Waste_kg": 4371.7, "Wastewater_L": 632029.9}, {"date": "2023-01-25", "Solid_Waste_kg": 1611.7, "Yarn_Waste_kg": 633.5, "Chemical_Waste_kg": 2682.1, "Wastewater_L": 354891.0}, {"date": "2023-01-26", "Solid_Waste_kg": 1264.0, "Yarn_Waste_kg": 884.0, "Chemical_Waste_kg": 3386.5, "Wastewater_L": 577734.9}, {"date": "2023-01-27", "Solid_Waste_kg": 1407.7, "Yarn_Waste_kg": 622.3, "Chemical_Waste_kg": 2022.4, "Wastewater_L": 391912.4}, {"date": "2023-01-28", "Solid_Waste_kg": 1539.8, "Yarn_Waste_kg": 731.0, "Chemical_Waste_kg": 1694.4, "Wastewater_L": 383577.7}, {"date": "2023-01-29", "Solid_Waste_kg": 2262.1, "Yarn_Waste_kg": 772.8, "Chemical_Waste_kg": 2764.5, "Wastewater_L": 522661.8}, {"date": "2023-01-30", "Solid_Waste_kg": 2009.8, "Yarn_Waste_kg": 821.0, "Chemical_Waste_kg": 3201.9, "Wastewater_L": 653401.5}, {"date": "2023-01-31", "Solid_Waste_kg": 2198.2, "Yarn_Waste_kg": 772.5, "Chemical_Waste_kg": 2814.6, "Wastewater_L": 429538.9}, {"date": "2023-02-01", "Solid_Waste_kg": 1701.2, "Yarn_Waste_kg": 827.1, "Chemical_Waste_kg": 2888.2, "Wastewater_L": 509736.6}, {"date": "2023-02-02", "Solid_Waste_kg": 1598.1, "Yarn_Waste_kg": 684.4, "Chemical_Waste_kg": 3572.2, "Wastewater_L": 523147.0}, {"date": "2023-02-03", "Solid_Waste_kg": 1536.3, "Yarn_Waste_kg": 706.2, "Chemical_Waste_kg": 2673.4, "Wastewater_L": 441478.0}, {"date": "2023-02-04", "Solid_Waste_kg": 1787.4, "Yarn_Waste_kg": 856.0, "Chemical_Waste_kg": 2667.8, "Wastewater_L": 524357.8}, {"date": "2023-02-05", "Solid_Waste_kg": 1497.1, "Yarn_Waste_kg": 763.4, "Chemical_Waste_kg": 3185.1, "Wastewater_L": 567726.6}, {"date": "2023-02-06", "Solid_Waste_kg": 1915.4, "Yarn_Waste_kg": 1049.3, "Chemical_Waste_kg": 3042.2, "Wastewater_L": 686777.5}, {"date": "2023-02-07", "Solid_Waste_kg": 1604.1, "Yarn_Waste_kg": 541.5, "Chemical_Waste_kg": 2679.9, "Wastewater_L": 453182.0}, {"date": "2023-02-08", "Solid_Waste_kg": 2334.2, "Yarn_Waste_kg": 889.5, "Chemical_Waste_kg": 3964.2, "Wastewater_L": 671820.1}, {"date": "2023-02-09", "Solid_Waste_kg": 1737.7, "Yarn_Waste_kg": 795.3, "Chemical_Waste_kg": 2910.3, "Wastewater_L": 576165.1}, {"date": "2023-02-10", "Solid_Waste_kg": 1803.1, "Yarn_Waste_kg": 769.6, "Chemical_Waste_kg": 3276.1, "Wastewater_L": 590352.2}, {"date": "2023-02-11", "Solid_Waste_kg": 1910.3, "Yarn_Waste_kg": 757.6, "Chemical_Waste_kg": 2674.2, "Wastewater_L": 366786.9}, {"date": "2023-02-12", "Solid_Waste_kg": 2537.8, "Yarn_Waste_kg": 832.4, "Chemical_Waste_kg": 3116.0, "Wastewater_L": 715603.4}, {"date": "2023-02-13", "Solid_Waste_kg": 2074.7, "Yarn_Waste_kg": 779.3, "Chemical_Waste_kg": 3039.9, "Wastewater_L": 592000.3}, {"date": "2023-02-14", "Solid_Waste_kg": 1606.7, "Yarn_Waste_kg": 627.1, "Chemical_Waste_kg": 2538.3, "Wastewater_L": 481605.2}, {"date": "2023-02-15", "Solid_Waste_kg": 2053.2, "Yarn_Waste_kg": 759.4, "Chemical_Waste_kg": 3225.9, "Wastewater_L": 537671.8}, {"date": "2023-02-16", "Solid_Waste_kg": 1241.9, "Yarn_Waste_kg": 641.7, "Chemical_Waste_kg": 2356.1, "Wastewater_L": 324492.4}, {"date": "2023-02-17", "Solid_Waste_kg": 2159.7, "Yarn_Waste_kg": 708.6, "Chemical_Waste_kg": 4111.6, "Wastewater_L": 587579.1}, {"date": "2023-02-18", "Solid_Waste_kg": 2197.2, "Yarn_Waste_kg": 1000.3, "Chemical_Waste_kg": 4010.0, "Wastewater_L": 616279.0}, {"date": "2023-02-19", "Solid_Waste_kg": 2098.1, "Yarn_Waste_kg": 747.1, "Chemical_Waste_kg": 3840.4, "Wastewater_L": 511608.0}, {"date": "2023-02-20", "Solid_Waste_kg": 2152.3, "Yarn_Waste_kg": 1084.5, "Chemical_Waste_kg": 3553.6, "Wastewater_L": 483297.0}, {"date": "2023-02-21", "Solid_Waste_kg": 2159.1, "Yarn_Waste_kg": 1046.5, "Chemical_Waste_kg": 3201.8, "Wastewater_L": 564726.4}, {"date": "2023-02-22", "Solid_Waste_kg": 1536.4, "Yarn_Waste_kg": 859.1, "Chemical_Waste_kg": 3402.2, "Wastewater_L": 430595.4}, {"date": "2023-02-23", "Solid_Waste_kg": 1146.9, "Yarn_Waste_kg": 454.1, "Chemical_Waste_kg": 2217.6, "Wastewater_L": 297072.0}, {"date": "2023-02-24", "Solid_Waste_kg": 2062.8, "Yarn_Waste_kg": 935.4, "Chemical_Waste_kg": 3914.9, "Wastewater_L": 517633.5}, {"date": "2023-02-25", "Solid_Waste_kg": 1512.0, "Yarn_Waste_kg": 541.0, "Chemical_Waste_kg": 2322.2, "Wastewater_L": 362977.0}, {"date": "2023-02-26", "Solid_Waste_kg": 2043.1, "Yarn_Waste_kg": 720.6, "Chemical_Waste_kg": 3672.3, "Wastewater_L": 509164.8}, {"date": "2023-02-27", "Solid_Waste_kg": 1578.9, "Yarn_Waste_kg": 614.3, "Chemical_Waste_kg": 3069.2, "Wastewater_L": 468090.8}, {"date": "2023-02-28", "Solid_Waste_kg": 2520.0, "Yarn_Waste_kg": 685.0, "Chemical_Waste_kg": 2100.4, "Wastewater_L": 586208.6}, {"date": "2023-03-01", "Solid_Waste_kg": 1533.7, "Yarn_Waste_kg": 602.6, "Chemical_Waste_kg": 2179.2, "Wastewater_L": 437101.7}, {"date": "2023-03-02", "Solid_Waste_kg": 1337.4, "Yarn_Waste_kg": 840.3, "Chemical_Waste_kg": 3447.7, "Wastewater_L": 562160.1}, {"date": "2023-03-03", "Solid_Waste_kg": 1992.4, "Yarn_Waste_kg": 991.4, "Chemical_Waste_kg": 3532.2, "Wastewater_L": 578328.9}, {"date": "2023-03-04", "Solid_Waste_kg": 1883.3, "Yarn_Waste_kg": 824.0, "Chemical_Waste_kg": 3565.6, "Wastewater_L": 623430.6}, {"date": "2023-03-05", "Solid_Waste_kg": 2411.1, "Yarn_Waste_kg": 722.7, "Chemical_Waste_kg": 3878.1, "Wastewater_L": 685966.5}, {"date": "2023-03-06", "Solid_Waste_kg": 2078.3, "Yarn_Waste_kg": 886.1, "Chemical_Waste_kg": 2553.0, "Wastewater_L": 558827.6}, {"date": "2023-03-07", "Solid_Waste_kg": 2327.8, "Yarn_Waste_kg": 856.2, "Chemical_Waste_kg": 3207.6, "Wastewater_L": 515192.5}, {"date": "2023-03-08", "Solid_Waste_kg": 1638.9, "Yarn_Waste_kg": 817.9, "Chemical_Waste_kg": 2580.4, "Wastewater_L": 575429.4}, {"date": "2023-03-09", "Solid_Waste_kg": 1784.3, "Yarn_Waste_kg": 771.8, "Chemical_Waste_kg": 3389.6, "Wastewater_L": 479780.7}, {"date": "2023-03-10", "Solid_Waste_kg": 1992.6, "Yarn_Waste_kg": 688.3, "Chemical_Waste_kg": 3374.9, "Wastewater_L": 618906.4}, {"date": "2023-03-11", "Solid_Waste_kg": 2313.4, "Yarn_Waste_kg": 967.3, "Chemical_Waste_kg": 3684.2, "Wastewater_L": 574499.5}, {"date": "2023-03-12", "Solid_Waste_kg": 2674.7, "Yarn_Waste_kg": 975.2, "Chemical_Waste_kg": 4194.1, "Wastewater_L": 598216.4}, {"date": "2023-03-13", "Solid_Waste_kg": 2522.1, "Yarn_Waste_kg": 1053.9, "Chemical_Waste_kg": 3839.7, "Wastewater_L": 735410.3}, {"date": "2023-03-14", "Solid_Waste_kg": 1970.6, "Yarn_Waste_kg": 621.5, "Chemical_Waste_kg": 1997.4, "Wastewater_L": 383805.8}, {"date": "2023-03-15", "Solid_Waste_kg": 2345.0, "Yarn_Waste_kg": 1112.4, "Chemical_Waste_kg": 3614.3, "Wastewater_L": 556062.4}, {"date": "2023-03-16", "Solid_Waste_kg": 1647.6, "Yarn_Waste_kg": 601.1, "Chemical_Waste_kg": 2412.6, "Wastewater_L": 423526.0}, {"date": "2023-03-17", "Solid_Waste_kg": 1834.1, "Yarn_Waste_kg": 837.9, "Chemical_Waste_kg": 2920.3, "Wastewater_L": 521070.7}, {"date": "2023-03-18", "Solid_Waste_kg": 2351.0, "Yarn_Waste_kg": 1049.1, "Chemical_Waste_kg": 4070.0, "Wastewater_L": 520307.0}, {"date": "2023-03-19", "Solid_Waste_kg": 2413.4, "Yarn_Waste_kg": 772.3, "Chemical_Waste_kg": 3794.8, "Wastewater_L": 661412.8}, {"date": "2023-03-20", "Solid_Waste_kg": 2089.4, "Yarn_Waste_kg": 794.5, "Chemical_Waste_kg": 3320.6, "Wastewater_L": 482107.3}, {"date": "2023-03-21", "Solid_Waste_kg": 1223.2, "Yarn_Waste_kg": 633.1, "Chemical_Waste_kg": 2819.7, "Wastewater_L": 486525.3}, {"date": "2023-03-22", "Solid_Waste_kg": 1546.6, "Yarn_Waste_kg": 964.4, "Chemical_Waste_kg": 3573.1, "Wastewater_L": 495239.2}, {"date": "2023-03-23", "Solid_Waste_kg": 2365.7, "Yarn_Waste_kg": 937.8, "Chemical_Waste_kg": 4477.1, "Wastewater_L": 629653.3}, {"date": "2023-03-24", "Solid_Waste_kg": 1810.6, "Yarn_Waste_kg": 895.3, "Chemical_Waste_kg": 3106.7, "Wastewater_L": 525876.2}, {"date": "2023-03-25", "Solid_Waste_kg": 1998.4, "Yarn_Waste_kg": 889.7, "Chemical_Waste_kg": 3385.1, "Wastewater_L": 606983.7}, {"date": "2023-03-26", "Solid_Waste_kg": 1820.5, "Yarn_Waste_kg": 855.0, "Chemical_Waste_kg": 3543.9, "Wastewater_L": 469969.0}, {"date": "2023-03-27", "Solid_Waste_kg": 3113.6, "Yarn_Waste_kg": 1039.6, "Chemical_Waste_kg": 4279.9, "Wastewater_L": 562422.6}, {"date": "2023-03-28", "Solid_Waste_kg": 2285.6, "Yarn_Waste_kg": 751.3, "Chemical_Waste_kg": 3487.5, "Wastewater_L": 658102.1}, {"date": "2023-03-29", "Solid_Waste_kg": 2669.2, "Yarn_Waste_kg": 999.9, "Chemical_Waste_kg": 3782.0, "Wastewater_L": 626084.3}, {"date": "2023-03-30", "Solid_Waste_kg": 3095.3, "Yarn_Waste_kg": 1051.7, "Chemical_Waste_kg": 3997.3, "Wastewater_L": 773818.5}, {"date": "2023-03-31", "Solid_Waste_kg": 2052.8, "Yarn_Waste_kg": 847.5, "Chemical_Waste_kg": 3510.8, "Wastewater_L": 714898.6}, {"date": "2023-04-01", "Solid_Waste_kg": 2620.4, "Yarn_Waste_kg": 976.2, "Chemical_Waste_kg": 3716.1, "Wastewater_L": 671256.0}, {"date": "2023-04-02", "Solid_Waste_kg": 2813.5, "Yarn_Waste_kg": 1070.8, "Chemical_Waste_kg": 3754.4, "Wastewater_L": 810612.6}, {"date": "2023-04-03", "Solid_Waste_kg": 1743.5, "Yarn_Waste_kg": 968.2, "Chemical_Waste_kg": 2561.0, "Wastewater_L": 518591.7}, {"date": "2023-04-04", "Solid_Waste_kg": 2251.7, "Yarn_Waste_kg": 1098.6, "Chemical_Waste_kg": 4118.5, "Wastewater_L": 725041.3}, {"date": "2023-04-05", "Solid_Waste_kg": 1700.1, "Yarn_Waste_kg": 710.9, "Chemical_Waste_kg": 2706.4, "Wastewater_L": 601055.9}, {"date": "2023-04-06", "Solid_Waste_kg": 3073.7, "Yarn_Waste_kg": 1081.9, "Chemical_Waste_kg": 4655.3, "Wastewater_L": 638324.0}, {"date": "2023-04-07", "Solid_Waste_kg": 2983.6, "Yarn_Waste_kg": 1070.2, "Chemical_Waste_kg": 4322.6, "Wastewater_L": 846050.0}, {"date": "2023-04-08", "Solid_Waste_kg": 2793.4, "Yarn_Waste_kg": 1054.8, "Chemical_Waste_kg": 3644.9, "Wastewater_L": 698959.3}, {"date": "2023-04-09", "Solid_Waste_kg": 2481.8, "Yarn_Waste_kg": 1045.7, "Chemical_Waste_kg": 3428.5, "Wastewater_L": 557128.4}, {"date": "2023-04-10", "Solid_Waste_kg": 1919.8, "Yarn_Waste_kg": 874.9, "Chemical_Waste_kg": 3135.2, "Wastewater_L": 581249.0}, {"date": "2023-04-11", "Solid_Waste_kg": 1904.2, "Yarn_Waste_kg": 1175.4, "Chemical_Waste_kg": 3415.1, "Wastewater_L": 685845.9}, {"date": "2023-04-12", "Solid_Waste_kg": 2508.5, "Yarn_Waste_kg": 744.1, "Chemical_Waste_kg": 4010.7, "Wastewater_L": 706061.9}, {"date": "2023-04-13", "Solid_Waste_kg": 1910.3, "Yarn_Waste_kg": 979.4, "Chemical_Waste_kg": 4368.8, "Wastewater_L": 676766.9}, {"date": "2023-04-14", "Solid_Waste_kg": 3305.3, "Yarn_Waste_kg": 1325.6, "Chemical_Waste_kg": 4087.2, "Wastewater_L": 670710.3}, {"date": "2023-04-15", "Solid_Waste_kg": 2820.1, "Yarn_Waste_kg": 779.5, "Chemical_Waste_kg": 3881.5, "Wastewater_L": 701444.3}, {"date": "2023-04-16", "Solid_Waste_kg": 2663.9, "Yarn_Waste_kg": 966.0, "Chemical_Waste_kg": 4210.2, "Wastewater_L": 750566.5}, {"date": "2023-04-17", "Solid_Waste_kg": 2329.2, "Yarn_Waste_kg": 983.6, "Chemical_Waste_kg": 5693.9, "Wastewater_L": 763511.8}, {"date": "2023-04-18", "Solid_Waste_kg": 2114.1, "Yarn_Waste_kg": 839.5, "Chemical_Waste_kg": 4412.4, "Wastewater_L": 643078.3}, {"date": "2023-04-19", "Solid_Waste_kg": 2192.2, "Yarn_Waste_kg": 1114.8, "Chemical_Waste_kg": 4544.9, "Wastewater_L": 515129.9}, {"date": "2023-04-20", "Solid_Waste_kg": 2727.1, "Yarn_Waste_kg": 1112.5, "Chemical_Waste_kg": 4065.2, "Wastewater_L": 670364.9}, {"date": "2023-04-21", "Solid_Waste_kg": 2206.6, "Yarn_Waste_kg": 894.5, "Chemical_Waste_kg": 3299.9, "Wastewater_L": 674156.1}, {"date": "2023-04-22", "Solid_Waste_kg": 1899.9, "Yarn_Waste_kg": 825.2, "Chemical_Waste_kg": 3567.5, "Wastewater_L": 655244.8}, {"date": "2023-04-23", "Solid_Waste_kg": 2281.0, "Yarn_Waste_kg": 846.9, "Chemical_Waste_kg": 4123.5, "Wastewater_L": 770505.5}, {"date": "2023-04-24", "Solid_Waste_kg": 2528.6, "Yarn_Waste_kg": 1078.3, "Chemical_Waste_kg": 3816.9, "Wastewater_L": 796159.0}, {"date": "2023-04-25", "Solid_Waste_kg": 3221.8, "Yarn_Waste_kg": 1170.2, "Chemical_Waste_kg": 4772.7, "Wastewater_L": 777940.5}, {"date": "2023-04-26", "Solid_Waste_kg": 2419.6, "Yarn_Waste_kg": 764.5, "Chemical_Waste_kg": 3424.9, "Wastewater_L": 742776.6}, {"date": "2023-04-27", "Solid_Waste_kg": 2717.2, "Yarn_Waste_kg": 1179.8, "Chemical_Waste_kg": 4188.5, "Wastewater_L": 615227.1}, {"date": "2023-04-28", "Solid_Waste_kg": 2374.3, "Yarn_Waste_kg": 1074.6, "Chemical_Waste_kg": 3598.3, "Wastewater_L": 763568.0}, {"date": "2023-04-29", "Solid_Waste_kg": 2472.1, "Yarn_Waste_kg": 816.2, "Chemical_Waste_kg": 3520.2, "Wastewater_L": 809221.8}, {"date": "2023-04-30", "Solid_Waste_kg": 2651.6, "Yarn_Waste_kg": 1130.2, "Chemical_Waste_kg": 4507.5, "Wastewater_L": 692487.1}, {"date": "2023-05-01", "Solid_Waste_kg": 2911.8, "Yarn_Waste_kg": 1256.1, "Chemical_Waste_kg": 5518.2, "Wastewater_L": 738399.3}, {"date": "2023-05-02", "Solid_Waste_kg": 2990.2, "Yarn_Waste_kg": 1161.9, "Chemical_Waste_kg": 4853.3, "Wastewater_L": 781055.1}, {"date": "2023-05-03", "Solid_Waste_kg": 2071.2, "Yarn_Waste_kg": 890.7, "Chemical_Waste_kg": 3830.4, "Wastewater_L": 857849.1}, {"date": "2023-05-04", "Solid_Waste_kg": 2279.3, "Yarn_Waste_kg": 785.0, "Chemical_Waste_kg": 4475.0, "Wastewater_L": 766606.4}, {"date": "2023-05-05", "Solid_Waste_kg": 1935.7, "Yarn_Waste_kg": 1055.2, "Chemical_Waste_kg": 4455.3, "Wastewater_L": 729250.3}, {"date": "2023-05-06", "Solid_Waste_kg": 1822.1, "Yarn_Waste_kg": 776.1, "Chemical_Waste_kg": 3321.1, "Wastewater_L": 700086.9}, {"date": "2023-05-07", "Solid_Waste_kg": 2238.0, "Yarn_Waste_kg": 891.6, "Chemical_Waste_kg": 4000.5, "Wastewater_L": 751194.8}, {"date": "2023-05-08", "Solid_Waste_kg": 3177.6, "Yarn_Waste_kg": 1059.6, "Chemical_Waste_kg": 5016.4, "Wastewater_L": 774408.7}, {"date": "2023-05-09", "Solid_Waste_kg": 2485.1, "Yarn_Waste_kg": 1018.0, "Chemical_Waste_kg": 4332.3, "Wastewater_L": 730943.4}, {"date": "2023-05-10", "Solid_Waste_kg": 1864.6, "Yarn_Waste_kg": 797.9, "Chemical_Waste_kg": 3529.8, "Wastewater_L": 621305.4}, {"date": "2023-05-11", "Solid_Waste_kg": 2699.4, "Yarn_Waste_kg": 972.0, "Chemical_Waste_kg": 4512.3, "Wastewater_L": 834065.4}, {"date": "2023-05-12", "Solid_Waste_kg": 2146.9, "Yarn_Waste_kg": 1363.2, "Chemical_Waste_kg": 5081.0, "Wastewater_L": 677228.9}, {"date": "2023-05-13", "Solid_Waste_kg": 2086.5, "Yarn_Waste_kg": 710.2, "Chemical_Waste_kg": 3502.1, "Wastewater_L": 710159.7}, {"date": "2023-05-14", "Solid_Waste_kg": 2844.5, "Yarn_Waste_kg": 820.6, "Chemical_Waste_kg": 3755.4, "Wastewater_L": 820645.6}, {"date": "2023-05-15", "Solid_Waste_kg": 2798.0, "Yarn_Waste_kg": 1018.4, "Chemical_Waste_kg": 3718.3, "Wastewater_L": 595164.0}, {"date": "2023-05-16", "Solid_Waste_kg": 1687.8, "Yarn_Waste_kg": 762.5, "Chemical_Waste_kg": 3716.9, "Wastewater_L": 762450.1}, {"date": "2023-05-17", "Solid_Waste_kg": 1977.3, "Yarn_Waste_kg": 1286.8, "Chemical_Waste_kg": 4459.5, "Wastewater_L": 779889.6}, {"date": "2023-05-18", "Solid_Waste_kg": 2494.1, "Yarn_Waste_kg": 1274.4, "Chemical_Waste_kg": 5326.4, "Wastewater_L": 863346.1}, {"date": "2023-05-19", "Solid_Waste_kg": 2846.1, "Yarn_Waste_kg": 893.1, "Chemical_Waste_kg": 5301.0, "Wastewater_L": 808532.8}, {"date": "2023-05-20", "Solid_Waste_kg": 3535.4, "Yarn_Waste_kg": 1241.2, "Chemical_Waste_kg": 4858.8, "Wastewater_L": 692542.1}, {"date": "2023-05-21", "Solid_Waste_kg": 3221.1, "Yarn_Waste_kg": 1083.4, "Chemical_Waste_kg": 4607.9, "Wastewater_L": 665856.1}, {"date": "2023-05-22", "Solid_Waste_kg": 1890.2, "Yarn_Waste_kg": 861.5, "Chemical_Waste_kg": 3930.9, "Wastewater_L": 463128.5}, {"date": "2023-05-23", "Solid_Waste_kg": 3126.1, "Yarn_Waste_kg": 1269.9, "Chemical_Waste_kg": 4773.6, "Wastewater_L": 715887.6}, {"date": "2023-05-24", "Solid_Waste_kg": 2211.1, "Yarn_Waste_kg": 910.0, "Chemical_Waste_kg": 4948.5, "Wastewater_L": 699626.9}, {"date": "2023-05-25", "Solid_Waste_kg": 1971.0, "Yarn_Waste_kg": 775.8, "Chemical_Waste_kg": 3352.0, "Wastewater_L": 542395.3}, {"date": "2023-05-26", "Solid_Waste_kg": 2602.0, "Yarn_Waste_kg": 1240.3, "Chemical_Waste_kg": 3881.0, "Wastewater_L": 772266.4}, {"date": "2023-05-27", "Solid_Waste_kg": 2915.6, "Yarn_Waste_kg": 1250.0, "Chemical_Waste_kg": 3837.4, "Wastewater_L": 876385.2}, {"date": "2023-05-28", "Solid_Waste_kg": 2320.7, "Yarn_Waste_kg": 830.1, "Chemical_Waste_kg": 4238.8, "Wastewater_L": 614426.6}, {"date": "2023-05-29", "Solid_Waste_kg": 2347.7, "Yarn_Waste_kg": 1050.8, "Chemical_Waste_kg": 4903.4, "Wastewater_L": 687793.3}, {"date": "2023-05-30", "Solid_Waste_kg": 2033.7, "Yarn_Waste_kg": 731.3, "Chemical_Waste_kg": 2992.6, "Wastewater_L": 673918.1}, {"date": "2023-05-31", "Solid_Waste_kg": 1800.7, "Yarn_Waste_kg": 843.1, "Chemical_Waste_kg": 3060.8, "Wastewater_L": 494813.1}, {"date": "2023-06-01", "Solid_Waste_kg": 2447.1, "Yarn_Waste_kg": 1020.5, "Chemical_Waste_kg": 3905.9, "Wastewater_L": 762762.2}, {"date": "2023-06-02", "Solid_Waste_kg": 2305.9, "Yarn_Waste_kg": 879.6, "Chemical_Waste_kg": 3665.4, "Wastewater_L": 675138.0}, {"date": "2023-06-03", "Solid_Waste_kg": 2398.1, "Yarn_Waste_kg": 1307.5, "Chemical_Waste_kg": 4341.0, "Wastewater_L": 612251.5}, {"date": "2023-06-04", "Solid_Waste_kg": 2469.3, "Yarn_Waste_kg": 852.1, "Chemical_Waste_kg": 3611.2, "Wastewater_L": 676470.3}, {"date": "2023-06-05", "Solid_Waste_kg": 2705.8, "Yarn_Waste_kg": 1243.3, "Chemical_Waste_kg": 4747.4, "Wastewater_L": 924480.6}, {"date": "2023-06-06", "Solid_Waste_kg": 3134.7, "Yarn_Waste_kg": 924.6, "Chemical_Waste_kg": 3558.6, "Wastewater_L": 554759.3}, {"date": "2023-06-07", "Solid_Waste_kg": 3010.0, "Yarn_Waste_kg": 1090.0, "Chemical_Waste_kg": 4228.5, "Wastewater_L": 812644.0}, {"date": "2023-06-08", "Solid_Waste_kg": 2504.0, "Yarn_Waste_kg": 1014.9, "Chemical_Waste_kg": 4047.6, "Wastewater_L": 681711.0}, {"date": "2023-06-09", "Solid_Waste_kg": 3808.8, "Yarn_Waste_kg": 1366.0, "Chemical_Waste_kg": 4690.2, "Wastewater_L": 988129.8}, {"date": "2023-06-10", "Solid_Waste_kg": 3217.0, "Yarn_Waste_kg": 925.6, "Chemical_Waste_kg": 3756.1, "Wastewater_L": 764683.1}, {"date": "2023-06-11", "Solid_Waste_kg": 2963.1, "Yarn_Waste_kg": 1090.8, "Chemical_Waste_kg": 3554.6, "Wastewater_L": 709336.2}, {"date": "2023-06-12", "Solid_Waste_kg": 2860.5, "Yarn_Waste_kg": 1260.2, "Chemical_Waste_kg": 4907.7, "Wastewater_L": 711949.2}, {"date": "2023-06-13", "Solid_Waste_kg": 3086.2, "Yarn_Waste_kg": 880.0, "Chemical_Waste_kg": 4723.6, "Wastewater_L": 837371.1}, {"date": "2023-06-14", "Solid_Waste_kg": 2977.4, "Yarn_Waste_kg": 903.4, "Chemical_Waste_kg": 4933.2, "Wastewater_L": 622588.6}, {"date": "2023-06-15", "Solid_Waste_kg": 2243.2, "Yarn_Waste_kg": 1186.9, "Chemical_Waste_kg": 4127.6, "Wastewater_L": 496978.9}, {"date": "2023-06-16", "Solid_Waste_kg": 2934.7, "Yarn_Waste_kg": 1149.8, "Chemical_Waste_kg": 4537.3, "Wastewater_L": 592234.8}, {"date": "2023-06-17", "Solid_Waste_kg": 2075.8, "Yarn_Waste_kg": 1187.9, "Chemical_Waste_kg": 3813.5, "Wastewater_L": 685130.4}, {"date": "2023-06-18", "Solid_Waste_kg": 3224.6, "Yarn_Waste_kg": 1295.6, "Chemical_Waste_kg": 5389.2, "Wastewater_L": 936391.5}, {"date": "2023-06-19", "Solid_Waste_kg": 3391.7, "Yarn_Waste_kg": 1243.1, "Chemical_Waste_kg": 4856.0, "Wastewater_L": 706952.6}, {"date": "2023-06-20", "Solid_Waste_kg": 2533.2, "Yarn_Waste_kg": 1255.6, "Chemical_Waste_kg": 4435.6, "Wastewater_L": 748540.4}, {"date": "2023-06-21", "Solid_Waste_kg": 3292.5, "Yarn_Waste_kg": 1153.8, "Chemical_Waste_kg": 4774.8, "Wastewater_L": 656333.0}, {"date": "2023-06-22", "Solid_Waste_kg": 3037.0, "Yarn_Waste_kg": 1066.9, "Chemical_Waste_kg": 5446.1, "Wastewater_L": 682725.2}, {"date": "2023-06-23", "Solid_Waste_kg": 3051.4, "Yarn_Waste_kg": 1243.8, "Chemical_Waste_kg": 4687.1, "Wastewater_L": 693247.5}, {"date": "2023-06-24", "Solid_Waste_kg": 2655.0, "Yarn_Waste_kg": 996.6, "Chemical_Waste_kg": 3298.8, "Wastewater_L": 640485.9}, {"date": "2023-06-25", "Solid_Waste_kg": 1646.6, "Yarn_Waste_kg": 1019.3, "Chemical_Waste_kg": 4593.1, "Wastewater_L": 786136.0}, {"date": "2023-06-26", "Solid_Waste_kg": 2032.2, "Yarn_Waste_kg": 731.2, "Chemical_Waste_kg": 3562.1, "Wastewater_L": 682433.0}, {"date": "2023-06-27", "Solid_Waste_kg": 3688.4, "Yarn_Waste_kg": 1299.9, "Chemical_Waste_kg": 4022.6, "Wastewater_L": 777873.3}, {"date": "2023-06-28", "Solid_Waste_kg": 2519.2, "Yarn_Waste_kg": 959.7, "Chemical_Waste_kg": 4192.9, "Wastewater_L": 694733.6}, {"date": "2023-06-29", "Solid_Waste_kg": 1890.9, "Yarn_Waste_kg": 1013.7, "Chemical_Waste_kg": 4976.5, "Wastewater_L": 606830.5}, {"date": "2023-06-30", "Solid_Waste_kg": 3170.7, "Yarn_Waste_kg": 1259.9, "Chemical_Waste_kg": 5302.4, "Wastewater_L": 601909.8}, {"date": "2023-07-01", "Solid_Waste_kg": 2369.4, "Yarn_Waste_kg": 989.3, "Chemical_Waste_kg": 4010.7, "Wastewater_L": 801215.5}, {"date": "2023-07-02", "Solid_Waste_kg": 1877.6, "Yarn_Waste_kg": 745.0, "Chemical_Waste_kg": 3824.4, "Wastewater_L": 588331.2}, {"date": "2023-07-03", "Solid_Waste_kg": 2108.6, "Yarn_Waste_kg": 1075.6, "Chemical_Waste_kg": 3979.3, "Wastewater_L": 796801.9}, {"date": "2023-07-04", "Solid_Waste_kg": 2167.2, "Yarn_Waste_kg": 1088.1, "Chemical_Waste_kg": 4067.7, "Wastewater_L": 597004.7}, {"date": "2023-07-05", "Solid_Waste_kg": 2461.1, "Yarn_Waste_kg": 1024.1, "Chemical_Waste_kg": 4695.3, "Wastewater_L": 582793.7}, {"date": "2023-07-06", "Solid_Waste_kg": 2951.4, "Yarn_Waste_kg": 927.7, "Chemical_Waste_kg": 3872.7, "Wastewater_L": 614499.1}, {"date": "2023-07-07", "Solid_Waste_kg": 2657.0, "Yarn_Waste_kg": 957.7, "Chemical_Waste_kg": 3956.1, "Wastewater_L": 885670.0}, {"date": "2023-07-08", "Solid_Waste_kg": 3521.1, "Yarn_Waste_kg": 1304.3, "Chemical_Waste_kg": 4622.2, "Wastewater_L": 635640.1}, {"date": "2023-07-09", "Solid_Waste_kg": 3060.6, "Yarn_Waste_kg": 939.5, "Chemical_Waste_kg": 5670.7, "Wastewater_L": 758841.1}, {"date": "2023-07-10", "Solid_Waste_kg": 2775.8, "Yarn_Waste_kg": 871.4, "Chemical_Waste_kg": 4578.4, "Wastewater_L": 763067.1}, {"date": "2023-07-11", "Solid_Waste_kg": 1651.1, "Yarn_Waste_kg": 1210.0, "Chemical_Waste_kg": 3617.9, "Wastewater_L": 687430.4}, {"date": "2023-07-12", "Solid_Waste_kg": 2100.8, "Yarn_Waste_kg": 1160.8, "Chemical_Waste_kg": 4782.1, "Wastewater_L": 670037.0}, {"date": "2023-07-13", "Solid_Waste_kg": 3217.4, "Yarn_Waste_kg": 1148.5, "Chemical_Waste_kg": 5444.4, "Wastewater_L": 710573.5}, {"date": "2023-07-14", "Solid_Waste_kg": 2899.6, "Yarn_Waste_kg": 1381.6, "Chemical_Waste_kg": 4370.5, "Wastewater_L": 954791.0}, {"date": "2023-07-15", "Solid_Waste_kg": 2110.2, "Yarn_Waste_kg": 966.2, "Chemical_Waste_kg": 3803.7, "Wastewater_L": 841092.5}, {"date": "2023-07-16", "Solid_Waste_kg": 3308.4, "Yarn_Waste_kg": 1423.7, "Chemical_Waste_kg": 6349.7, "Wastewater_L": 1029946.5}, {"date": "2023-07-17", "Solid_Waste_kg": 2554.4, "Yarn_Waste_kg": 1195.9, "Chemical_Waste_kg": 4095.0, "Wastewater_L": 634192.3}, {"date": "2023-07-18", "Solid_Waste_kg": 1947.2, "Yarn_Waste_kg": 858.4, "Chemical_Waste_kg": 2797.3, "Wastewater_L": 669882.4}, {"date": "2023-07-19", "Solid_Waste_kg": 2705.7, "Yarn_Waste_kg": 1271.9, "Chemical_Waste_kg": 4430.1, "Wastewater_L": 594058.2}, {"date": "2023-07-20", "Solid_Waste_kg": 2204.7, "Yarn_Waste_kg": 1160.0, "Chemical_Waste_kg": 4517.7, "Wastewater_L": 641704.7}, {"date": "2023-07-21", "Solid_Waste_kg": 2802.8, "Yarn_Waste_kg": 939.8, "Chemical_Waste_kg": 3866.3, "Wastewater_L": 624664.6}, {"date": "2023-07-22", "Solid_Waste_kg": 3120.9, "Yarn_Waste_kg": 1051.4, "Chemical_Waste_kg": 4806.4, "Wastewater_L": 652081.5}, {"date": "2023-07-23", "Solid_Waste_kg": 3248.0, "Yarn_Waste_kg": 1458.7, "Chemical_Waste_kg": 5208.6, "Wastewater_L": 992349.7}, {"date": "2023-07-24", "Solid_Waste_kg": 2912.9, "Yarn_Waste_kg": 989.3, "Chemical_Waste_kg": 5056.7, "Wastewater_L": 688616.9}, {"date": "2023-07-25", "Solid_Waste_kg": 2094.7, "Yarn_Waste_kg": 1156.4, "Chemical_Waste_kg": 4124.7, "Wastewater_L": 715154.6}, {"date": "2023-07-26", "Solid_Waste_kg": 2064.0, "Yarn_Waste_kg": 723.0, "Chemical_Waste_kg": 3208.2, "Wastewater_L": 561313.4}, {"date": "2023-07-27", "Solid_Waste_kg": 2483.2, "Yarn_Waste_kg": 791.5, "Chemical_Waste_kg": 4076.4, "Wastewater_L": 611450.7}, {"date": "2023-07-28", "Solid_Waste_kg": 2470.7, "Yarn_Waste_kg": 969.6, "Chemical_Waste_kg": 2851.2, "Wastewater_L": 666065.2}, {"date": "2023-07-29", "Solid_Waste_kg": 2589.5, "Yarn_Waste_kg": 999.9, "Chemical_Waste_kg": 3496.9, "Wastewater_L": 769916.4}, {"date": "2023-07-30", "Solid_Waste_kg": 2296.0, "Yarn_Waste_kg": 889.6, "Chemical_Waste_kg": 4255.8, "Wastewater_L": 730697.3}, {"date": "2023-07-31", "Solid_Waste_kg": 2279.0, "Yarn_Waste_kg": 1047.5, "Chemical_Waste_kg": 4435.6, "Wastewater_L": 670811.5}, {"date": "2023-08-01", "Solid_Waste_kg": 2018.0, "Yarn_Waste_kg": 743.2, "Chemical_Waste_kg": 3134.5, "Wastewater_L": 607540.1}, {"date": "2023-08-02", "Solid_Waste_kg": 2637.8, "Yarn_Waste_kg": 1220.9, "Chemical_Waste_kg": 4577.7, "Wastewater_L": 727129.6}, {"date": "2023-08-03", "Solid_Waste_kg": 1593.7, "Yarn_Waste_kg": 820.6, "Chemical_Waste_kg": 3592.1, "Wastewater_L": 527022.3}, {"date": "2023-08-04", "Solid_Waste_kg": 2090.9, "Yarn_Waste_kg": 761.9, "Chemical_Waste_kg": 3923.7, "Wastewater_L": 661423.7}, {"date": "2023-08-05", "Solid_Waste_kg": 1503.5, "Yarn_Waste_kg": 1031.3, "Chemical_Waste_kg": 3040.0, "Wastewater_L": 748738.3}, {"date": "2023-08-06", "Solid_Waste_kg": 1283.3, "Yarn_Waste_kg": 724.5, "Chemical_Waste_kg": 2943.6, "Wastewater_L": 448598.4}, {"date": "2023-08-07", "Solid_Waste_kg": 2233.5, "Yarn_Waste_kg": 833.8, "Chemical_Waste_kg": 3628.2, "Wastewater_L": 570641.1}, {"date": "2023-08-08", "Solid_Waste_kg": 2891.2, "Yarn_Waste_kg": 1126.7, "Chemical_Waste_kg": 4921.0, "Wastewater_L": 536988.4}, {"date": "2023-08-09", "Solid_Waste_kg": 2432.7, "Yarn_Waste_kg": 877.6, "Chemical_Waste_kg": 4276.7, "Wastewater_L": 817796.1}, {"date": "2023-08-10", "Solid_Waste_kg": 3397.9, "Yarn_Waste_kg": 1121.2, "Chemical_Waste_kg": 3725.5, "Wastewater_L": 825908.1}, {"date": "2023-08-11", "Solid_Waste_kg": 2630.4, "Yarn_Waste_kg": 907.6, "Chemical_Waste_kg": 3611.1, "Wastewater_L": 749875.1}, {"date": "2023-08-12", "Solid_Waste_kg": 2428.9, "Yarn_Waste_kg": 968.7, "Chemical_Waste_kg": 3977.1, "Wastewater_L": 564470.2}, {"date": "2023-08-13", "Solid_Waste_kg": 2556.4, "Yarn_Waste_kg": 973.5, "Chemical_Waste_kg": 3935.8, "Wastewater_L": 752466.9}, {"date": "2023-08-14", "Solid_Waste_kg": 1889.4, "Yarn_Waste_kg": 777.1, "Chemical_Waste_kg": 3892.9, "Wastewater_L": 595882.4}, {"date": "2023-08-15", "Solid_Waste_kg": 3118.3, "Yarn_Waste_kg": 1386.6, "Chemical_Waste_kg": 6110.9, "Wastewater_L": 868845.0}, {"date": "2023-08-16", "Solid_Waste_kg": 2242.3, "Yarn_Waste_kg": 983.3, "Chemical_Waste_kg": 4191.3, "Wastewater_L": 600476.6}, {"date": "2023-08-17", "Solid_Waste_kg": 2960.7, "Yarn_Waste_kg": 851.7, "Chemical_Waste_kg": 4849.1, "Wastewater_L": 615602.9}, {"date": "2023-08-18", "Solid_Waste_kg": 2538.1, "Yarn_Waste_kg": 1234.3, "Chemical_Waste_kg": 4957.6, "Wastewater_L": 755717.3}, {"date": "2023-08-19", "Solid_Waste_kg": 1815.6, "Yarn_Waste_kg": 714.3, "Chemical_Waste_kg": 3222.5, "Wastewater_L": 571942.0}, {"date": "2023-08-20", "Solid_Waste_kg": 2625.8, "Yarn_Waste_kg": 823.6, "Chemical_Waste_kg": 4042.4, "Wastewater_L": 742238.8}, {"date": "2023-08-21", "Solid_Waste_kg": 3431.5, "Yarn_Waste_kg": 1107.4, "Chemical_Waste_kg": 4319.1, "Wastewater_L": 722592.6}, {"date": "2023-08-22", "Solid_Waste_kg": 2089.1, "Yarn_Waste_kg": 794.9, "Chemical_Waste_kg": 3056.9, "Wastewater_L": 423221.3}, {"date": "2023-08-23", "Solid_Waste_kg": 2761.5, "Yarn_Waste_kg": 1008.6, "Chemical_Waste_kg": 3731.3, "Wastewater_L": 653710.7}, {"date": "2023-08-24", "Solid_Waste_kg": 1666.5, "Yarn_Waste_kg": 722.3, "Chemical_Waste_kg": 3937.9, "Wastewater_L": 627443.4}, {"date": "2023-08-25", "Solid_Waste_kg": 2563.9, "Yarn_Waste_kg": 926.5, "Chemical_Waste_kg": 3340.1, "Wastewater_L": 724810.9}, {"date": "2023-08-26", "Solid_Waste_kg": 1879.6, "Yarn_Waste_kg": 1064.1, "Chemical_Waste_kg": 2908.2, "Wastewater_L": 596568.0}, {"date": "2023-08-27", "Solid_Waste_kg": 2620.4, "Yarn_Waste_kg": 1009.1, "Chemical_Waste_kg": 3932.4, "Wastewater_L": 553221.1}, {"date": "2023-08-28", "Solid_Waste_kg": 2026.2, "Yarn_Waste_kg": 967.9, "Chemical_Waste_kg": 4274.6, "Wastewater_L": 691922.6}, {"date": "2023-08-29", "Solid_Waste_kg": 2576.8, "Yarn_Waste_kg": 1078.7, "Chemical_Waste_kg": 3349.4, "Wastewater_L": 737546.6}, {"date": "2023-08-30", "Solid_Waste_kg": 2372.2, "Yarn_Waste_kg": 1101.2, "Chemical_Waste_kg": 4199.5, "Wastewater_L": 513144.9}, {"date": "2023-08-31", "Solid_Waste_kg": 2474.0, "Yarn_Waste_kg": 1109.9, "Chemical_Waste_kg": 4908.8, "Wastewater_L": 539335.2}, {"date": "2023-09-01", "Solid_Waste_kg": 1638.1, "Yarn_Waste_kg": 639.9, "Chemical_Waste_kg": 2680.0, "Wastewater_L": 441276.7}, {"date": "2023-09-02", "Solid_Waste_kg": 2150.6, "Yarn_Waste_kg": 682.3, "Chemical_Waste_kg": 3652.7, "Wastewater_L": 618140.2}, {"date": "2023-09-03", "Solid_Waste_kg": 2734.4, "Yarn_Waste_kg": 1223.7, "Chemical_Waste_kg": 4307.7, "Wastewater_L": 646114.7}, {"date": "2023-09-04", "Solid_Waste_kg": 2393.7, "Yarn_Waste_kg": 812.6, "Chemical_Waste_kg": 3010.0, "Wastewater_L": 569276.2}, {"date": "2023-09-05", "Solid_Waste_kg": 2603.0, "Yarn_Waste_kg": 1035.5, "Chemical_Waste_kg": 4330.1, "Wastewater_L": 733771.5}, {"date": "2023-09-06", "Solid_Waste_kg": 1745.0, "Yarn_Waste_kg": 828.8, "Chemical_Waste_kg": 3601.7, "Wastewater_L": 472270.3}, {"date": "2023-09-07", "Solid_Waste_kg": 1811.4, "Yarn_Waste_kg": 660.0, "Chemical_Waste_kg": 3347.1, "Wastewater_L": 416959.6}, {"date": "2023-09-08", "Solid_Waste_kg": 1721.8, "Yarn_Waste_kg": 860.9, "Chemical_Waste_kg": 3575.5, "Wastewater_L": 720232.0}, {"date": "2023-09-09", "Solid_Waste_kg": 1786.2, "Yarn_Waste_kg": 960.4, "Chemical_Waste_kg": 3686.6, "Wastewater_L": 586113.2}, {"date": "2023-09-10", "Solid_Waste_kg": 2114.0, "Yarn_Waste_kg": 767.7, "Chemical_Waste_kg": 3104.4, "Wastewater_L": 490081.9}, {"date": "2023-09-11", "Solid_Waste_kg": 1832.0, "Yarn_Waste_kg": 756.6, "Chemical_Waste_kg": 2532.5, "Wastewater_L": 426469.7}, {"date": "2023-09-12", "Solid_Waste_kg": 2534.8, "Yarn_Waste_kg": 799.6, "Chemical_Waste_kg": 3668.2, "Wastewater_L": 602204.0}, {"date": "2023-09-13", "Solid_Waste_kg": 1922.0, "Yarn_Waste_kg": 800.8, "Chemical_Waste_kg": 2775.0, "Wastewater_L": 519307.8}, {"date": "2023-09-14", "Solid_Waste_kg": 2719.5, "Yarn_Waste_kg": 1002.6, "Chemical_Waste_kg": 2779.0, "Wastewater_L": 644750.2}, {"date": "2023-09-15", "Solid_Waste_kg": 1495.7, "Yarn_Waste_kg": 626.3, "Chemical_Waste_kg": 2472.7, "Wastewater_L": 453909.9}, {"date": "2023-09-16", "Solid_Waste_kg": 1787.6, "Yarn_Waste_kg": 740.0, "Chemical_Waste_kg": 3489.3, "Wastewater_L": 480040.4}, {"date": "2023-09-17", "Solid_Waste_kg": 1625.5, "Yarn_Waste_kg": 704.9, "Chemical_Waste_kg": 3396.9, "Wastewater_L": 523507.3}, {"date": "2023-09-18", "Solid_Waste_kg": 2198.7, "Yarn_Waste_kg": 1229.8, "Chemical_Waste_kg": 4153.5, "Wastewater_L": 621066.5}, {"date": "2023-09-19", "Solid_Waste_kg": 1688.6, "Yarn_Waste_kg": 842.6, "Chemical_Waste_kg": 3231.3, "Wastewater_L": 556417.8}, {"date": "2023-09-20", "Solid_Waste_kg": 1488.0, "Yarn_Waste_kg": 645.5, "Chemical_Waste_kg": 3349.2, "Wastewater_L": 458295.2}, {"date": "2023-09-21", "Solid_Waste_kg": 1824.9, "Yarn_Waste_kg": 640.9, "Chemical_Waste_kg": 2640.9, "Wastewater_L": 547222.6}, {"date": "2023-09-22", "Solid_Waste_kg": 2282.7, "Yarn_Waste_kg": 899.0, "Chemical_Waste_kg": 2484.3, "Wastewater_L": 476158.3}, {"date": "2023-09-23", "Solid_Waste_kg": 2323.0, "Yarn_Waste_kg": 1126.5, "Chemical_Waste_kg": 3442.8, "Wastewater_L": 702684.7}, {"date": "2023-09-24", "Solid_Waste_kg": 1313.6, "Yarn_Waste_kg": 788.2, "Chemical_Waste_kg": 2777.3, "Wastewater_L": 598013.4}, {"date": "2023-09-25", "Solid_Waste_kg": 1437.7, "Yarn_Waste_kg": 675.6, "Chemical_Waste_kg": 2734.1, "Wastewater_L": 520653.2}, {"date": "2023-09-26", "Solid_Waste_kg": 1545.6, "Yarn_Waste_kg": 583.2, "Chemical_Waste_kg": 2945.4, "Wastewater_L": 471288.4}, {"date": "2023-09-27", "Solid_Waste_kg": 2016.4, "Yarn_Waste_kg": 830.7, "Chemical_Waste_kg": 2527.0, "Wastewater_L": 561500.8}, {"date": "2023-09-28", "Solid_Waste_kg": 1883.9, "Yarn_Waste_kg": 690.1, "Chemical_Waste_kg": 3578.4, "Wastewater_L": 581197.2}, {"date": "2023-09-29", "Solid_Waste_kg": 2086.8, "Yarn_Waste_kg": 779.2, "Chemical_Waste_kg": 4159.4, "Wastewater_L": 641150.2}, {"date": "2023-09-30", "Solid_Waste_kg": 1976.7, "Yarn_Waste_kg": 801.4, "Chemical_Waste_kg": 4051.5, "Wastewater_L": 610101.6}, {"date": "2023-10-01", "Solid_Waste_kg": 1910.1, "Yarn_Waste_kg": 614.6, "Chemical_Waste_kg": 2343.4, "Wastewater_L": 537844.0}, {"date": "2023-10-02", "Solid_Waste_kg": 2582.5, "Yarn_Waste_kg": 824.3, "Chemical_Waste_kg": 2634.2, "Wastewater_L": 565952.6}, {"date": "2023-10-03", "Solid_Waste_kg": 1837.8, "Yarn_Waste_kg": 728.2, "Chemical_Waste_kg": 3108.8, "Wastewater_L": 615000.8}, {"date": "2023-10-04", "Solid_Waste_kg": 2185.6, "Yarn_Waste_kg": 826.2, "Chemical_Waste_kg": 4012.4, "Wastewater_L": 524429.3}, {"date": "2023-10-05", "Solid_Waste_kg": 1564.7, "Yarn_Waste_kg": 763.8, "Chemical_Waste_kg": 3030.8, "Wastewater_L": 547405.9}, {"date": "2023-10-06", "Solid_Waste_kg": 2125.1, "Yarn_Waste_kg": 710.4, "Chemical_Waste_kg": 3488.3, "Wastewater_L": 612993.4}, {"date": "2023-10-07", "Solid_Waste_kg": 2284.1, "Yarn_Waste_kg": 867.8, "Chemical_Waste_kg": 3406.6, "Wastewater_L": 609402.9}, {"date": "2023-10-08", "Solid_Waste_kg": 1953.0, "Yarn_Waste_kg": 792.4, "Chemical_Waste_kg": 3433.6, "Wastewater_L": 548070.9}, {"date": "2023-10-09", "Solid_Waste_kg": 1320.6, "Yarn_Waste_kg": 590.0, "Chemical_Waste_kg": 3045.6, "Wastewater_L": 494674.1}, {"date": "2023-10-10", "Solid_Waste_kg": 2701.3, "Yarn_Waste_kg": 1044.4, "Chemical_Waste_kg": 4827.8, "Wastewater_L": 817021.8}, {"date": "2023-10-11", "Solid_Waste_kg": 1631.1, "Yarn_Waste_kg": 635.6, "Chemical_Waste_kg": 2994.7, "Wastewater_L": 442768.7}, {"date": "2023-10-12", "Solid_Waste_kg": 1695.4, "Yarn_Waste_kg": 764.3, "Chemical_Waste_kg": 2945.4, "Wastewater_L": 524477.2}, {"date": "2023-10-13", "Solid_Waste_kg": 2008.8, "Yarn_Waste_kg": 785.9, "Chemical_Waste_kg": 3748.1, "Wastewater_L": 577808.6}, {"date": "2023-10-14", "Solid_Waste_kg": 921.4, "Yarn_Waste_kg": 467.9, "Chemical_Waste_kg": 2677.5, "Wastewater_L": 350889.2}, {"date": "2023-10-15", "Solid_Waste_kg": 1736.9, "Yarn_Waste_kg": 818.7, "Chemical_Waste_kg": 2329.0, "Wastewater_L": 479432.5}, {"date": "2023-10-16", "Solid_Waste_kg": 1867.3, "Yarn_Waste_kg": 937.3, "Chemical_Waste_kg": 3620.8, "Wastewater_L": 496815.4}, {"date": "2023-10-17", "Solid_Waste_kg": 2124.6, "Yarn_Waste_kg": 979.0, "Chemical_Waste_kg": 3059.1, "Wastewater_L": 558501.9}, {"date": "2023-10-18", "Solid_Waste_kg": 2030.1, "Yarn_Waste_kg": 624.5, "Chemical_Waste_kg": 2739.9, "Wastewater_L": 510583.4}, {"date": "2023-10-19", "Solid_Waste_kg": 1696.3, "Yarn_Waste_kg": 688.4, "Chemical_Waste_kg": 2745.2, "Wastewater_L": 318100.6}, {"date": "2023-10-20", "Solid_Waste_kg": 1896.1, "Yarn_Waste_kg": 779.5, "Chemical_Waste_kg": 4387.3, "Wastewater_L": 588720.0}, {"date": "2023-10-21", "Solid_Waste_kg": 1968.2, "Yarn_Waste_kg": 836.4, "Chemical_Waste_kg": 3582.6, "Wastewater_L": 543902.2}, {"date": "2023-10-22", "Solid_Waste_kg": 1550.1, "Yarn_Waste_kg": 762.4, "Chemical_Waste_kg": 2841.0, "Wastewater_L": 551604.3}, {"date": "2023-10-23", "Solid_Waste_kg": 2294.2, "Yarn_Waste_kg": 897.3, "Chemical_Waste_kg": 3986.0, "Wastewater_L": 657653.0}, {"date": "2023-10-24", "Solid_Waste_kg": 1872.0, "Yarn_Waste_kg": 1001.0, "Chemical_Waste_kg": 4340.7, "Wastewater_L": 584052.3}, {"date": "2023-10-25", "Solid_Waste_kg": 2375.7, "Yarn_Waste_kg": 715.8, "Chemical_Waste_kg": 3016.2, "Wastewater_L": 631150.4}, {"date": "2023-10-26", "Solid_Waste_kg": 2048.8, "Yarn_Waste_kg": 984.7, "Chemical_Waste_kg": 4059.9, "Wastewater_L": 464764.1}, {"date": "2023-10-27", "Solid_Waste_kg": 2225.9, "Yarn_Waste_kg": 767.7, "Chemical_Waste_kg": 2987.7, "Wastewater_L": 484011.5}, {"date": "2023-10-28", "Solid_Waste_kg": 1517.1, "Yarn_Waste_kg": 547.9, "Chemical_Waste_kg": 2732.5, "Wastewater_L": 328768.4}, {"date": "2023-10-29", "Solid_Waste_kg": 1698.9, "Yarn_Waste_kg": 616.8, "Chemical_Waste_kg": 2524.4, "Wastewater_L": 586836.8}, {"date": "2023-10-30", "Solid_Waste_kg": 2117.5, "Yarn_Waste_kg": 858.7, "Chemical_Waste_kg": 3678.2, "Wastewater_L": 572415.3}, {"date": "2023-10-31", "Solid_Waste_kg": 1843.4, "Yarn_Waste_kg": 659.1, "Chemical_Waste_kg": 3516.8, "Wastewater_L": 606299.0}, {"date": "2023-11-01", "Solid_Waste_kg": 1415.7, "Yarn_Waste_kg": 765.3, "Chemical_Waste_kg": 2242.1, "Wastewater_L": 344105.0}, {"date": "2023-11-02", "Solid_Waste_kg": 2353.1, "Yarn_Waste_kg": 784.0, "Chemical_Waste_kg": 3705.5, "Wastewater_L": 658882.0}, {"date": "2023-11-03", "Solid_Waste_kg": 1154.2, "Yarn_Waste_kg": 528.7, "Chemical_Waste_kg": 2006.9, "Wastewater_L": 279865.1}, {"date": "2023-11-04", "Solid_Waste_kg": 2148.4, "Yarn_Waste_kg": 563.1, "Chemical_Waste_kg": 2988.5, "Wastewater_L": 507036.8}, {"date": "2023-11-05", "Solid_Waste_kg": 1788.9, "Yarn_Waste_kg": 613.1, "Chemical_Waste_kg": 2176.4, "Wastewater_L": 375411.6}, {"date": "2023-11-06", "Solid_Waste_kg": 1911.3, "Yarn_Waste_kg": 635.3, "Chemical_Waste_kg": 2172.6, "Wastewater_L": 457199.4}, {"date": "2023-11-07", "Solid_Waste_kg": 2387.5, "Yarn_Waste_kg": 705.9, "Chemical_Waste_kg": 3705.8, "Wastewater_L": 601847.5}, {"date": "2023-11-08", "Solid_Waste_kg": 1916.5, "Yarn_Waste_kg": 563.9, "Chemical_Waste_kg": 1938.9, "Wastewater_L": 514748.3}, {"date": "2023-11-09", "Solid_Waste_kg": 2315.0, "Yarn_Waste_kg": 926.0, "Chemical_Waste_kg": 3407.6, "Wastewater_L": 495259.1}, {"date": "2023-11-10", "Solid_Waste_kg": 1491.2, "Yarn_Waste_kg": 738.6, "Chemical_Waste_kg": 3016.7, "Wastewater_L": 554115.6}, {"date": "2023-11-11", "Solid_Waste_kg": 1480.7, "Yarn_Waste_kg": 819.9, "Chemical_Waste_kg": 2896.8, "Wastewater_L": 385411.2}, {"date": "2023-11-12", "Solid_Waste_kg": 1890.4, "Yarn_Waste_kg": 589.6, "Chemical_Waste_kg": 2128.5, "Wastewater_L": 472371.5}, {"date": "2023-11-13", "Solid_Waste_kg": 2093.9, "Yarn_Waste_kg": 790.1, "Chemical_Waste_kg": 2928.6, "Wastewater_L": 489126.5}, {"date": "2023-11-14", "Solid_Waste_kg": 1312.3, "Yarn_Waste_kg": 492.2, "Chemical_Waste_kg": 2266.8, "Wastewater_L": 391705.9}, {"date": "2023-11-15", "Solid_Waste_kg": 1207.2, "Yarn_Waste_kg": 497.2, "Chemical_Waste_kg": 2391.4, "Wastewater_L": 425639.4}, {"date": "2023-11-16", "Solid_Waste_kg": 1410.6, "Yarn_Waste_kg": 626.4, "Chemical_Waste_kg": 2226.7, "Wastewater_L": 435917.1}, {"date": "2023-11-17", "Solid_Waste_kg": 2249.1, "Yarn_Waste_kg": 754.7, "Chemical_Waste_kg": 3371.7, "Wastewater_L": 430431.8}, {"date": "2023-11-18", "Solid_Waste_kg": 1774.1, "Yarn_Waste_kg": 762.8, "Chemical_Waste_kg": 2997.6, "Wastewater_L": 486302.1}, {"date": "2023-11-19", "Solid_Waste_kg": 1497.3, "Yarn_Waste_kg": 754.2, "Chemical_Waste_kg": 2742.8, "Wastewater_L": 414531.7}, {"date": "2023-11-20", "Solid_Waste_kg": 1622.3, "Yarn_Waste_kg": 631.6, "Chemical_Waste_kg": 3109.8, "Wastewater_L": 505777.3}, {"date": "2023-11-21", "Solid_Waste_kg": 2183.1, "Yarn_Waste_kg": 717.4, "Chemical_Waste_kg": 2653.9, "Wastewater_L": 615428.2}, {"date": "2023-11-22", "Solid_Waste_kg": 2116.6, "Yarn_Waste_kg": 765.1, "Chemical_Waste_kg": 2871.2, "Wastewater_L": 565683.7}, {"date": "2023-11-23", "Solid_Waste_kg": 1114.0, "Yarn_Waste_kg": 589.1, "Chemical_Waste_kg": 2048.2, "Wastewater_L": 341725.8}, {"date": "2023-11-24", "Solid_Waste_kg": 1263.8, "Yarn_Waste_kg": 648.4, "Chemical_Waste_kg": 2631.8, "Wastewater_L": 473343.7}, {"date": "2023-11-25", "Solid_Waste_kg": 2221.8, "Yarn_Waste_kg": 933.8, "Chemical_Waste_kg": 4521.8, "Wastewater_L": 621111.8}, {"date": "2023-11-26", "Solid_Waste_kg": 1621.0, "Yarn_Waste_kg": 723.3, "Chemical_Waste_kg": 2649.6, "Wastewater_L": 549158.6}, {"date": "2023-11-27", "Solid_Waste_kg": 2055.2, "Yarn_Waste_kg": 778.8, "Chemical_Waste_kg": 2836.9, "Wastewater_L": 486338.4}, {"date": "2023-11-28", "Solid_Waste_kg": 1587.4, "Yarn_Waste_kg": 658.3, "Chemical_Waste_kg": 2093.6, "Wastewater_L": 425125.5}, {"date": "2023-11-29", "Solid_Waste_kg": 1776.5, "Yarn_Waste_kg": 762.5, "Chemical_Waste_kg": 3040.4, "Wastewater_L": 548123.0}, {"date": "2023-11-30", "Solid_Waste_kg": 1387.5, "Yarn_Waste_kg": 552.9, "Chemical_Waste_kg": 2049.0, "Wastewater_L": 364665.4}, {"date": "2023-12-01", "Solid_Waste_kg": 2010.9, "Yarn_Waste_kg": 582.9, "Chemical_Waste_kg": 2491.5, "Wastewater_L": 398703.9}, {"date": "2023-12-02", "Solid_Waste_kg": 1863.0, "Yarn_Waste_kg": 694.8, "Chemical_Waste_kg": 2962.0, "Wastewater_L": 432106.2}, {"date": "2023-12-03", "Solid_Waste_kg": 1230.9, "Yarn_Waste_kg": 564.9, "Chemical_Waste_kg": 2415.5, "Wastewater_L": 563736.8}, {"date": "2023-12-04", "Solid_Waste_kg": 1725.1, "Yarn_Waste_kg": 894.6, "Chemical_Waste_kg": 2996.1, "Wastewater_L": 698445.6}, {"date": "2023-12-05", "Solid_Waste_kg": 2523.9, "Yarn_Waste_kg": 923.0, "Chemical_Waste_kg": 3777.5, "Wastewater_L": 667373.0}, {"date": "2023-12-06", "Solid_Waste_kg": 2165.3, "Yarn_Waste_kg": 675.1, "Chemical_Waste_kg": 3295.6, "Wastewater_L": 377786.0}, {"date": "2023-12-07", "Solid_Waste_kg": 1389.2, "Yarn_Waste_kg": 695.9, "Chemical_Waste_kg": 2138.1, "Wastewater_L": 364136.1}, {"date": "2023-12-08", "Solid_Waste_kg": 2118.3, "Yarn_Waste_kg": 771.6, "Chemical_Waste_kg": 3185.9, "Wastewater_L": 556131.6}, {"date": "2023-12-09", "Solid_Waste_kg": 1641.7, "Yarn_Waste_kg": 723.3, "Chemical_Waste_kg": 2312.7, "Wastewater_L": 528872.8}, {"date": "2023-12-10", "Solid_Waste_kg": 1479.4, "Yarn_Waste_kg": 630.6, "Chemical_Waste_kg": 2923.2, "Wastewater_L": 630613.8}, {"date": "2023-12-11", "Solid_Waste_kg": 2699.1, "Yarn_Waste_kg": 980.5, "Chemical_Waste_kg": 2759.6, "Wastewater_L": 462712.1}, {"date": "2023-12-12", "Solid_Waste_kg": 2008.0, "Yarn_Waste_kg": 820.3, "Chemical_Waste_kg": 3293.1, "Wastewater_L": 432971.3}, {"date": "2023-12-13", "Solid_Waste_kg": 1539.7, "Yarn_Waste_kg": 936.9, "Chemical_Waste_kg": 3337.0, "Wastewater_L": 516429.7}, {"date": "2023-12-14", "Solid_Waste_kg": 2421.9, "Yarn_Waste_kg": 711.8, "Chemical_Waste_kg": 4190.0, "Wastewater_L": 511453.4}, {"date": "2023-12-15", "Solid_Waste_kg": 2224.1, "Yarn_Waste_kg": 896.0, "Chemical_Waste_kg": 3839.2, "Wastewater_L": 466619.1}, {"date": "2023-12-16", "Solid_Waste_kg": 1594.1, "Yarn_Waste_kg": 624.4, "Chemical_Waste_kg": 2723.9, "Wastewater_L": 506252.7}, {"date": "2023-12-17", "Solid_Waste_kg": 1527.2, "Yarn_Waste_kg": 805.3, "Chemical_Waste_kg": 2668.5, "Wastewater_L": 367898.9}, {"date": "2023-12-18", "Solid_Waste_kg": 1344.2, "Yarn_Waste_kg": 773.5, "Chemical_Waste_kg": 2896.1, "Wastewater_L": 408214.0}, {"date": "2023-12-19", "Solid_Waste_kg": 1981.8, "Yarn_Waste_kg": 691.2, "Chemical_Waste_kg": 3248.0, "Wastewater_L": 557208.3}, {"date": "2023-12-20", "Solid_Waste_kg": 1160.1, "Yarn_Waste_kg": 580.1, "Chemical_Waste_kg": 2617.8, "Wastewater_L": 348033.0}, {"date": "2023-12-21", "Solid_Waste_kg": 1946.5, "Yarn_Waste_kg": 806.7, "Chemical_Waste_kg": 3085.5, "Wastewater_L": 511232.0}, {"date": "2023-12-22", "Solid_Waste_kg": 1767.0, "Yarn_Waste_kg": 698.6, "Chemical_Waste_kg": 2496.4, "Wastewater_L": 417496.7}, {"date": "2023-12-23", "Solid_Waste_kg": 1525.7, "Yarn_Waste_kg": 604.0, "Chemical_Waste_kg": 2233.5, "Wastewater_L": 321191.3}, {"date": "2023-12-24", "Solid_Waste_kg": 1641.3, "Yarn_Waste_kg": 685.1, "Chemical_Waste_kg": 2722.7, "Wastewater_L": 441391.0}, {"date": "2023-12-25", "Solid_Waste_kg": 1457.8, "Yarn_Waste_kg": 681.9, "Chemical_Waste_kg": 2733.8, "Wastewater_L": 539837.3}, {"date": "2023-12-26", "Solid_Waste_kg": 1895.7, "Yarn_Waste_kg": 736.6, "Chemical_Waste_kg": 2689.0, "Wastewater_L": 557702.0}, {"date": "2023-12-27", "Solid_Waste_kg": 2020.3, "Yarn_Waste_kg": 985.6, "Chemical_Waste_kg": 3933.5, "Wastewater_L": 615988.3}, {"date": "2023-12-28", "Solid_Waste_kg": 1969.6, "Yarn_Waste_kg": 707.1, "Chemical_Waste_kg": 3730.2, "Wastewater_L": 507619.1}, {"date": "2023-12-29", "Solid_Waste_kg": 1862.8, "Yarn_Waste_kg": 653.3, "Chemical_Waste_kg": 2797.6, "Wastewater_L": 448628.2}, {"date": "2023-12-30", "Solid_Waste_kg": 1422.5, "Yarn_Waste_kg": 821.6, "Chemical_Waste_kg": 2941.5, "Wastewater_L": 450835.1}, {"date": "2023-12-31", "Solid_Waste_kg": 1723.0, "Yarn_Waste_kg": 923.5, "Chemical_Waste_kg": 3173.3, "Wastewater_L": 455538.9}], "2024": [{"date": "2024-01-01", "Solid_Waste_kg": 2358.8, "Yarn_Waste_kg": 778.1, "Chemical_Waste_kg": 3108.1, "Wastewater_L": 471984.8}, {"date": "2024-01-02", "Solid_Waste_kg": 1273.8, "Yarn_Waste_kg": 714.8, "Chemical_Waste_kg": 2526.4, "Wastewater_L": 429913.8}, {"date": "2024-01-03", "Solid_Waste_kg": 1410.8, "Yarn_Waste_kg": 523.8, "Chemical_Waste_kg": 2561.8, "Wastewater_L": 469913.6}, {"date": "2024-01-04", "Solid_Waste_kg": 1200.2, "Yarn_Waste_kg": 626.9, "Chemical_Waste_kg": 2007.9, "Wastewater_L": 352058.6}, {"date": "2024-01-05", "Solid_Waste_kg": 2363.7, "Yarn_Waste_kg": 996.7, "Chemical_Waste_kg": 3536.9, "Wastewater_L": 563222.8}, {"date": "2024-01-06", "Solid_Waste_kg": 1404.2, "Yarn_Waste_kg": 542.8, "Chemical_Waste_kg": 2635.2, "Wastewater_L": 383240.3}, {"date": "2024-01-07", "Solid_Waste_kg": 2487.4, "Yarn_Waste_kg": 752.9, "Chemical_Waste_kg": 2891.4, "Wastewater_L": 472765.3}, {"date": "2024-01-08", "Solid_Waste_kg": 1245.1, "Yarn_Waste_kg": 608.0, "Chemical_Waste_kg": 1988.1, "Wastewater_L": 401821.9}, {"date": "2024-01-09", "Solid_Waste_kg": 1914.5, "Yarn_Waste_kg": 550.0, "Chemical_Waste_kg": 2258.3, "Wastewater_L": 525696.4}, {"date": "2024-01-10", "Solid_Waste_kg": 1084.8, "Yarn_Waste_kg": 486.1, "Chemical_Waste_kg": 2454.9, "Wastewater_L": 421076.7}, {"date": "2024-01-11", "Solid_Waste_kg": 2243.3, "Yarn_Waste_kg": 881.8, "Chemical_Waste_kg": 2904.1, "Wastewater_L": 538059.1}, {"date": "2024-01-12", "Solid_Waste_kg": 2067.4, "Yarn_Waste_kg": 733.5, "Chemical_Waste_kg": 3083.4, "Wastewater_L": 340579.3}, {"date": "2024-01-13", "Solid_Waste_kg": 1854.6, "Yarn_Waste_kg": 783.4, "Chemical_Waste_kg": 2786.4, "Wastewater_L": 465299.5}, {"date": "2024-01-14", "Solid_Waste_kg": 1488.1, "Yarn_Waste_kg": 485.5, "Chemical_Waste_kg": 2475.8, "Wastewater_L": 335336.9}, {"date": "2024-01-15", "Solid_Waste_kg": 1852.6, "Yarn_Waste_kg": 697.7, "Chemical_Waste_kg": 2307.1, "Wastewater_L": 403953.2}, {"date": "2024-01-16", "Solid_Waste_kg": 1525.2, "Yarn_Waste_kg": 705.3, "Chemical_Waste_kg": 2941.3, "Wastewater_L": 478228.4}, {"date": "2024-01-17", "Solid_Waste_kg": 2012.4, "Yarn_Waste_kg": 788.8, "Chemical_Waste_kg": 3190.9, "Wastewater_L": 408253.8}, {"date": "2024-01-18", "Solid_Waste_kg": 1386.9, "Yarn_Waste_kg": 462.0, "Chemical_Waste_kg": 2065.2, "Wastewater_L": 442741.5}, {"date": "2024-01-19", "Solid_Waste_kg": 2543.6, "Yarn_Waste_kg": 814.9, "Chemical_Waste_kg": 2397.0, "Wastewater_L": 461961.7}, {"date": "2024-01-20", "Solid_Waste_kg": 1668.5, "Yarn_Waste_kg": 777.3, "Chemical_Waste_kg": 2703.0, "Wastewater_L": 426381.2}, {"date": "2024-01-21", "Solid_Waste_kg": 1102.1, "Yarn_Waste_kg": 631.9, "Chemical_Waste_kg": 2130.4, "Wastewater_L": 449799.9}, {"date": "2024-01-22", "Solid_Waste_kg": 1745.1, "Yarn_Waste_kg": 556.6, "Chemical_Waste_kg": 2079.0, "Wastewater_L": 315102.8}, {"date": "2024-01-23", "Solid_Waste_kg": 1460.3, "Yarn_Waste_kg": 969.2, "Chemical_Waste_kg": 3600.9, "Wastewater_L": 547282.8}, {"date": "2024-01-24", "Solid_Waste_kg": 1962.1, "Yarn_Waste_kg": 801.0, "Chemical_Waste_kg": 2689.1, "Wastewater_L": 612478.1}, {"date": "2024-01-25", "Solid_Waste_kg": 1709.6, "Yarn_Waste_kg": 622.4, "Chemical_Waste_kg": 3013.2, "Wastewater_L": 554724.3}, {"date": "2024-01-26", "Solid_Waste_kg": 2180.7, "Yarn_Waste_kg": 811.7, "Chemical_Waste_kg": 3462.7, "Wastewater_L": 571763.5}, {"date": "2024-01-27", "Solid_Waste_kg": 1618.0, "Yarn_Waste_kg": 753.3, "Chemical_Waste_kg": 3078.1, "Wastewater_L": 630478.0}, {"date": "2024-01-28", "Solid_Waste_kg": 1498.1, "Yarn_Waste_kg": 667.8, "Chemical_Waste_kg": 3238.0, "Wastewater_L": 477475.1}, {"date": "2024-01-29", "Solid_Waste_kg": 1913.8, "Yarn_Waste_kg": 880.9, "Chemical_Waste_kg": 3540.9, "Wastewater_L": 586620.6}, {"date": "2024-01-30", "Solid_Waste_kg": 2228.4, "Yarn_Waste_kg": 816.7, "Chemical_Waste_kg": 3061.8, "Wastewater_L": 429546.6}, {"date": "2024-01-31", "Solid_Waste_kg": 1514.7, "Yarn_Waste_kg": 786.6, "Chemical_Waste_kg": 2596.4, "Wastewater_L": 502274.0}, {"date": "2024-02-01", "Solid_Waste_kg": 1938.7, "Yarn_Waste_kg": 646.4, "Chemical_Waste_kg": 2668.8, "Wastewater_L": 483143.4}, {"date": "2024-02-02", "Solid_Waste_kg": 1658.3, "Yarn_Waste_kg": 630.8, "Chemical_Waste_kg": 3386.7, "Wastewater_L": 515536.4}, {"date": "2024-02-03", "Solid_Waste_kg": 1542.5, "Yarn_Waste_kg": 723.1, "Chemical_Waste_kg": 2836.0, "Wastewater_L": 355317.5}, {"date": "2024-02-04", "Solid_Waste_kg": 1472.8, "Yarn_Waste_kg": 587.4, "Chemical_Waste_kg": 2930.4, "Wastewater_L": 476960.9}, {"date": "2024-02-05", "Solid_Waste_kg": 2304.3, "Yarn_Waste_kg": 840.5, "Chemical_Waste_kg": 2817.0, "Wastewater_L": 386320.0}, {"date": "2024-02-06", "Solid_Waste_kg": 1698.1, "Yarn_Waste_kg": 747.8, "Chemical_Waste_kg": 2882.9, "Wastewater_L": 403715.9}, {"date": "2024-02-07", "Solid_Waste_kg": 1988.9, "Yarn_Waste_kg": 839.0, "Chemical_Waste_kg": 3375.2, "Wastewater_L": 462136.3}, {"date": "2024-02-08", "Solid_Waste_kg": 1598.3, "Yarn_Waste_kg": 555.2, "Chemical_Waste_kg": 2719.0, "Wastewater_L": 508105.9}, {"date": "2024-02-09", "Solid_Waste_kg": 1728.0, "Yarn_Waste_kg": 602.7, "Chemical_Waste_kg": 3071.2, "Wastewater_L": 539448.9}, {"date": "2024-02-10", "Solid_Waste_kg": 2106.4, "Yarn_Waste_kg": 769.8, "Chemical_Waste_kg": 2808.8, "Wastewater_L": 583962.1}, {"date": "2024-02-11", "Solid_Waste_kg": 1926.9, "Yarn_Waste_kg": 811.7, "Chemical_Waste_kg": 2544.8, "Wastewater_L": 409251.5}, {"date": "2024-02-12", "Solid_Waste_kg": 1980.3, "Yarn_Waste_kg": 941.6, "Chemical_Waste_kg": 3511.4, "Wastewater_L": 663972.4}, {"date": "2024-02-13", "Solid_Waste_kg": 1977.0, "Yarn_Waste_kg": 889.9, "Chemical_Waste_kg": 3446.3, "Wastewater_L": 420869.4}, {"date": "2024-02-14", "Solid_Waste_kg": 1563.0, "Yarn_Waste_kg": 647.7, "Chemical_Waste_kg": 2206.4, "Wastewater_L": 455466.4}, {"date": "2024-02-15", "Solid_Waste_kg": 2024.1, "Yarn_Waste_kg": 715.4, "Chemical_Waste_kg": 2795.1, "Wastewater_L": 454827.5}, {"date": "2024-02-16", "Solid_Waste_kg": 2216.2, "Yarn_Waste_kg": 574.3, "Chemical_Waste_kg": 2339.2, "Wastewater_L": 382067.0}, {"date": "2024-02-17", "Solid_Waste_kg": 1659.1, "Yarn_Waste_kg": 743.7, "Chemical_Waste_kg": 3343.1, "Wastewater_L": 512709.4}, {"date": "2024-02-18", "Solid_Waste_kg": 1275.0, "Yarn_Waste_kg": 629.5, "Chemical_Waste_kg": 2068.2, "Wastewater_L": 609028.2}, {"date": "2024-02-19", "Solid_Waste_kg": 1615.9, "Yarn_Waste_kg": 815.8, "Chemical_Waste_kg": 3309.9, "Wastewater_L": 530851.5}, {"date": "2024-02-20", "Solid_Waste_kg": 1709.6, "Yarn_Waste_kg": 721.3, "Chemical_Waste_kg": 2587.7, "Wastewater_L": 474681.2}, {"date": "2024-02-21", "Solid_Waste_kg": 2388.6, "Yarn_Waste_kg": 976.5, "Chemical_Waste_kg": 3840.1, "Wastewater_L": 521591.6}, {"date": "2024-02-22", "Solid_Waste_kg": 1376.4, "Yarn_Waste_kg": 740.8, "Chemical_Waste_kg": 3118.1, "Wastewater_L": 464536.3}, {"date": "2024-02-23", "Solid_Waste_kg": 1444.3, "Yarn_Waste_kg": 758.1, "Chemical_Waste_kg": 2776.1, "Wastewater_L": 317203.5}, {"date": "2024-02-24", "Solid_Waste_kg": 1885.5, "Yarn_Waste_kg": 759.4, "Chemical_Waste_kg": 3214.4, "Wastewater_L": 457003.8}, {"date": "2024-02-25", "Solid_Waste_kg": 2061.6, "Yarn_Waste_kg": 774.6, "Chemical_Waste_kg": 3715.6, "Wastewater_L": 482968.9}, {"date": "2024-02-26", "Solid_Waste_kg": 1727.0, "Yarn_Waste_kg": 789.5, "Chemical_Waste_kg": 2614.0, "Wastewater_L": 456729.6}, {"date": "2024-02-27", "Solid_Waste_kg": 2438.1, "Yarn_Waste_kg": 1204.0, "Chemical_Waste_kg": 4898.9, "Wastewater_L": 772911.4}, {"date": "2024-02-28", "Solid_Waste_kg": 1970.5, "Yarn_Waste_kg": 1010.9, "Chemical_Waste_kg": 3155.0, "Wastewater_L": 540558.8}, {"date": "2024-02-29", "Solid_Waste_kg": 1725.7, "Yarn_Waste_kg": 620.1, "Chemical_Waste_kg": 3339.0, "Wastewater_L": 509693.5}, {"date": "2024-03-01", "Solid_Waste_kg": 2799.5, "Yarn_Waste_kg": 942.0, "Chemical_Waste_kg": 2946.8, "Wastewater_L": 578076.5}, {"date": "2024-03-02", "Solid_Waste_kg": 3223.6, "Yarn_Waste_kg": 965.1, "Chemical_Waste_kg": 3700.2, "Wastewater_L": 695281.4}, {"date": "2024-03-03", "Solid_Waste_kg": 2074.4, "Yarn_Waste_kg": 916.3, "Chemical_Waste_kg": 4336.6, "Wastewater_L": 547167.2}, {"date": "2024-03-04", "Solid_Waste_kg": 2238.5, "Yarn_Waste_kg": 964.2, "Chemical_Waste_kg": 3967.4, "Wastewater_L": 581828.7}, {"date": "2024-03-05", "Solid_Waste_kg": 2050.8, "Yarn_Waste_kg": 1118.2, "Chemical_Waste_kg": 4391.2, "Wastewater_L": 571500.7}, {"date": "2024-03-06", "Solid_Waste_kg": 2012.7, "Yarn_Waste_kg": 1128.8, "Chemical_Waste_kg": 3297.0, "Wastewater_L": 583040.5}, {"date": "2024-03-07", "Solid_Waste_kg": 1718.4, "Yarn_Waste_kg": 725.9, "Chemical_Waste_kg": 2890.3, "Wastewater_L": 590197.3}, {"date": "2024-03-08", "Solid_Waste_kg": 1645.1, "Yarn_Waste_kg": 561.4, "Chemical_Waste_kg": 2896.1, "Wastewater_L": 411211.4}, {"date": "2024-03-09", "Solid_Waste_kg": 2289.4, "Yarn_Waste_kg": 946.4, "Chemical_Waste_kg": 3108.1, "Wastewater_L": 586529.7}, {"date": "2024-03-10", "Solid_Waste_kg": 1913.2, "Yarn_Waste_kg": 656.3, "Chemical_Waste_kg": 2352.3, "Wastewater_L": 607652.3}, {"date": "2024-03-11", "Solid_Waste_kg": 1840.3, "Yarn_Waste_kg": 946.4, "Chemical_Waste_kg": 3183.0, "Wastewater_L": 613167.8}, {"date": "2024-03-12", "Solid_Waste_kg": 2560.1, "Yarn_Waste_kg": 883.4, "Chemical_Waste_kg": 3396.8, "Wastewater_L": 685645.0}, {"date": "2024-03-13", "Solid_Waste_kg": 1646.8, "Yarn_Waste_kg": 527.5, "Chemical_Waste_kg": 2254.1, "Wastewater_L": 465922.0}, {"date": "2024-03-14", "Solid_Waste_kg": 1469.7, "Yarn_Waste_kg": 794.2, "Chemical_Waste_kg": 2970.3, "Wastewater_L": 550402.4}, {"date": "2024-03-15", "Solid_Waste_kg": 1999.2, "Yarn_Waste_kg": 1003.2, "Chemical_Waste_kg": 3638.5, "Wastewater_L": 639056.7}, {"date": "2024-03-16", "Solid_Waste_kg": 1638.3, "Yarn_Waste_kg": 672.3, "Chemical_Waste_kg": 3054.7, "Wastewater_L": 455883.6}, {"date": "2024-03-17", "Solid_Waste_kg": 1778.9, "Yarn_Waste_kg": 893.7, "Chemical_Waste_kg": 3195.2, "Wastewater_L": 343790.6}, {"date": "2024-03-18", "Solid_Waste_kg": 2129.9, "Yarn_Waste_kg": 821.3, "Chemical_Waste_kg": 4221.8, "Wastewater_L": 652851.3}, {"date": "2024-03-19", "Solid_Waste_kg": 2189.7, "Yarn_Waste_kg": 849.7, "Chemical_Waste_kg": 4586.0, "Wastewater_L": 669301.6}, {"date": "2024-03-20", "Solid_Waste_kg": 2376.4, "Yarn_Waste_kg": 859.5, "Chemical_Waste_kg": 3729.4, "Wastewater_L": 606701.8}, {"date": "2024-03-21", "Solid_Waste_kg": 1473.0, "Yarn_Waste_kg": 871.9, "Chemical_Waste_kg": 3444.7, "Wastewater_L": 600898.0}, {"date": "2024-03-22", "Solid_Waste_kg": 1964.1, "Yarn_Waste_kg": 765.5, "Chemical_Waste_kg": 3651.9, "Wastewater_L": 530172.4}, {"date": "2024-03-23", "Solid_Waste_kg": 1779.7, "Yarn_Waste_kg": 792.2, "Chemical_Waste_kg": 3473.5, "Wastewater_L": 539392.9}, {"date": "2024-03-24", "Solid_Waste_kg": 1286.9, "Yarn_Waste_kg": 524.3, "Chemical_Waste_kg": 1829.0, "Wastewater_L": 320425.9}, {"date": "2024-03-25", "Solid_Waste_kg": 1557.4, "Yarn_Waste_kg": 808.8, "Chemical_Waste_kg": 3353.3, "Wastewater_L": 592654.9}, {"date": "2024-03-26", "Solid_Waste_kg": 1526.8, "Yarn_Waste_kg": 618.7, "Chemical_Waste_kg": 3299.6, "Wastewater_L": 536981.9}, {"date": "2024-03-27", "Solid_Waste_kg": 2102.2, "Yarn_Waste_kg": 929.8, "Chemical_Waste_kg": 3868.1, "Wastewater_L": 533408.2}, {"date": "2024-03-28", "Solid_Waste_kg": 1657.8, "Yarn_Waste_kg": 730.5, "Chemical_Waste_kg": 2855.0, "Wastewater_L": 455165.4}, {"date": "2024-03-29", "Solid_Waste_kg": 2015.8, "Yarn_Waste_kg": 636.5, "Chemical_Waste_kg": 2870.2, "Wastewater_L": 464592.2}, {"date": "2024-03-30", "Solid_Waste_kg": 2319.3, "Yarn_Waste_kg": 807.8, "Chemical_Waste_kg": 3585.6, "Wastewater_L": 713021.3}, {"date": "2024-03-31", "Solid_Waste_kg": 1647.3, "Yarn_Waste_kg": 893.4, "Chemical_Waste_kg": 2825.2, "Wastewater_L": 585292.8}, {"date": "2024-04-01", "Solid_Waste_kg": 2395.9, "Yarn_Waste_kg": 812.8, "Chemical_Waste_kg": 3624.7, "Wastewater_L": 580611.1}, {"date": "2024-04-02", "Solid_Waste_kg": 1999.7, "Yarn_Waste_kg": 714.8, "Chemical_Waste_kg": 2425.2, "Wastewater_L": 540003.8}, {"date": "2024-04-03", "Solid_Waste_kg": 1985.7, "Yarn_Waste_kg": 832.5, "Chemical_Waste_kg": 3399.6, "Wastewater_L": 483503.4}, {"date": "2024-04-04", "Solid_Waste_kg": 2098.8, "Yarn_Waste_kg": 938.9, "Chemical_Waste_kg": 4469.0, "Wastewater_L": 528136.1}, {"date": "2024-04-05", "Solid_Waste_kg": 1899.3, "Yarn_Waste_kg": 826.0, "Chemical_Waste_kg": 3604.5, "Wastewater_L": 539843.5}, {"date": "2024-04-06", "Solid_Waste_kg": 2986.4, "Yarn_Waste_kg": 1390.6, "Chemical_Waste_kg": 4275.5, "Wastewater_L": 650422.9}, {"date": "2024-04-07", "Solid_Waste_kg": 2865.9, "Yarn_Waste_kg": 1319.6, "Chemical_Waste_kg": 4531.8, "Wastewater_L": 822996.1}, {"date": "2024-04-08", "Solid_Waste_kg": 2766.1, "Yarn_Waste_kg": 1016.4, "Chemical_Waste_kg": 3458.2, "Wastewater_L": 640966.3}, {"date": "2024-04-09", "Solid_Waste_kg": 2385.1, "Yarn_Waste_kg": 991.1, "Chemical_Waste_kg": 3848.2, "Wastewater_L": 632896.1}, {"date": "2024-04-10", "Solid_Waste_kg": 2396.4, "Yarn_Waste_kg": 930.7, "Chemical_Waste_kg": 2942.9, "Wastewater_L": 603369.9}, {"date": "2024-04-11", "Solid_Waste_kg": 2150.4, "Yarn_Waste_kg": 699.0, "Chemical_Waste_kg": 3222.8, "Wastewater_L": 422022.3}, {"date": "2024-04-12", "Solid_Waste_kg": 2641.2, "Yarn_Waste_kg": 964.5, "Chemical_Waste_kg": 2747.1, "Wastewater_L": 729107.5}, {"date": "2024-04-13", "Solid_Waste_kg": 2440.6, "Yarn_Waste_kg": 963.6, "Chemical_Waste_kg": 2629.7, "Wastewater_L": 514352.5}, {"date": "2024-04-14", "Solid_Waste_kg": 2065.0, "Yarn_Waste_kg": 1007.4, "Chemical_Waste_kg": 3190.6, "Wastewater_L": 645438.9}, {"date": "2024-04-15", "Solid_Waste_kg": 2049.8, "Yarn_Waste_kg": 931.3, "Chemical_Waste_kg": 3482.9, "Wastewater_L": 548085.4}, {"date": "2024-04-16", "Solid_Waste_kg": 2411.1, "Yarn_Waste_kg": 1037.7, "Chemical_Waste_kg": 5279.2, "Wastewater_L": 798735.4}, {"date": "2024-04-17", "Solid_Waste_kg": 1989.3, "Yarn_Waste_kg": 1000.2, "Chemical_Waste_kg": 2920.9, "Wastewater_L": 596485.0}, {"date": "2024-04-18", "Solid_Waste_kg": 2133.3, "Yarn_Waste_kg": 988.3, "Chemical_Waste_kg": 4692.7, "Wastewater_L": 649986.9}, {"date": "2024-04-19", "Solid_Waste_kg": 2950.0, "Yarn_Waste_kg": 1066.1, "Chemical_Waste_kg": 5000.9, "Wastewater_L": 796245.7}, {"date": "2024-04-20", "Solid_Waste_kg": 2907.2, "Yarn_Waste_kg": 1118.3, "Chemical_Waste_kg": 2827.6, "Wastewater_L": 660698.7}, {"date": "2024-04-21", "Solid_Waste_kg": 2363.4, "Yarn_Waste_kg": 913.1, "Chemical_Waste_kg": 4251.0, "Wastewater_L": 651023.0}, {"date": "2024-04-22", "Solid_Waste_kg": 2029.0, "Yarn_Waste_kg": 979.6, "Chemical_Waste_kg": 3287.2, "Wastewater_L": 521575.6}, {"date": "2024-04-23", "Solid_Waste_kg": 1999.5, "Yarn_Waste_kg": 1110.9, "Chemical_Waste_kg": 3222.9, "Wastewater_L": 694306.8}, {"date": "2024-04-24", "Solid_Waste_kg": 2133.8, "Yarn_Waste_kg": 916.9, "Chemical_Waste_kg": 3446.4, "Wastewater_L": 531319.0}, {"date": "2024-04-25", "Solid_Waste_kg": 2755.2, "Yarn_Waste_kg": 860.7, "Chemical_Waste_kg": 3988.7, "Wastewater_L": 618538.9}, {"date": "2024-04-26", "Solid_Waste_kg": 2184.5, "Yarn_Waste_kg": 1137.3, "Chemical_Waste_kg": 3949.5, "Wastewater_L": 610497.3}, {"date": "2024-04-27", "Solid_Waste_kg": 2306.4, "Yarn_Waste_kg": 767.9, "Chemical_Waste_kg": 4883.5, "Wastewater_L": 727292.8}, {"date": "2024-04-28", "Solid_Waste_kg": 2942.3, "Yarn_Waste_kg": 1245.2, "Chemical_Waste_kg": 4607.7, "Wastewater_L": 828533.4}, {"date": "2024-04-29", "Solid_Waste_kg": 2587.8, "Yarn_Waste_kg": 817.8, "Chemical_Waste_kg": 3928.3, "Wastewater_L": 714125.4}, {"date": "2024-04-30", "Solid_Waste_kg": 2554.7, "Yarn_Waste_kg": 891.5, "Chemical_Waste_kg": 4959.1, "Wastewater_L": 598323.0}, {"date": "2024-05-01", "Solid_Waste_kg": 3167.4, "Yarn_Waste_kg": 1193.2, "Chemical_Waste_kg": 4135.0, "Wastewater_L": 780483.4}, {"date": "2024-05-02", "Solid_Waste_kg": 2270.7, "Yarn_Waste_kg": 875.7, "Chemical_Waste_kg": 3166.1, "Wastewater_L": 566042.0}, {"date": "2024-05-03", "Solid_Waste_kg": 1987.2, "Yarn_Waste_kg": 966.6, "Chemical_Waste_kg": 4191.9, "Wastewater_L": 775540.5}, {"date": "2024-05-04", "Solid_Waste_kg": 2480.8, "Yarn_Waste_kg": 1212.5, "Chemical_Waste_kg": 4634.4, "Wastewater_L": 647289.0}, {"date": "2024-05-05", "Solid_Waste_kg": 2234.3, "Yarn_Waste_kg": 1012.7, "Chemical_Waste_kg": 3224.1, "Wastewater_L": 443937.8}, {"date": "2024-05-06", "Solid_Waste_kg": 2003.3, "Yarn_Waste_kg": 753.9, "Chemical_Waste_kg": 4144.6, "Wastewater_L": 751800.5}, {"date": "2024-05-07", "Solid_Waste_kg": 2371.1, "Yarn_Waste_kg": 1097.8, "Chemical_Waste_kg": 3686.0, "Wastewater_L": 536868.2}, {"date": "2024-05-08", "Solid_Waste_kg": 2729.0, "Yarn_Waste_kg": 1060.9, "Chemical_Waste_kg": 3088.5, "Wastewater_L": 736700.2}, {"date": "2024-05-09", "Solid_Waste_kg": 2554.8, "Yarn_Waste_kg": 1141.3, "Chemical_Waste_kg": 4346.7, "Wastewater_L": 648374.8}, {"date": "2024-05-10", "Solid_Waste_kg": 2374.9, "Yarn_Waste_kg": 1149.3, "Chemical_Waste_kg": 3150.0, "Wastewater_L": 497029.0}, {"date": "2024-05-11", "Solid_Waste_kg": 2305.3, "Yarn_Waste_kg": 925.6, "Chemical_Waste_kg": 3609.5, "Wastewater_L": 643866.7}, {"date": "2024-05-12", "Solid_Waste_kg": 3378.3, "Yarn_Waste_kg": 1003.8, "Chemical_Waste_kg": 2925.7, "Wastewater_L": 639858.3}, {"date": "2024-05-13", "Solid_Waste_kg": 2615.9, "Yarn_Waste_kg": 1053.6, "Chemical_Waste_kg": 3587.8, "Wastewater_L": 612864.4}, {"date": "2024-05-14", "Solid_Waste_kg": 2057.0, "Yarn_Waste_kg": 683.9, "Chemical_Waste_kg": 2781.4, "Wastewater_L": 439058.8}, {"date": "2024-05-15", "Solid_Waste_kg": 2510.2, "Yarn_Waste_kg": 1030.1, "Chemical_Waste_kg": 4129.0, "Wastewater_L": 840665.9}, {"date": "2024-05-16", "Solid_Waste_kg": 1982.6, "Yarn_Waste_kg": 857.9, "Chemical_Waste_kg": 4105.1, "Wastewater_L": 715604.8}, {"date": "2024-05-17", "Solid_Waste_kg": 2444.5, "Yarn_Waste_kg": 1017.3, "Chemical_Waste_kg": 3803.6, "Wastewater_L": 581491.9}, {"date": "2024-05-18", "Solid_Waste_kg": 2152.9, "Yarn_Waste_kg": 1203.3, "Chemical_Waste_kg": 4827.7, "Wastewater_L": 914961.9}, {"date": "2024-05-19", "Solid_Waste_kg": 3055.6, "Yarn_Waste_kg": 1170.3, "Chemical_Waste_kg": 4390.6, "Wastewater_L": 614651.5}, {"date": "2024-05-20", "Solid_Waste_kg": 2173.1, "Yarn_Waste_kg": 1318.9, "Chemical_Waste_kg": 4553.6, "Wastewater_L": 648816.8}, {"date": "2024-05-21", "Solid_Waste_kg": 2632.6, "Yarn_Waste_kg": 1268.7, "Chemical_Waste_kg": 5057.4, "Wastewater_L": 938959.7}, {"date": "2024-05-22", "Solid_Waste_kg": 2791.4, "Yarn_Waste_kg": 908.6, "Chemical_Waste_kg": 3479.1, "Wastewater_L": 585919.7}, {"date": "2024-05-23", "Solid_Waste_kg": 2458.7, "Yarn_Waste_kg": 1148.3, "Chemical_Waste_kg": 4354.0, "Wastewater_L": 777734.2}, {"date": "2024-05-24", "Solid_Waste_kg": 1956.0, "Yarn_Waste_kg": 1140.3, "Chemical_Waste_kg": 4214.4, "Wastewater_L": 765063.9}, {"date": "2024-05-25", "Solid_Waste_kg": 2226.5, "Yarn_Waste_kg": 897.9, "Chemical_Waste_kg": 3694.4, "Wastewater_L": 689712.1}, {"date": "2024-05-26", "Solid_Waste_kg": 1956.8, "Yarn_Waste_kg": 698.0, "Chemical_Waste_kg": 3755.3, "Wastewater_L": 611339.3}, {"date": "2024-05-27", "Solid_Waste_kg": 2504.4, "Yarn_Waste_kg": 1171.2, "Chemical_Waste_kg": 4030.1, "Wastewater_L": 495215.8}, {"date": "2024-05-28", "Solid_Waste_kg": 2436.7, "Yarn_Waste_kg": 939.4, "Chemical_Waste_kg": 4607.2, "Wastewater_L": 676054.7}, {"date": "2024-05-29", "Solid_Waste_kg": 2978.4, "Yarn_Waste_kg": 847.1, "Chemical_Waste_kg": 4225.7, "Wastewater_L": 677136.6}, {"date": "2024-05-30", "Solid_Waste_kg": 2120.3, "Yarn_Waste_kg": 856.3, "Chemical_Waste_kg": 3528.7, "Wastewater_L": 642294.5}, {"date": "2024-05-31", "Solid_Waste_kg": 2245.7, "Yarn_Waste_kg": 883.7, "Chemical_Waste_kg": 4946.0, "Wastewater_L": 550025.4}, {"date": "2024-06-01", "Solid_Waste_kg": 3372.5, "Yarn_Waste_kg": 1135.9, "Chemical_Waste_kg": 3864.9, "Wastewater_L": 709054.0}, {"date": "2024-06-02", "Solid_Waste_kg": 2074.9, "Yarn_Waste_kg": 936.0, "Chemical_Waste_kg": 4548.8, "Wastewater_L": 742737.3}, {"date": "2024-06-03", "Solid_Waste_kg": 2347.5, "Yarn_Waste_kg": 1384.0, "Chemical_Waste_kg": 4618.1, "Wastewater_L": 692178.7}, {"date": "2024-06-04", "Solid_Waste_kg": 2141.4, "Yarn_Waste_kg": 945.0, "Chemical_Waste_kg": 3412.1, "Wastewater_L": 698040.4}, {"date": "2024-06-05", "Solid_Waste_kg": 2598.9, "Yarn_Waste_kg": 1227.7, "Chemical_Waste_kg": 4129.7, "Wastewater_L": 791231.6}, {"date": "2024-06-06", "Solid_Waste_kg": 1899.5, "Yarn_Waste_kg": 1097.5, "Chemical_Waste_kg": 3579.9, "Wastewater_L": 537418.1}, {"date": "2024-06-07", "Solid_Waste_kg": 3619.5, "Yarn_Waste_kg": 1181.8, "Chemical_Waste_kg": 5075.1, "Wastewater_L": 841565.0}, {"date": "2024-06-08", "Solid_Waste_kg": 3039.7, "Yarn_Waste_kg": 1558.2, "Chemical_Waste_kg": 5593.2, "Wastewater_L": 973879.4}, {"date": "2024-06-09", "Solid_Waste_kg": 2492.4, "Yarn_Waste_kg": 1216.3, "Chemical_Waste_kg": 4694.7, "Wastewater_L": 813977.2}, {"date": "2024-06-10", "Solid_Waste_kg": 3093.0, "Yarn_Waste_kg": 1357.8, "Chemical_Waste_kg": 5126.1, "Wastewater_L": 723639.6}, {"date": "2024-06-11", "Solid_Waste_kg": 3079.0, "Yarn_Waste_kg": 1392.6, "Chemical_Waste_kg": 3843.1, "Wastewater_L": 726926.5}, {"date": "2024-06-12", "Solid_Waste_kg": 2822.6, "Yarn_Waste_kg": 1218.7, "Chemical_Waste_kg": 4105.8, "Wastewater_L": 812093.8}, {"date": "2024-06-13", "Solid_Waste_kg": 3213.8, "Yarn_Waste_kg": 1505.4, "Chemical_Waste_kg": 6305.8, "Wastewater_L": 987261.4}, {"date": "2024-06-14", "Solid_Waste_kg": 2406.5, "Yarn_Waste_kg": 1283.7, "Chemical_Waste_kg": 5676.5, "Wastewater_L": 1063956.9}, {"date": "2024-06-15", "Solid_Waste_kg": 2702.1, "Yarn_Waste_kg": 1160.4, "Chemical_Waste_kg": 4341.7, "Wastewater_L": 730355.5}, {"date": "2024-06-16", "Solid_Waste_kg": 1980.1, "Yarn_Waste_kg": 1083.8, "Chemical_Waste_kg": 4507.7, "Wastewater_L": 955585.8}, {"date": "2024-06-17", "Solid_Waste_kg": 2582.6, "Yarn_Waste_kg": 972.6, "Chemical_Waste_kg": 3858.1, "Wastewater_L": 690996.6}, {"date": "2024-06-18", "Solid_Waste_kg": 3521.1, "Yarn_Waste_kg": 990.8, "Chemical_Waste_kg": 4474.0, "Wastewater_L": 740661.5}, {"date": "2024-06-19", "Solid_Waste_kg": 2731.6, "Yarn_Waste_kg": 1342.1, "Chemical_Waste_kg": 4297.1, "Wastewater_L": 890793.5}, {"date": "2024-06-20", "Solid_Waste_kg": 2759.4, "Yarn_Waste_kg": 952.3, "Chemical_Waste_kg": 4563.7, "Wastewater_L": 873356.2}, {"date": "2024-06-21", "Solid_Waste_kg": 2447.3, "Yarn_Waste_kg": 855.1, "Chemical_Waste_kg": 3866.1, "Wastewater_L": 678960.8}, {"date": "2024-06-22", "Solid_Waste_kg": 2058.7, "Yarn_Waste_kg": 952.3, "Chemical_Waste_kg": 3782.7, "Wastewater_L": 590155.2}, {"date": "2024-06-23", "Solid_Waste_kg": 2708.1, "Yarn_Waste_kg": 918.6, "Chemical_Waste_kg": 5567.6, "Wastewater_L": 774602.7}, {"date": "2024-06-24", "Solid_Waste_kg": 2293.4, "Yarn_Waste_kg": 977.3, "Chemical_Waste_kg": 3391.3, "Wastewater_L": 559245.6}, {"date": "2024-06-25", "Solid_Waste_kg": 2108.8, "Yarn_Waste_kg": 881.1, "Chemical_Waste_kg": 3443.4, "Wastewater_L": 774191.5}, {"date": "2024-06-26", "Solid_Waste_kg": 2115.2, "Yarn_Waste_kg": 991.4, "Chemical_Waste_kg": 3329.5, "Wastewater_L": 506502.2}, {"date": "2024-06-27", "Solid_Waste_kg": 2575.2, "Yarn_Waste_kg": 863.2, "Chemical_Waste_kg": 3624.9, "Wastewater_L": 652269.7}, {"date": "2024-06-28", "Solid_Waste_kg": 2246.5, "Yarn_Waste_kg": 1256.3, "Chemical_Waste_kg": 3925.1, "Wastewater_L": 730061.6}, {"date": "2024-06-29", "Solid_Waste_kg": 2019.5, "Yarn_Waste_kg": 849.8, "Chemical_Waste_kg": 3979.8, "Wastewater_L": 809596.5}, {"date": "2024-06-30", "Solid_Waste_kg": 2972.3, "Yarn_Waste_kg": 1045.4, "Chemical_Waste_kg": 4697.6, "Wastewater_L": 875364.2}, {"date": "2024-07-01", "Solid_Waste_kg": 2874.8, "Yarn_Waste_kg": 1130.9, "Chemical_Waste_kg": 4207.2, "Wastewater_L": 694904.8}, {"date": "2024-07-02", "Solid_Waste_kg": 2712.0, "Yarn_Waste_kg": 1020.8, "Chemical_Waste_kg": 4458.7, "Wastewater_L": 781522.3}, {"date": "2024-07-03", "Solid_Waste_kg": 2719.6, "Yarn_Waste_kg": 1122.3, "Chemical_Waste_kg": 5337.2, "Wastewater_L": 785474.8}, {"date": "2024-07-04", "Solid_Waste_kg": 2334.9, "Yarn_Waste_kg": 781.2, "Chemical_Waste_kg": 4188.1, "Wastewater_L": 615199.5}, {"date": "2024-07-05", "Solid_Waste_kg": 2212.6, "Yarn_Waste_kg": 1219.7, "Chemical_Waste_kg": 5047.6, "Wastewater_L": 901574.7}, {"date": "2024-07-06", "Solid_Waste_kg": 2655.1, "Yarn_Waste_kg": 1070.1, "Chemical_Waste_kg": 4766.1, "Wastewater_L": 759440.0}, {"date": "2024-07-07", "Solid_Waste_kg": 1558.1, "Yarn_Waste_kg": 980.6, "Chemical_Waste_kg": 3634.5, "Wastewater_L": 682577.1}, {"date": "2024-07-08", "Solid_Waste_kg": 2206.2, "Yarn_Waste_kg": 1284.5, "Chemical_Waste_kg": 5149.7, "Wastewater_L": 734302.9}, {"date": "2024-07-09", "Solid_Waste_kg": 1871.4, "Yarn_Waste_kg": 937.0, "Chemical_Waste_kg": 2759.8, "Wastewater_L": 674594.2}, {"date": "2024-07-10", "Solid_Waste_kg": 2465.7, "Yarn_Waste_kg": 1269.3, "Chemical_Waste_kg": 4125.6, "Wastewater_L": 793305.5}, {"date": "2024-07-11", "Solid_Waste_kg": 2388.9, "Yarn_Waste_kg": 818.8, "Chemical_Waste_kg": 4862.1, "Wastewater_L": 660703.2}, {"date": "2024-07-12", "Solid_Waste_kg": 2099.5, "Yarn_Waste_kg": 1010.7, "Chemical_Waste_kg": 4505.5, "Wastewater_L": 542850.7}, {"date": "2024-07-13", "Solid_Waste_kg": 2698.8, "Yarn_Waste_kg": 977.2, "Chemical_Waste_kg": 5264.0, "Wastewater_L": 892150.4}, {"date": "2024-07-14", "Solid_Waste_kg": 3067.4, "Yarn_Waste_kg": 1166.2, "Chemical_Waste_kg": 4695.3, "Wastewater_L": 510062.7}, {"date": "2024-07-15", "Solid_Waste_kg": 3307.7, "Yarn_Waste_kg": 1003.9, "Chemical_Waste_kg": 4655.3, "Wastewater_L": 648474.1}, {"date": "2024-07-16", "Solid_Waste_kg": 2645.6, "Yarn_Waste_kg": 1320.7, "Chemical_Waste_kg": 4873.8, "Wastewater_L": 833022.7}, {"date": "2024-07-17", "Solid_Waste_kg": 2814.2, "Yarn_Waste_kg": 953.2, "Chemical_Waste_kg": 4423.6, "Wastewater_L": 712998.6}, {"date": "2024-07-18", "Solid_Waste_kg": 2933.6, "Yarn_Waste_kg": 956.6, "Chemical_Waste_kg": 4309.5, "Wastewater_L": 956607.3}, {"date": "2024-07-19", "Solid_Waste_kg": 2174.1, "Yarn_Waste_kg": 1215.1, "Chemical_Waste_kg": 5441.5, "Wastewater_L": 711419.4}, {"date": "2024-07-20", "Solid_Waste_kg": 2691.1, "Yarn_Waste_kg": 1280.5, "Chemical_Waste_kg": 4666.0, "Wastewater_L": 687349.5}, {"date": "2024-07-21", "Solid_Waste_kg": 2790.6, "Yarn_Waste_kg": 1044.2, "Chemical_Waste_kg": 3720.6, "Wastewater_L": 503053.6}, {"date": "2024-07-22", "Solid_Waste_kg": 3217.7, "Yarn_Waste_kg": 1005.9, "Chemical_Waste_kg": 3321.9, "Wastewater_L": 663777.0}, {"date": "2024-07-23", "Solid_Waste_kg": 2959.1, "Yarn_Waste_kg": 1162.4, "Chemical_Waste_kg": 3830.5, "Wastewater_L": 871107.0}, {"date": "2024-07-24", "Solid_Waste_kg": 2541.1, "Yarn_Waste_kg": 1105.1, "Chemical_Waste_kg": 5160.4, "Wastewater_L": 501754.2}, {"date": "2024-07-25", "Solid_Waste_kg": 2735.6, "Yarn_Waste_kg": 1337.7, "Chemical_Waste_kg": 4866.3, "Wastewater_L": 698130.2}, {"date": "2024-07-26", "Solid_Waste_kg": 3168.9, "Yarn_Waste_kg": 1414.2, "Chemical_Waste_kg": 4407.6, "Wastewater_L": 840013.8}, {"date": "2024-07-27", "Solid_Waste_kg": 2121.2, "Yarn_Waste_kg": 918.8, "Chemical_Waste_kg": 3516.4, "Wastewater_L": 561614.3}, {"date": "2024-07-28", "Solid_Waste_kg": 2521.6, "Yarn_Waste_kg": 1094.5, "Chemical_Waste_kg": 4993.6, "Wastewater_L": 951222.8}, {"date": "2024-07-29", "Solid_Waste_kg": 1961.4, "Yarn_Waste_kg": 973.2, "Chemical_Waste_kg": 4664.5, "Wastewater_L": 650749.9}, {"date": "2024-07-30", "Solid_Waste_kg": 2462.0, "Yarn_Waste_kg": 1079.3, "Chemical_Waste_kg": 3433.3, "Wastewater_L": 614085.6}, {"date": "2024-07-31", "Solid_Waste_kg": 2722.1, "Yarn_Waste_kg": 747.7, "Chemical_Waste_kg": 4396.8, "Wastewater_L": 720969.9}, {"date": "2024-08-01", "Solid_Waste_kg": 2628.4, "Yarn_Waste_kg": 937.6, "Chemical_Waste_kg": 4327.6, "Wastewater_L": 540274.7}, {"date": "2024-08-02", "Solid_Waste_kg": 2586.3, "Yarn_Waste_kg": 1085.2, "Chemical_Waste_kg": 4272.7, "Wastewater_L": 603539.5}, {"date": "2024-08-03", "Solid_Waste_kg": 2639.6, "Yarn_Waste_kg": 1033.6, "Chemical_Waste_kg": 3932.6, "Wastewater_L": 486555.5}, {"date": "2024-08-04", "Solid_Waste_kg": 2870.4, "Yarn_Waste_kg": 1236.7, "Chemical_Waste_kg": 4745.1, "Wastewater_L": 578522.8}, {"date": "2024-08-05", "Solid_Waste_kg": 2607.0, "Yarn_Waste_kg": 1189.9, "Chemical_Waste_kg": 3841.9, "Wastewater_L": 734475.9}, {"date": "2024-08-06", "Solid_Waste_kg": 2805.5, "Yarn_Waste_kg": 1235.4, "Chemical_Waste_kg": 3093.4, "Wastewater_L": 603823.1}, {"date": "2024-08-07", "Solid_Waste_kg": 2208.4, "Yarn_Waste_kg": 1007.4, "Chemical_Waste_kg": 3682.6, "Wastewater_L": 564817.3}, {"date": "2024-08-08", "Solid_Waste_kg": 1499.8, "Yarn_Waste_kg": 973.1, "Chemical_Waste_kg": 4147.8, "Wastewater_L": 652593.8}, {"date": "2024-08-09", "Solid_Waste_kg": 2315.1, "Yarn_Waste_kg": 926.1, "Chemical_Waste_kg": 3884.9, "Wastewater_L": 712256.3}, {"date": "2024-08-10", "Solid_Waste_kg": 3185.3, "Yarn_Waste_kg": 877.6, "Chemical_Waste_kg": 4784.1, "Wastewater_L": 666657.4}, {"date": "2024-08-11", "Solid_Waste_kg": 1904.3, "Yarn_Waste_kg": 1068.8, "Chemical_Waste_kg": 3413.6, "Wastewater_L": 555793.6}, {"date": "2024-08-12", "Solid_Waste_kg": 2302.6, "Yarn_Waste_kg": 1135.1, "Chemical_Waste_kg": 4031.9, "Wastewater_L": 677654.0}, {"date": "2024-08-13", "Solid_Waste_kg": 2589.2, "Yarn_Waste_kg": 910.8, "Chemical_Waste_kg": 3348.0, "Wastewater_L": 620412.9}, {"date": "2024-08-14", "Solid_Waste_kg": 2222.2, "Yarn_Waste_kg": 963.0, "Chemical_Waste_kg": 4621.6, "Wastewater_L": 634036.3}, {"date": "2024-08-15", "Solid_Waste_kg": 2535.3, "Yarn_Waste_kg": 890.0, "Chemical_Waste_kg": 4564.7, "Wastewater_L": 736750.5}, {"date": "2024-08-16", "Solid_Waste_kg": 1640.1, "Yarn_Waste_kg": 887.1, "Chemical_Waste_kg": 3039.1, "Wastewater_L": 554768.5}, {"date": "2024-08-17", "Solid_Waste_kg": 3164.2, "Yarn_Waste_kg": 1271.0, "Chemical_Waste_kg": 4635.5, "Wastewater_L": 629520.0}, {"date": "2024-08-18", "Solid_Waste_kg": 2076.3, "Yarn_Waste_kg": 877.7, "Chemical_Waste_kg": 3576.3, "Wastewater_L": 683342.3}, {"date": "2024-08-19", "Solid_Waste_kg": 2645.3, "Yarn_Waste_kg": 848.9, "Chemical_Waste_kg": 3636.0, "Wastewater_L": 690400.5}, {"date": "2024-08-20", "Solid_Waste_kg": 2823.1, "Yarn_Waste_kg": 877.4, "Chemical_Waste_kg": 3817.7, "Wastewater_L": 602833.4}, {"date": "2024-08-21", "Solid_Waste_kg": 2961.5, "Yarn_Waste_kg": 1158.1, "Chemical_Waste_kg": 4555.7, "Wastewater_L": 573856.2}, {"date": "2024-08-22", "Solid_Waste_kg": 2233.0, "Yarn_Waste_kg": 853.0, "Chemical_Waste_kg": 2968.1, "Wastewater_L": 642563.5}, {"date": "2024-08-23", "Solid_Waste_kg": 2345.6, "Yarn_Waste_kg": 1363.0, "Chemical_Waste_kg": 4110.7, "Wastewater_L": 864557.8}, {"date": "2024-08-24", "Solid_Waste_kg": 1754.1, "Yarn_Waste_kg": 785.5, "Chemical_Waste_kg": 4219.5, "Wastewater_L": 552632.6}, {"date": "2024-08-25", "Solid_Waste_kg": 2500.0, "Yarn_Waste_kg": 1138.7, "Chemical_Waste_kg": 4163.7, "Wastewater_L": 718115.0}, {"date": "2024-08-26", "Solid_Waste_kg": 2119.2, "Yarn_Waste_kg": 1354.1, "Chemical_Waste_kg": 4414.3, "Wastewater_L": 839681.7}, {"date": "2024-08-27", "Solid_Waste_kg": 2561.7, "Yarn_Waste_kg": 1053.0, "Chemical_Waste_kg": 4562.2, "Wastewater_L": 771907.7}, {"date": "2024-08-28", "Solid_Waste_kg": 2145.2, "Yarn_Waste_kg": 895.1, "Chemical_Waste_kg": 3696.2, "Wastewater_L": 670502.7}, {"date": "2024-08-29", "Solid_Waste_kg": 2224.5, "Yarn_Waste_kg": 843.9, "Chemical_Waste_kg": 3228.2, "Wastewater_L": 571864.7}, {"date": "2024-08-30", "Solid_Waste_kg": 2618.5, "Yarn_Waste_kg": 1149.9, "Chemical_Waste_kg": 3672.4, "Wastewater_L": 766108.7}, {"date": "2024-08-31", "Solid_Waste_kg": 3103.0, "Yarn_Waste_kg": 1053.5, "Chemical_Waste_kg": 4063.1, "Wastewater_L": 559294.5}, {"date": "2024-09-01", "Solid_Waste_kg": 2878.5, "Yarn_Waste_kg": 802.6, "Chemical_Waste_kg": 3037.0, "Wastewater_L": 448276.0}, {"date": "2024-09-02", "Solid_Waste_kg": 2052.8, "Yarn_Waste_kg": 704.3, "Chemical_Waste_kg": 3290.9, "Wastewater_L": 582665.4}, {"date": "2024-09-03", "Solid_Waste_kg": 2292.6, "Yarn_Waste_kg": 790.6, "Chemical_Waste_kg": 3264.3, "Wastewater_L": 469636.1}, {"date": "2024-09-04", "Solid_Waste_kg": 1757.2, "Yarn_Waste_kg": 672.8, "Chemical_Waste_kg": 3546.1, "Wastewater_L": 530750.4}, {"date": "2024-09-05", "Solid_Waste_kg": 2200.7, "Yarn_Waste_kg": 804.6, "Chemical_Waste_kg": 2910.8, "Wastewater_L": 628913.2}, {"date": "2024-09-06", "Solid_Waste_kg": 1963.3, "Yarn_Waste_kg": 805.0, "Chemical_Waste_kg": 2848.8, "Wastewater_L": 536872.5}, {"date": "2024-09-07", "Solid_Waste_kg": 1849.2, "Yarn_Waste_kg": 951.2, "Chemical_Waste_kg": 3332.9, "Wastewater_L": 527919.4}, {"date": "2024-09-08", "Solid_Waste_kg": 2593.2, "Yarn_Waste_kg": 926.4, "Chemical_Waste_kg": 3706.0, "Wastewater_L": 610885.5}, {"date": "2024-09-09", "Solid_Waste_kg": 1978.0, "Yarn_Waste_kg": 906.0, "Chemical_Waste_kg": 2800.0, "Wastewater_L": 467972.9}, {"date": "2024-09-10", "Solid_Waste_kg": 1935.6, "Yarn_Waste_kg": 835.2, "Chemical_Waste_kg": 3871.0, "Wastewater_L": 597599.5}, {"date": "2024-09-11", "Solid_Waste_kg": 2007.4, "Yarn_Waste_kg": 995.5, "Chemical_Waste_kg": 4995.2, "Wastewater_L": 781867.4}, {"date": "2024-09-12", "Solid_Waste_kg": 1272.7, "Yarn_Waste_kg": 690.1, "Chemical_Waste_kg": 2757.5, "Wastewater_L": 304724.9}, {"date": "2024-09-13", "Solid_Waste_kg": 1548.1, "Yarn_Waste_kg": 758.4, "Chemical_Waste_kg": 3171.2, "Wastewater_L": 578185.3}, {"date": "2024-09-14", "Solid_Waste_kg": 1791.5, "Yarn_Waste_kg": 760.6, "Chemical_Waste_kg": 3617.3, "Wastewater_L": 549451.0}, {"date": "2024-09-15", "Solid_Waste_kg": 1864.0, "Yarn_Waste_kg": 954.2, "Chemical_Waste_kg": 3381.4, "Wastewater_L": 458175.7}, {"date": "2024-09-16", "Solid_Waste_kg": 2175.2, "Yarn_Waste_kg": 910.0, "Chemical_Waste_kg": 3609.8, "Wastewater_L": 556930.1}, {"date": "2024-09-17", "Solid_Waste_kg": 2800.3, "Yarn_Waste_kg": 882.4, "Chemical_Waste_kg": 3916.7, "Wastewater_L": 760189.6}, {"date": "2024-09-18", "Solid_Waste_kg": 2864.8, "Yarn_Waste_kg": 1208.6, "Chemical_Waste_kg": 4200.3, "Wastewater_L": 663167.0}, {"date": "2024-09-19", "Solid_Waste_kg": 2103.2, "Yarn_Waste_kg": 875.5, "Chemical_Waste_kg": 3691.8, "Wastewater_L": 643884.4}, {"date": "2024-09-20", "Solid_Waste_kg": 2439.2, "Yarn_Waste_kg": 691.3, "Chemical_Waste_kg": 3371.6, "Wastewater_L": 636373.9}, {"date": "2024-09-21", "Solid_Waste_kg": 2282.1, "Yarn_Waste_kg": 1125.2, "Chemical_Waste_kg": 4240.9, "Wastewater_L": 631807.7}, {"date": "2024-09-22", "Solid_Waste_kg": 1946.9, "Yarn_Waste_kg": 757.4, "Chemical_Waste_kg": 3040.1, "Wastewater_L": 617552.2}, {"date": "2024-09-23", "Solid_Waste_kg": 2424.5, "Yarn_Waste_kg": 803.7, "Chemical_Waste_kg": 4076.5, "Wastewater_L": 746806.9}, {"date": "2024-09-24", "Solid_Waste_kg": 2312.1, "Yarn_Waste_kg": 861.4, "Chemical_Waste_kg": 2978.1, "Wastewater_L": 624015.9}, {"date": "2024-09-25", "Solid_Waste_kg": 1740.8, "Yarn_Waste_kg": 825.5, "Chemical_Waste_kg": 2800.2, "Wastewater_L": 506662.6}, {"date": "2024-09-26", "Solid_Waste_kg": 2052.7, "Yarn_Waste_kg": 965.6, "Chemical_Waste_kg": 5019.9, "Wastewater_L": 716209.5}, {"date": "2024-09-27", "Solid_Waste_kg": 2810.7, "Yarn_Waste_kg": 1155.7, "Chemical_Waste_kg": 4570.4, "Wastewater_L": 770173.8}, {"date": "2024-09-28", "Solid_Waste_kg": 2712.6, "Yarn_Waste_kg": 1190.9, "Chemical_Waste_kg": 3596.3, "Wastewater_L": 746585.7}, {"date": "2024-09-29", "Solid_Waste_kg": 1874.5, "Yarn_Waste_kg": 809.9, "Chemical_Waste_kg": 3259.4, "Wastewater_L": 582582.4}, {"date": "2024-09-30", "Solid_Waste_kg": 2351.0, "Yarn_Waste_kg": 666.1, "Chemical_Waste_kg": 2477.7, "Wastewater_L": 486023.5}, {"date": "2024-10-01", "Solid_Waste_kg": 2160.0, "Yarn_Waste_kg": 867.4, "Chemical_Waste_kg": 3961.5, "Wastewater_L": 414957.9}, {"date": "2024-10-02", "Solid_Waste_kg": 1848.7, "Yarn_Waste_kg": 891.9, "Chemical_Waste_kg": 3496.5, "Wastewater_L": 520716.3}, {"date": "2024-10-03", "Solid_Waste_kg": 1695.0, "Yarn_Waste_kg": 747.1, "Chemical_Waste_kg": 3022.9, "Wastewater_L": 632344.2}, {"date": "2024-10-04", "Solid_Waste_kg": 1356.0, "Yarn_Waste_kg": 480.6, "Chemical_Waste_kg": 2168.4, "Wastewater_L": 311582.9}, {"date": "2024-10-05", "Solid_Waste_kg": 1390.0, "Yarn_Waste_kg": 623.0, "Chemical_Waste_kg": 2876.3, "Wastewater_L": 525393.5}, {"date": "2024-10-06", "Solid_Waste_kg": 2245.5, "Yarn_Waste_kg": 1091.5, "Chemical_Waste_kg": 3776.4, "Wastewater_L": 658140.6}, {"date": "2024-10-07", "Solid_Waste_kg": 1798.8, "Yarn_Waste_kg": 726.5, "Chemical_Waste_kg": 3426.1, "Wastewater_L": 535227.9}, {"date": "2024-10-08", "Solid_Waste_kg": 1680.0, "Yarn_Waste_kg": 666.2, "Chemical_Waste_kg": 2176.5, "Wastewater_L": 368401.2}, {"date": "2024-10-09", "Solid_Waste_kg": 1989.6, "Yarn_Waste_kg": 754.3, "Chemical_Waste_kg": 3004.8, "Wastewater_L": 585568.7}, {"date": "2024-10-10", "Solid_Waste_kg": 1995.0, "Yarn_Waste_kg": 883.2, "Chemical_Waste_kg": 3304.0, "Wastewater_L": 404887.4}, {"date": "2024-10-11", "Solid_Waste_kg": 1607.8, "Yarn_Waste_kg": 702.9, "Chemical_Waste_kg": 2842.3, "Wastewater_L": 495571.0}, {"date": "2024-10-12", "Solid_Waste_kg": 2252.6, "Yarn_Waste_kg": 842.8, "Chemical_Waste_kg": 3042.2, "Wastewater_L": 544667.6}, {"date": "2024-10-13", "Solid_Waste_kg": 1931.7, "Yarn_Waste_kg": 721.3, "Chemical_Waste_kg": 3537.3, "Wastewater_L": 610309.4}, {"date": "2024-10-14", "Solid_Waste_kg": 2056.7, "Yarn_Waste_kg": 818.3, "Chemical_Waste_kg": 2739.4, "Wastewater_L": 532477.8}, {"date": "2024-10-15", "Solid_Waste_kg": 2859.6, "Yarn_Waste_kg": 821.9, "Chemical_Waste_kg": 3493.0, "Wastewater_L": 626643.4}, {"date": "2024-10-16", "Solid_Waste_kg": 1966.2, "Yarn_Waste_kg": 811.7, "Chemical_Waste_kg": 2800.8, "Wastewater_L": 402803.0}, {"date": "2024-10-17", "Solid_Waste_kg": 1265.3, "Yarn_Waste_kg": 579.6, "Chemical_Waste_kg": 2535.6, "Wastewater_L": 508760.9}, {"date": "2024-10-18", "Solid_Waste_kg": 1874.7, "Yarn_Waste_kg": 685.4, "Chemical_Waste_kg": 2850.9, "Wastewater_L": 514626.1}, {"date": "2024-10-19", "Solid_Waste_kg": 2263.9, "Yarn_Waste_kg": 1114.9, "Chemical_Waste_kg": 3787.9, "Wastewater_L": 644712.1}, {"date": "2024-10-20", "Solid_Waste_kg": 1659.9, "Yarn_Waste_kg": 616.8, "Chemical_Waste_kg": 2434.2, "Wastewater_L": 475661.8}, {"date": "2024-10-21", "Solid_Waste_kg": 2341.0, "Yarn_Waste_kg": 717.0, "Chemical_Waste_kg": 4345.3, "Wastewater_L": 541318.6}, {"date": "2024-10-22", "Solid_Waste_kg": 1990.8, "Yarn_Waste_kg": 635.8, "Chemical_Waste_kg": 2990.3, "Wastewater_L": 448805.2}, {"date": "2024-10-23", "Solid_Waste_kg": 2085.8, "Yarn_Waste_kg": 797.7, "Chemical_Waste_kg": 2534.6, "Wastewater_L": 436966.2}, {"date": "2024-10-24", "Solid_Waste_kg": 2285.1, "Yarn_Waste_kg": 956.9, "Chemical_Waste_kg": 2724.6, "Wastewater_L": 597013.0}, {"date": "2024-10-25", "Solid_Waste_kg": 1326.8, "Yarn_Waste_kg": 605.8, "Chemical_Waste_kg": 2213.8, "Wastewater_L": 370886.3}, {"date": "2024-10-26", "Solid_Waste_kg": 1931.9, "Yarn_Waste_kg": 825.8, "Chemical_Waste_kg": 3449.0, "Wastewater_L": 595519.9}, {"date": "2024-10-27", "Solid_Waste_kg": 1750.6, "Yarn_Waste_kg": 1157.1, "Chemical_Waste_kg": 4396.4, "Wastewater_L": 513787.3}, {"date": "2024-10-28", "Solid_Waste_kg": 2125.6, "Yarn_Waste_kg": 791.2, "Chemical_Waste_kg": 2702.9, "Wastewater_L": 432482.3}, {"date": "2024-10-29", "Solid_Waste_kg": 1962.4, "Yarn_Waste_kg": 1014.9, "Chemical_Waste_kg": 3707.1, "Wastewater_L": 459869.5}, {"date": "2024-10-30", "Solid_Waste_kg": 2395.8, "Yarn_Waste_kg": 959.7, "Chemical_Waste_kg": 3149.0, "Wastewater_L": 365483.8}, {"date": "2024-10-31", "Solid_Waste_kg": 1486.3, "Yarn_Waste_kg": 680.9, "Chemical_Waste_kg": 3005.0, "Wastewater_L": 534562.6}, {"date": "2024-11-01", "Solid_Waste_kg": 1821.7, "Yarn_Waste_kg": 697.9, "Chemical_Waste_kg": 3047.3, "Wastewater_L": 421753.4}, {"date": "2024-11-02", "Solid_Waste_kg": 1423.3, "Yarn_Waste_kg": 495.6, "Chemical_Waste_kg": 2050.6, "Wastewater_L": 365991.6}, {"date": "2024-11-03", "Solid_Waste_kg": 1501.2, "Yarn_Waste_kg": 720.8, "Chemical_Waste_kg": 2651.8, "Wastewater_L": 490810.8}, {"date": "2024-11-04", "Solid_Waste_kg": 1186.2, "Yarn_Waste_kg": 431.7, "Chemical_Waste_kg": 2413.9, "Wastewater_L": 410669.8}, {"date": "2024-11-05", "Solid_Waste_kg": 1837.2, "Yarn_Waste_kg": 681.4, "Chemical_Waste_kg": 2093.7, "Wastewater_L": 396036.4}, {"date": "2024-11-06", "Solid_Waste_kg": 1562.8, "Yarn_Waste_kg": 893.4, "Chemical_Waste_kg": 4101.9, "Wastewater_L": 704849.3}, {"date": "2024-11-07", "Solid_Waste_kg": 1983.5, "Yarn_Waste_kg": 866.3, "Chemical_Waste_kg": 4383.4, "Wastewater_L": 524986.1}, {"date": "2024-11-08", "Solid_Waste_kg": 1562.4, "Yarn_Waste_kg": 729.5, "Chemical_Waste_kg": 2067.8, "Wastewater_L": 397955.3}, {"date": "2024-11-09", "Solid_Waste_kg": 1828.5, "Yarn_Waste_kg": 616.5, "Chemical_Waste_kg": 3139.7, "Wastewater_L": 440599.8}, {"date": "2024-11-10", "Solid_Waste_kg": 2061.3, "Yarn_Waste_kg": 653.7, "Chemical_Waste_kg": 3327.5, "Wastewater_L": 502786.0}, {"date": "2024-11-11", "Solid_Waste_kg": 1239.8, "Yarn_Waste_kg": 505.6, "Chemical_Waste_kg": 1803.7, "Wastewater_L": 329210.4}, {"date": "2024-11-12", "Solid_Waste_kg": 1438.8, "Yarn_Waste_kg": 538.9, "Chemical_Waste_kg": 2353.7, "Wastewater_L": 386102.9}, {"date": "2024-11-13", "Solid_Waste_kg": 1875.9, "Yarn_Waste_kg": 805.7, "Chemical_Waste_kg": 2690.1, "Wastewater_L": 534276.7}, {"date": "2024-11-14", "Solid_Waste_kg": 1333.5, "Yarn_Waste_kg": 638.6, "Chemical_Waste_kg": 2202.9, "Wastewater_L": 421348.9}, {"date": "2024-11-15", "Solid_Waste_kg": 769.5, "Yarn_Waste_kg": 350.4, "Chemical_Waste_kg": 1425.0, "Wastewater_L": 275178.4}, {"date": "2024-11-16", "Solid_Waste_kg": 1380.4, "Yarn_Waste_kg": 456.8, "Chemical_Waste_kg": 1724.1, "Wastewater_L": 367876.0}, {"date": "2024-11-17", "Solid_Waste_kg": 1952.1, "Yarn_Waste_kg": 531.6, "Chemical_Waste_kg": 2190.9, "Wastewater_L": 410572.4}, {"date": "2024-11-18", "Solid_Waste_kg": 1578.7, "Yarn_Waste_kg": 546.6, "Chemical_Waste_kg": 2728.8, "Wastewater_L": 313892.1}, {"date": "2024-11-19", "Solid_Waste_kg": 1489.1, "Yarn_Waste_kg": 657.3, "Chemical_Waste_kg": 2435.9, "Wastewater_L": 486103.3}, {"date": "2024-11-20", "Solid_Waste_kg": 1345.7, "Yarn_Waste_kg": 507.6, "Chemical_Waste_kg": 1882.7, "Wastewater_L": 321189.2}, {"date": "2024-11-21", "Solid_Waste_kg": 1590.6, "Yarn_Waste_kg": 752.6, "Chemical_Waste_kg": 2191.5, "Wastewater_L": 501707.2}, {"date": "2024-11-22", "Solid_Waste_kg": 1670.7, "Yarn_Waste_kg": 569.8, "Chemical_Waste_kg": 2759.7, "Wastewater_L": 453327.4}, {"date": "2024-11-23", "Solid_Waste_kg": 1431.5, "Yarn_Waste_kg": 403.8, "Chemical_Waste_kg": 1828.8, "Wastewater_L": 348365.0}, {"date": "2024-11-24", "Solid_Waste_kg": 2008.2, "Yarn_Waste_kg": 989.2, "Chemical_Waste_kg": 3167.9, "Wastewater_L": 656641.1}, {"date": "2024-11-25", "Solid_Waste_kg": 1483.6, "Yarn_Waste_kg": 734.7, "Chemical_Waste_kg": 2557.0, "Wastewater_L": 394117.3}, {"date": "2024-11-26", "Solid_Waste_kg": 2085.2, "Yarn_Waste_kg": 741.9, "Chemical_Waste_kg": 3275.3, "Wastewater_L": 520783.6}, {"date": "2024-11-27", "Solid_Waste_kg": 1421.3, "Yarn_Waste_kg": 644.6, "Chemical_Waste_kg": 3040.7, "Wastewater_L": 447258.5}, {"date": "2024-11-28", "Solid_Waste_kg": 2074.1, "Yarn_Waste_kg": 830.1, "Chemical_Waste_kg": 2722.6, "Wastewater_L": 585454.8}, {"date": "2024-11-29", "Solid_Waste_kg": 1227.3, "Yarn_Waste_kg": 529.9, "Chemical_Waste_kg": 2469.0, "Wastewater_L": 303502.7}, {"date": "2024-11-30", "Solid_Waste_kg": 1406.0, "Yarn_Waste_kg": 501.3, "Chemical_Waste_kg": 1890.5, "Wastewater_L": 308015.4}, {"date": "2024-12-01", "Solid_Waste_kg": 1499.0, "Yarn_Waste_kg": 832.3, "Chemical_Waste_kg": 3060.0, "Wastewater_L": 420812.5}, {"date": "2024-12-02", "Solid_Waste_kg": 1862.9, "Yarn_Waste_kg": 713.7, "Chemical_Waste_kg": 2652.4, "Wastewater_L": 451932.3}, {"date": "2024-12-03", "Solid_Waste_kg": 1996.7, "Yarn_Waste_kg": 880.5, "Chemical_Waste_kg": 3469.3, "Wastewater_L": 519809.5}, {"date": "2024-12-04", "Solid_Waste_kg": 1430.7, "Yarn_Waste_kg": 497.4, "Chemical_Waste_kg": 2279.1, "Wastewater_L": 420386.7}, {"date": "2024-12-05", "Solid_Waste_kg": 1425.5, "Yarn_Waste_kg": 739.9, "Chemical_Waste_kg": 3227.3, "Wastewater_L": 433790.1}, {"date": "2024-12-06", "Solid_Waste_kg": 1520.7, "Yarn_Waste_kg": 566.9, "Chemical_Waste_kg": 2173.1, "Wastewater_L": 484337.2}, {"date": "2024-12-07", "Solid_Waste_kg": 2289.7, "Yarn_Waste_kg": 706.7, "Chemical_Waste_kg": 3246.5, "Wastewater_L": 549543.4}, {"date": "2024-12-08", "Solid_Waste_kg": 2382.7, "Yarn_Waste_kg": 959.8, "Chemical_Waste_kg": 2861.8, "Wastewater_L": 555815.7}, {"date": "2024-12-09", "Solid_Waste_kg": 1829.9, "Yarn_Waste_kg": 1001.9, "Chemical_Waste_kg": 3956.9, "Wastewater_L": 628271.4}, {"date": "2024-12-10", "Solid_Waste_kg": 1968.9, "Yarn_Waste_kg": 1002.5, "Chemical_Waste_kg": 3500.8, "Wastewater_L": 565910.7}, {"date": "2024-12-11", "Solid_Waste_kg": 1681.6, "Yarn_Waste_kg": 554.7, "Chemical_Waste_kg": 3132.4, "Wastewater_L": 516770.7}, {"date": "2024-12-12", "Solid_Waste_kg": 1393.2, "Yarn_Waste_kg": 541.0, "Chemical_Waste_kg": 1730.6, "Wastewater_L": 333647.9}, {"date": "2024-12-13", "Solid_Waste_kg": 1449.2, "Yarn_Waste_kg": 649.7, "Chemical_Waste_kg": 2778.7, "Wastewater_L": 510066.2}, {"date": "2024-12-14", "Solid_Waste_kg": 1083.6, "Yarn_Waste_kg": 533.6, "Chemical_Waste_kg": 2021.5, "Wastewater_L": 351567.2}, {"date": "2024-12-15", "Solid_Waste_kg": 1481.1, "Yarn_Waste_kg": 588.9, "Chemical_Waste_kg": 3029.7, "Wastewater_L": 414209.1}, {"date": "2024-12-16", "Solid_Waste_kg": 1920.4, "Yarn_Waste_kg": 765.5, "Chemical_Waste_kg": 3215.6, "Wastewater_L": 502244.0}, {"date": "2024-12-17", "Solid_Waste_kg": 1731.4, "Yarn_Waste_kg": 643.3, "Chemical_Waste_kg": 3189.6, "Wastewater_L": 528880.8}, {"date": "2024-12-18", "Solid_Waste_kg": 1260.7, "Yarn_Waste_kg": 679.8, "Chemical_Waste_kg": 2778.5, "Wastewater_L": 453909.3}, {"date": "2024-12-19", "Solid_Waste_kg": 1527.1, "Yarn_Waste_kg": 603.8, "Chemical_Waste_kg": 3304.1, "Wastewater_L": 538503.5}, {"date": "2024-12-20", "Solid_Waste_kg": 1866.4, "Yarn_Waste_kg": 914.1, "Chemical_Waste_kg": 4054.0, "Wastewater_L": 613620.8}, {"date": "2024-12-21", "Solid_Waste_kg": 1848.2, "Yarn_Waste_kg": 691.5, "Chemical_Waste_kg": 3234.6, "Wastewater_L": 442101.6}, {"date": "2024-12-22", "Solid_Waste_kg": 1444.1, "Yarn_Waste_kg": 779.2, "Chemical_Waste_kg": 3171.6, "Wastewater_L": 537174.0}, {"date": "2024-12-23", "Solid_Waste_kg": 1882.8, "Yarn_Waste_kg": 808.3, "Chemical_Waste_kg": 2373.0, "Wastewater_L": 367361.8}, {"date": "2024-12-24", "Solid_Waste_kg": 2148.2, "Yarn_Waste_kg": 583.4, "Chemical_Waste_kg": 2909.9, "Wastewater_L": 472311.6}, {"date": "2024-12-25", "Solid_Waste_kg": 1450.3, "Yarn_Waste_kg": 684.0, "Chemical_Waste_kg": 2529.7, "Wastewater_L": 530944.4}, {"date": "2024-12-26", "Solid_Waste_kg": 2246.0, "Yarn_Waste_kg": 718.8, "Chemical_Waste_kg": 3114.0, "Wastewater_L": 612651.5}, {"date": "2024-12-27", "Solid_Waste_kg": 2365.8, "Yarn_Waste_kg": 840.4, "Chemical_Waste_kg": 3872.0, "Wastewater_L": 592806.6}, {"date": "2024-12-28", "Solid_Waste_kg": 2011.1, "Yarn_Waste_kg": 931.1, "Chemical_Waste_kg": 3953.4, "Wastewater_L": 681285.6}, {"date": "2024-12-29", "Solid_Waste_kg": 1946.7, "Yarn_Waste_kg": 815.7, "Chemical_Waste_kg": 3233.7, "Wastewater_L": 501404.7}, {"date": "2024-12-30", "Solid_Waste_kg": 2310.0, "Yarn_Waste_kg": 684.2, "Chemical_Waste_kg": 3393.1, "Wastewater_L": 451015.8}, {"date": "2024-12-31", "Solid_Waste_kg": 1906.8, "Yarn_Waste_kg": 672.7, "Chemical_Waste_kg": 2288.0, "Wastewater_L": 499428.9}]};

// ─── Shift quality / Day-Type analysis (from dataset) ────────────────────────
const DAY_TYPE_DATA = [
  {
    "Day_Type": "bad",
    "Solid_Waste_%_of_Production": 16.368,
    "Yarn_Waste_%_of_Production": 6.533,
    "Chemical_Waste_kg_per_1000kg": 259.075,
    "Wastewater_L_per_kg": 41.548,
    "Machine_Utilization_%": 80.948,
    "Dye_Fixation_Ratio_%": 73.895
  },
  {
    "Day_Type": "good",
    "Solid_Waste_%_of_Production": 14.644,
    "Yarn_Waste_%_of_Production": 6.062,
    "Chemical_Waste_kg_per_1000kg": 245.638,
    "Wastewater_L_per_kg": 41.674,
    "Machine_Utilization_%": 85.784,
    "Dye_Fixation_Ratio_%": 76.329
  },
  {
    "Day_Type": "normal",
    "Solid_Waste_%_of_Production": 15.252,
    "Yarn_Waste_%_of_Production": 6.312,
    "Chemical_Waste_kg_per_1000kg": 252.313,
    "Wastewater_L_per_kg": 42.64,
    "Machine_Utilization_%": 83.209,
    "Dye_Fixation_Ratio_%": 74.774
  }
];

// ─── Pearson correlations (input features vs waste targets) ──────────────────
const CORRELATION_DATA = {
  "Solid_Waste_kg": [
    {
      "feature": "Production_Volume_ton",
      "correlation": 0.8042
    },
    {
      "feature": "Production_Volume_kg",
      "correlation": 0.8041
    },
    {
      "feature": "Total_Energy_kWh",
      "correlation": 0.7586
    },
    {
      "feature": "Total_Water_Consumption_L",
      "correlation": 0.7149
    },
    {
      "feature": "Salt_kg",
      "correlation": 0.6571
    },
    {
      "feature": "Number_of_Workers",
      "correlation": 0.6375
    },
    {
      "feature": "Monthly_Demand_Factor",
      "correlation": 0.626
    },
    {
      "feature": "Alkali_kg",
      "correlation": 0.5631
    },
    {
      "feature": "Auxiliary_kg",
      "correlation": 0.5616
    },
    {
      "feature": "Acid_kg",
      "correlation": 0.5178
    }
  ],
  "Yarn_Waste_kg": [
    {
      "feature": "Production_Volume_kg",
      "correlation": 0.8072
    },
    {
      "feature": "Production_Volume_ton",
      "correlation": 0.8072
    },
    {
      "feature": "Total_Energy_kWh",
      "correlation": 0.7595
    },
    {
      "feature": "Total_Water_Consumption_L",
      "correlation": 0.7353
    },
    {
      "feature": "Salt_kg",
      "correlation": 0.6884
    },
    {
      "feature": "Number_of_Workers",
      "correlation": 0.677
    },
    {
      "feature": "Monthly_Demand_Factor",
      "correlation": 0.6687
    },
    {
      "feature": "Dye_Used_kg",
      "correlation": 0.6098
    },
    {
      "feature": "Auxiliary_kg",
      "correlation": 0.5733
    },
    {
      "feature": "Acid_kg",
      "correlation": 0.5224
    }
  ],
  "Chemical_Waste_kg": [
    {
      "feature": "Production_Volume_ton",
      "correlation": 0.8217
    },
    {
      "feature": "Production_Volume_kg",
      "correlation": 0.8217
    },
    {
      "feature": "Salt_kg",
      "correlation": 0.7962
    },
    {
      "feature": "Total_Energy_kWh",
      "correlation": 0.7828
    },
    {
      "feature": "Total_Water_Consumption_L",
      "correlation": 0.7764
    },
    {
      "feature": "Number_of_Workers",
      "correlation": 0.7044
    },
    {
      "feature": "Monthly_Demand_Factor",
      "correlation": 0.6722
    },
    {
      "feature": "Dye_Used_kg",
      "correlation": 0.658
    },
    {
      "feature": "Auxiliary_kg",
      "correlation": 0.6406
    },
    {
      "feature": "Acid_kg",
      "correlation": 0.6015
    }
  ],
  "Wastewater_L": [
    {
      "feature": "Total_Water_Consumption_L",
      "correlation": 0.9194
    },
    {
      "feature": "Production_Volume_kg",
      "correlation": 0.8247
    },
    {
      "feature": "Production_Volume_ton",
      "correlation": 0.8247
    },
    {
      "feature": "Total_Energy_kWh",
      "correlation": 0.8001
    },
    {
      "feature": "Number_of_Workers",
      "correlation": 0.6882
    },
    {
      "feature": "Monthly_Demand_Factor",
      "correlation": 0.6741
    },
    {
      "feature": "Salt_kg",
      "correlation": 0.6691
    },
    {
      "feature": "Auxiliary_kg",
      "correlation": 0.6377
    },
    {
      "feature": "Alkali_kg",
      "correlation": 0.5379
    },
    {
      "feature": "Acid_kg",
      "correlation": 0.51
    }
  ]
};

// ─── Surrogate coefficients calibrated to 2024 XGBoost predictions ───────────
const XGB_SURROGATE = {
  "Solid_Waste_kg": {
    "intercept": 1290.2369,
    "coefs": {
      "Production_Volume_kg": 0.132091,
      "Machine_Utilization_%": -13.371269,
      "Number_of_Workers": -3.681873,
      "Total_Energy_kWh": 0.007104,
      "Total_Water_Consumption_L": 0.000382,
      "Dye_Fixation_Ratio_%": -4.870127,
      "Liquor_Ratio_1x": 11.396739,
      "Dye_Used_kg": 0.230585,
      "Salt_kg": 0.007834,
      "Alkali_kg": 0.228486,
      "Acid_kg": 0.072322,
      "Auxiliary_kg": -0.039036
    }
  },
  "Yarn_Waste_kg": {
    "intercept": 529.0048,
    "coefs": {
      "Production_Volume_kg": 0.053424,
      "Machine_Utilization_%": 1.029346,
      "Number_of_Workers": 0.302647,
      "Total_Energy_kWh": -0.00757,
      "Total_Water_Consumption_L": 0.000194,
      "Dye_Fixation_Ratio_%": -7.613032,
      "Liquor_Ratio_1x": -0.957408,
      "Dye_Used_kg": 0.184086,
      "Salt_kg": 0.017755,
      "Alkali_kg": -0.007531,
      "Acid_kg": -0.003851,
      "Auxiliary_kg": -0.017101
    }
  },
  "Chemical_Waste_kg": {
    "intercept": 1950.9738,
    "coefs": {
      "Production_Volume_kg": 0.076834,
      "Machine_Utilization_%": -6.410686,
      "Number_of_Workers": 1.801653,
      "Total_Energy_kWh": 0.00465,
      "Total_Water_Consumption_L": 0.000464,
      "Dye_Fixation_Ratio_%": -23.59219,
      "Liquor_Ratio_1x": 30.450327,
      "Dye_Used_kg": 0.982912,
      "Salt_kg": 0.297305,
      "Alkali_kg": 0.260283,
      "Acid_kg": 0.581323,
      "Auxiliary_kg": 0.087343
    }
  },
  "Wastewater_L": {
    "intercept": 81678.9736,
    "coefs": {
      "Production_Volume_kg": 8.571968,
      "Machine_Utilization_%": 355.54947,
      "Number_of_Workers": -763.16062,
      "Total_Energy_kWh": 2.078748,
      "Total_Water_Consumption_L": 0.643174,
      "Dye_Fixation_Ratio_%": -889.621402,
      "Liquor_Ratio_1x": -287.192616,
      "Dye_Used_kg": -58.534094,
      "Salt_kg": -2.106214,
      "Alkali_kg": -17.410477,
      "Acid_kg": -9.237158,
      "Auxiliary_kg": 33.135185
    }
  }
};

function runLocalPrediction(inputs) {
  const featVals = {
    "Production_Volume_kg":      inputs.Production_Volume_kg ?? 15300,
    "Machine_Utilization_%":     inputs.Machine_Utilization_pct ?? inputs["Machine_Utilization_%"] ?? 85.0,
    "Number_of_Workers":         inputs.Number_of_Workers ?? 100,
    "Total_Energy_kWh":          inputs.Total_Energy_kWh ?? 25000,
    "Total_Water_Consumption_L": inputs.Total_Water_Consumption_L ?? 750000,
    "Dye_Fixation_Ratio_%":      inputs.Dye_Fixation_Ratio_pct ?? inputs["Dye_Fixation_Ratio_%"] ?? 65.0,
    "Liquor_Ratio_1x":           inputs.Liquor_Ratio ?? inputs.Liquor_Ratio_1x ?? 8.5,
    "Dye_Used_kg":               inputs.Dye_Used_kg ?? 50.0,
    "Salt_kg":                   inputs.Salt_kg ?? 320.0,
    "Alkali_kg":                 inputs.Alkali_kg ?? 110.0,
    "Acid_kg":                   inputs.Acid_kg ?? 30.0,
    "Auxiliary_kg":              inputs.Auxiliary_kg ?? 80.0
  };

  const predictions = {};
  for (const [target, spec] of Object.entries(XGB_SURROGATE)) {
    let val = spec.intercept;
    for (const [fName, coef] of Object.entries(spec.coefs)) {
      val += coef * (featVals[fName] ?? 0);
    }
    const floor = target === "Wastewater_L" ? 1000 : 10;
    predictions[target] = Math.round(Math.max(floor, val) * 100) / 100;
  }

  const treated_wastewater_L = Math.round(
    Math.max(0, predictions.Wastewater_L - predictions.Chemical_Waste_kg * 0.8) * 100
  ) / 100;

  return { predictions, treated_wastewater_L };
}

// ─── Exact 3-Hub MILP Reverse Logistics Solver (mirrors optimizer/milp.py) ───
const DEFAULT_CCS = [
  {
    name: "CC_1", distance_km: 45, fixed_contract_cost_bdt: 5000,
    facilities: {
      solid:    { capacity_kg: 2000, revenue_per_kg: 8.0,  disposal_fee_per_kg: 0.0, process_days: 7, co2_per_kg: 0.8, circularity_rate: 0.85 },
      yarn:     { capacity_kg: 2000, revenue_per_kg: 12.0, disposal_fee_per_kg: 0.0, process_days: 7, co2_per_kg: 0.8, circularity_rate: 0.90 },
      chemical: { capacity_kg: 4000, revenue_per_kg: 0.0,  disposal_fee_per_kg: 5.0, process_days: 7, co2_per_kg: 0.8, circularity_rate: 0.20 }
    }
  },
  {
    name: "CC_2", distance_km: 30, fixed_contract_cost_bdt: 4000,
    facilities: {
      solid:    { capacity_kg: 1500, revenue_per_kg: 7.0,  disposal_fee_per_kg: 0.0, process_days: 6, co2_per_kg: 0.9, circularity_rate: 0.80 },
      yarn:     { capacity_kg: 1500, revenue_per_kg: 11.5, disposal_fee_per_kg: 0.0, process_days: 6, co2_per_kg: 0.9, circularity_rate: 0.88 },
      chemical: { capacity_kg: 4000, revenue_per_kg: 0.0,  disposal_fee_per_kg: 6.0, process_days: 6, co2_per_kg: 0.9, circularity_rate: 0.15 }
    }
  },
  {
    name: "CC_3", distance_km: 20, fixed_contract_cost_bdt: 3000,
    facilities: {
      solid:    { capacity_kg: 1000, revenue_per_kg: 6.5,  disposal_fee_per_kg: 0.0, process_days: 8, co2_per_kg: 1.0, circularity_rate: 0.75 },
      yarn:     { capacity_kg: 1000, revenue_per_kg: 10.0, disposal_fee_per_kg: 0.0, process_days: 8, co2_per_kg: 1.0, circularity_rate: 0.82 },
      chemical: { capacity_kg: 4000, revenue_per_kg: 0.0,  disposal_fee_per_kg: 4.5, process_days: 8, co2_per_kg: 1.0, circularity_rate: 0.25 }
    }
  }
];

const DEFAULT_COST = {
  holding:   { solid: 2.5, yarn: 3.0, chemical: 4.0 },
  transport: { solid: 0.05, yarn: 0.04, chemical: 0.08 },
  wc_rate: 0.0003,
  co2_penalty: 1.5,
  circularity_bonus: 2.0,
  etp: { recycling_rate: 0.75, reuse_value_per_L: 0.002, drain_cost_per_L: 0.0005 }
};

function runLocalMilp(solidKg, yarnKg, chemicalKg, wastewaterL, customCcs = null, customCost = null) {
  const ccs = customCcs || DEFAULT_CCS;
  const cost = customCost || DEFAULT_COST;
  const wastes = { solid: Math.max(0, solidKg), yarn: Math.max(0, yarnKg), chemical: Math.max(0, chemicalKg) };
  const streams = ["solid", "yarn", "chemical"];
  const n = ccs.length;

  function unitNet(w, j) {
    const fac = ccs[j].facilities[w];
    const t = cost.transport[w] * ccs[j].distance_km;
    const h = cost.holding[w];
    const rate = (w === "solid" || w === "yarn") ? fac.revenue_per_kg : fac.disposal_fee_per_kg;
    const wc = cost.wc_rate * fac.process_days * rate;
    const env = cost.co2_penalty * fac.co2_per_kg;
    const cir = cost.circularity_bonus * fac.circularity_rate;
    return (w === "solid" || w === "yarn")
      ? (fac.revenue_per_kg - t - h - wc - env + cir)
      : (-(fac.disposal_fee_per_kg + t + h + wc + env) + cir);
  }

  const recycled_L = wastewaterL * cost.etp.recycling_rate;
  const drained_L  = wastewaterL * (1 - cost.etp.recycling_rate);
  const etp_net    = recycled_L * cost.etp.reuse_value_per_L - drained_L * cost.etp.drain_cost_per_L;

  let bestZ = -Infinity;
  let bestActive = [];
  let bestAllocMap = null;

  // Exact enumeration over all 2^n - 1 non-empty subsets of Collection Centers
  for (let mask = 1; mask < (1 << n); mask++) {
    const active = [];
    let fixedCost = 0;
    for (let j = 0; j < n; j++) {
      if (mask & (1 << j)) {
        active.push(j);
        fixedCost += ccs[j].fixed_contract_cost_bdt;
      }
    }

    let feasible = true;
    let varProfit = 0;
    const allocMap = [];

    for (const w of streams) {
      let rem = wastes[w];
      const sorted = [...active].sort((a, b) => unitNet(w, b) - unitNet(w, a));
      for (const j of sorted) {
        if (rem <= 0) break;
        const cap = ccs[j].facilities[w].capacity_kg;
        const qty = Math.min(rem, cap);
        if (qty > 0.01) {
          varProfit += unitNet(w, j) * qty;
          allocMap.push({ w, j, qty });
          rem -= qty;
        }
      }
      if (rem > 0.01) {
        feasible = false;
        break;
      }
    }

    if (!feasible) continue;
    const Z = varProfit - fixedCost + etp_net;
    if (Z > bestZ) {
      bestZ = Z;
      bestActive = active;
      bestAllocMap = allocMap;
    }
  }

  if (!bestAllocMap) {
    throw new Error("Waste quantities exceed total network capacity across all Collection Centers.");
  }

  const allocations = bestAllocMap.map(({ w, j, qty }) => {
    const fac = ccs[j].facilities[w];
    const rev   = (w === "solid" || w === "yarn") ? fac.revenue_per_kg * qty : 0.0;
    const disp  = w === "chemical" ? fac.disposal_fee_per_kg * qty : 0.0;
    const trans = cost.transport[w] * ccs[j].distance_km * qty;
    const hold  = cost.holding[w] * qty;
    const rate  = (w === "solid" || w === "yarn") ? fac.revenue_per_kg : fac.disposal_fee_per_kg;
    const wc    = cost.wc_rate * fac.process_days * rate * qty;
    const env   = cost.co2_penalty * fac.co2_per_kg * qty;
    const circ  = cost.circularity_bonus * fac.circularity_rate * qty;
    const net   = rev - disp - trans - hold - wc - env + circ;
    return {
      waste_type: w,
      cc_name: ccs[j].name,
      cc_index: j,
      distance_km: ccs[j].distance_km,
      quantity_kg: Math.round(qty * 100) / 100,
      revenue_bdt: Math.round(rev * 100) / 100,
      disposal_fee_bdt: Math.round(disp * 100) / 100,
      transport_bdt: Math.round(trans * 100) / 100,
      holding_bdt: Math.round(hold * 100) / 100,
      wc_cost_bdt: Math.round(wc * 100) / 100,
      env_penalty_bdt: Math.round(env * 100) / 100,
      circ_bonus_bdt: Math.round(circ * 100) / 100,
      net_bdt: Math.round(net * 100) / 100,
      process_days: fac.process_days,
      circularity_rate: fac.circularity_rate,
      co2_kg: Math.round(fac.co2_per_kg * qty * 100) / 100
    };
  });

  const total_qty = allocations.reduce((s, a) => s + a.quantity_kg, 0);
  const total_co2 = allocations.reduce((s, a) => s + a.co2_kg, 0);
  const avg_circ  = total_qty ? allocations.reduce((s, a) => s + a.circularity_rate * a.quantity_kg, 0) / total_qty : 0;

  return {
    objective_bdt: Math.round(bestZ * 100) / 100,
    active_cc_indices: bestActive,
    active_cc_names: bestActive.map(j => ccs[j].name),
    allocations,
    etp: {
      treated_ww_L: Math.round((recycled_L + drained_L) * 100) / 100,
      recycled_L: Math.round(recycled_L * 100) / 100,
      drained_L: Math.round(drained_L * 100) / 100,
      net_value_bdt: Math.round(etp_net * 100) / 100
    },
    summary: {
      total_waste_kg: Math.round(total_qty * 100) / 100,
      total_co2_kg: Math.round(total_co2 * 100) / 100,
      avg_circularity_pct: Math.round(avg_circ * 10000) / 100,
      n_active_ccs: bestActive.length
    },
    financials: {
      revenue_bdt: Math.round(allocations.reduce((s, a) => s + a.revenue_bdt, 0) * 100) / 100,
      disposal_bdt: Math.round(allocations.reduce((s, a) => s + a.disposal_fee_bdt, 0) * 100) / 100,
      transport_bdt: Math.round(allocations.reduce((s, a) => s + a.transport_bdt, 0) * 100) / 100,
      holding_bdt: Math.round(allocations.reduce((s, a) => s + a.holding_bdt, 0) * 100) / 100,
      wc_cost_bdt: Math.round(allocations.reduce((s, a) => s + a.wc_cost_bdt, 0) * 100) / 100,
      env_penalty_bdt: Math.round(allocations.reduce((s, a) => s + a.env_penalty_bdt, 0) * 100) / 100,
      circ_bonus_bdt: Math.round(allocations.reduce((s, a) => s + a.circ_bonus_bdt, 0) * 100) / 100,
      etp_net_bdt: Math.round(etp_net * 100) / 100
    }
  };
}

// ─── Public client interface (all synchronous/Promise, zero network calls) ───
const Api = {
  getOverview() {
    return Promise.resolve(OVERVIEW_DATA);
  },

  getWastageTrend(year = 2024, stream = 'all', granularity = 'daily') {
    const y = String(year);
    if (granularity === 'monthly') {
      return Promise.resolve(TREND_MONTHLY[y] || TREND_MONTHLY["2024"]);
    }
    return Promise.resolve(TREND_DAILY[y] || TREND_DAILY["2024"]);
  },

  getDayTypeAnalysis() {
    return Promise.resolve(DAY_TYPE_DATA);
  },

  getChemicalBreakdown(year = 2024) {
    return Promise.resolve(MONTHLY_PROFILES[String(year)] || MONTHLY_PROFILES["2024"]);
  },

  getWastewaterEtp(year = 2024) {
    return Promise.resolve(ETP_PROFILES[String(year)] || ETP_PROFILES["2024"]);
  },

  getCorrelation(target = 'Solid_Waste_kg') {
    return Promise.resolve(CORRELATION_DATA[target] || CORRELATION_DATA['Solid_Waste_kg']);
  },

  predict(factoryInputs) {
    return Promise.resolve(runLocalPrediction(factoryInputs));
  },

  optimize(solidKg, yarnKg, chemicalKg, wastewaterL, ccs = null, cost = null) {
    return Promise.resolve(runLocalMilp(
      parseFloat(solidKg), parseFloat(yarnKg), parseFloat(chemicalKg), parseFloat(wastewaterL), ccs, cost
    ));
  },

  getModelMetrics() {
    return Promise.resolve(MODEL_METRICS);
  },

  getFeatureImportance(target = 'Solid_Waste_kg') {
    return Promise.resolve(FEATURE_IMPORTANCE[target] || FEATURE_IMPORTANCE['Solid_Waste_kg']);
  },

  getModelTargets() {
    return Promise.resolve(Object.keys(MODEL_METRICS));
  }
};

// ─── Global number formatters ─────────────────────────────────────────────────
const NumFormat = {
  format(val, decimals = 2) {
    if (val === undefined || val === null || isNaN(val)) return '-';
    return Number(val).toLocaleString('en-US', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    });
  },
  int(val) { return this.format(val, 0); },
  pct(val, decimals = 1) {
    if (val === undefined || val === null || isNaN(val)) return '-';
    return `${Number(val).toFixed(decimals)}%`;
  },
  currency(val, decimals = 2) {
    if (val === undefined || val === null || isNaN(val)) return '-';
    const sign = val >= 0 ? '+' : '';
    return `${sign}${this.format(val, decimals)} BDT`;
  }
};
