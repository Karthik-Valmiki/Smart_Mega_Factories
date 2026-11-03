/**
 * Textile Dyeing Waste Intelligence API client
 * Fetches dashboard telemetry, predictions, and MILP optimization
 */

const API_BASE = window.location.origin.includes(':8000')
  ? ''
  : (window.location.protocol.startsWith('http') ? 'http://localhost:8000' : 'http://localhost:8000');

const Api = {
  async fetchJson(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    try {
      const res = await fetch(url, {
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
          ...(options.headers || {})
        },
        ...options
      });
      if (!res.ok) {
        let errorDetail = `HTTP ${res.status}`;
        try {
          const errJson = await res.json();
          errorDetail = errJson.detail || errorDetail;
        } catch (_) { }
        throw new Error(errorDetail);
      }
      return await res.json();
    } catch (err) {
      console.warn(`[API] Failed ${endpoint}:`, err);
      throw err;
    }
  },

  // Health
  getHealth() {
    return this.fetchJson('/health');
  },

  // Dashboard Analytics
  getOverview() {
    return this.fetchJson('/dashboard/overview');
  },

  getWastageTrend(year = 2024, stream = 'all', granularity = 'daily') {
    return this.fetchJson(`/dashboard/wastage-trend?year=${year}&stream=${stream}&granularity=${granularity}`);
  },

  getDayTypeAnalysis() {
    return this.fetchJson('/dashboard/day-type-analysis');
  },

  getProductionVsWaste(year = 2024) {
    return this.fetchJson(`/dashboard/production-vs-waste?year=${year}`);
  },

  getMonthlySummary(year = 2024) {
    return this.fetchJson(`/dashboard/monthly-summary?year=${year}`);
  },

  getChemicalBreakdown(year = 2024) {
    return this.fetchJson(`/dashboard/chemical-input-breakdown?year=${year}`);
  },

  getWastewaterEtp(year = 2024) {
    return this.fetchJson(`/dashboard/wastewater-etp?year=${year}`);
  },

  getCorrelation(target = 'Solid_Waste_kg') {
    return this.fetchJson(`/dashboard/correlation?target=${encodeURIComponent(target)}`);
  },

  // Machine Learning
  predict(factoryInputs) {
    return this.fetchJson('/predict', {
      method: 'POST',
      body: JSON.stringify({ factory_inputs: factoryInputs })
    });
  },

  getModelMetrics() {
    return this.fetchJson('/model/metrics');
  },

  getFeatureImportance(target = 'Solid_Waste_kg') {
    return this.fetchJson(`/model/feature-importance?target=${encodeURIComponent(target)}`);
  },

  getModelTargets() {
    return this.fetchJson('/model/targets');
  },

  // MILP Optimizer
  getOptimizerDefaults() {
    return this.fetchJson('/optimizer/defaults');
  },

  optimize(solidKg, yarnKg, chemicalKg, wastewaterL, ccs = null, cost = null) {
    return this.fetchJson('/optimize', {
      method: 'POST',
      body: JSON.stringify({
        solid_kg: parseFloat(solidKg),
        yarn_kg: parseFloat(yarnKg),
        chemical_kg: parseFloat(chemicalKg),
        wastewater_L: parseFloat(wastewaterL),
        ccs: ccs,
        cost: cost
      })
    });
  },

  predictAndOptimize(factoryInputs, ccs = null, cost = null) {
    return this.fetchJson('/predict-and-optimize', {
      method: 'POST',
      body: JSON.stringify({
        factory_inputs: factoryInputs,
        ccs: ccs,
        cost: cost
      })
    });
  }
};

// Global formatters for tabular numbers (Roboto Mono digits)
const NumFormat = {
  format(val, decimals = 2) {
    if (val === undefined || val === null || isNaN(val)) return '-';
    return Number(val).toLocaleString('en-US', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    });
  },
  int(val) {
    return this.format(val, 0);
  },
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

// Auto-check health status on page load and update pill with Google Sans typography
document.addEventListener('DOMContentLoaded', () => {
  const healthPill = document.getElementById('systemHealthPill');
  const healthDot = document.getElementById('systemHealthDot');
  const healthText = document.getElementById('systemHealthText');

  if (healthPill || healthText) {
    Api.getHealth().then(data => {
      if (data.status === 'ok') {
        if (healthPill) {
          healthPill.className = 'h-8 flex items-center gap-2 px-3 rounded-md bg-emerald-50/80 border border-emerald-200 text-emerald-800 transition-colors shadow-xs';
        }
        if (healthDot) {
          healthDot.className = 'status-dot status-dot-live';
        }
        if (healthText) {
          healthText.textContent = data.model_loaded ? 'Model active · Online' : 'API connected';
          healthText.className = 'text-[12px] font-medium text-emerald-800';
        }
      }
    }).catch(() => {
      if (healthPill) {
        healthPill.className = 'h-8 flex items-center gap-2 px-3 rounded-md bg-amber-50/80 border border-amber-200 text-amber-800 transition-colors shadow-xs';
      }
      if (healthDot) {
        healthDot.className = 'status-dot status-dot-warning';
      }
      if (healthText) {
        healthText.textContent = 'API offline · Mock data';
        healthText.className = 'text-[12px] font-medium text-amber-800';
      }
    });
  }
});
