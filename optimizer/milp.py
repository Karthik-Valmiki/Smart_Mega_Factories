"""
MILP Reverse Logistics Optimizer.
Solver: IBM CPLEX (via DOcplex) with scipy.optimize.milp as fallback.

Network:
    Factory  ->  Collection Center j  ->  Processing Facility (within CC j)

Waste streams:
    solid, yarn, chemical  (sent to CCs)
    wastewater             (treated at in-factory ETP, not routed to CCs)

Decision variables:
    x[w, j]  ∈ ℝ⁺   kg of waste type w sent to CC j
    y[j]     ∈ {0,1} 1 if CC j is contracted

Objective (maximize):
    Revenue (solid + yarn)
    - Disposal fees (chemical)
    - Transport costs
    - Holding & prep costs
    - Fixed contract activation costs
    - Working capital / time-value costs
    - CO2 environmental penalty
    + Circularity bonus
    + ETP net savings (constant, wastewater recycled in-factory)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

def default_cc_params(n_ccs: int = 3) -> list[dict]:
    """
    Default Collection Center parameters for 3 CCs.
    Callers can override any field.
    """
    defaults = [
        {
            "name": "CC_1",
            "distance_km": 45,
            "fixed_contract_cost_bdt": 5000,
            "facilities": {
                "solid":    {"capacity_kg": 2000, "revenue_per_kg": 8.0,  "disposal_fee_per_kg": 0.0, "process_days": 7, "co2_per_kg": 0.8, "circularity_rate": 0.85},
                "yarn":     {"capacity_kg": 2000, "revenue_per_kg": 12.0, "disposal_fee_per_kg": 0.0, "process_days": 7, "co2_per_kg": 0.8, "circularity_rate": 0.90},
                "chemical": {"capacity_kg": 4000, "revenue_per_kg": 0.0,  "disposal_fee_per_kg": 5.0, "process_days": 7, "co2_per_kg": 0.8, "circularity_rate": 0.20},
            },
        },
        {
            "name": "CC_2",
            "distance_km": 30,
            "fixed_contract_cost_bdt": 4000,
            "facilities": {
                "solid":    {"capacity_kg": 1500, "revenue_per_kg": 7.0,  "disposal_fee_per_kg": 0.0, "process_days": 6, "co2_per_kg": 0.9, "circularity_rate": 0.80},
                "yarn":     {"capacity_kg": 1500, "revenue_per_kg": 11.5, "disposal_fee_per_kg": 0.0, "process_days": 6, "co2_per_kg": 0.9, "circularity_rate": 0.88},
                "chemical": {"capacity_kg": 4000, "revenue_per_kg": 0.0,  "disposal_fee_per_kg": 6.0, "process_days": 6, "co2_per_kg": 0.9, "circularity_rate": 0.15},
            },
        },
        {
            "name": "CC_3",
            "distance_km": 20,
            "fixed_contract_cost_bdt": 3000,
            "facilities": {
                "solid":    {"capacity_kg": 1000, "revenue_per_kg": 6.5,  "disposal_fee_per_kg": 0.0, "process_days": 8, "co2_per_kg": 1.0, "circularity_rate": 0.75},
                "yarn":     {"capacity_kg": 1000, "revenue_per_kg": 10.0, "disposal_fee_per_kg": 0.0, "process_days": 8, "co2_per_kg": 1.0, "circularity_rate": 0.82},
                "chemical": {"capacity_kg": 4000, "revenue_per_kg": 0.0,  "disposal_fee_per_kg": 4.5, "process_days": 8, "co2_per_kg": 1.0, "circularity_rate": 0.25},
            },
        },
    ]
    return defaults[:n_ccs]


def default_cost_params() -> dict:
    return {
        "holding": {"solid": 2.5, "yarn": 3.0, "chemical": 4.0},      # BDT/kg
        "transport": {"solid": 0.05, "yarn": 0.04, "chemical": 0.08}, # BDT/kg/km
        "wc_rate": 0.0003,       # daily working capital rate (10% annual / 365)
        "co2_penalty": 1.5,      # BDT per kg CO2-equivalent
        "circularity_bonus": 2.0, # BDT per kg × circularity rate
        "etp": {
            "recycling_rate": 0.75,
            "reuse_value_per_L": 0.002,
            "drain_cost_per_L": 0.0005,
        },
    }


# ---------------------------------------------------------------------------
# CPLEX solver
# ---------------------------------------------------------------------------

def solve_cplex(wastes: dict, treated_ww: float, ccs: list[dict], cost: dict) -> dict:
    from docplex.mp.model import Model

    W = list(wastes.keys())
    J = list(range(len(ccs)))
    holding   = cost["holding"]
    transport = cost["transport"]
    wc_rate   = cost["wc_rate"]
    co2_pen   = cost["co2_penalty"]
    circ_bonus = cost["circularity_bonus"]
    etp        = cost["etp"]

    mdl = Model(name="ReverseLogistics")
    mdl.context.cplex_parameters.timelimit = 60

    x = {
        (w, j): mdl.continuous_var(lb=0, ub=ccs[j]["facilities"][w]["capacity_kg"], name=f"x_{w}_{j}")
        for w in W for j in J
    }
    y = {j: mdl.binary_var(name=f"y_{j}") for j in J}

    def _wc(w, j):
        fac = ccs[j]["facilities"][w]
        rate = fac["revenue_per_kg"] if w in ("solid", "yarn") else fac["disposal_fee_per_kg"]
        return wc_rate * fac["process_days"] * rate

    revenue    = mdl.sum(ccs[j]["facilities"][w]["revenue_per_kg"]     * x[w, j] for w in ("solid", "yarn") for j in J)
    disposal   = mdl.sum(ccs[j]["facilities"]["chemical"]["disposal_fee_per_kg"] * x["chemical", j] for j in J)
    trans_cost = mdl.sum(transport[w] * ccs[j]["distance_km"] * x[w, j] for w in W for j in J)
    hold_cost  = mdl.sum(holding[w] * x[w, j] for w in W for j in J)
    fixed_cost = mdl.sum(ccs[j]["fixed_contract_cost_bdt"] * y[j] for j in J)
    wc_cost    = mdl.sum(_wc(w, j) * x[w, j] for w in W for j in J)
    env_cost   = mdl.sum(co2_pen * ccs[j]["facilities"][w]["co2_per_kg"] * x[w, j] for w in W for j in J)
    circ_gain  = mdl.sum(circ_bonus * ccs[j]["facilities"][w]["circularity_rate"] * x[w, j] for w in W for j in J)

    recycled_L = treated_ww * etp["recycling_rate"]
    drained_L  = treated_ww * (1 - etp["recycling_rate"])
    etp_net    = recycled_L * etp["reuse_value_per_L"] - drained_L * etp["drain_cost_per_L"]

    mdl.maximize(revenue - disposal - trans_cost - hold_cost - fixed_cost - wc_cost - env_cost + circ_gain + etp_net)

    # Full allocation
    for w in W:
        mdl.add_constraint(mdl.sum(x[w, j] for j in J) == wastes[w], ctname=f"alloc_{w}")

    # Capacity linked to activation
    for w in W:
        for j in J:
            mdl.add_constraint(x[w, j] <= ccs[j]["facilities"][w]["capacity_kg"] * y[j], ctname=f"cap_{w}_{j}")

    sol = mdl.solve(log_output=False)
    if sol is None:
        raise RuntimeError("CPLEX found no feasible solution. Check CC capacities vs waste quantities.")

    return _extract(sol.get_value, sol.objective_value, x, y, W, J, wastes, ccs, cost, etp_net, recycled_L, drained_L)


# ---------------------------------------------------------------------------
# SciPy fallback
# ---------------------------------------------------------------------------

def solve_scipy(wastes: dict, treated_ww: float, ccs: list[dict], cost: dict) -> dict:
    from scipy.optimize import LinearConstraint, Bounds, milp

    W = list(wastes.keys())
    J = list(range(len(ccs)))
    holding   = cost["holding"]
    transport = cost["transport"]
    wc_rate   = cost["wc_rate"]
    co2_pen   = cost["co2_penalty"]
    circ_bonus = cost["circularity_bonus"]
    etp        = cost["etp"]

    n_x = len(W) * len(J)
    n_y = len(J)
    n   = n_x + n_y

    def xi(w_idx, j): return w_idx * len(J) + j
    def yi(j):        return n_x + j

    c = np.zeros(n)
    for w_idx, w in enumerate(W):
        for j in J:
            fac  = ccs[j]["facilities"][w]
            t_   = transport[w] * ccs[j]["distance_km"]
            h    = holding[w]
            rate = fac["revenue_per_kg"] if w in ("solid", "yarn") else fac["disposal_fee_per_kg"]
            wc   = wc_rate * fac["process_days"] * rate
            env  = co2_pen * fac["co2_per_kg"]
            cir  = circ_bonus * fac["circularity_rate"]
            if w in ("solid", "yarn"):
                net = fac["revenue_per_kg"] - t_ - h - wc - env + cir
            else:
                net = -(fac["disposal_fee_per_kg"] + t_ + h + wc + env) + cir
            c[xi(w_idx, j)] = -net  # negate for minimisation

    for j in J:
        c[yi(j)] = ccs[j]["fixed_contract_cost_bdt"]

    integrality = np.zeros(n)
    for j in J:
        integrality[yi(j)] = 1

    lb = np.zeros(n)
    ub = np.full(n, np.inf)
    for j in J:
        ub[yi(j)] = 1.0
    for w_idx, w in enumerate(W):
        for j in J:
            ub[xi(w_idx, j)] = ccs[j]["facilities"][w]["capacity_kg"]

    rows, lb_r, ub_r = [], [], []
    for w_idx, w in enumerate(W):
        row = np.zeros(n)
        for j in J:
            row[xi(w_idx, j)] = 1.0
        rows.append(row); lb_r.append(wastes[w]); ub_r.append(wastes[w])

    for w_idx, w in enumerate(W):
        for j in J:
            cap = ccs[j]["facilities"][w]["capacity_kg"]
            row = np.zeros(n)
            row[xi(w_idx, j)] = 1.0
            row[yi(j)]        = -cap
            rows.append(row); lb_r.append(-np.inf); ub_r.append(0.0)

    result = milp(c, constraints=LinearConstraint(np.array(rows), lb_r, ub_r),
                  integrality=integrality, bounds=Bounds(lb, ub))
    if result.status != 0:
        raise RuntimeError(f"scipy MILP failed: {result.message}")

    recycled_L = treated_ww * etp["recycling_rate"]
    drained_L  = treated_ww * (1 - etp["recycling_rate"])
    etp_net    = recycled_L * etp["reuse_value_per_L"] - drained_L * etp["drain_cost_per_L"]
    Z          = -result.fun + etp_net

    def get_val(var):
        if isinstance(var, tuple):  # x[w, j]
            w_idx = list(wastes.keys()).index(var[0])
            return float(result.x[xi(w_idx, var[1])])
        return float(result.x[yi(var)])  # y[j]

    return _extract(get_val, Z, {k: k for k in [("s", 0)]},  # dummy, overridden below
                    None, W, J, wastes, ccs, cost, etp_net, recycled_L, drained_L,
                    _raw_x=result.x, _n_x=n_x, _xi_fn=xi, _yi_fn=yi)


def _extract(get_val_fn, Z, x, y, W, J, wastes, ccs, cost, etp_net, recycled_L, drained_L,
             _raw_x=None, _n_x=None, _xi_fn=None, _yi_fn=None):
    """Shared result extraction for both solvers."""
    holding    = cost["holding"]
    transport  = cost["transport"]
    wc_rate    = cost["wc_rate"]
    co2_pen    = cost["co2_penalty"]
    circ_bonus = cost["circularity_bonus"]

    allocations = []
    for w_idx, w in enumerate(W):
        for j in J:
            if _raw_x is not None:
                qty = float(_raw_x[_xi_fn(w_idx, j)])
            else:
                qty = float(get_val_fn(x[w, j]))

            if qty < 0.01:
                continue

            fac   = ccs[j]["facilities"][w]
            rev   = fac["revenue_per_kg"] * qty if w in ("solid", "yarn") else 0.0
            disp  = fac["disposal_fee_per_kg"] * qty if w == "chemical" else 0.0
            trans = transport[w] * ccs[j]["distance_km"] * qty
            hold  = holding[w] * qty
            rate  = fac["revenue_per_kg"] if w in ("solid", "yarn") else fac["disposal_fee_per_kg"]
            wc    = wc_rate * fac["process_days"] * rate * qty
            env   = co2_pen * fac["co2_per_kg"] * qty
            circ  = circ_bonus * fac["circularity_rate"] * qty
            net   = rev - disp - trans - hold - wc - env + circ

            allocations.append({
                "waste_type":       w,
                "cc_name":          ccs[j]["name"],
                "cc_index":         j,
                "distance_km":      ccs[j]["distance_km"],
                "quantity_kg":      round(qty, 2),
                "revenue_bdt":      round(rev, 2),
                "disposal_fee_bdt": round(disp, 2),
                "transport_bdt":    round(trans, 2),
                "holding_bdt":      round(hold, 2),
                "wc_cost_bdt":      round(wc, 2),
                "env_penalty_bdt":  round(env, 2),
                "circ_bonus_bdt":   round(circ, 2),
                "net_bdt":          round(net, 2),
                "process_days":     fac["process_days"],
                "circularity_rate": fac["circularity_rate"],
                "co2_kg":           round(fac["co2_per_kg"] * qty, 2),
            })

    if _raw_x is not None:
        active_ccs = [j for j in J if _raw_x[_yi_fn(j)] > 0.5]
    else:
        active_ccs = [j for j in J if get_val_fn(y[j]) > 0.5]

    total_qty  = sum(a["quantity_kg"] for a in allocations)
    total_co2  = sum(a["co2_kg"] for a in allocations)
    avg_circ   = (sum(a["circularity_rate"] * a["quantity_kg"] for a in allocations) / total_qty) if total_qty else 0

    return {
        "objective_bdt":     round(float(Z), 2),
        "active_cc_indices": active_ccs,
        "active_cc_names":   [ccs[j]["name"] for j in active_ccs],
        "allocations":        allocations,
        "etp": {
            "treated_ww_L":    round(recycled_L + drained_L, 2),
            "recycled_L":      round(recycled_L, 2),
            "drained_L":       round(drained_L, 2),
            "net_value_bdt":   round(float(etp_net), 2),
        },
        "summary": {
            "total_waste_kg":      round(total_qty, 2),
            "total_co2_kg":        round(total_co2, 2),
            "avg_circularity_pct": round(avg_circ * 100, 2),
            "n_active_ccs":        len(active_ccs),
        },
        "financials": {
            "revenue_bdt":      round(sum(a["revenue_bdt"]      for a in allocations), 2),
            "disposal_bdt":     round(sum(a["disposal_fee_bdt"] for a in allocations), 2),
            "transport_bdt":    round(sum(a["transport_bdt"]    for a in allocations), 2),
            "holding_bdt":      round(sum(a["holding_bdt"]      for a in allocations), 2),
            "wc_cost_bdt":      round(sum(a["wc_cost_bdt"]      for a in allocations), 2),
            "env_penalty_bdt":  round(sum(a["env_penalty_bdt"]  for a in allocations), 2),
            "circ_bonus_bdt":   round(sum(a["circ_bonus_bdt"]   for a in allocations), 2),
            "etp_net_bdt":      round(float(etp_net), 2),
        },
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def optimize(
    wastes: dict,
    treated_ww: float,
    ccs: list[dict] | None = None,
    cost: dict | None = None,
) -> dict:
    """
    Run the MILP optimizer.

    Parameters
    ----------
    wastes : {"solid": float, "yarn": float, "chemical": float}  (kg)
    treated_ww : float  — litres of wastewater to ETP
    ccs   : list of CC parameter dicts (defaults to 3-CC network)
    cost  : cost parameter dict (defaults to standard values)

    Returns
    -------
    Full result dict (objective, allocations, financials, ETP, summary)
    """
    if ccs is None:
        ccs = default_cc_params()
    if cost is None:
        cost = default_cost_params()

    try:
        from docplex.mp.model import Model  # noqa: F401
        return solve_cplex(wastes, treated_ww, ccs, cost)
    except ImportError:
        return solve_scipy(wastes, treated_ww, ccs, cost)

