import os
import numpy as np
import pandas as pd
import random
import joblib
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RationalQuadratic

from mealpy import FloatVar, RIME  # DE, GA, WOA not used here

# ========= Load model and scaler =========
# Expect: gpr_model2.pkl and scaler2.pkl exist in the working directory
gpr = joblib.load("gpr_model2.pkl")
scaler = joblib.load("scaler2.pkl")

# ========= Material properties (order matters and matches bounds) =========
material_properties = {
    "OPC":    {"co2": 0.84,   "energy": 4.727, "cost": 0.08},
    "BFS":    {"co2": 0.019,  "energy": 1.588, "cost": 0.10},
    "FlyAsh": {"co2": 0.009,  "energy": 0.833, "cost": 0.60},
    "Water":  {"co2": 0.0003, "energy": 0.006, "cost": 0.04},
    "SP":     {"co2": 0.75,   "energy": 18.3,  "cost": 1.41},
    "CA":     {"co2": 0.0048, "energy": 0.083, "cost": 0.01},
    "FA":     {"co2": 0.006,  "energy": 0.114, "cost": 0.02},
    "Age":    {"co2": 0.0,    "energy": 0.0,   "cost": 0.0},
}
material_keys = list(material_properties.keys())  # preserves insertion order

# ========= Prediction function =========
def predict_strength(features):
    """features: list/array of 8 values in the same order as material_keys."""
    Xs = scaler.transform([features])
    return float(gpr.predict(Xs)[0])

# ========= Storage across runs =========
all_solutions = []
all_histories = []
all_costs = []       # per-run logged costs (every 20 calls)
all_co2s = []        # per-run logged co2 (every 20 calls)

# ========= Optimizer wrapper =========
def Optimization_Concrete(target_strength):
    """Run one optimization and log per-20-call traces."""
    iteration_costs = []
    iteration_co2s = []
    call_counter = {"count": 0}

    def objective_multi(solution):
        # Count model calls
        call_counter["count"] += 1
    
        # Compute raw objectives
        cost = sum(solution[i] * material_properties[material_keys[i]]["cost"]   for i in range(len(material_keys)))
        co2 = sum(solution[i] * material_properties[material_keys[i]]["co2"]     for i in range(len(material_keys)))
    
        # Predict compressive strength
        pred = predict_strength(solution)
    
        # --- Penalty computation (cubic penalty) ---
        eps1 = 1.0
        if pred < target_strength:
            f2 = (target_strength - pred) / target_strength   # normalized violation
            fpenalty = (1 + eps1 * f2)**3
        else:
            fpenalty = 1.0  # no penalty if within target
    
        # Apply multiplicative penalty
        cost_pen = cost * fpenalty
        co2_pen = co2 * fpenalty
    
        # Log every 20 calls
        if (call_counter["count"] % 20) == 0:
            iteration_costs.append(cost_pen)
            iteration_co2s.append(co2_pen)
    
        # Return penalized objectives
        return [cost_pen,  co2_pen]

    # Bounds align 1-to-1 with material_keys order above
    lb = [102,   0.1,   0.1, 121.75,  0.1,  801,  594, 28]
    ub = [540, 359.4, 200.1, 247.00, 32.2, 1145, 992.6, 28]

    problem_multi = {
        "obj_func": objective_multi,
        "bounds": FloatVar(lb=lb, ub=ub),
        "minmax": "min",
        # Weighted-sum reduction of 3 objectives; here you effectively minimize CO₂ only.
        # Keep as you set, or e.g., use [0.4, 0.1, 0.5] for a blended trade-off.
        "obj_weights": [0.5, 0.5],
    }

    # Optimizer settings
    model = RIME.OriginalRIME(epoch=100, pop_size=20)
    _ = model.solve(problem_multi)

    # Return the solution, history (global best fitness), and per-20-call logs
    return (model.g_best.solution,
            model.history.list_global_best_fit,
            iteration_costs,
            iteration_co2s)

# ========= Simulation loop =========
target_strength = 50.0   # e.g., [35,.. 50] MPa
runtime = 1

for run in range(runtime):
    print(f"Running optimization {run + 1}/{runtime}...")
    solution, history, costs,  co2s = Optimization_Concrete(target_strength)
    all_solutions.append(solution)
    all_histories.append(history)
    all_costs.append(costs)
    all_co2s.append(co2s)

# ========= Convert to DataFrames (pad ragged logs) =========
solutions_df = pd.DataFrame(all_solutions, columns=material_keys)

# Histories are usually same length (epochs); still guard just in case
max_hist_len = max(len(h) for h in all_histories)
histories_padded = [h + [np.nan] * (max_hist_len - len(h)) for h in all_histories]
histories_df = pd.DataFrame(histories_padded, columns=[f"Iter_{i+1}" for i in range(max_hist_len)])

def pad_and_make_df(list_of_lists, prefix):
    if not list_of_lists:
        return pd.DataFrame()
    max_len = max((len(x) for x in list_of_lists), default=0)
    padded = [(x + [np.nan] * (max_len - len(x))) for x in list_of_lists]
    return pd.DataFrame(padded, columns=[f"{prefix}_{i+1}" for i in range(max_len)])

costs_df = pad_and_make_df(all_costs, "Iter")
co2_df = pad_and_make_df(all_co2s, "Iter")

# ========= Save to CSV =========
solutions_df.to_csv("AII_optimized_solutions_20runs_RIME.csv", index=False)
histories_df.to_csv("AII_optimization_histories_20runs_RIME.csv", index=False)
costs_df.to_csv("AII_iteration_costs_RIME.csv", index=False)
co2_df.to_csv("AII_iteration_co2_RIME.csv", index=False)

print("✅ All files saved successfully!")

