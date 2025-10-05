# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 21:50:35 2025

@author: Admin
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import joblib
from scipy.stats import norm, weibull_min
import time

# ----------- Setup & Model Loading ------------
def reset_random_seeds():
    os.environ['PYTHONHASHSEED'] = str(1)
    np.random.seed(1)
    random.seed(1)

def g_Probalistic(features, model_file='gpr_model2.pkl', scaler_file='scaler2.pkl'):
    gpr = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    input_data = pd.DataFrame([features], columns=scaler.feature_names_in_)
    input_data = scaler.transform(input_data)
    return gpr.predict(input_data)[0]

# ----------- MCS Function ---------------------
def Concrete_Find_Pf_Distribution_Optimized(A, N, COV, phi, distribution,YF):
    means = np.maximum(A, [102, 0.1, 0.1, 121.75, 0.1, 801, 594, 28])

    if distribution == "normal":
        stddevs = np.maximum(COV * means, 1e-6)
        samples = np.random.normal(means, stddevs, size=(N, len(A)))

    elif distribution == "lognormal":
        mu = np.log(means / np.sqrt(1 + COV**2))
        sigma = np.sqrt(np.log(1 + COV**2))
        samples = np.random.lognormal(mu, sigma, size=(N, len(A)))

    elif distribution == "weibull":
        shape_param = 2.0
        scale_param = means / np.exp(0.5772 / shape_param)
        samples = weibull_min.rvs(shape_param, scale=scale_param, size=(N, len(A)))

    else:
        raise ValueError("Unsupported distribution. Choose 'normal', 'lognormal', or 'weibull'.")

    gpr = joblib.load('gpr_model2.pkl')
    scaler = joblib.load('scaler2.pkl')
    samples_scaled = scaler.transform(samples)
    results = gpr.predict(samples_scaled)

    YF = YF
    reduced_threshold = phi * YF
    nF = np.sum(results < reduced_threshold)
    PF = nF / N

    return PF

# ----------- Subset Simulation Function -------
from ERANataf import ERANataf
from ERADist import ERADist
from SuS import SuS

def SuS_Concrete(A, N, COV, phi,YF):
    x_mean = np.maximum(A, [102, 0.1, 0.1, 121.75, 0.1, 801, 594, 28])
    x_std = np.minimum(COV * np.array(x_mean), [540, 359.4, 200.1, 247, 32.2, 1145, 992.6, 28])
    d = 8

    pi_pdf = [ERADist('normal', 'MOM', [x_mean[i], x_std[i]]) for i in range(d)]
    R = np.eye(d)
    pi_pdf = ERANataf(pi_pdf, R)

    YF = YF
    fc = phi * YF
    f_tilde = np.full(7, fc)

    f = lambda x: g_Probalistic([x[:, i] for i in range(d)])
    g = lambda x: np.sum(1 - (f(x) / f_tilde))

    print(f"\n--- Running SuS for phi={phi:.3f}, COV={COV} ---")
    p0 = 0.1
    sensitivity_analysis = 1
    samples_return = 2

    Pf_SuS= SuS(N, p0, g, pi_pdf, sensitivity_analysis, samples_return)
    return round(1-Pf_SuS, 6)

def compare_SuS_vs_MCS0(A_means, N_MCS, N, COV, phi_values, ax,YF, distribution="normal"):
    results = []

    for phi in phi_values:
        print(f"\n--- phi = {phi:.3f} ---")

        pf_mcs = Concrete_Find_Pf_Distribution_Optimized(A_means, N_MCS, COV, phi, distribution,YF)
        beta_mcs = -norm.ppf(max(pf_mcs, 1e-10))
        print(f"MCS: Pf = {pf_mcs:.6f}, β = {beta_mcs:.3f}")

        pf_sus = SuS_Concrete(A_means, N, COV, phi,YF)
        beta_sus = -norm.ppf(max(pf_sus, 1e-10))
        print(f"SuS: Pf = {pf_sus:.6f}, β = {beta_sus:.3f}")

        results.append([phi, pf_mcs, beta_mcs, pf_sus, beta_sus])

    df_results = pd.DataFrame(results, columns=["phi", "Pf_MCS", "beta_MCS", "Pf_SuS", "beta_SuS"])

    # Save CSV สำหรับแต่ละ COV
    csv_name = f"results40MPa_ver4_cov_{COV:.3f}.csv"
    df_results.to_csv(csv_name, index=False)
    print(f"📁 Results for COV={COV:.3f} saved to: {csv_name}")

    # Plotting
    ax.plot(df_results["phi"], df_results["beta_MCS"], marker='o', linestyle='--', label=f"MCS (COV={COV})")
    ax.plot(df_results["phi"], df_results["beta_SuS"], marker='s', linestyle='-', label=f"SuS (COV={COV})")

    return df_results



if __name__ == "__main__":
    start = time.time()
    
    YF=40 #MPa
    A_means = [203.00	,0.10,	0.10,	121.75,	0.61, 978.03,	594.00,28] #40 MPa
    #A_means = [299.08	,0.10,	0.10,	121.75,	0.10, 998.40,	632.52,28] #45 MPa
    # A_means = [369.60	,0.10,	0.10,	131.38,	0.10, 1019.49,	689.96,28] #50 MPa
    phi_values = np.linspace(0.5, 0.95, 50)
    COV_list = [0.10, 0.12, 0.15, 0.18]
    N_MCS = 1000000
    N = 2500

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for cov in COV_list:
        df = compare_SuS_vs_MCS0(A_means, N_MCS, N, cov, phi_values, ax,YF)

    ax.set_xlabel("Reduction Factor (phi)")
    ax.set_ylabel("Reliability Index (β)")
    ax.set_title("Comparison of β from MCS and SuS for Different COVs")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig("combined_sus_vs_mcs_plot40.png", dpi=300)
    plt.show()

    end = time.time()
    print(f"\n⏱️ Total Time: {round(end - start, 2)} seconds") #⏱️ Total Time: 12976.66 seconds


# def compare_SuS_vs_MCS1(A_means, N_MCS, N, COV, phi_values, ax,YF, distribution="lognormal"):
#     results = []

#     for phi in phi_values:
#         print(f"\n--- phi = {phi:.3f} ---")

#         pf_mcs = Concrete_Find_Pf_Distribution_Optimized(A_means, N_MCS, COV, phi, distribution,YF)
#         beta_mcs = -norm.ppf(max(pf_mcs, 1e-10))
#         print(f"MCS: Pf = {pf_mcs:.6f}, β = {beta_mcs:.3f}")

#         pf_sus = SuS_Concrete(A_means, N, COV, phi,YF)
#         beta_sus = -norm.ppf(max(pf_sus, 1e-10))
#         print(f"SuS: Pf = {pf_sus:.6f}, β = {beta_sus:.3f}")

#         results.append([phi, pf_mcs, beta_mcs, pf_sus, beta_sus])

#     df_results = pd.DataFrame(results, columns=["phi", "Pf_MCS", "beta_MCS", "Pf_SuS", "beta_SuS"])

#     # Save CSV สำหรับแต่ละ COV
#     csv_name = f"results45MPa_ver3_cov_{COV:.3f}.csv"
#     df_results.to_csv(csv_name, index=False)
#     print(f"📁 Results for COV={COV:.3f} saved to: {csv_name}")

#     # Plotting
#     ax.plot(df_results["phi"], df_results["beta_MCS"], marker='o', linestyle='--', label=f"MCS (COV={COV})")
#     ax.plot(df_results["phi"], df_results["beta_SuS"], marker='s', linestyle='-', label=f"SuS (COV={COV})")

#     return df_results



# if __name__ == "__main__":
#     start = time.time()
    
#     YF=45 #MPa
#     #A_means = [203.00	,0.10,	0.10,	121.75,	0.61, 978.03,	594.00,28] #40 MPa
#     A_means = [299.08	,0.10,	0.10,	121.75,	0.10, 998.40,	632.52,28] #45 MPa
#     # A_means = [369.60	,0.10,	0.10,	131.38,	0.10, 1019.49,	689.96,28] #50 MPa
#     phi_values = np.linspace(0.5, 0.95, 50)
#     COV_list = [0.10, 0.12, 0.15, 0.18]
#     N_MCS = 1000000
#     N = 100

#     plt.figure(figsize=(10, 6))
#     ax = plt.gca()

#     for cov in COV_list:
#         df = compare_SuS_vs_MCS1(A_means, N_MCS, N, cov, phi_values, ax,YF)

#     ax.set_xlabel("Reduction Factor (phi)")
#     ax.set_ylabel("Reliability Index (β)")
#     ax.set_title("Comparison of β from MCS and SuS for Different COVs")
#     ax.grid(True)
#     ax.legend()
#     plt.tight_layout()
#     plt.savefig("combined_sus_vs_mcs_plot45.png", dpi=300)
#     plt.show()

#     end = time.time()
#     print(f"\n⏱️ Total Time: {round(end - start, 2)} seconds") #⏱️ Total Time: 12976.66 seconds
    
# def compare_SuS_vs_MCS2(A_means, N_MCS, N, COV, phi_values, ax,YF, distribution="lognormal"):
#     results = []

#     for phi in phi_values:
#         print(f"\n--- phi = {phi:.3f} ---")

#         pf_mcs = Concrete_Find_Pf_Distribution_Optimized(A_means, N_MCS, COV, phi, distribution,YF)
#         beta_mcs = -norm.ppf(max(pf_mcs, 1e-10))
#         print(f"MCS: Pf = {pf_mcs:.6f}, β = {beta_mcs:.3f}")

#         pf_sus = SuS_Concrete(A_means, N, COV, phi,YF)
#         beta_sus = -norm.ppf(max(pf_sus, 1e-10))
#         print(f"SuS: Pf = {pf_sus:.6f}, β = {beta_sus:.3f}")

#         results.append([phi, pf_mcs, beta_mcs, pf_sus, beta_sus])

#     df_results = pd.DataFrame(results, columns=["phi", "Pf_MCS", "beta_MCS", "Pf_SuS", "beta_SuS"])

#     # Save CSV สำหรับแต่ละ COV
#     csv_name = f"results50MPa_ver3_cov_{COV:.3f}.csv"
#     df_results.to_csv(csv_name, index=False)
#     print(f"📁 Results for COV={COV:.3f} saved to: {csv_name}")

#     # Plotting
#     ax.plot(df_results["phi"], df_results["beta_MCS"], marker='o', linestyle='--', label=f"MCS (COV={COV})")
#     ax.plot(df_results["phi"], df_results["beta_SuS"], marker='s', linestyle='-', label=f"SuS (COV={COV})")

#     return df_results



# if __name__ == "__main__":
#     start = time.time()
    
#     YF=50 #MPa
#     #A_means = [203.00	,0.10,	0.10,	121.75,	0.61, 978.03,	594.00,28] #40 MPa
#     # A_means = [299.08	,0.10,	0.10,	121.75,	0.10, 998.40,	632.52,28] #45 MPa
#     A_means = [369.60	,0.10,	0.10,	131.38,	0.10, 1019.49,	689.96,28] #50 MPa ⏱️ Total Time: 660.38 seconds N_MCS = 100000
#     phi_values = np.linspace(0.5, 0.95, 50)
#     COV_list = [0.10, 0.12, 0.15, 0.18]
#     N_MCS = 1000000
#     N = 100

#     plt.figure(figsize=(10, 6))
#     ax = plt.gca()

#     for cov in COV_list:
#         df = compare_SuS_vs_MCS2(A_means, N_MCS, N, cov, phi_values, ax,YF)

#     ax.set_xlabel("Reduction Factor (phi)")
#     ax.set_ylabel("Reliability Index (β)")
#     ax.set_title("Comparison of β from MCS and SuS for Different COVs")
#     ax.grid(True)
#     ax.legend()
#     plt.tight_layout()
#     plt.savefig("combined_sus_vs_mcs_plot50.png", dpi=300)
#     plt.show()

#     end = time.time()
#     print(f"\n⏱️ Total Time: {round(end - start, 2)} seconds") #⏱️ Total Time: 12976.66 seconds



