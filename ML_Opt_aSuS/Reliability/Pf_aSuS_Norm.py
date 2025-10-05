# === Pf_aSuS_Norm_ready.py ===
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import joblib

# ---------- (A) จัดการ/ปิดคำเตือนจาก scikit-learn รุ่นไม่ตรง ----------
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ---------- (B) โหลดไลบรารี Subset Simulation / ERANataf ----------
from ERANataf import ERANataf
from ERADist import ERADist
from SuS import SuS  # ใช้ฟังก์ชัน SuS ของคุณ

# ---------- (C) ตั้งค่าชื่อไฟล์โมเดล ----------
GPR_MODEL_FILE = "gpr_model2.pkl"      # ถ้ารี-dump สำหรับ 1.4.2: "gpr_model_skl142.pkl"
SCALER_FILE    = "scaler2.pkl"         # ถ้ารี-dump สำหรับ 1.4.2: "scaler_skl142.pkl"

# ---------- (D) โหลดโมเดล & สเกลเลอร์ ครั้งเดียว ----------
gpr = joblib.load(GPR_MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

# ตรวจลำดับฟีเจอร์ให้ตรงกับตอนเทรน
FEATURE_ORDER = list(getattr(scaler, "feature_names_in_", [
    "OPC","BFS","FlyAsh","Water","SP","CA","FA","Age"
]))

# ---------- (E) แทนที่ surrogate ให้รับเวกเตอร์ ----------
def g_Probalistic(X):
    """
    พยากรณ์กำลังอัดด้วย GPR (รองรับ X ขนาด (n_samples, 8) หรือ (8,))
    คืนค่า y ขนาด (n_samples,)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    # ให้คอลัมน์ตรงกับ feature_names_in_
    df = pd.DataFrame(X, columns=FEATURE_ORDER)
    Xs = scaler.transform(df)
    y = gpr.predict(Xs)
    return np.asarray(y, dtype=float)

# ---------- (F) ตัวช่วยอ่านค่าจาก SuS แบบยืดหยุ่น ----------
def _extract_Pf_from_SuS_return(sus_ret):
    """
    ทำให้รองรับ SuS() ทั้งที่คืนค่า scalar (Pf_SuS) หรือ tuple หลายช่อง
    """
    if isinstance(sus_ret, (float, np.floating)):
        return float(sus_ret)
    if isinstance(sus_ret, (list, tuple, np.ndarray)):
        if len(sus_ret) == 0:
            raise RuntimeError("SuS returned an empty result.")
        return float(sus_ret[0])
    raise RuntimeError(f"Unexpected SuS return type: {type(sus_ret)}")

# ---------- (G) รัน SuS 1 ครั้งสำหรับ A, COV, phi ----------
def SuS_Concrete(A, N_samples, COV, phi):
    """
    A: เวกเตอร์ค่าเฉลี่ยความยาว 8 ตาม FEATURE_ORDER
    N_samples: จำนวน sample ต่อระดับใน SuS
    COV: coefficient of variation (ต่อแปร)
    phi: reduction factor (fc = phi * YF)
    """
    # ขอบเขต 
    lb = np.array([102,   0.01,  0.01, 121.75,  0.01,  801,  594, 28.0], dtype=float)
    ub = np.array([540, 359.4, 200.1, 247.00, 32.20, 1145, 992.6, 28.0], dtype=float)

    A = np.asarray(A, dtype=float)
    if A.shape != (8,):
        raise ValueError("A must be a length-8 vector in FEATURE_ORDER.")

    means = np.maximum(A, lb)
    # ใช้ std = COV * mean (กันศูนย์)
    stds  = np.maximum(1e-9, COV * means)

    # สร้างมาร์จินัลอิสระ
    d = 8
    pi = [ERADist('normal', 'MOM', [means[i], stds[i]]) for i in range(d)]
    R = np.eye(d)
    pi_pdf = ERANataf(pi, R)

    # limit-state: g(X) = f_pred - fc (fail ถ้า g <= 0)
    YF = 40.0
    fc = phi * YF

    def g_fun(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        f_pred = g_Probalistic(X)
        return f_pred - fc

    # พารามิเตอร์ SuS
    p0 = 0.1
    sensitivity_analysis = 1
    samples_return = 2

    print("\n\nSUBSET SIMULATION:")
    sus_ret = SuS(N_samples, p0, g_fun, pi_pdf, sensitivity_analysis, samples_return)
    Pf_SuS = _extract_Pf_from_SuS_return(sus_ret)
    return round(Pf_SuS, 6)

# ---------- (H) วิเคราะห์ phi-β และบันทึกผล ----------
def analyze_phi_vs_Beta_SuS(A, N_samples, COV_values, phi_values, fig_name="phi_vs_beta_sus_plot.png"):
    all_results = []
    plt.figure(figsize=(8, 5))

    for COV in COV_values:
        rows = []
        for phi in phi_values:
            pf_sus = SuS_Concrete(A, N_samples, COV, phi)
            beta = -norm.ppf(np.clip(pf_sus, 1e-12, 1-1e-12))
            rows.append((COV, phi, beta))
            print(f"COV: {COV:.3f}, phi: {phi:.3f}, Pf: {pf_sus:.6e}, β: {beta:.3f}")

        df = pd.DataFrame(rows, columns=["COV", "phi", "β"])
        all_results.append(df)
        plt.plot(df["phi"], df["β"], marker='o', label=f"COV = {COV}")

    plt.xlabel("Reduction Factor (phi)")
    plt.ylabel("Reliability Index (β)")
    plt.title("Reliability Index (β) vs. phi using Subset Simulation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_name, dpi=300)
    plt.show()

    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv("phi_vs_beta_results.csv", index=False)
    print("✅ Results saved to 'phi_vs_beta_results.csv'")
    return df_all

# ---------- (I) main ----------
if __name__ == "__main__":
    np.random.seed(12345)  # ทำให้ reproducible ขึ้นเล็กน้อย
    start = time.time()

    # ควรตรงกับลำดับ FEATURE_ORDER
    A_means = np.array([139.14, 37.89, 0.10, 121.75, 12.14, 801.00, 630.48, 28.0], dtype=float)

    phi_values = np.linspace(0.75, 0.95, 5)
    COV_values = [0.10]   # ลอง [0.05, 0.10, 0.15, 0.20]
    N = 2500              # samples ต่อระดับใน SuS

    df_sus_results = analyze_phi_vs_Beta_SuS(
        A=A_means, N_samples=N, COV_values=COV_values, phi_values=phi_values
    )

    end = time.time()
    print(f"\n⏱️ Total Time: {round(end - start, 2)} seconds")
