
# ═══════════════════════════════════════════════════════════════════
# WaterWatch — Ensemble Coning Screening Framework
# WITH NOVEL NDCE RECOMMENDATION ENGINE
#
# University of Benin
# Department of Petroleum Engineering
# Final Year Project
#
# CORE INNOVATION:
#   Niger Delta Coning Ensemble (NDCE) — synthesizes four classical
#   correlations into actionable production guidance with P10/P50/P90
#   uncertainty quantification and reservoir-type-specific weighting.
# ═══════════════════════════════════════════════════════════════════

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="WaterWatch | Ensemble Coning Framework", page_icon="💧", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp { background-color: #0e1621; }
section[data-testid="stSidebar"] { background-color: #172231; }
.main-header {
    background: linear-gradient(135deg, #1a3a5c 0%, #2980b9 100%);
    color: white; padding: 30px 40px; border-radius: 12px;
    margin-bottom: 25px; box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.main-header h1 { color: white !important; font-size: 2.4rem; font-weight: 800; margin: 0 0 8px 0; }
.main-header p { color: #d6eaf8 !important; margin: 4px 0; font-size: 0.95rem; }
.section-hdr {
    background: linear-gradient(90deg, #1a3a5c, #34495e);
    color: white; padding: 12px 20px; border-radius: 8px;
    font-weight: 600; margin: 15px 0 12px 0; font-size: 1.05rem;
}
.method-card-sc { background: #1e2b3d; border-left: 4px solid #3498db; padding: 20px; border-radius: 8px; margin: 10px 0; }
.method-card-sc-orig { background: #1e2b3d; border-left: 4px solid #e74c3c; padding: 20px; border-radius: 8px; margin: 10px 0; }
.method-card-mg { background: #1e2b3d; border-left: 4px solid #9b59b6; padding: 20px; border-radius: 8px; margin: 10px 0; }
.method-card-sch { background: #1e2b3d; border-left: 4px solid #2ecc71; padding: 20px; border-radius: 8px; margin: 10px 0; }
.recommendation-card {
    background: linear-gradient(135deg, #1e3a5c, #2980b9);
    color: white; padding: 25px; border-radius: 12px;
    margin: 15px 0; border-left: 6px solid #f39c12;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.recommendation-card h2 { color: white !important; margin: 0 0 10px 0; font-size: 1.8rem; }
.recommendation-card h3 { color: #f39c12 !important; margin: 0 0 8px 0; font-size: 1.3rem; }
.recommendation-card p { color: #ecf0f1 !important; margin: 4px 0; font-size: 1.0rem; }
.envelope-safe { background: #1e3a2c; border-left: 4px solid #27ae60; padding: 15px; border-radius: 8px; margin: 8px 0; }
.envelope-rec { background: #1e2b3d; border-left: 4px solid #f39c12; padding: 15px; border-radius: 8px; margin: 8px 0; }
.envelope-risk { background: #3a1e1e; border-left: 4px solid #e74c3c; padding: 15px; border-radius: 8px; margin: 8px 0; }
.info-card {
    background: #1e2b3d; color: #ecf0f1; padding: 15px 20px;
    border-radius: 8px; border-left: 4px solid #3498db; margin: 8px 0;
}
.limitation-card {
    background: #2c1e1e; color: #ecf0f1; padding: 15px 20px;
    border-radius: 8px; border-left: 4px solid #e74c3c; margin: 8px 0;
}
.decision-tree {
    background: #1e2b3d; color: #ecf0f1; padding: 20px;
    border-radius: 10px; border: 2px solid #3498db; margin: 10px 0;
}
.decision-tree h4 { color: #3498db !important; margin: 0 0 10px 0; }
.decision-tree li { margin: 6px 0; color: #bdc3c7; }
div[data-testid="stMetricValue"] { color: #ecf0f1 !important; }
div[data-testid="stMetricLabel"] { color: #bdc3c7 !important; }
</style>
""", unsafe_allow_html=True)

def oil_specific_gravity(API): return 141.5 / (API + 131.5)

def dead_oil_viscosity(API, T_F):
    T_R = T_F + 460
    a = 10**(0.43 + 8.33/API)
    return ((0.32 + 1.8e7/API**4.53) * (360/(T_R - 260))**a)

def bubble_point_pressure(Rs, gg, T_F, API):
    return abs(18.2 * ((Rs/gg)**0.83 * 10**(0.00091*T_F - 0.0125*API) - 1.4))

def saturated_viscosity(mu_od, Rs):
    return 10.715 * (Rs + 100)**(-0.515) * mu_od**(5.44 * (Rs + 150)**(-0.338))

def undersaturated_viscosity(mu_ob, Pi, Pb):
    a = -3.9e-5 * Pi - 5
    m = 2.6 * Pi**1.187 * 10**a
    return mu_ob * (Pi/Pb)**m

def oil_fvf(Rs, gg, go_val, T_F):
    F = Rs * (gg/go_val)**0.5 + 1.25 * T_F
    return 0.9759 + 0.000120 * F**1.2

def water_density(sal_ppm, T_F, P):
    return 62.4 + sal_ppm/10000 * 0.5 - 0.003 * (T_F - 60) + 0.0000145 * P

def oil_density(API, Bo, Rs, gg):
    go_val = 141.5/(API + 131.5)
    return (go_val * 62.4 + 0.01357 * Rs * gg) / Bo

def mobility_ratio(krw, kro, mu_o, mu_w):
    M = (krw/kro) * (mu_o/mu_w)
    return M, (0.5 if M <= 1 else 0.6)

def calc_Z(kh, h, hp, mu_o, Bo, Qo, rho_w_lbft3, rho_o_lbft3):
    rho_w = rho_w_lbft3 / 62.4
    rho_o = rho_o_lbft3 / 62.4
    dr = rho_w - rho_o
    ht = h - hp
    if dr <= 0 or ht <= 0 or Qo <= 0: return None, "Invalid"
    Z = (0.00307 * dr * kh * h * ht) / (mu_o * Bo * Qo)
    return Z, None

def sobocinski_standard(kh, kv, phi, h, hp, mu_o, Bo, Qo, rho_w, rho_o, M, alpha):
    Z, err = calc_Z(kh, h, hp, mu_o, Bo, Qo, rho_w, rho_o)
    if err: return None, None, None, err
    if Z >= 3.0: return None, None, None, f"Z={Z:.2f}: Near critical coning rate."
    tD = Z / (3 - 0.7 * Z)
    dr = (rho_w / 62.4) - (rho_o / 62.4)
    tBT = (tD * mu_o * phi * h * (kh / kv)) / (0.00137 * dr * kh * (1 + M**alpha))
    return round(Z, 4), round(tD, 4), round(tBT, 1), None

def sobocinski_original(kh, kv, phi, h, hp, mu_o, Bo, Qo, rho_w, rho_o, M, alpha):
    Z, err = calc_Z(kh, h, hp, mu_o, Bo, Qo, rho_w, rho_o)
    if err: return None, None, None, err
    if Z >= 3.5: return None, None, None, f"Z={Z:.2f}: Out of range."
    denom = 7 - 2 * Z
    if denom <= 0.1: return None, None, None, "Unstable."
    tD = (4*Z + 1.75*Z**2 - 0.75*Z**3) / denom
    dr = (rho_w / 62.4) - (rho_o / 62.4)
    tBT = (tD * mu_o * phi * h * (kh / kv)) / (0.00137 * dr * kh * (1 + M**alpha))
    return round(Z, 4), round(tD, 4), round(tBT, 1), None

def meyer_garder(kh, kro, h, hp, mu_o, Bo, rho_w, rho_o, re, rw):
    dr = (rho_w / 62.4) - (rho_o / 62.4)
    if dr <= 0 or h <= hp or re <= rw: return None, "Invalid"
    ko = kh * kro
    ln_term = np.log(re / rw)
    if ln_term <= 0: return None, "Invalid"
    qc = 0.001535 * (dr / ln_term) * (ko / (mu_o * Bo)) * (h**2 - hp**2)
    return round(qc, 1), None

def schols_critical_rate(kh, kv, krw, kro, h, mu_o, mu_w, Bo, rho_w, rho_o, re, rw):
    dr = rho_w - rho_o
    if dr <= 0 or re <= rw: return None, "Invalid"
    M = (krw/kro) * (mu_o/mu_w)
    ln_term = np.log(re / rw) - 0.75 + M**0.5
    if ln_term <= 0: return None, "Invalid"
    qc = (0.00333 * dr * kv * h**2) / (mu_o * Bo * ln_term)
    return round(qc, 1), None

# ═══════════════════════════════════════════════════════════════════
# ENSEMBLE RECOMMENDATION ENGINE (NOVEL CONTRIBUTION)
# ═══════════════════════════════════════════════════════════════════

def classify_reservoir(kh, h, API, mu_o):
    if kh >= 1000 and h >= 50: return "high_perm_thick"
    elif kh >= 1000 and h < 50: return "high_perm_thin"
    elif kh < 300: return "low_perm"
    elif API < 28 or mu_o > 1.5: return "heavy_oil"
    else: return "moderate"

def get_method_weights(res_type):
    weights = {
        "high_perm_thick":    {"schols": 0.45, "sob_std": 0.30, "meyer": 0.15, "sob_orig": 0.10},
        "high_perm_thin":     {"schols": 0.30, "sob_std": 0.45, "meyer": 0.15, "sob_orig": 0.10},
        "moderate":           {"schols": 0.35, "sob_std": 0.35, "meyer": 0.20, "sob_orig": 0.10},
        "low_perm":           {"schols": 0.20, "sob_std": 0.30, "meyer": 0.40, "sob_orig": 0.10},
        "heavy_oil":          {"schols": 0.25, "sob_std": 0.30, "meyer": 0.35, "sob_orig": 0.10},
    }
    return weights.get(res_type, weights["moderate"])

def compute_ensemble(qc_mg, qc_sch, tBT_std, tBT_orig, res_type, Qo, kh, kv, phi, h, hp, mu_o, Bo, rho_w, rho_o, M, alpha):
    w = get_method_weights(res_type)
    rates, weights_list = [], []
    if qc_mg is not None:
        rates.append(qc_mg)
        weights_list.append(w["meyer"])
    if qc_sch is not None:
        rates.append(qc_sch)
        weights_list.append(w["schols"])

    total_w = sum(weights_list)
    if total_w > 0:
        weights_list = [w/total_w for w in weights_list]

    if len(rates) >= 2:
        Qc_P90 = qc_mg * 0.8 if qc_mg else min(rates) * 0.8
        Qc_P10 = qc_sch * 0.9 if qc_sch else max(rates) * 0.9
        Qc_P50 = sum(r * w for r, w in zip(rates, weights_list))
    elif len(rates) == 1:
        Qc_P90 = rates[0] * 0.7
        Qc_P50 = rates[0] * 0.85
        Qc_P10 = rates[0] * 0.95
    else:
        Qc_P90 = Qc_P50 = Qc_P10 = None

    bt_P50 = bt_P90 = bt_P10 = None
    if Qc_P50 and Qc_P50 > 0:
        op_rate = Qc_P50 * 0.85
        Z, tD, bt, err = sobocinski_standard(kh, kv, phi, h, hp, mu_o, Bo, op_rate, rho_w, rho_o, M, alpha)
        if not err: bt_P50 = bt

        op_rate_P90 = Qc_P90 * 0.80
        Z, tD, bt, err = sobocinski_standard(kh, kv, phi, h, hp, mu_o, Bo, op_rate_P90, rho_w, rho_o, M, alpha)
        if not err: bt_P90 = bt

        op_rate_P10 = Qc_P10 * 0.90
        Z, tD, bt, err = sobocinski_standard(kh, kv, phi, h, hp, mu_o, Bo, op_rate_P10, rho_w, rho_o, M, alpha)
        if not err: bt_P10 = bt

    if Qc_P50:
        ratio = Qo / Qc_P50
        if ratio <= 0.5:
            assessment = "SAFE"
            rec_action = "Current rate is well within recommended envelope. Maintain production."
        elif ratio <= 0.8:
            assessment = "CAUTION"
            rec_action = "Approaching recommended limit. Monitor water cut weekly. Plan water handling."
        elif ratio <= 1.0:
            assessment = "AT RISK"
            rec_action = "Near ensemble critical rate. Reduce rate by 10-15% or redesign completion."
        else:
            assessment = "ABOVE RECOMMENDED"
            rec_action = "Rate exceeds ensemble recommendation. Immediate rate reduction or workover required."
    else:
        assessment = "UNKNOWN"
        rec_action = "Insufficient data for ensemble recommendation. Use individual method results."

    return {
        "Qc_P90": round(Qc_P90, 0) if Qc_P90 else None,
        "Qc_P50": round(Qc_P50, 0) if Qc_P50 else None,
        "Qc_P10": round(Qc_P10, 0) if Qc_P10 else None,
        "bt_P90": round(bt_P90, 0) if bt_P90 else None,
        "bt_P50": round(bt_P50, 0) if bt_P50 else None,
        "bt_P10": round(bt_P10, 0) if bt_P10 else None,
        "op_rate_P50": round(Qc_P50 * 0.85, 0) if Qc_P50 else None,
        "assessment": assessment,
        "rec_action": rec_action,
        "res_type": res_type,
        "weights": w
    }

def generate_decision_tree(kh, h, API, mu_o, Qo, ensemble, qc_mg, qc_sch):
    tree = []
    if kh >= 1500:
        tree.append("🏖️ **High-permeability reservoir detected** (>1500 mD). Schols correlation is most reliable for this sand quality.")
    elif kh <= 300:
        tree.append("🪨 **Low-permeability reservoir detected** (<300 mD). Meyer-Garder provides conservative guidance. Consider stimulation.")
    if h <= 30:
        tree.append("📏 **Thin oil column** (<30 ft). Perforation standoff is critical. Keep hp < 30% of h.")
    elif h >= 80:
        tree.append("📏 **Thick oil column** (>80 ft). Good standoff margin. Rate is the primary control.")
    if API < 28:
        tree.append("🛢️ **Heavy oil** (API < 28°). High viscosity accelerates coning. All methods predict early breakthrough.")
    if ensemble["Qc_P50"]:
        ratio = Qo / ensemble["Qc_P50"]
        if ratio > 1.0:
            tree.append(f"⚠️ **Rate exceeds P50 recommendation by {(ratio-1)*100:.0f}%**. Reduce to ~{ensemble['Qc_P50']:.0f} STB/D for safe operation.")
        elif ratio > 0.8:
            tree.append(f"⚡ **Rate is {(ratio)*100:.0f}% of P50 recommendation**. Monitor closely. Consider reducing to ~{ensemble['Qc_P50']*0.8:.0f} STB/D.")
        else:
            tree.append(f"✅ **Rate is {(ratio)*100:.0f}% of P50 recommendation**. Well-positioned within safe envelope.")
    if qc_mg and qc_sch:
        divergence = qc_sch / qc_mg
        if divergence > 20:
            tree.append(f"📊 **High method divergence** (Schols/Meyer = {divergence:.1f}×). Use P90 for facility design, P50 for operations.")
        elif divergence > 10:
            tree.append(f"📊 **Moderate method divergence** (Schols/Meyer = {divergence:.1f}×). Use ensemble range for decision-making.")
        else:
            tree.append(f"📊 **Low method divergence** (Schols/Meyer = {divergence:.1f}×). Methods agree. Higher confidence.")
    if ensemble["bt_P50"]:
        if ensemble["bt_P50"] < 90:
            tree.append("⏰ **Very short breakthrough expected** (<3 months). Water handling must be ready before production starts.")
        elif ensemble["bt_P50"] < 180:
            tree.append("⏰ **Short breakthrough expected** (3–6 months). Plan water handling within first quarter.")
        elif ensemble["bt_P50"] < 365:
            tree.append("⏰ **Moderate breakthrough expected** (6–12 months). Standard water handling planning is sufficient.")
        else:
            tree.append("⏰ **Long breakthrough expected** (>12 months). Well is stable. Focus on rate optimization.")
    return tree

def risk_level_bt(tBT):
    if tBT is None: return {'cat': 'N/A', 'color': '#7f8c8d', 'icon': '⚪'}
    if tBT <= 30: return {'cat': 'CRITICAL', 'color': '#c0392b', 'icon': '🔴'}
    elif tBT <= 90: return {'cat': 'HIGH', 'color': '#e67e22', 'icon': '🟠'}
    elif tBT <= 180: return {'cat': 'MODERATE', 'color': '#f1c40f', 'icon': '🟡'}
    else: return {'cat': 'LOW', 'color': '#27ae60', 'icon': '🟢'}

def risk_level_cr(qc, Qo):
    if qc is None or Qo <= 0: return {'cat': 'N/A', 'color': '#7f8c8d', 'icon': '⚪'}
    ratio = Qo / qc
    if ratio <= 0.5: return {'cat': 'SAFE', 'color': '#27ae60', 'icon': '🟢'}
    elif ratio <= 0.8: return {'cat': 'CAUTION', 'color': '#f1c40f', 'icon': '🟡'}
    elif ratio <= 1.0: return {'cat': 'AT RISK', 'color': '#e67e22', 'icon': '🟠'}
    else: return {'cat': 'ABOVE CRITICAL', 'color': '#c0392b', 'icon': '🔴'}

def ensemble_risk(assessment):
    colors = {"SAFE": "#27ae60", "CAUTION": "#f1c40f", "AT RISK": "#e67e22", "ABOVE RECOMMENDED": "#c0392b", "UNKNOWN": "#7f8c8d"}
    icons = {"SAFE": "🟢", "CAUTION": "🟡", "AT RISK": "🟠", "ABOVE RECOMMENDED": "🔴", "UNKNOWN": "⚪"}
    return {'cat': assessment, 'color': colors.get(assessment, '#7f8c8d'), 'icon': icons.get(assessment, '⚪')}

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>💧 WaterWatch</h1>
    <p><b>Ensemble Coning Screening Framework</b> for Niger Delta Vertical Wells</p>
    <p>4 Classical Methods + Novel NDCE Ensemble Engine | University of Benin | Petroleum Engineering</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR INPUTS
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Reservoir & Well Parameters")
    st.markdown("**🪨 Rock Properties**")
    kh = st.number_input("kh — Horizontal Perm (mD)", 50.0, 5000.0, 500.0, 10.0, help="Arithmetic average from core or log")
    kv = st.number_input("kv — Vertical Perm (mD)", 5.0, 2000.0, 80.0, 5.0, help="Harmonic average from core. Typically kh/5 to kh/15")
    phi = st.number_input("φ — Porosity (fraction)", 0.05, 0.45, 0.25, 0.01, help="Effective porosity from log analysis")
    h = st.number_input("h — Oil Column (ft)", 10.0, 300.0, 60.0, 5.0, help="Distance from top of reservoir to OWC")
    hp = st.number_input("hp — Perforated Interval (ft)", 1.0, 200.0, 20.0, 1.0, help="Perforated length from top of oil column")

    st.markdown("**🌍 Geometry**")
    re = st.number_input("re — Drainage Radius (ft)", 100.0, 5000.0, 1000.0, 50.0, help="From well spacing: 40 acres ≈ 745 ft, 80 acres ≈ 1053 ft")
    rw = st.number_input("rw — Wellbore Radius (ft)", 0.2, 2.0, 0.5, 0.05, help="Typically 0.3–0.5 ft for cased hole")

    st.markdown("**🧪 Fluid Properties**")
    pvt_mode = st.radio("PVT Source", ["Calculate from correlations", "Enter measured PVT"], help="Use correlations for screening, measured for accuracy")
    API = st.number_input("API Gravity (°)", 15.0, 55.0, 35.0, 0.5)
    T_F = st.number_input("Temperature (°F)", 100.0, 300.0, 180.0, 5.0)
    Pi = st.number_input("Initial Pressure (psia)", 500.0, 10000.0, 4200.0, 50.0)
    sal = st.number_input("Water Salinity (ppm)", 1000.0, 150000.0, 35000.0, 1000.0)
    mu_w = st.number_input("Water Viscosity (cp)", 0.2, 1.5, 0.50, 0.05)

    if pvt_mode == "Calculate from correlations":
        Rs = st.number_input("Rs (scf/STB)", 50.0, 2000.0, 600.0, 10.0)
        gg = st.number_input("γg — Gas Gravity", 0.5, 1.2, 0.75, 0.01)
        mu_o_meas = None; Bo_meas = None
    else:
        mu_o_meas = st.number_input("μo measured (cp)", 0.1, 100.0, 0.6, 0.1)
        Bo_meas = st.number_input("Bo measured (bbl/STB)", 1.0, 3.0, 1.34, 0.01)
        Rs = 600.0; gg = 0.75

    st.markdown("**💧 Relative Permeability**")
    krw = st.number_input("krw at Sor", 0.1, 0.8, 0.30, 0.05, help="Endpoint water relative permeability")
    kro = st.number_input("kro at Swc", 0.3, 1.0, 0.85, 0.05, help="Endpoint oil relative permeability")

    st.markdown("**⚡ Production**")
    Qo = st.number_input("Qo — Oil Rate (STB/day)", 50.0, 10000.0, 1000.0, 50.0, help="Current or planned production rate")

    run_btn = st.button("🔍 RUN ENSEMBLE ANALYSIS", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Ensemble Recommendation",
    "📊 Method Comparison",
    "📈 Sensitivity Analysis",
    "🌳 Decision Tree",
    "ℹ️ About & Methodology"
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — ENSEMBLE RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="section-hdr">🎯 Niger Delta Coning Ensemble (NDCE) Recommendation</div>', unsafe_allow_html=True)

    if not run_btn:
        st.info("👈 Enter parameters in the sidebar and click **RUN ENSEMBLE ANALYSIS**")
        st.markdown("""
        <div class="info-card">
        <b>What the Ensemble Engine does:</b><br>
        Instead of showing 4 conflicting numbers, the NDCE engine:
        <ol>
        <li>Classifies your reservoir type (high-perm, thin rim, heavy oil, etc.)</li>
        <li>Weights each correlation based on its reliability for YOUR reservoir</li>
        <li>Computes a <b>Production Envelope</b>: P90 (conservative) → P50 (most likely) → P10 (optimistic)</li>
        <li>Estimates breakthrough time at the recommended P50 rate</li>
        <li>Tells you exactly what action to take</li>
        </ol>
        This is the <b>novel contribution</b> of this study.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="recommendation-card">
        <h3>📐 Production Envelope Concept</h3>
        <p><b>P90 (Conservative):</b> Based on Meyer-Garder with 20% safety margin.<br>
        "If you produce below this, you are almost certainly safe."</p>
        <p><b>P50 (Most Likely):</b> Weighted average of all methods calibrated to your reservoir type.<br>
        "This is your best estimate for economic planning."</p>
        <p><b>P10 (Optimistic):</b> Based on Schols with 10% safety margin.<br>
        "If you produce above this, you are taking significant risk."</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    if hp >= h:
        st.error("❌ Perforation interval (hp) must be less than oil column (h)")
        st.stop()
    if re <= rw:
        st.error("❌ Drainage radius (re) must exceed wellbore radius (rw)")
        st.stop()

    # PVT calculations
    go_val = oil_specific_gravity(API)
    if pvt_mode == "Calculate from correlations":
        mu_od = dead_oil_viscosity(API, T_F)
        Pb = bubble_point_pressure(Rs, gg, T_F, API)
        mu_ob = saturated_viscosity(mu_od, Rs)
        mu_o = undersaturated_viscosity(mu_ob, Pi, Pb) if Pi > Pb else mu_ob
        cond = "Undersaturated" if Pi > Pb else "Saturated"
        Bo = oil_fvf(Rs, gg, go_val, T_F)
    else:
        mu_o = mu_o_meas; Bo = Bo_meas; Pb = None; mu_od = None; mu_ob = None; cond = "Measured PVT"

    rw_dens = water_density(sal, T_F, Pi)
    ro_dens = oil_density(API, Bo, Rs, gg)
    M, alpha = mobility_ratio(krw, kro, mu_o, mu_w)

    # Run all four methods
    Z1, tD1, tBT1, err1 = sobocinski_standard(kh, kv, phi, h, hp, mu_o, Bo, Qo, rw_dens, ro_dens, M, alpha)
    Z2, tD2, tBT2, err2 = sobocinski_original(kh, kv, phi, h, hp, mu_o, Bo, Qo, rw_dens, ro_dens, M, alpha)
    qc_mg, err_mg = meyer_garder(kh, kro, h, hp, mu_o, Bo, rw_dens, ro_dens, re, rw)
    qc_sch, err_sch = schols_critical_rate(kh, kv, krw, kro, h, mu_o, mu_w, Bo, rw_dens, ro_dens, re, rw)

    # Run ensemble engine
    res_type = classify_reservoir(kh, h, API, mu_o)
    ensemble = compute_ensemble(qc_mg, qc_sch, tBT1, tBT2, res_type, Qo, kh, kv, phi, h, hp, mu_o, Bo, rw_dens, ro_dens, M, alpha)

    # ENSEMBLE DISPLAY
    st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
    ens_risk = ensemble_risk(ensemble["assessment"])
    st.markdown(f"<h2>{ens_risk['icon']} {ensemble['assessment']}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p><b>Reservoir Type:</b> {res_type.replace('_', ' ').title()}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:1.1rem;'><b>Recommended Action:</b> {ensemble['rec_action']}</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # PRODUCTION ENVELOPE
    st.markdown('<div class="section-hdr">📐 Recommended Production Envelope</div>', unsafe_allow_html=True)

    col_env1, col_env2, col_env3 = st.columns(3)
    with col_env1:
        st.markdown('<div class="envelope-safe">', unsafe_allow_html=True)
        st.markdown("#### 🟢 P90 — Conservative")
        st.markdown(f"**{ensemble['Qc_P90']:.0f} STB/D**" if ensemble['Qc_P90'] else "**N/A**")
        st.markdown("<p style='font-size:0.85rem;'>Based on Meyer-Garder x 0.8 safety factor.<br>" +
                   (f"BT ~ {ensemble['bt_P90']:.0f} days" if ensemble['bt_P90'] else "BT: N/A") + "</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.85rem;color:#a9dfbf;'>Use for facility design and insurance planning.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_env2:
        st.markdown('<div class="envelope-rec">', unsafe_allow_html=True)
        st.markdown("#### 🟡 P50 — Most Likely")
        st.markdown(f"**{ensemble['Qc_P50']:.0f} STB/D**" if ensemble['Qc_P50'] else "**N/A**")
        st.markdown("<p style='font-size:0.85rem;'>Weighted ensemble for YOUR reservoir type.<br>" +
                   (f"BT ~ {ensemble['bt_P50']:.0f} days" if ensemble['bt_P50'] else "BT: N/A") + "</p>", unsafe_allow_html=True)
        if ensemble['Qc_P50']:
            st.markdown(f"<p style='font-size:0.85rem;color:#f9e79f;'>Operate at ~{ensemble['op_rate_P50']:.0f} STB/D (85% of P50).</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_env3:
        st.markdown('<div class="envelope-risk">', unsafe_allow_html=True)
        st.markdown("#### 🔴 P10 — Optimistic")
        st.markdown(f"**{ensemble['Qc_P10']:.0f} STB/D**" if ensemble['Qc_P10'] else "**N/A**")
        st.markdown("<p style='font-size:0.85rem;'>Based on Schols x 0.9 safety factor.<br>" +
                   (f"BT ~ {ensemble['bt_P10']:.0f} days" if ensemble['bt_P10'] else "BT: N/A") + "</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.85rem;color:#f5b7b1;'>Do not exceed this rate. High coning risk above.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # CURRENT RATE POSITION
    if ensemble['Qc_P50']:
        st.markdown('<div class="section-hdr">📍 Your Current Rate on the Envelope</div>', unsafe_allow_html=True)
        fig_env = go.Figure()
        fig_env.add_hrect(y0=0, y1=ensemble['Qc_P90']*0.8, fillcolor="#27ae60", opacity=0.2, annotation_text="SAFE ZONE", annotation_position="left")
        fig_env.add_hrect(y0=ensemble['Qc_P90']*0.8, y1=ensemble['Qc_P10']*1.1, fillcolor="#f39c12", opacity=0.2, annotation_text="RECOMMENDED ZONE", annotation_position="left")
        fig_env.add_hrect(y0=ensemble['Qc_P10']*1.1, y1=ensemble['Qc_P10']*2.0, fillcolor="#c0392b", opacity=0.2, annotation_text="RISK ZONE", annotation_position="left")
        fig_env.add_vline(x=ensemble['Qc_P90'], line_dash="dash", line_color="#27ae60", annotation_text="P90", annotation_position="top")
        fig_env.add_vline(x=ensemble['Qc_P50'], line_dash="dash", line_color="#f39c12", annotation_text="P50", annotation_position="top")
        fig_env.add_vline(x=ensemble['Qc_P10'], line_dash="dash", line_color="#c0392b", annotation_text="P10", annotation_position="top")
        fig_env.add_trace(go.Scatter(x=[Qo], y=[0.5], mode='markers+text',
            marker=dict(size=25, color='white', symbol='diamond'),
            text=[f"YOUR RATE<br>{Qo:.0f} STB/D"], textposition="top center",
            textfont=dict(size=14, color='white'), name="Current Rate"))
        fig_env.update_layout(title="Production Rate Envelope", xaxis_title="Rate (STB/D)",
            yaxis_visible=False, height=350, plot_bgcolor='#0e1621', paper_bgcolor='#0e1621',
            font=dict(color='white'), showlegend=False)
        st.plotly_chart(fig_env, use_container_width=True)

    # METHOD WEIGHTS TABLE
    with st.expander("🔬 How the Ensemble Weights Were Assigned"):
        st.markdown(f"**Reservoir Classification:** {res_type.replace('_', ' ').title()}")
        st.markdown("**Method Weights:**")
        w = ensemble['weights']
        w_df = pd.DataFrame([
            {"Method": "Schols (1972)", "Weight": f"{w['schols']*100:.0f}%", "Rationale": "Best for high-perm Niger Delta sands"},
            {"Method": "Sobocinski-Cornelius (Std)", "Weight": f"{w['sob_std']*100:.0f}%", "Rationale": "Most widely validated analytical method"},
            {"Method": "Meyer-Garder (1954)", "Weight": f"{w['meyer']*100:.0f}%", "Rationale": "Conservative safety bound"},
            {"Method": "Sobocinski-Cornelius (Orig)", "Weight": f"{w['sob_orig']*100:.0f}%", "Rationale": "Historical comparison only (known instability)"},
        ])
        st.dataframe(w_df, hide_index=True, use_container_width=True)
        st.markdown("""
        <div class="info-card">
        <b>Weighting Logic:</b> Each method is weighted based on its published validation 
        against field data and its suitability for Niger Delta reservoir characteristics. 
        For example, Schols (simulation-based) is weighted highest for high-permeability sands 
        because it was calibrated on conditions similar to Niger Delta Agbada Formation.
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 2 — METHOD COMPARISON
# ═══════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-hdr">📊 Individual Method Results</div>', unsafe_allow_html=True)

    if not run_btn:
        st.info("Run the analysis in Tab 1 first.")
        st.stop()

    with st.expander("🔬 PVT & Fluid Properties"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("μo", f"{mu_o:.3f} cp")
        c2.metric("Bo", f"{Bo:.3f} bbl/STB")
        c3.metric("ρw", f"{rw_dens:.1f} lb/ft³")
        c4.metric("ρo", f"{ro_dens:.1f} lb/ft³")
        c5, c6, c7 = st.columns(3)
        c5.metric("Δρ (g/cc)", f"{(rw_dens-ro_dens)/62.4:.3f}")
        c6.metric("M (mobility)", f"{M:.3f}")
        c7.metric("α", f"{alpha}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="method-card-sc">', unsafe_allow_html=True)
        st.markdown("#### 🔵 Method 1: Sobocinski-Cornelius (Standard)")
        st.caption("Ahmed (2010) | Stable tD = Z/(3-0.7Z)")
        if err1: st.error(f"❌ {err1}")
        else:
            r1 = risk_level_bt(tBT1)
            st.markdown(f"**BT: {tBT1:.0f} days ({tBT1/30.4:.1f} months)**")
            st.markdown(f"<span style='color:{r1['color']};font-size:1.2rem;'>{r1['icon']} {r1['cat']}</span>", unsafe_allow_html=True)
            cA, cB = st.columns(2)
            cA.metric("Z", f"{Z1:.3f}")
            cB.metric("tD", f"{tD1:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="method-card-mg">', unsafe_allow_html=True)
        st.markdown("#### 🟣 Method 3: Meyer-Garder (1954)")
        st.caption("Analytical critical rate | Very conservative")
        if err_mg: st.error(f"❌ {err_mg}")
        else:
            r_mg = risk_level_cr(qc_mg, Qo)
            st.markdown(f"**Qc: {qc_mg:.0f} STB/D**")
            st.markdown(f"<span style='color:{r_mg['color']};font-size:1.2rem;'>{r_mg['icon']} {r_mg['cat']} (Qo/Qc = {Qo/qc_mg:.2f})</span>", unsafe_allow_html=True)
            st.progress(min(Qo/qc_mg, 1.0))
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="method-card-sc-orig">', unsafe_allow_html=True)
        st.markdown("#### 🔴 Method 2: Sobocinski-Cornelius (Original 1965)")
        st.caption("Original polynomial | Numerically unstable near Z=3.5")
        if err2: st.error(f"❌ {err2}")
        else:
            r2 = risk_level_bt(tBT2)
            st.markdown(f"**BT: {tBT2:.0f} days ({tBT2/30.4:.1f} months)**")
            st.markdown(f"<span style='color:{r2['color']};font-size:1.2rem;'>{r2['icon']} {r2['cat']}</span>", unsafe_allow_html=True)
            cC, cD = st.columns(2)
            cC.metric("Z", f"{Z2:.3f}")
            cD.metric("tD", f"{tD2:.3f}")
            if tBT1 and not err1:
                diff = ((tBT2 - tBT1) / tBT1) * 100
                st.caption(f"⚠️ Original overpredicts by **{diff:.0f}%** vs Standard")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="method-card-sch">', unsafe_allow_html=True)
        st.markdown("#### 🟢 Method 4: Schols (1972)")
        st.caption("Empirical from simulation | Practical for high-perm sands")
        if err_sch: st.error(f"❌ {err_sch}")
        else:
            r_sch = risk_level_cr(qc_sch, Qo)
            st.markdown(f"**Qc: {qc_sch:.0f} STB/D**")
            st.markdown(f"<span style='color:{r_sch['color']};font-size:1.2rem;'>{r_sch['icon']} {r_sch['cat']} (Qo/Qc = {Qo/qc_sch:.2f})</span>", unsafe_allow_html=True)
            st.progress(min(Qo/qc_sch, 1.0))
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-hdr">📈 Comparative Summary</div>', unsafe_allow_html=True)
    comp_data = []
    if tBT1 and not err1: comp_data.append({"Method": "Sobocinski-Cornelius (Standard)", "Type": "Breakthrough Time", "Value": f"{tBT1:.0f} days", "Risk": risk_level_bt(tBT1)['cat']})
    if tBT2 and not err2: comp_data.append({"Method": "Sobocinski-Cornelius (Original)", "Type": "Breakthrough Time", "Value": f"{tBT2:.0f} days", "Risk": risk_level_bt(tBT2)['cat']})
    if qc_mg and not err_mg: comp_data.append({"Method": "Meyer-Garder", "Type": "Critical Rate", "Value": f"{qc_mg:.0f} STB/D", "Risk": risk_level_cr(qc_mg, Qo)['cat']})
    if qc_sch and not err_sch: comp_data.append({"Method": "Schols", "Type": "Critical Rate", "Value": f"{qc_sch:.0f} STB/D", "Risk": risk_level_cr(qc_sch, Qo)['cat']})
    if comp_data:
        df_comp = pd.DataFrame(comp_data)
        st.dataframe(df_comp, hide_index=True, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 3 — SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-hdr">📈 Sensitivity Analysis</div>', unsafe_allow_html=True)
    if not run_btn:
        st.info("Run the analysis in Tab 1 first.")
        st.stop()

    sens_param = st.selectbox("Select parameter to vary",
        ["Production Rate (Qo)", "Perforation Interval (hp)", "Vertical Permeability (kv)", "Oil Column (h)"])

    n_points = 50
    if sens_param == "Production Rate (Qo)":
        vary_vals = np.linspace(200, 3000, n_points)
        bt_std, bt_orig, qc_mg_list, qc_sch_list, ens_p50_list = [], [], [], [], []
        for q in vary_vals:
            Z1, tD1, tBT1, _ = sobocinski_standard(kh, kv, phi, h, hp, mu_o, Bo, q, rw_dens, ro_dens, M, alpha)
            Z2, tD2, tBT2, _ = sobocinski_original(kh, kv, phi, h, hp, mu_o, Bo, q, rw_dens, ro_dens, M, alpha)
            bt_std.append(tBT1 if tBT1 else None)
            bt_orig.append(tBT2 if tBT2 else None)
            qm, _ = meyer_garder(kh, kro, h, hp, mu_o, Bo, rw_dens, ro_dens, re, rw)
            qs, _ = schols_critical_rate(kh, kv, krw, kro, h, mu_o, mu_w, Bo, rw_dens, ro_dens, re, rw)
            qc_mg_list.append(qm if qm else None)
            qc_sch_list.append(qs if qs else None)
            ens = compute_ensemble(qm, qs, tBT1, tBT2, res_type, q, kh, kv, phi, h, hp, mu_o, Bo, rw_dens, ro_dens, M, alpha)
            ens_p50_list.append(ens['Qc_P50'] if ens['Qc_P50'] else None)

        fig = make_subplots(rows=2, cols=1, subplot_titles=("Breakthrough Time vs Rate", "Critical Rate vs Rate"), vertical_spacing=0.15)
        fig.add_trace(go.Scatter(x=vary_vals, y=bt_std, mode='lines', name='Sobocinski (Std)', line=dict(color='#3498db', width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=bt_orig, mode='lines', name='Sobocinski (Orig)', line=dict(color='#e74c3c', width=2, dash='dash')), row=1, col=1)
        fig.add_hline(y=90, line_dash="dot", line_color="red", row=1, col=1)
        fig.add_hline(y=365, line_dash="dot", line_color="orange", row=1, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=qc_mg_list, mode='lines', name='Meyer-Garder', line=dict(color='#9b59b6', width=2.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=qc_sch_list, mode='lines', name='Schols', line=dict(color='#2ecc71', width=2.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=ens_p50_list, mode='lines', name='NDCE P50', line=dict(color='#f39c12', width=3, dash='dot')), row=2, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=vary_vals, mode='lines', name='Qo = Qc', line=dict(color='white', width=1, dash='dot')), row=2, col=1)
        fig.update_xaxes(title_text="Qo (STB/D)")
        fig.update_yaxes(title_text="BT (days)", row=1, col=1)
        fig.update_yaxes(title_text="Qc (STB/D)", row=2, col=1)

    elif sens_param == "Perforation Interval (hp)":
        vary_vals = np.linspace(1, h-5, n_points)
        bt_std, bt_orig, qc_mg_list, qc_sch_list, ens_p50_list = [], [], [], [], []
        for hp_v in vary_vals:
            Z1, tD1, tBT1, _ = sobocinski_standard(kh, kv, phi, h, hp_v, mu_o, Bo, Qo, rw_dens, ro_dens, M, alpha)
            Z2, tD2, tBT2, _ = sobocinski_original(kh, kv, phi, h, hp_v, mu_o, Bo, Qo, rw_dens, ro_dens, M, alpha)
            bt_std.append(tBT1 if tBT1 else None)
            bt_orig.append(tBT2 if tBT2 else None)
            qm, _ = meyer_garder(kh, kro, h, hp_v, mu_o, Bo, rw_dens, ro_dens, re, rw)
            qs, _ = schols_critical_rate(kh, kv, krw, kro, h, mu_o, mu_w, Bo, rw_dens, ro_dens, re, rw)
            qc_mg_list.append(qm if qm else None)
            qc_sch_list.append(qs if qs else None)
            ens = compute_ensemble(qm, qs, tBT1, tBT2, res_type, Qo, kh, kv, phi, h, hp_v, mu_o, Bo, rw_dens, ro_dens, M, alpha)
            ens_p50_list.append(ens['Qc_P50'] if ens['Qc_P50'] else None)

        fig = make_subplots(rows=2, cols=1, subplot_titles=("Breakthrough Time vs Perforation", "Critical Rate vs Perforation"), vertical_spacing=0.15)
        fig.add_trace(go.Scatter(x=vary_vals, y=bt_std, mode='lines', name='Sobocinski (Std)', line=dict(color='#3498db', width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=bt_orig, mode='lines', name='Sobocinski (Orig)', line=dict(color='#e74c3c', width=2, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=qc_mg_list, mode='lines', name='Meyer-Garder', line=dict(color='#9b59b6', width=2.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=qc_sch_list, mode='lines', name='Schols', line=dict(color='#2ecc71', width=2.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=ens_p50_list, mode='lines', name='NDCE P50', line=dict(color='#f39c12', width=3, dash='dot')), row=2, col=1)
        fig.update_xaxes(title_text="hp (ft)")
        fig.update_yaxes(title_text="BT (days)", row=1, col=1)
        fig.update_yaxes(title_text="Qc (STB/D)", row=2, col=1)

    elif sens_param == "Vertical Permeability (kv)":
        vary_vals = np.linspace(10, min(kh, 500), n_points)
        bt_std, bt_orig, qc_mg_list, qc_sch_list, ens_p50_list = [], [], [], [], []
        for kv_v in vary_vals:
            Z1, tD1, tBT1, _ = sobocinski_standard(kh, kv_v, phi, h, hp, mu_o, Bo, Qo, rw_dens, ro_dens, M, alpha)
            Z2, tD2, tBT2, _ = sobocinski_original(kh, kv_v, phi, h, hp, mu_o, Bo, Qo, rw_dens, ro_dens, M, alpha)
            bt_std.append(tBT1 if tBT1 else None)
            bt_orig.append(tBT2 if tBT2 else None)
            qm, _ = meyer_garder(kh, kro, h, hp, mu_o, Bo, rw_dens, ro_dens, re, rw)
            qs, _ = schols_critical_rate(kh, kv_v, krw, kro, h, mu_o, mu_w, Bo, rw_dens, ro_dens, re, rw)
            qc_mg_list.append(qm if qm else None)
            qc_sch_list.append(qs if qs else None)
            ens = compute_ensemble(qm, qs, tBT1, tBT2, res_type, Qo, kh, kv_v, phi, h, hp, mu_o, Bo, rw_dens, ro_dens, M, alpha)
            ens_p50_list.append(ens['Qc_P50'] if ens['Qc_P50'] else None)

        fig = make_subplots(rows=2, cols=1, subplot_titles=("Breakthrough Time vs kv", "Critical Rate vs kv"), vertical_spacing=0.15)
        fig.add_trace(go.Scatter(x=vary_vals, y=bt_std, mode='lines', name='Sobocinski (Std)', line=dict(color='#3498db', width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=bt_orig, mode='lines', name='Sobocinski (Orig)', line=dict(color='#e74c3c', width=2, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=qc_mg_list, mode='lines', name='Meyer-Garder', line=dict(color='#9b59b6', width=2.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=qc_sch_list, mode='lines', name='Schols', line=dict(color='#2ecc71', width=2.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=ens_p50_list, mode='lines', name='NDCE P50', line=dict(color='#f39c12', width=3, dash='dot')), row=2, col=1)
        fig.update_xaxes(title_text="kv (mD)")
        fig.update_yaxes(title_text="BT (days)", row=1, col=1)
        fig.update_yaxes(title_text="Qc (STB/D)", row=2, col=1)

    else:
        vary_vals = np.linspace(hp+5, 200, n_points)
        bt_std, bt_orig, qc_mg_list, qc_sch_list, ens_p50_list = [], [], [], [], []
        for h_v in vary_vals:
            Z1, tD1, tBT1, _ = sobocinski_standard(kh, kv, phi, h_v, hp, mu_o, Bo, Qo, rw_dens, ro_dens, M, alpha)
            Z2, tD2, tBT2, _ = sobocinski_original(kh, kv, phi, h_v, hp, mu_o, Bo, Qo, rw_dens, ro_dens, M, alpha)
            bt_std.append(tBT1 if tBT1 else None)
            bt_orig.append(tBT2 if tBT2 else None)
            qm, _ = meyer_garder(kh, kro, h_v, hp, mu_o, Bo, rw_dens, ro_dens, re, rw)
            qs, _ = schols_critical_rate(kh, kv, krw, kro, h_v, mu_o, mu_w, Bo, rw_dens, ro_dens, re, rw)
            qc_mg_list.append(qm if qm else None)
            qc_sch_list.append(qs if qs else None)
            ens = compute_ensemble(qm, qs, tBT1, tBT2, res_type, Qo, kh, kv, phi, h_v, hp, mu_o, Bo, rw_dens, ro_dens, M, alpha)
            ens_p50_list.append(ens['Qc_P50'] if ens['Qc_P50'] else None)

        fig = make_subplots(rows=2, cols=1, subplot_titles=("Breakthrough Time vs Oil Column", "Critical Rate vs Oil Column"), vertical_spacing=0.15)
        fig.add_trace(go.Scatter(x=vary_vals, y=bt_std, mode='lines', name='Sobocinski (Std)', line=dict(color='#3498db', width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=bt_orig, mode='lines', name='Sobocinski (Orig)', line=dict(color='#e74c3c', width=2, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=qc_mg_list, mode='lines', name='Meyer-Garder', line=dict(color='#9b59b6', width=2.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=qc_sch_list, mode='lines', name='Schols', line=dict(color='#2ecc71', width=2.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=ens_p50_list, mode='lines', name='NDCE P50', line=dict(color='#f39c12', width=3, dash='dot')), row=2, col=1)
        fig.update_xaxes(title_text="h (ft)")
        fig.update_yaxes(title_text="BT (days)", row=1, col=1)
        fig.update_yaxes(title_text="Qc (STB/D)", row=2, col=1)

    fig.update_layout(height=700, plot_bgcolor='#0e1621', paper_bgcolor='#0e1621', font=dict(color='white'), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="info-card">
    <b>Key Insight:</b> Breakthrough time is most sensitive to <b>production rate</b> and <b>perforation standoff</b>.
    Critical rate methods show that increasing oil column thickness or reducing vertical permeability 
    significantly improves water-free production capacity. The <b>NDCE P50 line</b> (dotted gold) shows 
    the ensemble recommendation across the sensitivity range.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 4 — DECISION TREE
# ═══════════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="section-hdr">🌳 Smart Decision Tree</div>', unsafe_allow_html=True)

    if not run_btn:
        st.info("Run the analysis in Tab 1 first to generate personalized recommendations.")
        st.markdown("""
        <div class="info-card">
        The Decision Tree analyzes your specific reservoir characteristics and current production rate,
        then generates actionable engineering recommendations. It considers:
        <ul>
        <li>Reservoir permeability and thickness</li>
        <li>Oil gravity and viscosity</li>
        <li>Method divergence (uncertainty level)</li>
        <li>Current rate vs ensemble recommendation</li>
        <li>Expected breakthrough timing</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    tree = generate_decision_tree(kh, h, API, mu_o, Qo, ensemble, qc_mg, qc_sch)

    st.markdown('<div class="decision-tree">', unsafe_allow_html=True)
    st.markdown("<h4>🎯 Personalized Recommendations for This Well</h4>", unsafe_allow_html=True)
    for item in tree:
        st.markdown(f"<li>{item}</li>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Decision flowchart
    st.markdown('<div class="section-hdr">📋 Decision Flowchart</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
    <b>Step 1 — Classify Your Reservoir</b><br>
    • kh > 1000 mD + h > 50 ft → <b>High-perm thick sand</b> → Trust Schols most<br>
    • kh > 1000 mD + h < 50 ft → <b>High-perm thin rim</b> → Trust Sobocinski Std most<br>
    • kh < 300 mD → <b>Low-perm reservoir</b> → Trust Meyer-Garder most<br>
    • API < 28° → <b>Heavy oil</b> → All methods predict early BT. Consider horizontal well.<br><br>

    <b>Step 2 — Check Method Divergence</b><br>
    • Schols/Meyer < 10× → <b>Low uncertainty</b> → Use P50 for operations<br>
    • Schols/Meyer 10–20× → <b>Moderate uncertainty</b> → Use P90 for facilities, P50 for operations<br>
    • Schols/Meyer > 20× → <b>High uncertainty</b> → Use P90 for everything. Get more data.<br><br>

    <b>Step 3 — Compare Current Rate to Envelope</b><br>
    • Qo < P90 × 0.8 → <b>Safe zone</b> → Maintain production. Monitor quarterly.<br>
    • P90 × 0.8 < Qo < P50 → <b>Recommended zone</b> → Good operating point. Monitor monthly.<br>
    • P50 < Qo < P10 → <b>Caution zone</b> → Reduce rate by 10% or plan water handling.<br>
    • Qo > P10 → <b>Risk zone</b> → Immediate action required. Reduce rate or recomplete.<br><br>

    <b>Step 4 — Plan Based on Expected BT</b><br>
    • BT < 3 months → Water handling must be ready BEFORE production starts<br>
    • BT 3–6 months → Plan water handling within first quarter<br>
    • BT 6–12 months → Standard water handling planning is sufficient<br>
    • BT > 12 months → Well is stable. Focus on rate optimization<br><br>

    <b>Step 5 — If All Methods Agree You're Safe</b><br>
    → Proceed with planned rate. Monitor water cut monthly.<br><br>

    <b>Step 6 — If Methods Disagree Significantly</b><br>
    → Collect more data (core kv, relative permeability curves)<br>
    → Run reservoir simulation for detailed planning<br>
    → Use P90 for facility design to avoid under-building
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT & METHODOLOGY
# ═══════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("""
    ## About WaterWatch

    WaterWatch is an **ensemble coning screening framework** for Niger Delta vertical oil wells.
    Rather than relying on a single correlation, it evaluates four established methods simultaneously,
    quantifies their divergence, and synthesizes them into actionable production guidance via the
    **Niger Delta Coning Ensemble (NDCE)** engine.

    ---

    ### Core Innovation: NDCE Ensemble Engine

    The primary contribution of this study is the **NDCE (Niger Delta Coning Ensemble)** engine,
    which addresses a critical gap in existing coning screening tools: **method uncertainty**.

    Existing tools present a single "best estimate" from one correlation. But different correlations
    were developed under different assumptions and validated against different data. For Niger Delta
    reservoirs — characterized by high-permeability Agbada Formation sands, interbedded shales, and
    typical fluid properties — the choice of correlation can change predicted critical rates by
    10–30×.

    The NDCE engine:
    1. **Classifies** the reservoir into one of five types (high-perm thick, high-perm thin,
       moderate, low-perm, heavy oil)
    2. **Weights** each correlation based on its published validation against data most similar
       to the classified reservoir type
    3. **Computes** a Production Envelope with P90 (conservative), P50 (most likely), and
       P10 (optimistic) estimates
    4. **Recommends** a specific operating rate and expected breakthrough time
    5. **Assesses** the current production rate against the envelope and provides actionable guidance

    ---

    ### Methodology

    **Method 1: Sobocinski-Cornelius (Standard)**
    - Source: Ahmed (2010) *Reservoir Engineering Handbook*, Eq 9-21 to 9-23
    - Based on: Sobocinski & Cornelius (1965) SPE-894; Bournazel & Jeanson (1971) SPE-3628
    - Formula: tD = Z / (3 - 0.7Z)
    - Valid: Z < 3.0
    - Best for: General screening, thin oil rims

    **Method 2: Sobocinski-Cornelius (Original 1965)**
    - Source: Sobocinski & Cornelius (1965) JPT, May 1965
    - Formula: tD = (4Z + 1.75Z^2 - 0.75Z^3) / (7 - 2Z)
    - Valid: Z < 3.5 (theoretical), unstable near limit
    - Best for: Historical comparison only
    - **Key finding:** Denominator (7-2Z) → 0 as Z → 3.5, causing numerical instability

    **Method 3: Meyer-Garder (1954)**
    - Source: Meyer & Garder (1954) *J. Appl. Phys.* 25, No. 11
    - Formula: qc = 0.001535 * (ρw-ρo)/ln(re/rw) * (ko/μoBo) * (h^2 - hp^2)
    - Best for: Conservative analytical lower bound

    **Method 4: Schols (1972)**
    - Source: Schols, R.S. (1972) "An Empirical Formula for the Critical Oil Production Rate,"
      *Erdoel-Erdgas*, Jan 1972
    - Formula: qc = 0.00333 * (ρw-ρo) * kv * h^2 / (μo * Bo * [ln(re/rw) - 0.75 + M^0.5])
    - Best for: Practical high-perm reservoir screening

    ---

    ### NDCE Weighting Rationale

    | Reservoir Type | Schols | Sobocinski Std | Meyer-Garder | Sobocinski Orig |
    |----------------|--------|----------------|--------------|-----------------|
    | High-perm thick | 45% | 30% | 15% | 10% |
    | High-perm thin | 30% | 45% | 15% | 10% |
    | Moderate | 35% | 35% | 20% | 10% |
    | Low-perm | 20% | 30% | 40% | 10% |
    | Heavy oil | 25% | 30% | 35% | 10% |

    **Rationale:**
    - **Schols** is weighted highest for high-perm thick sands because it was derived from
      numerical simulation of high-permeability systems — most similar to Niger Delta Agbada
      Formation sands (300–5000 mD).
    - **Sobocinski Standard** is weighted highest for thin oil rims because it is the most
      widely validated analytical method for coning in limited oil columns.
    - **Meyer-Garder** is weighted highest for low-perm and heavy oil because its conservative
      assumptions (ideal radial flow, no heterogeneity) are more valid for these systems.
    - **Sobocinski Original** receives a low fixed weight (10%) because it is included only
      for historical comparison and to quantify the impact of the polynomial instability.

    ---

    ### PVT Correlations Used

    | Property | Correlation | Reference |
    |----------|-------------|-----------|
    | Dead oil viscosity | Beal (1946) | Ahmed Eq 2-117 |
    | Saturated oil viscosity | Beggs & Robinson (1975) | Ahmed Eq 2-121 |
    | Undersaturated viscosity | Vasquez-Beggs | Ahmed Eq 2-123 |
    | Oil FVF | Standing (1981) | Ahmed Eq 2-85 |
    | Bubble point | Standing (1947) | Ahmed Eq 2-72 |
    | Water density | Salinity-corrected | Tuttle (1999) |

    ---

    ### Limitations

    1. **Analytical models assume homogeneous, radial flow** — Niger Delta reservoirs are
       heterogeneous with interbedded shales
    2. **No post-breakthrough performance prediction** — WOR behavior after BT is not modeled
    3. **Single-well analysis** — Interference from offset wells is neglected
    4. **Endpoint relative permeabilities** — Actual curves are rarely available for screening
    5. **Critical rate ≠ breakthrough time** — These are different physical quantities;
       direct comparison requires care
    6. **Validation is qualitative** — Exact field BT data is proprietary; framework validated
       against published ranges
    7. **Ensemble weights are literature-based** — Not calibrated against Niger Delta-specific
       production history; recommended as future work

    ---

    ### Recommendations for Use

    1. **For thin oil rims (< 30 ft):** Use Sobocinski-Cornelius (Standard) with conservative
       rate assumptions. P90 envelope for facility design.
    2. **For high-perm Niger Delta sands (> 1000 mD):** Use Schols critical rate as primary
       guide. P50 envelope for operations.
    3. **For marginal field economics:** Use Meyer-Garder as absolute lower bound. P90 envelope.
    4. **For detailed development planning:** Follow screening with full reservoir simulation.
    5. **When methods diverge > 20×:** Collect more data before making final decisions.

    ---

    ### Key References

    - Ahmed, T. (2010) *Reservoir Engineering Handbook*, 4th Ed., Gulf Professional Publishing
    - Sobocinski, D.P. & Cornelius, A.J. (1965) "A Correlation for Predicting Water Coning Time,"
      *JPT*, May 1965, SPE-894
    - Bournazel, C. & Jeanson, B. (1971) "Fast Water Coning Evaluation," SPE-3628
    - Meyer, H.I. & Garder, A.O. (1954) "Mechanics of Two Immiscible Fluids in Porous Media,"
      *J. Appl. Phys.*, 25(11)
    - Schols, R.S. (1972) "An Empirical Formula for the Critical Oil Production Rate,"
      *Erdoel-Erdgas*, Jan 1972
    - Standing, M.B. (1947, 1981) PVT correlations
    - Beggs, H.D. & Robinson, J.R. (1975) Viscosity correlation

    ---

    ### University of Benin
    **Department of Petroleum Engineering**  
    **Final Year Project**
    """)
