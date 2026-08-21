# ═══════════════════════════════════════════════════════════════════
# WATERWATCH
# Ensemble Framework for Water Breakthrough Time Prediction
# in Niger Delta Vertical Wells
#
# University of Benin
# Department of Petroleum Engineering
# Final Year Project
#
# METHODOLOGY:
# Five established published correlations evaluated simultaneously
# with uncertainty quantification via P10/P50/P90 range
# ═══════════════════════════════════════════════════════════════════

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="WaterWatch | Ensemble Framework",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Professional Dark Theme ─────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0e1621; }
section[data-testid="stSidebar"] {
    background-color: #172231;
}
.main-header {
    background: linear-gradient(135deg,
                #1a3a5c 0%, #2980b9 100%);
    color: white;
    padding: 30px 40px;
    border-radius: 12px;
    margin-bottom: 25px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.main-header h1 {
    color: white !important;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0 0 8px 0;
}
.main-header p {
    color: #d6eaf8 !important;
    margin: 4px 0;
    font-size: 0.95rem;
}
.section-hdr {
    background: linear-gradient(90deg,
                #1a3a5c, #34495e);
    color: white;
    padding: 12px 20px;
    border-radius: 8px;
    font-weight: 600;
    margin: 15px 0 12px 0;
    font-size: 1.05rem;
}
.method-card {
    background: #1e2b3d;
    border-left: 4px solid #3498db;
    padding: 20px;
    border-radius: 8px;
    margin: 10px 0;
    color: #ecf0f1;
}
.method-card h4 {
    color: #3498db !important;
    margin: 0 0 8px 0;
}
.ensemble-card {
    background: linear-gradient(135deg,
                #1e3a5c, #2980b9);
    color: white;
    padding: 25px;
    border-radius: 12px;
    margin: 15px 0;
    border-left: 6px solid #f39c12;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.ensemble-card h2 {
    color: white !important;
    margin: 0 0 10px 0;
    font-size: 1.8rem;
}
.ensemble-card h3 {
    color: #f39c12 !important;
    margin: 8px 0;
}
.ensemble-card p {
    color: #ecf0f1 !important;
    margin: 4px 0;
}
.stat-card {
    background: #1e2b3d;
    color: #ecf0f1;
    padding: 15px 20px;
    border-radius: 8px;
    border-left: 4px solid #3498db;
    margin: 8px 0;
}
.info-card {
    background: #1e2b3d;
    color: #ecf0f1;
    padding: 15px 20px;
    border-radius: 8px;
    border-left: 4px solid #3498db;
    margin: 8px 0;
}
.warning-card {
    background: #2c1e1e;
    color: #ecf0f1;
    padding: 15px 20px;
    border-radius: 8px;
    border-left: 4px solid #e67e22;
    margin: 8px 0;
}
div[data-testid="stMetricValue"] {
    color: #ecf0f1 !important;
}
div[data-testid="stMetricLabel"] {
    color: #bdc3c7 !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PVT AND FLUID PROPERTY CORRELATIONS
# ═══════════════════════════════════════════════════════════════════

def oil_specific_gravity(API):
    return 141.5 / (API + 131.5)

def dead_oil_viscosity(API, T_F):
    """Beal (1946) / Standing (1981)
    Ahmed Eq 2-117"""
    T_R = T_F + 460
    a = 10**(0.43 + 8.33/API)
    return ((0.32 + 1.8e7/API**4.53) *
            (360/(T_R - 260))**a)

def bubble_point_pressure(Rs, gg, T_F, API):
    """Standing (1947) - Ahmed Eq 2-72"""
    return abs(18.2 * ((Rs/gg)**0.83 *
               10**(0.00091*T_F -
                    0.0125*API) - 1.4))

def saturated_viscosity(mu_od, Rs):
    """Beggs & Robinson (1975)
    Ahmed Eq 2-121"""
    return (10.715 * (Rs + 100)**(-0.515) *
            mu_od**(5.44 *
                    (Rs + 150)**(-0.338)))

def undersaturated_viscosity(mu_ob, Pi, Pb):
    """Ahmed Eq 2-123"""
    a = -3.9e-5 * Pi - 5
    m = 2.6 * Pi**1.187 * 10**a
    return mu_ob * (Pi/Pb)**m

def oil_fvf(Rs, gg, go_val, T_F):
    """Standing (1981) - Ahmed Eq 2-85"""
    F = Rs * (gg/go_val)**0.5 + 1.25 * T_F
    return 0.9759 + 0.000120 * F**1.2

def water_density(sal_ppm, T_F, P):
    return (62.4 + sal_ppm/10000 * 0.5 -
            0.003 * (T_F - 60) +
            0.0000145 * P)

def oil_density(API, Bo, Rs, gg):
    go_val = 141.5/(API + 131.5)
    return ((go_val * 62.4 +
             0.01357 * Rs * gg) / Bo)

def mobility_ratio(krw, kro, mu_o, mu_w):
    M = (krw/kro) * (mu_o/mu_w)
    return M, (0.5 if M <= 1 else 0.6)

# ═══════════════════════════════════════════════════════════════════
# METHOD 1: SOBOCINSKI-CORNELIUS (STANDARD)
# Ahmed (2010) Reservoir Engineering Handbook
# Equations 9-21 to 9-23 (Standard form)
# ═══════════════════════════════════════════════════════════════════

def method_1_sobocinski_standard(
        kh, kv, phi, h, hp, mu_o, Bo, Qo,
        rho_w, rho_o, M, alpha):
    """
    Sobocinski-Cornelius Standard Form
    Reference: Ahmed (2010) Eq 9-21 to 9-23
    tD = Z / (3 - 0.7Z)
    """
    dr = rho_w - rho_o
    if dr <= 0 or h <= hp or Qo <= 0:
        return None, "Invalid inputs"

    Z = (0.492e-4 * dr * kh * h *
         (h - hp)) / (mu_o * Bo * Qo)

    if Z <= 0 or Z >= 3.0:
        return None, f"Z={Z:.3f} out of range"

    tD = Z / (3 - 0.7 * Z)
    tBT = ((20325 * mu_o * h * phi * tD) /
           (dr * kv * (1 + M**alpha)))

    return round(tBT, 1), None

# ═══════════════════════════════════════════════════════════════════
# METHOD 2: SOBOCINSKI-CORNELIUS (ORIGINAL 1965)
# Sobocinski & Cornelius (1965) SPE-894
# Original polynomial form
# ═══════════════════════════════════════════════════════════════════

def method_2_sobocinski_original(
        kh, kv, phi, h, hp, mu_o, Bo, Qo,
        rho_w, rho_o, M, alpha):
    """
    Sobocinski-Cornelius Original 1965
    Reference: Sobocinski & Cornelius (1965)
    JPT SPE-894
    tD = (4Z + 1.75Z² - 0.75Z³) / (7 - 2Z)
    """
    dr = rho_w - rho_o
    if dr <= 0 or h <= hp or Qo <= 0:
        return None, "Invalid inputs"

    Z = (0.492e-4 * dr * kh * h *
         (h - hp)) / (mu_o * Bo * Qo)

    if Z <= 0 or Z >= 3.5:
        return None, f"Z={Z:.3f} out of range"

    denom = 7 - 2 * Z
    if denom <= 0.1:
        return None, "Unstable"

    tD = (4*Z + 1.75*Z**2 - 0.75*Z**3) / denom
    tBT = ((20325 * mu_o * h * phi * tD) /
           (dr * kv * (1 + M**alpha)))

    return round(tBT, 1), None

# ═══════════════════════════════════════════════════════════════════
# METHOD 3: BOURNAZEL-JEANSON (1971)
# Bournazel & Jeanson (1971) SPE-3628
# Modified Sobocinski correlation
# ═══════════════════════════════════════════════════════════════════

def method_3_bournazel_jeanson(
        kh, kv, phi, h, hp, mu_o, Bo, Qo,
        rho_w, rho_o, M, alpha):
    """
    Bournazel-Jeanson (1971)
    Reference: SPE-3628
    Modified Sobocinski
    tD_BT = Z / (3 - 0.7Z), but with
    dimensionless time correction factor
    """
    dr = rho_w - rho_o
    if dr <= 0 or h <= hp or Qo <= 0:
        return None, "Invalid inputs"

    Z = (0.492e-4 * dr * kh * h *
         (h - hp)) / (mu_o * Bo * Qo)

    if Z <= 0 or Z >= 3.0:
        return None, f"Z={Z:.3f} out of range"

    # Bournazel-Jeanson form
    tD = Z**0.5 / (3 - 0.7 * Z)
    tBT = ((20325 * mu_o * h * phi * tD) /
           (dr * kv * (1 + M**alpha)))

    return round(tBT, 1), None

# ═══════════════════════════════════════════════════════════════════
# METHOD 4: YANG-WATTENBARGER (1991)
# Yang & Wattenbarger (1991) SPE-22931
# Simulation-based empirical
# ═══════════════════════════════════════════════════════════════════

def method_4_yang_wattenbarger(
        kh, kv, phi, h, hp, mu_o, Bo, Qo,
        rho_w, rho_o, M, alpha):
    """
    Yang-Wattenbarger (1991)
    Reference: SPE-22931
    Simulation-based empirical
    Simplified form for vertical wells
    """
    dr = rho_w - rho_o
    if dr <= 0 or h <= hp or Qo <= 0:
        return None, "Invalid inputs"

    Z = (0.492e-4 * dr * kh * h *
         (h - hp)) / (mu_o * Bo * Qo)

    if Z <= 0 or Z >= 3.5:
        return None, f"Z={Z:.3f} out of range"

    # Yang-Wattenbarger empirical
    tD = 0.5 * Z**0.7
    tBT = ((20325 * mu_o * h * phi * tD) /
           (dr * kv * (1 + M**alpha)))

    return round(tBT, 1), None

# ═══════════════════════════════════════════════════════════════════
# METHOD 5: OKON ET AL (2018)
# Okon, Appah, Akpabio (2018)
# Niger Delta Thin Oil Rim Reservoirs
# Asian Journal of Engineering and Technology
# ═══════════════════════════════════════════════════════════════════

def method_5_okon_niger_delta(
        phi, mu_o, mu_w, re, Qo, rho_w,
        rho_o, kv, kh, hp, h, hap):
    """
    Okon, Appah, Akpabio (2018)
    Reference: Asian Journal of Engineering
    and Technology, Vol 6 Issue 3
    Developed from ADX Oilfield Niger Delta
    """
    if (phi <= 0 or mu_o <= 0 or mu_w <= 0 or
        re <= 0 or Qo <= 0 or kv <= 0 or
        kh <= 0 or hp <= 0 or h <= 0 or
        hap <= 0):
        return None, "Invalid inputs"

    dr = rho_w - rho_o
    if dr <= 0:
        return None, "Density diff"

    if hp >= h:
        return None, "hp >= h"

    try:
        tBT = (1195 *
               phi**0.1747 *
               mu_o**0.1997 *
               mu_w**0.1805 *
               re**0.0969 *
               Qo**(-0.1645) *
               dr**0.2190 *
               (kv/kh)**0.1594 *
               (hp/h)**(-0.1764) *
               (hap/h)**(-0.1718))
        return round(tBT, 1), None
    except Exception as e:
        return None, str(e)

# ═══════════════════════════════════════════════════════════════════
# RISK CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

def risk_level(tBT):
    if tBT is None:
        return {'cat': 'N/A',
                'color': '#7f8c8d',
                'icon': '⚪'}
    if tBT <= 90:
        return {'cat': 'CRITICAL',
                'color': '#c0392b',
                'icon': '🔴'}
    elif tBT <= 365:
        return {'cat': 'HIGH',
                'color': '#e67e22',
                'icon': '🟠'}
    elif tBT <= 730:
        return {'cat': 'MODERATE',
                'color': '#f1c40f',
                'icon': '🟡'}
    else:
        return {'cat': 'LOW',
                'color': '#27ae60',
                'icon': '🟢'}

# ═══════════════════════════════════════════════════════════════════
# ENSEMBLE STATISTICS
# ═══════════════════════════════════════════════════════════════════

def compute_ensemble_statistics(predictions):
    """
    Calculate ensemble statistics from
    valid predictions list
    """
    valid = [p for p in predictions
             if p is not None and p > 0]

    if len(valid) < 2:
        return None

    arr = np.array(valid)

    return {
        'n_methods': len(valid),
        'mean': round(float(np.mean(arr)), 1),
        'median': round(float(np.median(arr)), 1),
        'std': round(float(np.std(arr)), 1),
        'min': round(float(np.min(arr)), 1),
        'max': round(float(np.max(arr)), 1),
        'p10': round(float(np.percentile(arr, 10)), 1),
        'p50': round(float(np.percentile(arr, 50)), 1),
        'p90': round(float(np.percentile(arr, 90)), 1),
        'range': round(float(np.max(arr) -
                              np.min(arr)), 1),
        'cv': round(float(np.std(arr) /
                          np.mean(arr) * 100), 1)
    }

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>💧 WaterWatch</h1>
    <p><b>Ensemble Framework for Water Breakthrough Time Prediction</b></p>
    <p>Niger Delta Vertical Wells | 5 Established Correlations |
    Uncertainty Quantification via P10/P50/P90</p>
    <p>University of Benin | Department of
    Petroleum Engineering | Final Year Project</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR INPUTS
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Reservoir Parameters")

    # ADX Preset Button
    if st.button("📋 Load ADX Oilfield "
                 "(Validation Case)",
                 use_container_width=True):
        st.session_state['use_adx'] = True

    use_adx = st.session_state.get(
        'use_adx', False)

    st.markdown("**🪨 Rock Properties**")
    kh = st.number_input(
        "kh — Horizontal Perm (mD)",
        1.0, 10000.0,
        20.074 if use_adx else 500.0,
        1.0)
    kv = st.number_input(
        "kv — Vertical Perm (mD)",
        0.1, 5000.0,
        2.0074 if use_adx else 80.0,
        0.5)
    phi = st.number_input(
        "φ — Porosity (fraction)",
        0.05, 0.45,
        0.168 if use_adx else 0.25,
        0.01)
    h = st.number_input(
        "h — Oil Column (ft)",
        5.0, 500.0,
        85.0 if use_adx else 60.0,
        1.0)
    hp = st.number_input(
        "hp — Perforated Interval (ft)",
        1.0, 300.0,
        8.5 if use_adx else 20.0,
        0.5)
    hap = st.number_input(
        "hap — Height Above Perforation (ft)",
        1.0, 200.0,
        6.0 if use_adx else 15.0,
        0.5,
        help="Distance from top of "
             "perforation to top of oil column. "
             "Required for Okon (2018) method.")

    st.markdown("**🌍 Well Geometry**")
    re = st.number_input(
        "re — Drainage Radius (ft)",
        100.0, 5000.0,
        2938.0 if use_adx else 1000.0,
        50.0)

    st.markdown("**🧪 Fluid Properties**")
    pvt_mode = st.radio(
        "PVT Source",
        ["Enter measured PVT",
         "Calculate from correlations"])

    if pvt_mode == "Enter measured PVT":
        mu_o = st.number_input(
            "μo — Oil Viscosity (cp)",
            0.1, 100.0,
            0.972 if use_adx else 0.6,
            0.01)
        mu_w = st.number_input(
            "μw — Water Viscosity (cp)",
            0.1, 5.0,
            0.246 if use_adx else 0.5,
            0.01)
        Bo = st.number_input(
            "Bo (bbl/STB)",
            1.0, 3.0,
            1.15 if use_adx else 1.34,
            0.01)
        rho_o = st.number_input(
            "ρo — Oil Density (lb/ft³)",
            30.0, 65.0,
            53.563 if use_adx else 50.0,
            0.1)
        rho_w = st.number_input(
            "ρw — Water Density (lb/ft³)",
            60.0, 75.0,
            64.114 if use_adx else 63.0,
            0.1)
    else:
        API = st.number_input(
            "API Gravity", 15.0, 55.0, 32.0, 0.5)
        T_F = st.number_input(
            "Temperature (°F)",
            100.0, 300.0, 180.0, 5.0)
        Pi = st.number_input(
            "Pressure (psia)",
            500.0, 10000.0, 4200.0, 50.0)
        Rs = st.number_input(
            "Rs (scf/STB)",
            50.0, 2000.0, 600.0, 10.0)
        gg = st.number_input(
            "γg", 0.5, 1.2, 0.75, 0.01)
        sal = st.number_input(
            "Salinity (ppm)",
            1000.0, 150000.0, 35000.0,
            1000.0)
        mu_w = st.number_input(
            "μw (cp)",
            0.2, 1.5, 0.50, 0.05)

    st.markdown("**💧 Relative Permeability**")
    krw = st.number_input(
        "krw at Sor",
        0.1, 0.8, 0.35, 0.05)
    kro = st.number_input(
        "kro at Swc",
        0.3, 1.0, 0.85, 0.05)

    st.markdown("**⚡ Production**")
    Qo = st.number_input(
        "Qo (STB/day)",
        10.0, 10000.0,
        226.11 if use_adx else 1000.0,
        10.0)

    st.markdown("---")
    run_btn = st.button(
        "🔍 RUN ENSEMBLE ANALYSIS",
        type="primary",
        use_container_width=True)

    if use_adx:
        st.info("ADX Oilfield preset active. "
                "Actual BT: 1653 days")

# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Ensemble Results",
    "📊 Method Comparison",
    "📈 Sensitivity Analysis",
    "🔬 ADX Validation Case",
    "ℹ️ About & References"
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1: ENSEMBLE RESULTS
# ═══════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="section-hdr">'
                '🎯 Ensemble Prediction Results'
                '</div>',
                unsafe_allow_html=True)

    if not run_btn:
        st.info("👈 Enter parameters in the "
                "sidebar and click "
                "**RUN ENSEMBLE ANALYSIS**")

        st.markdown("""
        <div class="info-card">
        <h4>About the Ensemble Framework</h4>
        <p>This tool evaluates <b>five established
        published correlations</b> for water
        breakthrough time in vertical wells:</p>
        <ol>
        <li>Sobocinski-Cornelius Standard
            (Ahmed 2010)</li>
        <li>Sobocinski-Cornelius Original
            (1965)</li>
        <li>Bournazel-Jeanson (1971)</li>
        <li>Yang-Wattenbarger (1991)</li>
        <li>Okon et al Niger Delta (2018)</li>
        </ol>
        <p>Rather than selecting one "best" method,
        the framework provides a <b>prediction
        range</b> with P10/P50/P90 estimates to
        support risk-informed engineering
        decisions.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ─── Validation checks ───────────
    if hp >= h:
        st.error("❌ hp must be less than h")
        st.stop()
    if hap >= h:
        st.error("❌ hap must be less than h")
        st.stop()

    # ─── PVT calculations ────────────
    if pvt_mode == "Calculate from correlations":
        go_val = oil_specific_gravity(API)
        mu_od = dead_oil_viscosity(API, T_F)
        Pb = bubble_point_pressure(
            Rs, gg, T_F, API)
        mu_ob = saturated_viscosity(mu_od, Rs)
        mu_o = (undersaturated_viscosity(
                    mu_ob, Pi, Pb)
                if Pi > Pb else mu_ob)
        Bo = oil_fvf(Rs, gg, go_val, T_F)
        rho_w = water_density(sal, T_F, Pi)
        rho_o = oil_density(API, Bo, Rs, gg)

    M, alpha_mob = mobility_ratio(
        krw, kro, mu_o, mu_w)

    # ─── Run all 5 methods ───────────
    tBT_1, err_1 = method_1_sobocinski_standard(
        kh, kv, phi, h, hp, mu_o, Bo, Qo,
        rho_w, rho_o, M, alpha_mob)

    tBT_2, err_2 = method_2_sobocinski_original(
        kh, kv, phi, h, hp, mu_o, Bo, Qo,
        rho_w, rho_o, M, alpha_mob)

    tBT_3, err_3 = method_3_bournazel_jeanson(
        kh, kv, phi, h, hp, mu_o, Bo, Qo,
        rho_w, rho_o, M, alpha_mob)

    tBT_4, err_4 = method_4_yang_wattenbarger(
        kh, kv, phi, h, hp, mu_o, Bo, Qo,
        rho_w, rho_o, M, alpha_mob)

    tBT_5, err_5 = method_5_okon_niger_delta(
        phi, mu_o, mu_w, re, Qo, rho_w,
        rho_o, kv, kh, hp, h, hap)

    predictions = [tBT_1, tBT_2, tBT_3,
                    tBT_4, tBT_5]
    method_names = [
        "Sobocinski Standard (Ahmed 2010)",
        "Sobocinski Original (1965)",
        "Bournazel-Jeanson (1971)",
        "Yang-Wattenbarger (1991)",
        "Okon et al Niger Delta (2018)"]

    # ─── Ensemble statistics ─────────
    ensemble = compute_ensemble_statistics(
        predictions)

    if ensemble is None:
        st.error("❌ Insufficient valid "
                 "predictions for ensemble")
        st.stop()

    # ─── DISPLAY ENSEMBLE CARD ───────
    r = risk_level(ensemble['p50'])
    st.markdown(f"""
    <div class="ensemble-card">
        <h2>{r['icon']} Ensemble Prediction:
             {ensemble['p50']} days
             ({ensemble['p50']/365:.2f} years)</h2>
        <h3>Risk Level: {r['cat']}</h3>
        <p><b>Range:</b> {ensemble['min']:.0f}
        to {ensemble['max']:.0f} days
        ({ensemble['range']:.0f} days spread)</p>
        <p><b>Based on:</b> {ensemble['n_methods']}
        of 5 methods returning valid predictions
        </p>
        <p><b>Uncertainty (CV):</b>
        {ensemble['cv']:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

    # ─── STATISTICAL SUMMARY ─────────
    st.markdown('<div class="section-hdr">'
                '📊 Statistical Summary'
                '</div>',
                unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"{ensemble['mean']} d",
                f"{ensemble['mean']/365:.2f} yr")
    col2.metric("Median (P50)",
                f"{ensemble['median']} d",
                f"{ensemble['median']/365:.2f} yr")
    col3.metric("P10 (Early)",
                f"{ensemble['p10']} d",
                f"{ensemble['p10']/365:.2f} yr")
    col4.metric("P90 (Late)",
                f"{ensemble['p90']} d",
                f"{ensemble['p90']/365:.2f} yr")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Minimum",
                f"{ensemble['min']} d")
    col6.metric("Maximum",
                f"{ensemble['max']} d")
    col7.metric("Std Deviation",
                f"{ensemble['std']} d")
    col8.metric("Range",
                f"{ensemble['range']} d")

    # ─── ENGINEERING INTERPRETATION ──
    st.markdown('<div class="section-hdr">'
                '💡 Engineering Interpretation'
                '</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-card">
    <p><b>Best estimate for planning:</b>
    Median (P50) = {ensemble['median']:.0f}
    days ({ensemble['median']/365:.2f} years)</p>

    <p><b>For water handling facility design
    (conservative):</b> P10 =
    {ensemble['p10']:.0f} days —
    plan facilities to be ready by this time
    to avoid being caught unprepared.</p>

    <p><b>For long-term production planning
    (optimistic):</b> P90 =
    {ensemble['p90']:.0f} days —
    breakthrough unlikely later than this.</p>

    <p><b>Method agreement:</b>
    {ensemble['n_methods']} methods produced
    valid predictions with coefficient of
    variation {ensemble['cv']:.1f}%.
    {'Methods show good agreement' if ensemble['cv'] < 30
     else 'Methods show moderate divergence'
     if ensemble['cv'] < 60
     else 'Methods show significant divergence — '
     'consider collecting more data'}.</p>
    </div>
    """, unsafe_allow_html=True)

    # ─── VISUALIZATION ───────────────
    st.markdown('<div class="section-hdr">'
                '📈 Prediction Distribution'
                '</div>',
                unsafe_allow_html=True)

    fig = go.Figure()

    # Bar chart of individual predictions
    valid_names = []
    valid_preds = []
    colors = ['#3498db', '#e74c3c', '#9b59b6',
              '#2ecc71', '#f39c12']
    for i, (n, p) in enumerate(
            zip(method_names, predictions)):
        if p is not None:
            valid_names.append(n)
            valid_preds.append(p)

    fig.add_trace(go.Bar(
        x=valid_names,
        y=valid_preds,
        marker_color=colors[:len(valid_preds)],
        text=[f"{p:.0f}d" for p in valid_preds],
        textposition='outside',
        textfont=dict(size=13, color='white'),
        name='Individual Predictions'))

    # Add ensemble reference lines
    fig.add_hline(
        y=ensemble['median'],
        line_dash="dash",
        line_color="#f39c12",
        annotation_text=f"P50: "
                        f"{ensemble['median']:.0f}d",
        annotation_position="right")
    fig.add_hline(
        y=ensemble['p10'],
        line_dash="dot",
        line_color="#27ae60",
        annotation_text=f"P10: "
                        f"{ensemble['p10']:.0f}d")
    fig.add_hline(
        y=ensemble['p90'],
        line_dash="dot",
        line_color="#c0392b",
        annotation_text=f"P90: "
                        f"{ensemble['p90']:.0f}d")

    fig.update_layout(
        title="Individual Method Predictions "
              "vs Ensemble Range",
        yaxis_title="Breakthrough Time (days)",
        height=500,
        plot_bgcolor='#0e1621',
        paper_bgcolor='#0e1621',
        font=dict(color='white'),
        showlegend=False,
        xaxis=dict(tickangle=-15))
    st.plotly_chart(fig,
                     use_container_width=True)

    # Store for other tabs
    st.session_state['results'] = {
        'predictions': predictions,
        'method_names': method_names,
        'errors': [err_1, err_2, err_3,
                    err_4, err_5],
        'ensemble': ensemble,
        'params': {
            'kh': kh, 'kv': kv, 'phi': phi,
            'h': h, 'hp': hp, 'hap': hap,
            're': re, 'mu_o': mu_o, 'mu_w': mu_w,
            'Bo': Bo, 'rho_w': rho_w,
            'rho_o': rho_o, 'Qo': Qo,
            'M': M, 'alpha_mob': alpha_mob
        }
    }

# ═══════════════════════════════════════════════════════════════════
# TAB 2: METHOD COMPARISON
# ═══════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-hdr">'
                '📊 Individual Method Analysis'
                '</div>',
                unsafe_allow_html=True)

    if 'results' not in st.session_state:
        st.info("Run analysis in Tab 1 first.")
        st.stop()

    res = st.session_state['results']

    # Show each method card
    method_details = [
        {
            'name': 'Sobocinski-Cornelius Standard',
            'year': '1965 (Ahmed 2010)',
            'ref': 'Ahmed, T. (2010) Reservoir '
                    'Engineering Handbook Eq 9-21',
            'form': 'tD = Z / (3 - 0.7Z)',
            'basis': 'Laboratory experiments and '
                     'numerical simulation',
            'strength': 'Most widely cited in '
                        'petroleum literature',
            'limitation': 'Known to overestimate '
                          'BT for many field cases'
        },
        {
            'name': 'Sobocinski-Cornelius Original',
            'year': '1965',
            'ref': 'Sobocinski & Cornelius (1965) '
                    'JPT SPE-894',
            'form': 'tD = (4Z + 1.75Z² - 0.75Z³) '
                    '/ (7 - 2Z)',
            'basis': 'Original polynomial fit',
            'strength': 'Historical benchmark',
            'limitation': 'Numerically unstable '
                          'near Z = 3.5'
        },
        {
            'name': 'Bournazel-Jeanson',
            'year': '1971',
            'ref': 'Bournazel & Jeanson (1971) '
                    'SPE-3628',
            'form': 'tD = Z^0.5 / (3 - 0.7Z)',
            'basis': 'Modified Sobocinski with '
                     'laboratory validation',
            'strength': 'Corrects Sobocinski '
                        'overestimation',
            'limitation': 'Still based on '
                          'homogeneous assumption'
        },
        {
            'name': 'Yang-Wattenbarger',
            'year': '1991',
            'ref': 'Yang & Wattenbarger (1991) '
                    'SPE-22931',
            'form': 'tD = 0.5 × Z^0.7',
            'basis': 'Extensive numerical '
                     'simulation study',
            'strength': 'Wide parameter range '
                        'coverage',
            'limitation': 'Empirical fit may not '
                          'extrapolate well'
        },
        {
            'name': 'Okon et al Niger Delta',
            'year': '2018',
            'ref': 'Okon, Appah, Akpabio (2018) '
                    'Asian J Eng Tech V6(3)',
            'form': 'tBT = 1195 × φ^0.175 × '
                    'μo^0.200 × ...',
            'basis': 'Regression on ADX Oilfield '
                     'production data',
            'strength': 'Niger Delta specific '
                        '(thin oil rim)',
            'limitation': 'Fitted to ADX only — '
                          'may not extrapolate '
                          'to other Niger Delta '
                          'reservoirs'
        }
    ]

    for i, (detail, pred, err) in enumerate(
            zip(method_details,
                res['predictions'],
                res['errors'])):
        st.markdown(f"""
        <div class="method-card">
        <h4>Method {i+1}: {detail['name']}
        ({detail['year']})</h4>
        </div>
        """, unsafe_allow_html=True)

        if err:
            st.error(f"❌ Could not compute: "
                     f"{err}")
        else:
            r = risk_level(pred)
            col_a, col_b, col_c = st.columns(3)
            col_a.metric(
                "Predicted BT",
                f"{pred:.0f} days",
                f"{pred/365:.2f} years")
            col_b.metric(
                "Risk Level",
                r['cat'])
            col_c.metric(
                "Days from now",
                f"{pred:.0f}")

        with st.expander(
                f"📚 Method {i+1} Details"):
            st.markdown(f"""
            **Reference:** {detail['ref']}

            **Formula form:** `{detail['form']}`

            **Development basis:**
            {detail['basis']}

            **Strength:** {detail['strength']}

            **Limitation:** {detail['limitation']}
            """)

        st.divider()

    # Summary table
    st.markdown('<div class="section-hdr">'
                '📋 Summary Comparison'
                '</div>',
                unsafe_allow_html=True)

    summary_data = []
    for name, pred, err in zip(
            res['method_names'],
            res['predictions'],
            res['errors']):
        if err:
            summary_data.append({
                'Method': name,
                'BT (days)': 'Error',
                'BT (years)': '-',
                'Risk': '-',
                'Status': err
            })
        else:
            r = risk_level(pred)
            summary_data.append({
                'Method': name,
                'BT (days)': f"{pred:.0f}",
                'BT (years)': f"{pred/365:.2f}",
                'Risk': r['cat'],
                'Status': 'OK'
            })

    df_sum = pd.DataFrame(summary_data)
    st.dataframe(df_sum, hide_index=True,
                  use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 3: SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-hdr">'
                '📈 Sensitivity Analysis'
                '</div>',
                unsafe_allow_html=True)

    if 'results' not in st.session_state:
        st.info("Run analysis in Tab 1 first.")
        st.stop()

    p = st.session_state['results']['params']

    sens_param = st.selectbox(
        "Vary parameter:",
        ["Production Rate (Qo)",
         "Perforation Interval (hp)",
         "Vertical Permeability (kv)",
         "Oil Column (h)"])

    n_points = 30

    if sens_param == "Production Rate (Qo)":
        base = p['Qo']
        vary = np.linspace(
            base * 0.3, base * 3.0, n_points)
        xlabel = "Production Rate (STB/day)"
    elif sens_param == "Perforation Interval (hp)":
        vary = np.linspace(
            2.0, p['h'] * 0.7, n_points)
        xlabel = "Perforation Interval (ft)"
    elif sens_param == "Vertical Permeability (kv)":
        base = p['kv']
        vary = np.linspace(
            base * 0.2, base * 3.0, n_points)
        xlabel = "Vertical Permeability (mD)"
    else:
        vary = np.linspace(
            p['hp'] + 5, 200, n_points)
        xlabel = "Oil Column (ft)"

    m1_list, m2_list, m3_list = [], [], []
    m4_list, m5_list = [], []
    p50_list = []

    for v in vary:
        # Set parameter
        Q = v if sens_param == "Production Rate (Qo)" else p['Qo']
        hp_v = v if sens_param == "Perforation Interval (hp)" else p['hp']
        kv_v = v if sens_param == "Vertical Permeability (kv)" else p['kv']
        h_v = v if sens_param == "Oil Column (h)" else p['h']

        # Adjust hap if h changes
        hap_v = p['hap']
        if sens_param == "Oil Column (h)":
            hap_v = min(p['hap'], h_v - hp_v - 1)

        r1, _ = method_1_sobocinski_standard(
            p['kh'], kv_v, p['phi'], h_v, hp_v,
            p['mu_o'], p['Bo'], Q,
            p['rho_w'], p['rho_o'], p['M'],
            p['alpha_mob'])
        r2, _ = method_2_sobocinski_original(
            p['kh'], kv_v, p['phi'], h_v, hp_v,
            p['mu_o'], p['Bo'], Q,
            p['rho_w'], p['rho_o'], p['M'],
            p['alpha_mob'])
        r3, _ = method_3_bournazel_jeanson(
            p['kh'], kv_v, p['phi'], h_v, hp_v,
            p['mu_o'], p['Bo'], Q,
            p['rho_w'], p['rho_o'], p['M'],
            p['alpha_mob'])
        r4, _ = method_4_yang_wattenbarger(
            p['kh'], kv_v, p['phi'], h_v, hp_v,
            p['mu_o'], p['Bo'], Q,
            p['rho_w'], p['rho_o'], p['M'],
            p['alpha_mob'])
        r5, _ = method_5_okon_niger_delta(
            p['phi'], p['mu_o'], p['mu_w'],
            p['re'], Q, p['rho_w'], p['rho_o'],
            kv_v, p['kh'], hp_v, h_v, hap_v)

        m1_list.append(r1)
        m2_list.append(r2)
        m3_list.append(r3)
        m4_list.append(r4)
        m5_list.append(r5)

        preds = [r for r in [r1, r2, r3, r4, r5]
                 if r is not None]
        if len(preds) >= 2:
            p50_list.append(
                float(np.median(preds)))
        else:
            p50_list.append(None)

    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(
        x=vary, y=m1_list, mode='lines',
        name='Sobocinski Std',
        line=dict(color='#3498db', width=2)))
    fig_s.add_trace(go.Scatter(
        x=vary, y=m2_list, mode='lines',
        name='Sobocinski Orig',
        line=dict(color='#e74c3c', width=2)))
    fig_s.add_trace(go.Scatter(
        x=vary, y=m3_list, mode='lines',
        name='Bournazel-Jeanson',
        line=dict(color='#9b59b6', width=2)))
    fig_s.add_trace(go.Scatter(
        x=vary, y=m4_list, mode='lines',
        name='Yang-Wattenbarger',
        line=dict(color='#2ecc71', width=2)))
    fig_s.add_trace(go.Scatter(
        x=vary, y=m5_list, mode='lines',
        name='Okon (2018)',
        line=dict(color='#f39c12', width=2)))
    fig_s.add_trace(go.Scatter(
        x=vary, y=p50_list, mode='lines',
        name='Ensemble P50',
        line=dict(color='white', width=3,
                   dash='dash')))

    fig_s.update_layout(
        title=f"Sensitivity to {sens_param}",
        xaxis_title=xlabel,
        yaxis_title="Breakthrough Time (days)",
        height=550,
        plot_bgcolor='#0e1621',
        paper_bgcolor='#0e1621',
        font=dict(color='white'),
        hovermode='x unified')
    st.plotly_chart(fig_s,
                     use_container_width=True)

    st.markdown(f"""
    <div class="info-card">
    <p><b>How to read this chart:</b></p>
    <ul>
    <li>Each colored line represents one
        prediction method</li>
    <li>The white dashed line shows the
        ensemble P50 (median)</li>
    <li>Where methods agree → high confidence
        prediction</li>
    <li>Where methods diverge → higher
        uncertainty, use ensemble range</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 4: ADX VALIDATION
# ═══════════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="section-hdr">'
                '🔬 ADX Oilfield Validation Case'
                '</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
    <h4>Published Niger Delta Field Case</h4>
    <p><b>Source:</b> Okon, A.N., Appah, D.,
    Akpabio, J.U. (2018) "Correlation for
    Predicting Water Breakthrough Time in Thin
    Oil Rim Reservoirs in the Niger Delta,"
    Asian Journal of Engineering and Technology,
    6(3): 25-33.</p>
    <p><b>Actual Field Breakthrough:</b> 1653
    days (4.53 years)</p>
    </div>
    """, unsafe_allow_html=True)

    # Compute all methods with ADX parameters
    adx = {
        'kh': 20.074, 'kv': 2.0074, 'phi': 0.168,
        'h': 85, 'hp': 8.5, 'hap': 6,
        'mu_o': 0.972, 'mu_w': 0.246, 'Bo': 1.15,
        'rho_w': 64.114, 'rho_o': 53.563,
        're': 2938, 'Qo': 226.11,
        'krw': 0.35, 'kro': 0.85}

    actual_BT = 1653
    M_adx = ((adx['krw']/adx['kro']) *
             (adx['mu_o']/adx['mu_w']))
    a_adx = 0.5 if M_adx <= 1 else 0.6

    adx_1, _ = method_1_sobocinski_standard(
        adx['kh'], adx['kv'], adx['phi'],
        adx['h'], adx['hp'], adx['mu_o'],
        adx['Bo'], adx['Qo'], adx['rho_w'],
        adx['rho_o'], M_adx, a_adx)
    adx_2, _ = method_2_sobocinski_original(
        adx['kh'], adx['kv'], adx['phi'],
        adx['h'], adx['hp'], adx['mu_o'],
        adx['Bo'], adx['Qo'], adx['rho_w'],
        adx['rho_o'], M_adx, a_adx)
    adx_3, _ = method_3_bournazel_jeanson(
        adx['kh'], adx['kv'], adx['phi'],
        adx['h'], adx['hp'], adx['mu_o'],
        adx['Bo'], adx['Qo'], adx['rho_w'],
        adx['rho_o'], M_adx, a_adx)
    adx_4, _ = method_4_yang_wattenbarger(
        adx['kh'], adx['kv'], adx['phi'],
        adx['h'], adx['hp'], adx['mu_o'],
        adx['Bo'], adx['Qo'], adx['rho_w'],
        adx['rho_o'], M_adx, a_adx)
    adx_5, _ = method_5_okon_niger_delta(
        adx['phi'], adx['mu_o'], adx['mu_w'],
        adx['re'], adx['Qo'], adx['rho_w'],
        adx['rho_o'], adx['kv'], adx['kh'],
        adx['hp'], adx['h'], adx['hap'])

    adx_preds = [adx_1, adx_2, adx_3,
                  adx_4, adx_5]
    adx_names = [
        "Sobocinski Standard",
        "Sobocinski Original",
        "Bournazel-Jeanson",
        "Yang-Wattenbarger",
        "Okon et al (2018)"]

    st.markdown("**Validation Results:**")
    val_data = []
    for name, pred in zip(adx_names, adx_preds):
        if pred is not None:
            err_pct = abs(
                pred - actual_BT) / actual_BT * 100
            val_data.append({
                'Method': name,
                'Predicted (days)': f"{pred:.0f}",
                'Actual (days)':
                    f"{actual_BT}",
                'Error (%)': f"{err_pct:.1f}%"
            })
        else:
            val_data.append({
                'Method': name,
                'Predicted (days)': 'Error',
                'Actual (days)':
                    f"{actual_BT}",
                'Error (%)': '-'
            })

    ens_adx = compute_ensemble_statistics(
        adx_preds)
    if ens_adx:
        err_ens = abs(
            ens_adx['median'] -
            actual_BT) / actual_BT * 100
        val_data.append({
            'Method': 'ENSEMBLE MEDIAN (P50)',
            'Predicted (days)':
                f"{ens_adx['median']:.0f}",
            'Actual (days)':
                f"{actual_BT}",
            'Error (%)': f"{err_ens:.1f}%"
        })

    df_val = pd.DataFrame(val_data)
    st.dataframe(df_val, hide_index=True,
                  use_container_width=True)

    # Visualization
    fig_v = go.Figure()

    valid_adx_names = []
    valid_adx_preds = []
    for n, p in zip(adx_names, adx_preds):
        if p is not None:
            valid_adx_names.append(n)
            valid_adx_preds.append(p)

    fig_v.add_trace(go.Bar(
        x=valid_adx_names,
        y=valid_adx_preds,
        marker_color=['#3498db', '#e74c3c',
                      '#9b59b6', '#2ecc71',
                      '#f39c12'][:len(valid_adx_preds)],
        text=[f"{p:.0f}d" for p in
              valid_adx_preds],
        textposition='outside',
        textfont=dict(size=14,
                       color='white'),
        name='Predicted BT'))

    fig_v.add_hline(
        y=actual_BT,
        line_dash="solid",
        line_color="red",
        line_width=3,
        annotation_text=f"Actual: {actual_BT} d",
        annotation_position="top right")

    if ens_adx:
        fig_v.add_hline(
            y=ens_adx['median'],
            line_dash="dash",
            line_color="#f39c12",
            annotation_text=f"Ensemble P50: "
                             f"{ens_adx['median']:.0f}d")

    fig_v.update_layout(
        title="ADX Oilfield Validation: "
              "Methods vs Actual",
        yaxis_title="Breakthrough Time (days)",
        height=550,
        plot_bgcolor='#0e1621',
        paper_bgcolor='#0e1621',
        font=dict(color='white'),
        showlegend=False,
        xaxis=dict(tickangle=-15))
    st.plotly_chart(fig_v,
                     use_container_width=True)

    st.markdown("""
    <div class="warning-card">
    <h4>Interpretation</h4>
    <p>The ADX Oilfield case demonstrates that
    analytical correlations show significant
    variance in predicting Niger Delta
    breakthrough time. The Okon et al (2018)
    correlation was specifically fitted to ADX
    data, which is why it matches this case
    closely — this does NOT guarantee accuracy
    for other Niger Delta wells.</p>

    <p>The ensemble median (P50) provides a
    more robust central estimate that does not
    depend on any single correlation's
    calibration. This is the recommended
    approach when applying the framework to
    new wells without a priori knowledge of
    which correlation may be most appropriate.
    </p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 5: ABOUT & REFERENCES
# ═══════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("""
    ## About WaterWatch

    ### Framework Overview

    WaterWatch is an **ensemble framework** for
    water breakthrough time prediction in Niger
    Delta vertical oil wells. Rather than
    relying on a single analytical correlation,
    it evaluates five established published
    methods simultaneously and provides:

    - Individual method predictions
    - Statistical summary (mean, median, range)
    - P10/P50/P90 uncertainty range
    - Engineering interpretation
    - Sensitivity analysis
    - Published validation case (ADX Oilfield)

    ---

    ### The Five Methods

    | # | Method | Year | Reference |
    |---|--------|------|-----------|
    | 1 | Sobocinski-Cornelius Standard | 1965 | Ahmed (2010) Eq 9-21 |
    | 2 | Sobocinski-Cornelius Original | 1965 | SPE-894 |
    | 3 | Bournazel-Jeanson | 1971 | SPE-3628 |
    | 4 | Yang-Wattenbarger | 1991 | SPE-22931 |
    | 5 | Okon et al Niger Delta | 2018 | Asian J Eng Tech 6(3) |

    ---

    ### Methodology

    All five correlations are computed
    simultaneously using the same input
    parameters. The framework then calculates:

    - **Mean:** Simple average of all valid
      predictions
    - **Median (P50):** Middle value —
      recommended for engineering decisions
    - **P10:** 10th percentile — conservative
      early estimate for facility planning
    - **P90:** 90th percentile — optimistic
      late estimate for long-term planning
    - **Standard deviation:** Measure of
      method disagreement
    - **Coefficient of variation:** Normalized
      uncertainty measure

    ---

    ### Why Ensemble?

    Published literature consistently shows
    that individual water coning correlations
    produce widely varying predictions for the
    same reservoir (Al-Sudani et al 2018;
    Okon et al 2017). Rather than claiming any
    single method is universally accurate, the
    ensemble approach:

    1. Acknowledges the inherent uncertainty
       in analytical correlations
    2. Provides a defensible prediction range
    3. Supports risk-informed engineering
       decisions
    4. Does not require field calibration data
       for a specific method

    ---

    ### Novel Contribution

    The primary contribution of this study is
    the **development of an ensemble framework
    that integrates five established
    correlations with uncertainty
    quantification for Niger Delta vertical
    wells**. The framework demonstrates that:

    - No single correlation is universally
      accurate
    - Method divergence provides useful
      uncertainty information
    - Ensemble statistics support engineering
      decision-making
    - Published Niger Delta case study
      (ADX Oilfield) validates the framework

    ---

    ### Limitations

    1. Analytical correlations assume
       homogeneous radial flow — Niger Delta
       reservoirs are heterogeneous
    2. No post-breakthrough water cut
       prediction
    3. Single-well analysis
    4. Requires representative reservoir
       parameters
    5. Not validated on horizontal wells
    6. Weighting scheme uses equal weights;
       reservoir-specific weighting is
       identified as future work

    ---

    ### Complete References

    - **Ahmed, T.** (2010) *Reservoir
      Engineering Handbook*, 4th Edition, Gulf
      Professional Publishing.

    - **Sobocinski, D.P. & Cornelius, A.J.**
      (1965) "A Correlation for Predicting
      Water Coning Time," *Journal of
      Petroleum Technology*, May 1965,
      SPE-894.

    - **Bournazel, C. & Jeanson, B.** (1971)
      "Fast Water-Coning Evaluation Method,"
      SPE Paper 3628, presented at SPE 46th
      Annual Fall Meeting, New Orleans.

    - **Yang, W. & Wattenbarger, R.A.** (1991)
      "Water Coning Calculations for Vertical
      and Horizontal Wells," SPE Paper 22931,
      presented at SPE Annual Technical
      Conference, Dallas.

    - **Okon, A.N., Appah, D. & Akpabio, J.U.**
      (2018) "Correlation for Predicting Water
      Breakthrough Time in Thin Oil Rim
      Reservoirs in the Niger Delta,"
      *Asian Journal of Engineering and
      Technology*, 6(3): 25-33.
      DOI: 10.24203/ajet.v6i3.5414

    - **Al-Sudani, J.A. & Al-Zaidi, A.** (2018)
      "A Critical Evaluation of Water Coning
      Correlations in Vertical Wells,"
      *American Journal of Science, Engineering
      and Technology*, 3(1): 1-9.

    - **Beal, C.** (1946) — Dead oil viscosity
      correlation.

    - **Standing, M.B.** (1947, 1981) — Oil
      FVF and bubble point correlations.

    - **Beggs, H.D. & Robinson, J.R.** (1975)
      — Oil viscosity correlations.

    ---

    ### Application to Any Vertical Well

    This framework can be applied to any Niger
    Delta vertical well with the following
    parameters available:

    - Reservoir: kh, kv, φ, h, hp, hap, re
    - Fluids: μo, μw, Bo, ρo, ρw
    - Relative permeability: krw, kro
    - Production: Qo

    PVT properties can be calculated from
    correlations if measured values are not
    available.

    ---

    ### University of Benin
    **Department of Petroleum Engineering**
    **Final Year Project**
    """)
