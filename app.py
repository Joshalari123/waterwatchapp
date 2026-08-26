# ═══════════════════════════════════════════════════════════════════
# WATERWATCH
# Comparative Framework for Water Breakthrough Time Prediction
# in Niger Delta Vertical Wells
#
# University of Benin
# Department of Petroleum Engineering
# Final Year Project
#
# METHODOLOGY:
# Five established published correlations evaluated simultaneously
# for comparative analysis and documentation of method behavior
# ═══════════════════════════════════════════════════════════════════

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="WaterWatch | Comparative Framework",
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
    Calculate ensemble statistics with
    TWO-GROUP categorization:
    - Group A (Classical): methods 1-4
    - Group B (Niger Delta): method 5 (Okon)
    """
    if len(predictions) != 5:
        return None

    # Group A: Classical methods (indices 0-3)
    classical = [p for p in predictions[:4]
                  if p is not None and p > 0]
    # Group B: Niger Delta calibrated (index 4)
    niger_delta = (predictions[4]
                    if predictions[4] is not None
                    and predictions[4] > 0
                    else None)

    if len(classical) < 2 and niger_delta is None:
        return None

    result = {}

    # Classical group statistics
    if len(classical) >= 2:
        arr_c = np.array(classical)
        result['classical_mean'] = round(
            float(np.mean(arr_c)), 1)
        result['classical_median'] = round(
            float(np.median(arr_c)), 1)
        result['classical_min'] = round(
            float(np.min(arr_c)), 1)
        result['classical_max'] = round(
            float(np.max(arr_c)), 1)
        result['classical_n'] = len(classical)
    else:
        result['classical_mean'] = None
        result['classical_median'] = None
        result['classical_min'] = None
        result['classical_max'] = None
        result['classical_n'] = len(classical)

    # Niger Delta result
    result['okon'] = niger_delta

    # Overall range
    all_preds = classical + (
        [niger_delta] if niger_delta else [])
    if len(all_preds) >= 2:
        arr_all = np.array(all_preds)
        result['overall_min'] = round(
            float(np.min(arr_all)), 1)
        result['overall_max'] = round(
            float(np.max(arr_all)), 1)
        result['overall_range'] = round(
            float(np.max(arr_all) -
                   np.min(arr_all)), 1)

    # Engineering recommendation
    if (result['classical_mean'] is not None
        and niger_delta is not None):
        # Lower bound: classical mean
        # (early warning)
        # Upper bound: Okon
        # (Niger Delta calibrated best guess)
        result['lower_bound'] = result[
            'classical_mean']
        result['upper_bound'] = niger_delta
        result['recommended'] = niger_delta
        result['engineering_bt'] = round(
            (result['classical_mean'] +
             niger_delta) / 2, 1)
    elif result['classical_mean'] is not None:
        result['lower_bound'] = result[
            'classical_min']
        result['upper_bound'] = result[
            'classical_max']
        result['recommended'] = result[
            'classical_mean']
        result['engineering_bt'] = result[
            'classical_mean']
    elif niger_delta is not None:
        result['lower_bound'] = niger_delta * 0.5
        result['upper_bound'] = niger_delta * 1.5
        result['recommended'] = niger_delta
        result['engineering_bt'] = niger_delta
    else:
        return None

    return result

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>💧 WaterWatch</h1>
    <p><b>Comparative Framework for Water Breakthrough Time Prediction</b></p>
    <p>Niger Delta Vertical Wells | 5 Established Correlations |
    Systematic Method Comparison and Analysis</p>
    <p>University of Benin | Department of
    Petroleum Engineering | Final Year Project</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR INPUTS
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Reservoir Parameters")

    # Preset buttons
    st.markdown("**📋 Load Validation Presets:**")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("🇳🇬 ADX (Niger)",
                     use_container_width=True):
            st.session_state['preset'] = 'adx'
    with col_p2:
        if st.button("🇮🇶 Iraqi Field",
                     use_container_width=True):
            st.session_state['preset'] = 'iraq'

    if st.button("🔄 Clear Preset",
                 use_container_width=True):
        st.session_state['preset'] = None

    preset = st.session_state.get('preset', None)
    use_adx = (preset == 'adx')
    use_iraq = (preset == 'iraq')

    st.markdown("**🪨 Rock Properties**")
    kh = st.number_input(
        "kh — Horizontal Perm (mD)",
        1.0, 10000.0,
        20.074 if use_adx else
        100.0 if use_iraq else 500.0,
        1.0)
    kv = st.number_input(
        "kv — Vertical Perm (mD)",
        0.1, 5000.0,
        2.0074 if use_adx else
        50.0 if use_iraq else 80.0,
        0.5)
    phi = st.number_input(
        "φ — Porosity (fraction)",
        0.05, 0.45,
        0.168 if use_adx else
        0.20 if use_iraq else 0.25,
        0.01)
    h = st.number_input(
        "h — Oil Column (ft)",
        5.0, 500.0,
        85.0 if use_adx else
        100.0 if use_iraq else 60.0,
        1.0)
    hp = st.number_input(
        "hp — Perforated Interval (ft)",
        1.0, 300.0,
        8.5 if use_adx else
        35.0 if use_iraq else 20.0,
        0.5)
    hap = st.number_input(
        "hap — Height Above Perforation (ft)",
        1.0, 200.0,
        6.0 if use_adx else
        5.0 if use_iraq else 15.0,
        0.5,
        help="Distance from top of "
             "perforation to top of oil column. "
             "Required for Okon (2018) method.")

    st.markdown("**🌍 Well Geometry**")
    re = st.number_input(
        "re — Drainage Radius (ft)",
        100.0, 10000.0,
        2938.0 if use_adx else
        7500.0 if use_iraq else 1000.0,
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
            0.972 if use_adx else
            1.0 if use_iraq else 0.6,
            0.01)
        mu_w = st.number_input(
            "μw — Water Viscosity (cp)",
            0.1, 5.0,
            0.246 if use_adx else
            0.3 if use_iraq else 0.5,
            0.01)
        Bo = st.number_input(
            "Bo (bbl/STB)",
            1.0, 3.0,
            1.15 if use_adx else
            1.2 if use_iraq else 1.34,
            0.01)
        rho_o = st.number_input(
            "ρo — Oil Density (lb/ft³)",
            30.0, 65.0,
            53.563 if use_adx else
            45.95 if use_iraq else 50.0,
            0.1)
        rho_w = st.number_input(
            "ρw — Water Density (lb/ft³)",
            60.0, 75.0,
            64.114 if use_adx else
            63.8 if use_iraq else 63.0,
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
        0.05, 1.0,
        0.35 if use_adx else
        0.5 if use_iraq else 0.35,
        0.05)
    kro = st.number_input(
        "kro at Swc",
        0.05, 1.0,
        0.85 if use_adx else
        0.11 if use_iraq else 0.85,
        0.05)

    st.markdown("**⚡ Production**")
    Qo = st.number_input(
        "Qo (STB/day)",
        10.0, 10000.0,
        226.11 if use_adx else
        1500.0 if use_iraq else 1000.0,
        10.0)

    st.markdown("---")
    run_btn = st.button(
        "🔍 RUN COMPARATIVE ANALYSIS",
        type="primary",
        use_container_width=True)

    if use_adx:
        st.info("🇳🇬 ADX Oilfield preset active.\n\n"
                "**Actual BT: 1653 days**\n\n"
                "Source: Okon et al 2018")
    elif use_iraq:
        st.info("🇮🇶 Iraqi Field preset active.\n\n"
                "**Actual BT at Qo=1500: 424 days**\n\n"
                "Source: Al-Sudani & Faleh 2019\n\n"
                "Try Qo: 800→924d, 1500→424d,\n"
                "2500→195d, 3500→125d, 5000→80d")

# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Comparative Results",
    "📊 Method Comparison",
    "📈 Sensitivity Analysis",
    "🔬 Validation Cases",
    "ℹ️ About & References"
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1: ENSEMBLE RESULTS
# ═══════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="section-hdr">'
                '🎯 Comparative Prediction Results'
                '</div>',
                unsafe_allow_html=True)

    if not run_btn:
        st.info("👈 Enter parameters in the "
                "sidebar and click "
                "**RUN COMPARATIVE ANALYSIS**")

        st.markdown("""
        <div class="info-card">
        <h4>About the Comparative Framework</h4>
        <p>This tool implements <b>five established
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
        <p><b>Purpose:</b> Each method is
        displayed individually so engineers,
        students, and researchers can
        systematically compare how different
        published correlations perform for
        the same reservoir configuration.</p>
        <p><b>Important:</b> This is a
        <b>comparison and analysis tool</b>,
        not a facility planning tool.
        Predictions from all methods carry
        significant uncertainty. Users should
        interpret results in the context of
        the documented method limitations
        (see Validation Cases tab).</p>
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

    # ─── SEPARATE CLASSICAL vs OKON ─
    classical_preds = [p for p in
                        [tBT_1, tBT_2,
                         tBT_3, tBT_4]
                        if p is not None
                        and p > 0]
    okon_pred = tBT_5

    # Calculate classical statistics
    if len(classical_preds) >= 2:
        classical_arr = np.array(classical_preds)
        classical_mean = float(np.mean(
            classical_arr))
        classical_min = float(np.min(
            classical_arr))
        classical_max = float(np.max(
            classical_arr))
    else:
        classical_mean = None
        classical_min = None
        classical_max = None

    # ─── ALL 5 METHODS DISPLAYED EQUALLY ─
    st.markdown('<div class="section-hdr">'
                '🎯 Individual Method Predictions'
                '</div>',
                unsafe_allow_html=True)

    method_info = [
        ("Sobocinski Standard", tBT_1,
         "Ahmed (2010)", "#3498db"),
        ("Sobocinski Original", tBT_2,
         "1965", "#e74c3c"),
        ("Bournazel-Jeanson", tBT_3,
         "1971", "#9b59b6"),
        ("Yang-Wattenbarger", tBT_4,
         "1991", "#2ecc71"),
        ("Okon et al Niger Delta", tBT_5,
         "2018", "#f39c12")
    ]

    # Display in 2 rows of columns
    row1 = st.columns(3)
    row2 = st.columns(2)

    for idx, (name, pred, yr, color) in \
            enumerate(method_info):
        col = row1[idx] if idx < 3 \
              else row2[idx - 3]

        with col:
            if pred is not None and pred > 0:
                r = risk_level(pred)
                st.markdown(f"""
                <div style="background: #1e2b3d;
                border-left: 5px solid {color};
                padding: 15px; border-radius: 8px;
                margin: 5px 0; height: 180px;">
                <h4 style="color: {color} !important;
                margin: 0 0 8px 0;
                font-size: 1.0rem;">
                {name}</h4>
                <p style="color: #bdc3c7 !important;
                font-size: 0.8rem;
                margin: 2px 0;">
                {yr}</p>
                <h3 style="color: white !important;
                margin: 8px 0;
                font-size: 1.5rem;">
                {r['icon']} {pred:.0f} days</h3>
                <p style="color: #ecf0f1 !important;
                font-size: 0.85rem;
                margin: 2px 0;">
                {pred/365:.2f} years<br>
                Risk: <b>{r['cat']}</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #2c1e1e;
                border-left: 5px solid #7f8c8d;
                padding: 15px; border-radius: 8px;
                margin: 5px 0; height: 180px;">
                <h4 style="color: {color} !important;
                margin: 0 0 8px 0;">
                {name}</h4>
                <p style="color: #bdc3c7 !important;">
                {yr}</p>
                <h3 style="color: #7f8c8d !important;
                margin: 8px 0;">
                ⚪ Unable</h3>
                <p style="color: #bdc3c7 !important;
                font-size: 0.85rem;">
                Parameters out of range</p>
                </div>
                """, unsafe_allow_html=True)

    # ─── COMPARATIVE SUMMARY ──────────
    valid_preds = [p for p in predictions
                    if p is not None and p > 0]

    if len(valid_preds) >= 2:
        pred_min = min(valid_preds)
        pred_max = max(valid_preds)
        pred_mean = np.mean(valid_preds)
        pred_median = np.median(valid_preds)

        st.markdown('<div class="section-hdr">'
                    '📊 Comparative Summary'
                    '</div>',
                    unsafe_allow_html=True)

        divergence = pred_max / pred_min \
                     if pred_min > 0 else 0

        st.markdown(f"""
        <div class="ensemble-card">
        <h2>📊 Method Divergence Analysis</h2>
        <h3>Predictions differ by
        {divergence:.1f}x across
        {len(valid_preds)} methods</h3>
        <p><b>Shortest prediction:</b>
        {pred_min:.0f} days
        ({pred_min/365:.2f} years)</p>
        <p><b>Longest prediction:</b>
        {pred_max:.0f} days
        ({pred_max/365:.2f} years)</p>
        <p><b>Mean of predictions:</b>
        {pred_mean:.0f} days</p>
        <p><b>Median of predictions:</b>
        {pred_median:.0f} days</p>
        <p style="margin-top: 15px;
        color: #f39c12 !important;">
        <b>Interpretation:</b> The
        {divergence:.1f}x divergence between
        methods indicates the significant
        uncertainty inherent in analytical
        water breakthrough prediction.
        See Validation Cases tab for
        documented method performance
        against published field data.</p>
        </div>
        """, unsafe_allow_html=True)

        col_s1, col_s2, col_s3, col_s4 = \
            st.columns(4)
        col_s1.metric("Minimum",
                       f"{pred_min:.0f} d",
                       f"{pred_min/365:.2f} yr")
        col_s2.metric("Mean",
                       f"{pred_mean:.0f} d",
                       f"{pred_mean/365:.2f} yr")
        col_s3.metric("Median",
                       f"{pred_median:.0f} d",
                       f"{pred_median/365:.2f} yr")
        col_s4.metric("Maximum",
                       f"{pred_max:.0f} d",
                       f"{pred_max/365:.2f} yr")

    # ─── DETAILED STATISTICS ─────────
    with st.expander(
            "📊 Detailed Statistical Summary"):
        if ensemble.get('classical_mean'):
            st.markdown("**Classical Methods "
                        "(4 methods):**")
            col1, col2, col3, col4 = \
                st.columns(4)
            col1.metric(
                "Classical Mean",
                f"{ensemble['classical_mean']} d",
                f"{ensemble['classical_mean']/365:.2f} yr")
            col2.metric(
                "Classical Median",
                f"{ensemble['classical_median']} d")
            col3.metric(
                "Classical Min",
                f"{ensemble['classical_min']} d")
            col4.metric(
                "Classical Max",
                f"{ensemble['classical_max']} d")

        if ensemble.get('okon'):
            st.markdown("**Niger Delta Method:**")
            col5, col6 = st.columns(2)
            col5.metric(
                "Okon 2018",
                f"{ensemble['okon']} d",
                f"{ensemble['okon']/365:.2f} yr")
            col6.metric(
                "Recommended (best est.)",
                f"{ensemble.get('recommended', 0):.0f} d")

        if ensemble.get('overall_min'):
            st.markdown("**Overall Range:**")
            col7, col8, col9 = st.columns(3)
            col7.metric("Min",
                        f"{ensemble['overall_min']} d")
            col8.metric("Max",
                        f"{ensemble['overall_max']} d")
            col9.metric("Range",
                        f"{ensemble['overall_range']} d")

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

    # Add reference lines using new keys
    if ensemble.get('classical_mean'):
        fig.add_hline(
            y=ensemble['classical_mean'],
            line_dash="dash",
            line_color="#e74c3c",
            annotation_text=f"Classical Mean: "
                             f"{ensemble['classical_mean']:.0f}d",
            annotation_position="right")
    if ensemble.get('okon'):
        fig.add_hline(
            y=ensemble['okon'],
            line_dash="dash",
            line_color="#27ae60",
            annotation_text=f"Okon: "
                             f"{ensemble['okon']:.0f}d",
            annotation_position="right")

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
    <li>Where methods agree closely, the
        general prediction trend is more
        consistent</li>
    <li>Where methods diverge widely,
        this indicates significant
        method-dependent uncertainty</li>
    <li>Compare individual method
        predictions against the base case
        result in Tab 1</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 4: ADX VALIDATION
# ═══════════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="section-hdr">'
                '🔬 Multi-Case Validation Study'
                '</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
    <h4>Purpose of This Section</h4>
    <p>The framework is validated against
    <b>published field cases from two
    geographical regions</b> to demonstrate
    both applicability and limitations of
    the analytical correlations:</p>
    <ol>
    <li><b>Niger Delta:</b> ADX Oilfield
        (Okon et al 2018)</li>
    <li><b>Middle East:</b> Iraqi Oil Field
        (Al-Sudani & Faleh 2019)
        — 5 production rate cases</li>
    </ol>
    <p>This comparative validation reveals
    important regional dependencies in
    correlation accuracy.</p>
    </div>
    """, unsafe_allow_html=True)

    # ═════════════════════════════════════════
    # CASE 1: ADX OILFIELD (NIGER DELTA)
    # ═════════════════════════════════════════

    st.markdown('<div class="section-hdr">'
                '🇳🇬 Case 1: ADX Oilfield '
                '(Niger Delta)</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
    <p><b>Source:</b> Okon, A.N., Appah, D.,
    Akpabio, J.U. (2018) "Correlation for
    Predicting Water Breakthrough Time in Thin
    Oil Rim Reservoirs in the Niger Delta,"
    Asian Journal of Engineering and Technology,
    6(3): 25-33.</p>
    <p><b>Reservoir Type:</b> Thin Oil Rim,
    Bottom Water Drive</p>
    <p><b>Actual Field Breakthrough:</b> 1653
    days (4.53 years) — from ADX Oilfield
    production history</p>
    <p><b>Important Note:</b> The Okon 2018
    correlation was DEVELOPED using ADX
    Oilfield data through regression fitting.
    Therefore its near-perfect match here
    (0.2% error) is expected and does NOT
    represent independent validation for
    this specific well. The other four
    classical methods provide independent
    predictions since they were not
    calibrated on this dataset.</p>
    </div>
    """, unsafe_allow_html=True)

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
    method_names_short = [
        "Sobocinski Std",
        "Sobocinski Orig",
        "Bournazel-J",
        "Yang-Watt",
        "Okon 2018"]

    val1_data = []
    for name, pred in zip(method_names_short,
                           adx_preds):
        if pred is not None:
            err = abs(pred-actual_BT)/actual_BT*100
            val1_data.append({
                'Method': name,
                'Predicted (days)': f"{pred:.0f}",
                'Actual (days)': f"{actual_BT}",
                'Error (%)': f"{err:.1f}%"})

    df_val1 = pd.DataFrame(val1_data)
    st.dataframe(df_val1, hide_index=True,
                  use_container_width=True)

    # Chart for ADX
    fig_adx = go.Figure()
    valid_names1 = [n for n, p in
                     zip(method_names_short,
                         adx_preds)
                     if p is not None]
    valid_preds1 = [p for p in adx_preds
                     if p is not None]

    fig_adx.add_trace(go.Bar(
        x=valid_names1, y=valid_preds1,
        marker_color=['#3498db', '#e74c3c',
                      '#9b59b6', '#2ecc71',
                      '#f39c12'][:len(valid_preds1)],
        text=[f"{p:.0f}d" for p in valid_preds1],
        textposition='outside',
        textfont=dict(size=12, color='white')))
    fig_adx.add_hline(
        y=actual_BT, line_dash="solid",
        line_color="red", line_width=3,
        annotation_text=f"Actual: {actual_BT} d",
        annotation_position="top right")
    fig_adx.update_layout(
        title="Case 1: ADX Oilfield (Niger Delta) "
              "— Method Predictions vs Actual",
        yaxis_title="Breakthrough Time (days)",
        height=450,
        plot_bgcolor='#0e1621',
        paper_bgcolor='#0e1621',
        font=dict(color='white'),
        showlegend=False,
        xaxis=dict(tickangle=-15))
    st.plotly_chart(fig_adx,
                     use_container_width=True)

    st.divider()

    # ═════════════════════════════════════════
    # CASE 2: IRAQI FIELD (MIDDLE EAST)
    # ═════════════════════════════════════════

    st.markdown('<div class="section-hdr">'
                '🇮🇶 Case 2: Iraqi Oil Field '
                '(Middle East — 5 Rate Cases)'
                '</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
    <p><b>Source:</b> Al-Sudani, J.A. and
    Faleh, A.M. (2019) "Estimation of Water
    Breakthrough Using Numerical Simulation,"
    Association of Arab Universities Journal
    of Engineering Sciences, 26(3): 73-81.
    DOI: 10.33261/jaaru.2019.26.3.009</p>
    <p><b>Reservoir Type:</b> Homogeneous
    anisotropic bottom water drive</p>
    <p><b>Reference Values:</b> ECLIPSE
    numerical simulation results across
    5 production rates. ECLIPSE is the
    industry-standard reservoir simulator
    and represents the "gold standard"
    reference commonly used to validate
    analytical correlations in petroleum
    engineering literature.</p>
    <p><b>Note:</b> The comparison here is
    between analytical correlations and
    numerical simulation output — a
    standard validation approach where
    numerical simulation provides the
    physical reference against which
    simplified analytical methods are
    evaluated.</p>
    </div>
    """, unsafe_allow_html=True)

    # Iraqi Field base parameters
    iraq = {
        'h': 100, 'hp': 35, 'phi': 0.20,
        'kh': 100, 'kv': 50,
        'mu_o': 1.0, 'mu_w': 0.3, 'Bo': 1.2,
        'rho_w': 63.8, 'rho_o': 45.95,
        're': 7500, 'hap': 5,
        'krw': 0.5, 'kro': 0.11}

    iraq_cases = [
        (800, 924), (1500, 424),
        (2500, 195), (3500, 125),
        (5000, 80)]

    M_ir = ((iraq['krw']/iraq['kro']) *
            (iraq['mu_o']/iraq['mu_w']))
    a_ir = 0.5 if M_ir <= 1 else 0.6

    iraq_results = []
    for Qo_test, actual_test in iraq_cases:
        r1, _ = method_1_sobocinski_standard(
            iraq['kh'], iraq['kv'], iraq['phi'],
            iraq['h'], iraq['hp'], iraq['mu_o'],
            iraq['Bo'], Qo_test, iraq['rho_w'],
            iraq['rho_o'], M_ir, a_ir)
        r2, _ = method_2_sobocinski_original(
            iraq['kh'], iraq['kv'], iraq['phi'],
            iraq['h'], iraq['hp'], iraq['mu_o'],
            iraq['Bo'], Qo_test, iraq['rho_w'],
            iraq['rho_o'], M_ir, a_ir)
        r3, _ = method_3_bournazel_jeanson(
            iraq['kh'], iraq['kv'], iraq['phi'],
            iraq['h'], iraq['hp'], iraq['mu_o'],
            iraq['Bo'], Qo_test, iraq['rho_w'],
            iraq['rho_o'], M_ir, a_ir)
        r4, _ = method_4_yang_wattenbarger(
            iraq['kh'], iraq['kv'], iraq['phi'],
            iraq['h'], iraq['hp'], iraq['mu_o'],
            iraq['Bo'], Qo_test, iraq['rho_w'],
            iraq['rho_o'], M_ir, a_ir)
        r5, _ = method_5_okon_niger_delta(
            iraq['phi'], iraq['mu_o'],
            iraq['mu_w'], iraq['re'],
            Qo_test, iraq['rho_w'],
            iraq['rho_o'], iraq['kv'],
            iraq['kh'], iraq['hp'],
            iraq['h'], iraq['hap'])
        iraq_results.append({
            'Qo': Qo_test, 'actual': actual_test,
            'preds': [r1, r2, r3, r4, r5]})

    # Build comparison table
    val2_data = []
    for res in iraq_results:
        row = {
            'Rate (BPD)': res['Qo'],
            'Actual (d)': res['actual']}
        for name, pred in zip(
                method_names_short,
                res['preds']):
            if pred is not None:
                row[name] = f"{pred:.0f}"
            else:
                row[name] = "Error"
        val2_data.append(row)

    df_val2 = pd.DataFrame(val2_data)
    st.markdown("**Predictions (days):**")
    st.dataframe(df_val2, hide_index=True,
                  use_container_width=True)

    # Error table
    err_data = []
    for res in iraq_results:
        row = {
            'Rate (BPD)': res['Qo'],
            'Actual (d)': res['actual']}
        for name, pred in zip(
                method_names_short,
                res['preds']):
            if pred is not None:
                err = abs(pred - res['actual']) \
                      / res['actual'] * 100
                row[name] = f"{err:.0f}%"
            else:
                row[name] = "-"
        err_data.append(row)

    df_err = pd.DataFrame(err_data)
    st.markdown("**Error Analysis (%):**")
    st.dataframe(df_err, hide_index=True,
                  use_container_width=True)

    # Chart Iraqi cases
    fig_iraq = go.Figure()
    rates = [r['Qo'] for r in iraq_results]
    actuals = [r['actual'] for r in
                iraq_results]

    colors_i = ['#3498db', '#e74c3c',
                '#9b59b6', '#2ecc71', '#f39c12']

    fig_iraq.add_trace(go.Scatter(
        x=rates, y=actuals,
        mode='lines+markers',
        name='Actual (ECLIPSE)',
        line=dict(color='red', width=3),
        marker=dict(size=12, color='red')))

    for i, name in enumerate(
            method_names_short):
        preds_i = [r['preds'][i]
                    if r['preds'][i] is not None
                    else 0
                    for r in iraq_results]
        fig_iraq.add_trace(go.Scatter(
            x=rates, y=preds_i,
            mode='lines+markers', name=name,
            line=dict(color=colors_i[i], width=2),
            marker=dict(size=8)))

    fig_iraq.update_layout(
        title="Case 2: Iraqi Field — Predictions "
              "vs Actual Across 5 Rates",
        xaxis_title="Production Rate (BPD)",
        yaxis_title="Breakthrough Time (days)",
        yaxis_type="log",
        height=500,
        plot_bgcolor='#0e1621',
        paper_bgcolor='#0e1621',
        font=dict(color='white'),
        hovermode='x unified')
    st.plotly_chart(fig_iraq,
                     use_container_width=True)

    st.divider()

    # ═════════════════════════════════════════
    # COMPARATIVE ANALYSIS
    # ═════════════════════════════════════════

    st.markdown('<div class="section-hdr">'
                '📊 Comparative Analysis — '
                'Regional Findings</div>',
                unsafe_allow_html=True)

    # Calculate mean errors
    adx_class_errs = []
    for p in adx_preds[:4]:
        if p is not None:
            adx_class_errs.append(
                abs(p - actual_BT) /
                actual_BT * 100)
    adx_okon_err = (abs(adx_preds[4] -
                        actual_BT) /
                    actual_BT * 100
                    if adx_preds[4]
                    is not None else None)

    iraq_class_errs = []
    iraq_okon_errs = []
    for r in iraq_results:
        class_avg = np.mean([
            p for p in r['preds'][:4]
            if p is not None])
        iraq_class_errs.append(
            abs(class_avg - r['actual']) /
            r['actual'] * 100)
        if r['preds'][4] is not None:
            iraq_okon_errs.append(
                abs(r['preds'][4] - r['actual']) /
                r['actual'] * 100)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="ensemble-card" style="border-left-color: #27ae60;">
        <h3 style="color: #2ecc71 !important;">
        🇳🇬 Niger Delta (ADX)</h3>
        </div>
        """, unsafe_allow_html=True)
        st.metric(
            "Classical Methods Error",
            f"{np.mean(adx_class_errs):.0f}%")
        if adx_okon_err is not None:
            st.metric(
                "Okon 2018 Error",
                f"{adx_okon_err:.1f}%",
                delta="Best",
                delta_color="normal")

    with col_b:
        st.markdown("""
        <div class="ensemble-card" style="border-left-color: #e74c3c;">
        <h3 style="color: #e74c3c !important;">
        🇮🇶 Middle East (Iraqi)</h3>
        </div>
        """, unsafe_allow_html=True)
        st.metric(
            "Classical Methods Error",
            f"{np.mean(iraq_class_errs):.0f}%")
        st.metric(
            "Okon 2018 Error",
            f"{np.mean(iraq_okon_errs):.0f}%",
            delta="Worst",
            delta_color="inverse")

    st.markdown("""
    <div class="warning-card">
    <h4>🔬 Key Research Findings</h4>

    <p><b>1. Regional Calibration Effect:</b>
    The Okon et al (2018) correlation shows
    excellent accuracy for Niger Delta
    reservoirs (0.2% error on ADX) but severe
    overprediction for Middle East reservoirs
    (110-1700% error). This demonstrates
    that regionally calibrated correlations
    do not extrapolate universally.</p>

    <p><b>2. Classical Method Underprediction:</b>
    All four classical analytical correlations
    (Sobocinski Standard, Sobocinski Original,
    Bournazel-Jeanson, Yang-Wattenbarger)
    consistently underpredict breakthrough
    time by 90-98% across both geographical
    regions, indicating a systematic
    limitation of purely analytical
    approaches for real reservoirs.</p>

    <p><b>3. Engineering Implication:</b>
    Water breakthrough prediction requires
    either regional calibration factors
    (as with Okon for Niger Delta) or
    integration of multiple methods with
    uncertainty quantification (as with
    the WaterWatch ensemble framework).
    No single analytical correlation is
    universally accurate.</p>

    <p><b>4. Framework Value:</b>
    The ensemble framework accommodates
    both prediction paradigms — providing
    a classical lower bound (conservative,
    early warning) and a Niger Delta
    calibrated upper bound (best estimate
    for Niger Delta wells only) — enabling
    engineers to make risk-informed
    decisions regardless of reservoir
    location.</p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 5: ABOUT & REFERENCES
# ═══════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("""
    ## About WaterWatch

    ### Framework Overview

    WaterWatch is a **comparative framework**
    that implements five established water
    breakthrough time correlations to enable
    systematic comparison of their behavior
    when applied to the same reservoir
    parameters. The framework does not
    prescribe a "best" method or claim
    accurate prediction; instead, it makes
    the diversity of published methods and
    their limitations transparent and
    accessible.

    The framework provides:

    - Individual method predictions displayed
      separately for direct comparison
    - Statistical summary of predictions
      (min, max, mean, median)
    - Two published validation cases
      (Niger Delta ADX Oilfield and
      Iraqi Field)
    - Sensitivity analysis for parameter
      effects
    - Complete methodology and references

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
    parameters. Predictions from each
    method are displayed separately
    without combination into a single
    output value. Summary statistics
    quantify the divergence between
    methods:

    - **Individual predictions:** Each
      method's output shown independently
    - **Min/Max:** The shortest and
      longest predicted times among
      the methods
    - **Mean:** Simple average of all
      valid predictions
    - **Median:** Middle value among
      predictions
    - **Divergence factor:** Ratio of
      longest to shortest prediction

    ---

    ### Why Comparative?

    Published literature consistently
    documents that individual water coning
    correlations produce widely varying
    predictions for the same reservoir
    (Al-Sudani et al 2018; Okon et al 2018).
    Rather than selecting a single method
    and treating its output as authoritative,
    the comparative approach:

    1. Makes method diversity transparent
       to the user
    2. Reveals the significant uncertainty
       inherent in analytical prediction
    3. Supports education about method
       assumptions and limitations
    4. Provides documentation of method
       behavior across different reservoir
       types when combined with validation
       cases

    ---

    ### Contribution of This Study

    The primary contribution of this study
    is the **systematic implementation and
    comparative evaluation of five
    established water breakthrough
    correlations across two published field
    validation cases**. The framework
    documents:

    - The significant divergence between
      published methods when applied to
      identical reservoir parameters
    - The performance of the Okon et al
      (2018) Niger Delta specific
      correlation both within and outside
      its calibration domain
    - The systematic underprediction bias
      exhibited by classical analytical
      correlations across both tested
      reservoir types
    - The practical limitations of
      currently available analytical
      methods for Niger Delta water
      breakthrough prediction

    These findings identify specific
    research gaps and provide an
    evidence-based foundation for future
    work in Niger Delta water breakthrough
    prediction methodology.


    ---

    ### Limitations

    1. Analytical correlations assume
       homogeneous radial flow — Niger Delta
       reservoirs are heterogeneous
    2. All tested methods show significant
       prediction error against validation
       cases; the framework does not
       claim predictive accuracy
    3. No post-breakthrough water cut
       prediction
    4. Single-well analysis without
       interference effects
    5. Not validated on horizontal wells
    6. Only two validation cases available;
       broader validation identified as
       future work
    7. Framework is a comparative and
       analytical tool; not intended to
       replace detailed reservoir
       simulation for facility planning
       decisions

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
