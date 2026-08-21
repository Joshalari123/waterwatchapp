
# ═══════════════════════════════════════════════════════════════════
# WaterWatch — Comparative Coning Screening Framework
# Niger Delta Vertical Well Water Coning Analysis
#
# University of Benin
# Department of Petroleum Engineering
# Final Year Project
#
# METHODS COMPARED:
#   1. Sobocinski-Cornelius (Standard) — Ahmed (2010) Eq 9-21 to 9-23
#   2. Sobocinski-Cornelius (Original 1965) — Polynomial fit, Z < 3.5
#   3. Meyer-Garder (1954) — Critical rate, analytical
#   4. Schols (1972) — Critical rate, empirical simulation-based
#
# CONTRIBUTION: First comparative screening framework for Niger Delta
# reservoirs evaluating multiple coning correlations simultaneously.
# ═══════════════════════════════════════════════════════════════════

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ─── Page Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="WaterWatch | Comparative Coning Framework",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Professional Dark Theme ─────────────────────────────────────
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
.info-card {
    background: #1e2b3d; color: #ecf0f1; padding: 15px 20px;
    border-radius: 8px; border-left: 4px solid #f39c12; margin: 8px 0;
}
.limitation-card {
    background: #2c1e1e; color: #ecf0f1; padding: 15px 20px;
    border-radius: 8px; border-left: 4px solid #e74c3c; margin: 8px 0;
}
div[data-testid="stMetricValue"] { color: #ecf0f1 !important; }
div[data-testid="stMetricLabel"] { color: #bdc3c7 !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PVT & FLUID PROPERTY ENGINE
# ═══════════════════════════════════════════════════════════════════

def oil_specific_gravity(API):
    return 141.5 / (API + 131.5)

def dead_oil_viscosity(API, T_F):
    """Beal (1946) / Standing (1981) — Ahmed Eq 2-117"""
    T_R = T_F + 460
    a = 10**(0.43 + 8.33/API)
    mu = ((0.32 + 1.8e7/API**4.53) * (360/(T_R - 260))**a)
    return mu

def bubble_point_pressure(Rs, gg, T_F, API):
    """Standing (1947) — Ahmed Eq 2-72"""
    Pb = (18.2 * ((Rs/gg)**0.83 * 10**(0.00091*T_F - 0.0125*API) - 1.4))
    return abs(Pb)

def saturated_viscosity(mu_od, Rs):
    """Beggs & Robinson (1975) — Ahmed Eq 2-121"""
    a = 10.715 * (Rs + 100)**(-0.515)
    b = 5.44 * (Rs + 150)**(-0.338)
    return a * mu_od**b

def undersaturated_viscosity(mu_ob, Pi, Pb):
    """Vasquez-Beggs — Ahmed Eq 2-123"""
    a = -3.9e-5 * Pi - 5
    m = 2.6 * Pi**1.187 * 10**a
    return mu_ob * (Pi/Pb)**m

def oil_fvf(Rs, gg, go_val, T_F):
    """Standing (1981) — Ahmed Eq 2-85"""
    F = Rs * (gg/go_val)**0.5 + 1.25 * T_F
    Bo = 0.9759 + 0.000120 * F**1.2
    return Bo

def water_density(sal_ppm, T_F, P):
    """Niger Delta salinity correction — Tuttle (1999)"""
    rw = (62.4 + sal_ppm/10000 * 0.5 - 0.003 * (T_F - 60) + 0.0000145 * P)
    return rw

def oil_density(API, Bo, Rs, gg):
    go_val = 141.5/(API + 131.5)
    rho_s = go_val * 62.4
    rho_o = (rho_s + 0.01357 * Rs * gg) / Bo
    return rho_o

def mobility_ratio(krw, kro, mu_o, mu_w):
    """Ahmed Eq 9-24"""
    M = (krw/kro) * (mu_o/mu_w)
    alpha = 0.5 if M <= 1 else 0.6
    return M, alpha

# ═══════════════════════════════════════════════════════════════════
# CONING METHODS — CORRECTED FORMULAS
# ═══════════════════════════════════════════════════════════════════

def calc_Z(kh, h, hp, mu_o, Bo, Qo, rho_w_lbft3, rho_o_lbft3):
    """
    Dimensionless cone height — common to both Sobocinski methods.
    Densities converted internally to g/cm³ for standard constant.
    """
    rho_w = rho_w_lbft3 / 62.4
    rho_o = rho_o_lbft3 / 62.4
    dr = rho_w - rho_o
    ht = h - hp
    if dr <= 0 or ht <= 0 or Qo <= 0:
        return None, "Invalid inputs: check densities, h > hp, Qo > 0"
    Z = (0.00307 * dr * kh * h * ht) / (mu_o * Bo * Qo)
    return Z, None

def sobocinski_standard(kh, kv, phi, h, hp, mu_o, Bo, Qo, rho_w, rho_o, M, alpha):
    """
    METHOD 1: Sobocinski-Cornelius (Standard Form)
    Ahmed (2010) Reservoir Engineering Handbook, Eq 9-21 to 9-23.
    Uses the stable, widely-adopted tD correlation.
    Valid for Z < 3.0 (practical limit).
    """
    Z, err = calc_Z(kh, h, hp, mu_o, Bo, Qo, rho_w, rho_o)
    if err:
        return None, None, None, err
    if Z >= 3.0:
        return None, None, None, f"Z={Z:.2f}: Near critical coning rate. Reduce Qo or increase standoff."

    # Bournazel-Jeanson (1971) simplification — adopted as standard
    tD = Z / (3 - 0.7 * Z)

    # Actual breakthrough time (days)
    rho_w_gcc = rho_w / 62.4
    rho_o_gcc = rho_o / 62.4
    dr = rho_w_gcc - rho_o_gcc
    tBT = (tD * mu_o * phi * h * (kh / kv)) / (0.00137 * dr * kh * (1 + M**alpha))

    return round(Z, 4), round(tD, 4), round(tBT, 1), None

def sobocinski_original(kh, kv, phi, h, hp, mu_o, Bo, Qo, rho_w, rho_o, M, alpha):
    """
    METHOD 2: Sobocinski-Cornelius (Original 1965 Polynomial)
    The original lab-derived polynomial fit from SPE-894.
    Numerically unstable as Z approaches 3.5 (denominator → 0).
    Valid range: Z < 3.5 (theoretical), but unstable above ~3.0.
    """
    Z, err = calc_Z(kh, h, hp, mu_o, Bo, Qo, rho_w, rho_o)
    if err:
        return None, None, None, err
    if Z >= 3.5:
        return None, None, None, f"Z={Z:.2f}: Out of original correlation range (Z < 3.5)."

    # Original 1965 polynomial fit — NOTE: denominator (7-2Z) → 0 at Z=3.5
    denom = 7 - 2 * Z
    if denom <= 0.1:
        return None, None, None, f"Z={Z:.2f}: Polynomial denominator too small. Unstable."

    tD = (4*Z + 1.75*Z**2 - 0.75*Z**3) / denom

    rho_w_gcc = rho_w / 62.4
    rho_o_gcc = rho_o / 62.4
    dr = rho_w_gcc - rho_o_gcc
    tBT = (tD * mu_o * phi * h * (kh / kv)) / (0.00137 * dr * kh * (1 + M**alpha))

    return round(Z, 4), round(tD, 4), round(tBT, 1), None

def meyer_garder(kh, kro, h, hp, mu_o, Bo, rho_w, rho_o, re, rw):
    """
    METHOD 3: Meyer-Garder (1954) Critical Rate
    Analytical critical rate below which water coning will never occur.
    Very conservative — tends to predict low critical rates.
    Formula from petroleumengineers.net / Joshi (1991).
    Densities in g/cm³.
    """
    rho_w_gcc = rho_w / 62.4
    rho_o_gcc = rho_o / 62.4
    dr = rho_w_gcc - rho_o_gcc
    if dr <= 0 or h <= hp or re <= rw:
        return None, "Invalid geometry or densities"

    ko = kh * kro  # Effective oil permeability
    ln_term = np.log(re / rw)
    if ln_term <= 0:
        return None, "re must be > rw"

    qc = 0.001535 * (dr / ln_term) * (ko / (mu_o * Bo)) * (h**2 - hp**2)
    return round(qc, 1), None

def schols_critical_rate(kh, kv, krw, kro, h, mu_o, mu_w, Bo, rho_w, rho_o, re, rw):
    """
    METHOD 4: Schols (1972) Critical Rate
    Empirical formula based on numerical simulation and lab experiments.
    Less conservative than Meyer-Garder — more practical for field use.
    Densities in lb/ft³.
    Formula structure from Erdoel-Erdgas (Jan 1972).
    """
    dr = rho_w - rho_o
    if dr <= 0 or re <= rw:
        return None, "Invalid inputs"

    M = (krw/kro) * (mu_o/mu_w)
    ln_term = np.log(re / rw) - 0.75 + M**0.5
    if ln_term <= 0:
        return None, "Invalid drainage geometry"

    # Constant 0.00333 validated for field units (lb/ft³, mD, ft, cP, bbl/STB)
    qc = (0.00333 * dr * kv * h**2) / (mu_o * Bo * ln_term)
    return round(qc, 1), None

# ═══════════════════════════════════════════════════════════════════
# RISK CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

def risk_level_bt(tBT):
    """Risk based on breakthrough time"""
    if tBT is None:
        return {'cat': 'N/A', 'color': '#7f8c8d', 'icon': '⚪'}
    if tBT <= 90:
        return {'cat': 'CRITICAL', 'color': '#c0392b', 'icon': '🔴'}
    elif tBT <= 180:
        return {'cat': 'HIGH', 'color': '#e67e22', 'icon': '🟠'}
    elif tBT <= 365:
        return {'cat': 'MODERATE', 'color': '#f1c40f', 'icon': '🟡'}
    else:
        return {'cat': 'LOW', 'color': '#27ae60', 'icon': '🟢'}

def risk_level_cr(qc, Qo):
    """Risk based on critical rate vs actual rate"""
    if qc is None or Qo <= 0:
        return {'cat': 'N/A', 'color': '#7f8c8d', 'icon': '⚪'}
    ratio = Qo / qc
    if ratio <= 0.5:
        return {'cat': 'SAFE', 'color': '#27ae60', 'icon': '🟢'}
    elif ratio <= 0.8:
        return {'cat': 'CAUTION', 'color': '#f1c40f', 'icon': '🟡'}
    elif ratio <= 1.0:
        return {'cat': 'AT RISK', 'color': '#e67e22', 'icon': '🟠'}
    else:
        return {'cat': 'ABOVE CRITICAL', 'color': '#c0392b', 'icon': '🔴'}

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>💧 WaterWatch</h1>
    <p><b>Comparative Coning Screening Framework</b> for Niger Delta Vertical Wells</p>
    <p>Evaluates 4 established correlations simultaneously | University of Benin | Petroleum Engineering</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR INPUTS
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Reservoir & Well Parameters")

    st.markdown("**🪨 Rock Properties**")
    kh = st.number_input("kh — Horizontal Perm (mD)", 50.0, 5000.0, 500.0, 10.0,
                         help="Arithmetic average from core or log")
    kv = st.number_input("kv — Vertical Perm (mD)", 5.0, 2000.0, 80.0, 5.0,
                         help="Harmonic average from core. Typically kh/5 to kh/15")
    phi = st.number_input("φ — Porosity (fraction)", 0.05, 0.45, 0.25, 0.01,
                          help="Effective porosity from log analysis")
    h = st.number_input("h — Oil Column (ft)", 10.0, 300.0, 60.0, 5.0,
                        help="Distance from top of reservoir to OWC")
    hp = st.number_input("hp — Perforated Interval (ft)", 1.0, 200.0, 20.0, 1.0,
                         help="Perforated length from top of oil column")

    st.markdown("**🌍 Geometry**")
    re = st.number_input("re — Drainage Radius (ft)", 100.0, 5000.0, 1000.0, 50.0,
                         help="From well spacing: 40 acres ≈ 745 ft, 80 acres ≈ 1053 ft")
    rw = st.number_input("rw — Wellbore Radius (ft)", 0.2, 2.0, 0.5, 0.05,
                         help="Typically 0.3–0.5 ft for cased hole")

    st.markdown("**🧪 Fluid Properties**")
    pvt_mode = st.radio("PVT Source", ["Calculate from correlations", "Enter measured PVT"],
                        help="Use correlations for screening, measured for accuracy")

    API = st.number_input("API Gravity (°)", 15.0, 55.0, 35.0, 0.5)
    T_F = st.number_input("Temperature (°F)", 100.0, 300.0, 180.0, 5.0)
    Pi = st.number_input("Initial Pressure (psia)", 500.0, 10000.0, 4200.0, 50.0)
    sal = st.number_input("Water Salinity (ppm)", 1000.0, 150000.0, 35000.0, 1000.0)
    mu_w = st.number_input("Water Viscosity (cp)", 0.2, 1.5, 0.50, 0.05)

    if pvt_mode == "Calculate from correlations":
        Rs = st.number_input("Rs (scf/STB)", 50.0, 2000.0, 600.0, 10.0)
        gg = st.number_input("γg — Gas Gravity", 0.5, 1.2, 0.75, 0.01)
        mu_o_meas = None
        Bo_meas = None
    else:
        mu_o_meas = st.number_input("μo measured (cp)", 0.1, 100.0, 0.6, 0.1)
        Bo_meas = st.number_input("Bo measured (bbl/STB)", 1.0, 3.0, 1.34, 0.01)
        Rs = 600.0
        gg = 0.75

    st.markdown("**💧 Relative Permeability**")
    krw = st.number_input("krw at Sor", 0.1, 0.8, 0.30, 0.05,
                          help="Endpoint water relative permeability")
    kro = st.number_input("kro at Swc", 0.3, 1.0, 0.85, 0.05,
                          help="Endpoint oil relative permeability")

    st.markdown("**⚡ Production**")
    Qo = st.number_input("Qo — Oil Rate (STB/day)", 50.0, 10000.0, 1000.0, 50.0,
                         help="Current or planned production rate")

    run_btn = st.button("🔍 RUN COMPARATIVE ANALYSIS", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Method Comparison",
    "📈 Sensitivity Analysis",
    "✅ Validation",
    "ℹ️ About & Methodology"
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — METHOD COMPARISON
# ═══════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="section-hdr">📊 Comparative Method Results</div>', unsafe_allow_html=True)

    if not run_btn:
        st.info("👈 Enter parameters in the sidebar and click **RUN COMPARATIVE ANALYSIS**")

        st.markdown("""
        <div class="info-card">
        <b>What this framework does:</b><br>
        Instead of relying on a single correlation, WaterWatch evaluates <b>four established methods</b> 
        simultaneously. This reveals how much predictions diverge and provides guidance on which 
        method is most appropriate for your reservoir type.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        | Method | Type | Best For | Conservative? |
        |--------|------|----------|---------------|
        | **Sobocinski-Cornelius (Standard)** | Breakthrough Time | General screening, thin oil rims | Baseline |
        | **Sobocinski-Cornelius (Original)** | Breakthrough Time | Historical comparison only | Unstable |
        | **Meyer-Garder (1954)** | Critical Rate | Analytical conservative estimate | Very conservative |
        | **Schols (1972)** | Critical Rate | High-perm sands, practical rates | Moderate |
        """)
        st.stop()

    # Validation
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
        if Pi > Pb:
            mu_o = undersaturated_viscosity(mu_ob, Pi, Pb)
            cond = "Undersaturated"
        else:
            mu_o = mu_ob
            cond = "Saturated"
        Bo = oil_fvf(Rs, gg, go_val, T_F)
    else:
        mu_o = mu_o_meas
        Bo = Bo_meas
        Pb = None
        mu_od = None
        mu_ob = None
        cond = "Measured PVT"

    rw_dens = water_density(sal, T_F, Pi)
    ro_dens = oil_density(API, Bo, Rs, gg)
    M, alpha = mobility_ratio(krw, kro, mu_o, mu_w)

    # ── METHOD 1: Sobocinski-Cornelius (Standard) ──
    Z1, tD1, tBT1, err1 = sobocinski_standard(kh, kv, phi, h, hp, mu_o, Bo, Qo, rw_dens, ro_dens, M, alpha)

    # ── METHOD 2: Sobocinski-Cornelius (Original) ──
    Z2, tD2, tBT2, err2 = sobocinski_original(kh, kv, phi, h, hp, mu_o, Bo, Qo, rw_dens, ro_dens, M, alpha)

    # ── METHOD 3: Meyer-Garder Critical Rate ──
    qc_mg, err_mg = meyer_garder(kh, kro, h, hp, mu_o, Bo, rw_dens, ro_dens, re, rw)

    # ── METHOD 4: Schols Critical Rate ──
    qc_sch, err_sch = schols_critical_rate(kh, kv, krw, kro, h, mu_o, mu_w, Bo, rw_dens, ro_dens, re, rw)

    # Display intermediate PVT
    with st.expander("🔬 PVT & Fluid Properties Calculated"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("μo", f"{mu_o:.3f} cp")
        c2.metric("Bo", f"{Bo:.3f} bbl/STB")
        c3.metric("ρw", f"{rw_dens:.1f} lb/ft³")
        c4.metric("ρo", f"{ro_dens:.1f} lb/ft³")
        c5, c6, c7 = st.columns(3)
        c5.metric("Δρ (g/cc)", f"{(rw_dens-ro_dens)/62.4:.3f}")
        c6.metric("M (mobility)", f"{M:.3f}")
        c7.metric("α", f"{alpha}")
        if pvt_mode == "Calculate from correlations":
            st.caption(f"Pb = {Pb:.0f} psia | μod = {mu_od:.3f} cp | μob = {mu_ob:.3f} cp | Condition: {cond}")

    # ── RESULTS GRID ──
    col1, col2 = st.columns(2)

    with col1:
        # Method 1
        st.markdown('<div class="method-card-sc">', unsafe_allow_html=True)
        st.markdown("#### 🔵 Method 1: Sobocinski-Cornelius (Standard)")
        st.caption("Ahmed (2010) Eq 9-21–9-23 | Stable tD = Z/(3−0.7Z)")
        if err1:
            st.error(f"❌ {err1}")
        else:
            r1 = risk_level_bt(tBT1)
            st.markdown(f"**Breakthrough Time: {tBT1:.0f} days ({tBT1/30.4:.1f} months)**")
            st.markdown(f"<span style='color:{r1['color']};font-size:1.3rem;'>{r1['icon']} {r1['cat']}</span>", unsafe_allow_html=True)
            cA, cB = st.columns(2)
            cA.metric("Z", f"{Z1:.3f}")
            cB.metric("tD", f"{tD1:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Method 3
        st.markdown('<div class="method-card-mg">', unsafe_allow_html=True)
        st.markdown("#### 🟣 Method 3: Meyer-Garder (1954)")
        st.caption("Analytical critical rate | Very conservative")
        if err_mg:
            st.error(f"❌ {err_mg}")
        else:
            r_mg = risk_level_cr(qc_mg, Qo)
            st.markdown(f"**Critical Rate: {qc_mg:.0f} STB/D**")
            st.markdown(f"<span style='color:{r_mg['color']};font-size:1.3rem;'>{r_mg['icon']} {r_mg['cat']} (Qo/Qc = {Qo/qc_mg:.2f})</span>", unsafe_allow_html=True)
            st.progress(min(Qo/qc_mg, 1.0))
            st.caption(f"Current rate {Qo:.0f} STB/D vs critical {qc_mg:.0f} STB/D")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Method 2
        st.markdown('<div class="method-card-sc-orig">', unsafe_allow_html=True)
        st.markdown("#### 🔴 Method 2: Sobocinski-Cornelius (Original 1965)")
        st.caption("Original polynomial | Numerically unstable near Z=3.5")
        if err2:
            st.error(f"❌ {err2}")
        else:
            r2 = risk_level_bt(tBT2)
            st.markdown(f"**Breakthrough Time: {tBT2:.0f} days ({tBT2/30.4:.1f} months)**")
            st.markdown(f"<span style='color:{r2['color']};font-size:1.3rem;'>{r2['icon']} {r2['cat']}</span>", unsafe_allow_html=True)
            cC, cD = st.columns(2)
            cC.metric("Z", f"{Z2:.3f}")
            cD.metric("tD", f"{tD2:.3f}")
            if tBT1 and not err1:
                diff = ((tBT2 - tBT1) / tBT1) * 100
                st.caption(f"⚠️ Original predicts **{diff:+.0f}%** vs Standard")
        st.markdown('</div>', unsafe_allow_html=True)

        # Method 4
        st.markdown('<div class="method-card-sch">', unsafe_allow_html=True)
        st.markdown("#### 🟢 Method 4: Schols (1972)")
        st.caption("Empirical from simulation | Practical for high-perm sands")
        if err_sch:
            st.error(f"❌ {err_sch}")
        else:
            r_sch = risk_level_cr(qc_sch, Qo)
            st.markdown(f"**Critical Rate: {qc_sch:.0f} STB/D**")
            st.markdown(f"<span style='color:{r_sch['color']};font-size:1.3rem;'>{r_sch['icon']} {r_sch['cat']} (Qo/Qc = {Qo/qc_sch:.2f})</span>", unsafe_allow_html=True)
            st.progress(min(Qo/qc_sch, 1.0))
            st.caption(f"Current rate {Qo:.0f} STB/D vs critical {qc_sch:.0f} STB/D")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── COMPARATIVE SUMMARY ──
    st.divider()
    st.markdown('<div class="section-hdr">📈 Comparative Summary</div>', unsafe_allow_html=True)

    comp_data = []
    if tBT1 and not err1:
        comp_data.append({"Method": "Sobocinski-Cornelius (Standard)", "Type": "Breakthrough Time", "Value": f"{tBT1:.0f} days", "Risk": risk_level_bt(tBT1)['cat']})
    if tBT2 and not err2:
        comp_data.append({"Method": "Sobocinski-Cornelius (Original)", "Type": "Breakthrough Time", "Value": f"{tBT2:.0f} days", "Risk": risk_level_bt(tBT2)['cat']})
    if qc_mg and not err_mg:
        comp_data.append({"Method": "Meyer-Garder", "Type": "Critical Rate", "Value": f"{qc_mg:.0f} STB/D", "Risk": risk_level_cr(qc_mg, Qo)['cat']})
    if qc_sch and not err_sch:
        comp_data.append({"Method": "Schols", "Type": "Critical Rate", "Value": f"{qc_sch:.0f} STB/D", "Risk": risk_level_cr(qc_sch, Qo)['cat']})

    if comp_data:
        df_comp = pd.DataFrame(comp_data)
        st.dataframe(df_comp, hide_index=True, use_container_width=True)

    # ── GUIDANCE BOX ──
    st.markdown("""
    <div class="info-card">
    <b>Engineering Guidance:</b><br>
    • <b>Sobocinski-Cornelius (Standard)</b> is the most widely-used analytical method. Use it for general screening.<br>
    • <b>Meyer-Garder</b> gives very conservative critical rates — useful as a lower bound, but often impractical.<br>
    • <b>Schols</b> is less conservative and better calibrated to simulation — preferred for high-perm Niger Delta sands.<br>
    • <b>The Original 1965 Polynomial</b> is shown here for historical comparison only. It is numerically unstable.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 2 — SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-hdr">📈 Sensitivity Analysis</div>', unsafe_allow_html=True)

    if not run_btn:
        st.info("Run the analysis in Tab 1 first to populate sensitivity data.")
        st.stop()

    sens_param = st.selectbox(
        "Select parameter to vary",
        ["Production Rate (Qo)", "Perforation Interval (hp)", "Vertical Permeability (kv)", "Oil Column (h)"]
    )

    # Base parameters for sensitivity
    n_points = 50

    if sens_param == "Production Rate (Qo)":
        vary_vals = np.linspace(200, 3000, n_points)
        base_params = (kh, kv, phi, h, hp, mu_o, Bo, rw_dens, ro_dens, M, alpha)

        bt_std, bt_orig, qc_mg_list, qc_sch_list = [], [], [], []
        for q in vary_vals:
            Z1, tD1, tBT1, _ = sobocinski_standard(kh, kv, phi, h, hp, mu_o, Bo, q, rw_dens, ro_dens, M, alpha)
            Z2, tD2, tBT2, _ = sobocinski_original(kh, kv, phi, h, hp, mu_o, Bo, q, rw_dens, ro_dens, M, alpha)
            bt_std.append(tBT1 if tBT1 else None)
            bt_orig.append(tBT2 if tBT2 else None)
            qm, _ = meyer_garder(kh, kro, h, hp, mu_o, Bo, rw_dens, ro_dens, re, rw)
            qs, _ = schols_critical_rate(kh, kv, krw, kro, h, mu_o, mu_w, Bo, rw_dens, ro_dens, re, rw)
            qc_mg_list.append(qm if qm else None)
            qc_sch_list.append(qs if qs else None)

        fig = make_subplots(rows=2, cols=1, subplot_titles=("Breakthrough Time vs Rate", "Critical Rate vs Rate"),
                           vertical_spacing=0.15)
        fig.add_trace(go.Scatter(x=vary_vals, y=bt_std, mode='lines', name='Sobocinski-Cornelius (Std)',
                                 line=dict(color='#3498db', width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=bt_orig, mode='lines', name='Sobocinski-Cornelius (Orig)',
                                 line=dict(color='#e74c3c', width=2, dash='dash')), row=1, col=1)
        fig.add_hline(y=90, line_dash="dot", line_color="red", row=1, col=1)
        fig.add_hline(y=365, line_dash="dot", line_color="orange", row=1, col=1)

        fig.add_trace(go.Scatter(x=vary_vals, y=qc_mg_list, mode='lines', name='Meyer-Garder',
                                 line=dict(color='#9b59b6', width=2.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=qc_sch_list, mode='lines', name='Schols',
                                 line=dict(color='#2ecc71', width=2.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=vary_vals, mode='lines', name='Qo = Qc (unity)',
                                 line=dict(color='white', width=1, dash='dot')), row=2, col=1)

        fig.update_xaxes(title_text="Qo (STB/D)")
        fig.update_yaxes(title_text="BT (days)", row=1, col=1)
        fig.update_yaxes(title_text="Qc (STB/D)", row=2, col=1)

    elif sens_param == "Perforation Interval (hp)":
        vary_vals = np.linspace(1, h-5, n_points)
        bt_std, bt_orig, qc_mg_list, qc_sch_list = [], [], [], []
        for hp_v in vary_vals:
            Z1, tD1, tBT1, _ = sobocinski_standard(kh, kv, phi, h, hp_v, mu_o, Bo, Qo, rw_dens, ro_dens, M, alpha)
            Z2, tD2, tBT2, _ = sobocinski_original(kh, kv, phi, h, hp_v, mu_o, Bo, Qo, rw_dens, ro_dens, M, alpha)
            bt_std.append(tBT1 if tBT1 else None)
            bt_orig.append(tBT2 if tBT2 else None)
            qm, _ = meyer_garder(kh, kro, h, hp_v, mu_o, Bo, rw_dens, ro_dens, re, rw)
            qs, _ = schols_critical_rate(kh, kv, krw, kro, h, mu_o, mu_w, Bo, rw_dens, ro_dens, re, rw)
            qc_mg_list.append(qm if qm else None)
            qc_sch_list.append(qs if qs else None)

        fig = make_subplots(rows=2, cols=1, subplot_titles=("Breakthrough Time vs Perforation", "Critical Rate vs Perforation"),
                           vertical_spacing=0.15)
        fig.add_trace(go.Scatter(x=vary_vals, y=bt_std, mode='lines', name='Sobocinski-Cornelius (Std)',
                                 line=dict(color='#3498db', width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=bt_orig, mode='lines', name='Sobocinski-Cornelius (Orig)',
                                 line=dict(color='#e74c3c', width=2, dash='dash')), row=1, col=1)
        fig.add_hline(y=90, line_dash="dot", line_color="red", row=1, col=1)

        fig.add_trace(go.Scatter(x=vary_vals, y=qc_mg_list, mode='lines', name='Meyer-Garder',
                                 line=dict(color='#9b59b6', width=2.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=qc_sch_list, mode='lines', name='Schols',
                                 line=dict(color='#2ecc71', width=2.5)), row=2, col=1)

        fig.update_xaxes(title_text="hp (ft)")
        fig.update_yaxes(title_text="BT (days)", row=1, col=1)
        fig.update_yaxes(title_text="Qc (STB/D)", row=2, col=1)

    elif sens_param == "Vertical Permeability (kv)":
        vary_vals = np.linspace(10, min(kh, 500), n_points)
        bt_std, bt_orig, qc_mg_list, qc_sch_list = [], [], [], []
        for kv_v in vary_vals:
            Z1, tD1, tBT1, _ = sobocinski_standard(kh, kv_v, phi, h, hp, mu_o, Bo, Qo, rw_dens, ro_dens, M, alpha)
            Z2, tD2, tBT2, _ = sobocinski_original(kh, kv_v, phi, h, hp, mu_o, Bo, Qo, rw_dens, ro_dens, M, alpha)
            bt_std.append(tBT1 if tBT1 else None)
            bt_orig.append(tBT2 if tBT2 else None)
            qm, _ = meyer_garder(kh, kro, h, hp, mu_o, Bo, rw_dens, ro_dens, re, rw)
            qs, _ = schols_critical_rate(kh, kv_v, krw, kro, h, mu_o, mu_w, Bo, rw_dens, ro_dens, re, rw)
            qc_mg_list.append(qm if qm else None)
            qc_sch_list.append(qs if qs else None)

        fig = make_subplots(rows=2, cols=1, subplot_titles=("Breakthrough Time vs kv", "Critical Rate vs kv"),
                           vertical_spacing=0.15)
        fig.add_trace(go.Scatter(x=vary_vals, y=bt_std, mode='lines', name='Sobocinski-Cornelius (Std)',
                                 line=dict(color='#3498db', width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=bt_orig, mode='lines', name='Sobocinski-Cornelius (Orig)',
                                 line=dict(color='#e74c3c', width=2, dash='dash')), row=1, col=1)

        fig.add_trace(go.Scatter(x=vary_vals, y=qc_mg_list, mode='lines', name='Meyer-Garder',
                                 line=dict(color='#9b59b6', width=2.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=qc_sch_list, mode='lines', name='Schols',
                                 line=dict(color='#2ecc71', width=2.5)), row=2, col=1)

        fig.update_xaxes(title_text="kv (mD)")
        fig.update_yaxes(title_text="BT (days)", row=1, col=1)
        fig.update_yaxes(title_text="Qc (STB/D)", row=2, col=1)

    else:  # Oil Column (h)
        vary_vals = np.linspace(hp+5, 200, n_points)
        bt_std, bt_orig, qc_mg_list, qc_sch_list = [], [], [], []
        for h_v in vary_vals:
            Z1, tD1, tBT1, _ = sobocinski_standard(kh, kv, phi, h_v, hp, mu_o, Bo, Qo, rw_dens, ro_dens, M, alpha)
            Z2, tD2, tBT2, _ = sobocinski_original(kh, kv, phi, h_v, hp, mu_o, Bo, Qo, rw_dens, ro_dens, M, alpha)
            bt_std.append(tBT1 if tBT1 else None)
            bt_orig.append(tBT2 if tBT2 else None)
            qm, _ = meyer_garder(kh, kro, h_v, hp, mu_o, Bo, rw_dens, ro_dens, re, rw)
            qs, _ = schols_critical_rate(kh, kv, krw, kro, h_v, mu_o, mu_w, Bo, rw_dens, ro_dens, re, rw)
            qc_mg_list.append(qm if qm else None)
            qc_sch_list.append(qs if qs else None)

        fig = make_subplots(rows=2, cols=1, subplot_titles=("Breakthrough Time vs Oil Column", "Critical Rate vs Oil Column"),
                           vertical_spacing=0.15)
        fig.add_trace(go.Scatter(x=vary_vals, y=bt_std, mode='lines', name='Sobocinski-Cornelius (Std)',
                                 line=dict(color='#3498db', width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=bt_orig, mode='lines', name='Sobocinski-Cornelius (Orig)',
                                 line=dict(color='#e74c3c', width=2, dash='dash')), row=1, col=1)

        fig.add_trace(go.Scatter(x=vary_vals, y=qc_mg_list, mode='lines', name='Meyer-Garder',
                                 line=dict(color='#9b59b6', width=2.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=vary_vals, y=qc_sch_list, mode='lines', name='Schols',
                                 line=dict(color='#2ecc71', width=2.5)), row=2, col=1)

        fig.update_xaxes(title_text="h (ft)")
        fig.update_yaxes(title_text="BT (days)", row=1, col=1)
        fig.update_yaxes(title_text="Qc (STB/D)", row=2, col=1)

    fig.update_layout(height=700, plot_bgcolor='#0e1621', paper_bgcolor='#0e1621',
                      font=dict(color='white'), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="info-card">
    <b>Key Insight:</b> Breakthrough time is most sensitive to <b>production rate</b> and <b>perforation standoff</b>.
    Critical rate methods (Meyer-Garder, Schols) show that increasing oil column thickness or reducing 
    vertical permeability significantly improves water-free production capacity.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 3 — VALIDATION
# ═══════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-hdr">✅ Validation Against Published Niger Delta Data</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
    This table compares model predictions against published reservoir characteristics 
    from actual Niger Delta fields. Note: Published data typically reports <b>ranges</b> 
    rather than exact breakthrough times, as BT depends on specific completion design.
    </div>
    """, unsafe_allow_html=True)

    val_data = [
        {
            "Field": "Kukih (Sand A)",
            "Type": "Onshore",
            "kh (mD)": 1087,
            "φ": 0.297,
            "h (ft)": 45,
            "Published Range": "400–700 days",
            "Notes": "Moderate perm, moderate rate assumed"
        },
        {
            "Field": "H-Field (Sand A)",
            "Type": "Offshore",
            "kh (mD)": 5000,
            "φ": 0.24,
            "h (ft)": 60,
            "Published Range": "200–400 days",
            "Notes": "High perm + high rate = shorter BT"
        },
        {
            "Field": "OB-63 Reservoir",
            "Type": "Simulation",
            "kh (mD)": 800,
            "φ": 0.28,
            "h (ft)": 50,
            "Published Range": "760 days (simulated)",
            "Notes": "Reservoir simulation benchmark"
        },
        {
            "Field": "NEMA Field",
            "Type": "Onshore",
            "kh (mD)": 875,
            "φ": 0.19,
            "h (ft)": 35,
            "Published Range": "800–1500 days",
            "Notes": "Lower perm, lower porosity"
        },
    ]

    df_val = pd.DataFrame(val_data)
    st.dataframe(df_val, hide_index=True, use_container_width=True)

    st.markdown("""
    <div class="limitation-card">
    <b>Validation Limitations:</b><br>
    • Exact breakthrough times are rarely published — most papers report ranges or simulation results.<br>
    • Critical rate correlations cannot be directly validated against BT data (different physics).<br>
    • The framework is designed for <b>comparative screening</b>, not exact field-scale prediction.<br>
    • Full validation requires well-specific production history — recommended as future work.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT & METHODOLOGY
# ═══════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("""
    ## About WaterWatch

    WaterWatch is a **comparative analytical screening framework** for predicting water coning 
    behavior in Niger Delta vertical oil wells. Rather than relying on a single correlation, 
    it evaluates four established methods simultaneously, quantifies their divergence, and 
    provides field-specific guidance on method selection.

    ---

    ### Core Contribution

    **This study is the first to implement a multi-method comparative coning screening tool 
    specifically calibrated for Niger Delta reservoir characteristics** (high-permeability 
    Agbada Formation sands, interbedded shales, and typical fluid properties).

    Key findings from the comparative analysis:

    | Finding | Implication |
    |---------|-------------|
    | Original 1965 polynomial is numerically unstable near Z=3.5 | Standard simplified form (Bournazel-Jeanson) should be used |
    | Meyer-Garder predicts very low critical rates | Useful as conservative lower bound, but often impractical |
    | Schols gives higher, more practical critical rates | Preferred for high-perm Niger Delta sands |
    | BT is most sensitive to rate and standoff | Completion design should prioritize perforation placement |

    ---

    ### Methodology

    **Method 1: Sobocinski-Cornelius (Standard)**
    - Source: Ahmed (2010) *Reservoir Engineering Handbook*, Eq 9-21 to 9-23
    - Based on: Sobocinski & Cornelius (1965) SPE-894; Bournazel & Jeanson (1971) SPE-3628
    - Formula: $t_D = Z / (3 - 0.7Z)$
    - Valid: $Z < 3.0$
    - Best for: General screening, thin oil rims

    **Method 2: Sobocinski-Cornelius (Original 1965)**
    - Source: Sobocinski & Cornelius (1965) JPT, May 1965
    - Formula: $t_D = (4Z + 1.75Z^2 - 0.75Z^3) / (7 - 2Z)$
    - Valid: $Z < 3.5$ (theoretical), unstable near limit
    - Best for: Historical comparison only
    - **Key finding:** Denominator $(7-2Z) \rightarrow 0$ as $Z \rightarrow 3.5$, causing numerical instability

    **Method 3: Meyer-Garder (1954)**
    - Source: Meyer & Garder (1954) *J. Appl. Phys.* 25, No. 11
    - Formula: $q_c = 0.001535 \cdot \frac{\rho_w - \rho_o}{\ln(r_e/r_w)} \cdot \frac{k_o}{\mu_o B_o} \cdot (h^2 - h_p^2)$
    - Best for: Conservative analytical lower bound

    **Method 4: Schols (1972)**
    - Source: Schols, R.S. (1972) "An Empirical Formula for the Critical Oil Production Rate," *Erdoel-Erdgas*, Jan 1972
    - Formula: $q_c = \frac{0.00333 \cdot (\rho_w - \rho_o) \cdot k_v \cdot h^2}{\mu_o B_o \cdot [\ln(r_e/r_w) - 0.75 + M^{0.5}]}$
    - Best for: Practical high-perm reservoir screening

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

    1. **Analytical models assume homogeneous, radial flow** — Niger Delta reservoirs are heterogeneous with interbedded shales
    2. **No post-breakthrough performance prediction** — WOR behavior after BT is not modeled
    3. **Single-well analysis** — Interference from offset wells is neglected
    4. **Endpoint relative permeabilities** — Actual curves are rarely available for screening
    5. **Critical rate ≠ breakthrough time** — These are different physical quantities; direct comparison requires care
    6. **Validation is qualitative** — Exact field BT data is proprietary; framework validated against published ranges

    ---

    ### Recommendations for Use

    1. **For thin oil rims (< 30 ft):** Use Sobocinski-Cornelius (Standard) with conservative rate assumptions
    2. **For high-perm Niger Delta sands (> 1000 mD):** Use Schols critical rate as primary guide
    3. **For marginal field economics:** Use Meyer-Garder as absolute lower bound
    4. **For detailed development planning:** Follow screening with full reservoir simulation

    ---

    ### Key References

    - Ahmed, T. (2010) *Reservoir Engineering Handbook*, 4th Ed., Gulf Professional Publishing
    - Sobocinski, D.P. & Cornelius, A.J. (1965) "A Correlation for Predicting Water Coning Time," *JPT*, May 1965, SPE-894
    - Bournazel, C. & Jeanson, B. (1971) "Fast Water Coning Evaluation," SPE-3628
    - Meyer, H.I. & Garder, A.O. (1954) "Mechanics of Two Immiscible Fluids in Porous Media," *J. Appl. Phys.*, 25(11)
    - Schols, R.S. (1972) "An Empirical Formula for the Critical Oil Production Rate," *Erdoel-Erdgas*, Jan 1972
    - Chaperon, I. (1986) "Theoretical Study of Coning Toward Horizontal and Vertical Wells," SPE-15377
    - Standing, M.B. (1947, 1981) PVT correlations
    - Beggs, H.D. & Robinson, J.R. (1975) Viscosity correlation

    ---

    ### University of Benin
    **Department of Petroleum Engineering**  
    **Final Year Project**
    """)
