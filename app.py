# ══════════════════════════════════════════════
# WATERWATCH — Enhanced Sobocinski-Cornelius
# Aquifer Water Breakthrough Prediction Tool
# Niger Delta Oil Reservoirs
# University of Benin Final Year Project
# ══════════════════════════════════════════════

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ── Page Configuration ─────────────────────────
st.set_page_config(
    page_title="WaterWatch",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom Styling ─────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
    }
    * {
        font-family: 'Arial', sans-serif;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 10px;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 20px;
    }
       .result-box-red {
        background-color: #c0392b;
        border-left: 8px solid #ff0000;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
        color: #ffffff;
    }
    .result-box-red h2 {
        color: #ffffff !important;
        font-size: 2rem;
    }
    .result-box-red h3 {
        color: #ffffff !important;
        font-size: 1.5rem;
    }
    .result-box-red p {
        color: #ffffff !important;
    }
    .result-box-yellow {
        background-color: #d35400;
        border-left: 8px solid #ffa500;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
        color: #ffffff;
    }
    .result-box-yellow h2 {
        color: #ffffff !important;
        font-size: 2rem;
    }
    .result-box-yellow h3 {
        color: #ffffff !important;
        font-size: 1.5rem;
    }
    .result-box-yellow p {
        color: #ffffff !important;
    }
    .result-box-green {
        background-color: #27ae60;
        border-left: 8px solid #00ff00;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
        color: #ffffff;
    }
    .result-box-green h2 {
        color: #ffffff !important;
        font-size: 2rem;
    }
    .result-box-green h3 {
        color: #ffffff !important;
        font-size: 1.5rem;
    }
    .result-box-green p {
        color: #ffffff !important;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# CORE CALCULATION FUNCTIONS
# ══════════════════════════════════════════════

def calculate_oil_specific_gravity(API):
    return 141.5 / (API + 131.5)

def calculate_dead_oil_viscosity(API, T_F):
    """
    Beal (1946) / Standing (1981)
    Ahmed (2010) Equation 2-117
    """
    T_R   = T_F + 460
    a     = 10**(0.43 + 8.33/API)
    mu_od = (0.32 + (1.8e7/API**4.53)) * \
            (360/(T_R - 260))**a
    return round(mu_od, 4)

def calculate_bubble_point(Rs, gamma_g, T_F, API):
    """
    Standing (1947)
    Ahmed (2010) Equation 2-72
    """
    Pb = 18.2 * ((Rs/gamma_g)**0.83 *
                  10**(0.00091*T_F -
                       0.0125*API) - 1.4)
    return round(abs(Pb), 1)

def calculate_saturated_viscosity(mu_od, Rs):
    """
    Beggs and Robinson (1975)
    Ahmed (2010) Equation 2-121
    """
    a     = 10.715 * (Rs + 100)**(-0.515)
    b     = 5.44   * (Rs + 150)**(-0.338)
    mu_ob = a * (mu_od**b)
    return round(mu_ob, 4)

def calculate_undersaturated_viscosity(mu_ob,
                                        Pi, Pb):
    """
    Ahmed (2010) Equation 2-123
    """
    a    = -3.9e-5 * Pi - 5
    m    = 2.6 * Pi**1.187 * 10**a
    mu_o = mu_ob * (Pi/Pb)**m
    return round(mu_o, 4)

def calculate_oil_fvf(Rs, gamma_g,
                       gamma_o, T_F):
    """
    Standing (1947/1981)
    Ahmed (2010) Equation 2-85
    """
    T_R = T_F + 460
    F   = Rs * (gamma_g/gamma_o)**0.5 + \
          1.25 * (T_R - 460)
    Bo  = 0.9759 + 0.000120 * (F**1.2)
    return round(Bo, 4)

def calculate_water_density(salinity_ppm,
                             T_F, P_psia):
    """
    Niger Delta formation water density
    Source: Tuttle et al (1999)
    """
    rho_w = (62.4 +
             (salinity_ppm/10000 * 0.5) -
             0.003 * (T_F - 60) +
             0.0000145 * P_psia)
    return round(rho_w, 3)

def calculate_oil_density(API, Bo, Rs, gamma_g):
    """
    Oil density at reservoir conditions
    Ahmed (2010) Chapter 2
    """
    gamma_o  = 141.5 / (API + 131.5)
    rho_surf = gamma_o * 62.4
    rho_o    = (rho_surf +
                0.01357 * Rs * gamma_g) / Bo
    return round(rho_o, 3)

def apply_niger_delta_corrections(kh_mean,
        kv_matrix, phi_log, depth_ft,
        V_DP, Vsh, SCI):
    """
    Niger Delta Rock Corrections:
    1. Dykstra-Parsons - Tuttle et al (1999)
    2. SCI - Novel contribution this study
       Short & Stauble (1967)
    3. Athy compaction - Athy (1930)
       Doust & Omatsola (1990)
    """
    kh_eff  = kh_mean * (1 - V_DP)
    kv_eff  = kv_matrix * (1 - Vsh * SCI)
    phi_eff = phi_log * np.exp(-0.000025 *
                                depth_ft)
    return {
        'kh_eff':  round(kh_eff, 2),
        'kv_eff':  round(kv_eff, 4),
        'phi_eff': round(phi_eff, 4)
    }

def calculate_mobility_ratio(krw_sor, kro_swc,
                              mu_o, mu_w):
    """Ahmed (2010) Equation 9-24"""
    M = (krw_sor/kro_swc) * (mu_o/mu_w)
    alpha = 0.5 if M <= 1 else 0.6
    return round(M, 4), alpha

def sobocinski_cornelius(kh_eff, kv_eff,
        phi_eff, h, hp, mu_o, Bo, Qo,
        rho_w, rho_o, M, alpha):
    """
    Sobocinski-Cornelius (1965)
    Ahmed (2010) Equations 9-21 to 9-23
    Bottom water drive — vertical wells
    """
    delta_rho = rho_w - rho_o
    if delta_rho <= 0:
        return None, None, None, \
               "Error: Water must be denser than oil"

    Z = (0.492e-4 * delta_rho * kh_eff *
         h * (h - hp)) / (mu_o * Bo * Qo)

    if Z <= 0:
        return None, None, None, \
               "Error: Z≤0 — check h > hp"
    if Z >= 3.5:
        return None, None, None, \
               "Warning: Z≥3.5 outside valid range"

    tD_BT = (4*Z + 1.75*Z**2 - 0.75*Z**3) / \
            (7 - 2*Z)

    tBT = (20325 * mu_o * h * phi_eff * tD_BT) / \
          (delta_rho * kv_eff * (1 + M**alpha))

    return (round(Z, 4),
            round(tD_BT, 4),
            round(tBT, 1), None)

def bournazel_jeanson(kh_eff, kv_eff,
        phi_eff, h, hp, mu_o, Bo, Qo,
        rho_w, rho_o, M):
    """
    Bournazel and Jeanson (1971) SPE-3628
    Edge water drive — vertical wells
    Valid: 0.14 < M ≤ 7.3
    """
    delta_rho = rho_w - rho_o
    alpha_bj  = 0.7

    Z = (0.492e-4 * delta_rho * kh_eff *
         h * (h - hp)) / (mu_o * Bo * Qo)

    if Z <= 0 or (3 - 0.7*Z) <= 0:
        return None, None, None, \
               "Outside valid range"

    tD_BT_BJ = Z / (3 - 0.7*Z)

    tBT_BJ = (20325 * mu_o * h * phi_eff *
               tD_BT_BJ) / \
              (delta_rho * kv_eff *
               (1 + M**alpha_bj))

    return (round(Z, 4),
            round(tD_BT_BJ, 4),
            round(tBT_BJ, 1), None)

def classify_risk(tBT):
    if tBT <= 365:
        return {
            'category': 'HIGH RISK',
            'symbol':   '🔴',
            'color':    'red',
            'box':      'result-box-red',
            'action':   'Begin water handling '
                       'facility planning '
                       'immediately. Consider '
                       'rate reduction.'
        }
    elif tBT <= 730:
        return {
            'category': 'MEDIUM RISK',
            'symbol':   '🟡',
            'color':    'orange',
            'box':      'result-box-yellow',
            'action':   'Plan water handling '
                       'within 6 months. '
                       'Monitor production closely.'
        }
    else:
        return {
            'category': 'LOW RISK',
            'symbol':   '🟢',
            'color':    'green',
            'box':      'result-box-green',
            'action':   'Monitor quarterly. '
                       'No immediate action required.'
        }

def run_sci_sensitivity(kh_eff, kv_matrix,
        phi_eff, h, hp, mu_o, Bo, Qo,
        rho_w, rho_o, M, alpha, Vsh):
    """SCI Sensitivity Analysis"""
    SCI_values = [0.0, 0.2, 0.4,
                  0.6, 0.8, 1.0]
    results = []
    for SCI in SCI_values:
        kv_s = kv_matrix * (1 - Vsh * SCI)
        _, _, tBT_s, err = sobocinski_cornelius(
            kh_eff, kv_s, phi_eff,
            h, hp, mu_o, Bo, Qo,
            rho_w, rho_o, M, alpha
        )
        if tBT_s:
            risk = classify_risk(tBT_s)
            results.append({
                'SCI':      SCI,
                'kv_eff':   round(kv_s, 2),
                'tBT_days': tBT_s,
                'tBT_years':round(tBT_s/365, 2),
                'Risk':     risk['category']
            })
    return pd.DataFrame(results)

# ══════════════════════════════════════════════
# APP HEADER
# ══════════════════════════════════════════════

st.markdown(
    '<div class="main-header">💧 WaterWatch</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-header">'
    'Enhanced Sobocinski-Cornelius Aquifer '
    'Water Breakthrough Prediction Tool<br>'
    'Niger Delta Oil Reservoirs | '
    'University of Benin Final Year Project'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ══════════════════════════════════════════════
# SIDEBAR — INPUT PARAMETERS
# ══════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Input Parameters")

    # PVT Option
    pvt_option = st.radio(
        "PVT Data Availability",
        ["Calculate from correlations",
         "Enter measured PVT values"],
        help="Select if you have a PVT report"
    )

    st.subheader("🪨 Rock Properties")
    kh_mean   = st.number_input(
        "Horizontal Permeability kh (md)",
        min_value=10.0, max_value=5000.0,
        value=1800.0, step=10.0)
    kv_matrix = st.number_input(
        "Vertical Permeability kv (md)",
        min_value=1.0, max_value=1000.0,
        value=270.0, step=5.0)
    phi_log   = st.number_input(
        "Log Porosity φ (fraction)",
        min_value=0.05, max_value=0.45,
        value=0.28, step=0.01)
    depth_ft  = st.number_input(
        "Reservoir Depth (ft)",
        min_value=1000.0, max_value=15000.0,
        value=8500.0, step=100.0)
    h         = st.number_input(
        "Oil Column Height h (ft)",
        min_value=5.0, max_value=300.0,
        value=80.0, step=1.0)
    hp        = st.number_input(
        "Perforated Interval hp (ft)",
        min_value=1.0, max_value=200.0,
        value=25.0, step=1.0)

    st.subheader("🌍 Niger Delta Corrections")
    V_DP = st.slider(
        "Dykstra-Parsons Coefficient V",
        min_value=0.0, max_value=0.9,
        value=0.45, step=0.05,
        help="Niger Delta range: 0.3-0.6 "
             "(Tuttle et al 1999)")
    Vsh  = st.slider(
        "Shale Volume Fraction Vsh",
        min_value=0.0, max_value=0.5,
        value=0.15, step=0.01,
        help="From Gamma Ray log. "
             "Niger Delta clean sands: 0.05-0.25 "
             "(Avbovbo 1978)")
    SCI  = st.slider(
        "Shale Continuity Index SCI ★",
        min_value=0.0, max_value=1.0,
        value=0.50, step=0.05,
        help="★ Novel parameter proposed in "
             "this study. "
             "0=discontinuous shales, "
             "1=fully continuous shales. "
             "Estimate: SCI = 1 - NTG")

    st.subheader("🧪 Fluid Properties")
    if pvt_option == "Calculate from correlations":
        API     = st.number_input(
            "API Gravity (°)",
            min_value=15.0, max_value=55.0,
            value=35.0, step=0.5)
        T_F     = st.number_input(
            "Reservoir Temperature (°F)",
            min_value=100.0, max_value=300.0,
            value=180.0, step=5.0)
        Rs      = st.number_input(
            "Solution GOR Rs (scf/STB)",
            min_value=50.0, max_value=2000.0,
            value=600.0, step=10.0)
        gamma_g = st.number_input(
            "Gas Specific Gravity γg",
            min_value=0.5, max_value=1.2,
            value=0.75, step=0.01)
        Pi      = st.number_input(
            "Initial Reservoir Pressure (psia)",
            min_value=500.0, max_value=10000.0,
            value=4200.0, step=50.0)
        salinity_ppm = st.number_input(
            "Formation Water Salinity (ppm)",
            min_value=1000.0, max_value=150000.0,
            value=35000.0, step=1000.0)
        mu_w    = st.number_input(
            "Water Viscosity μw (cp)",
            min_value=0.2, max_value=1.5,
            value=0.50, step=0.05)
        mu_o_measured = None
        Bo_measured   = None
        Pb_measured   = None
    else:
        API     = st.number_input(
            "API Gravity (°)",
            min_value=15.0, max_value=55.0,
            value=35.0, step=0.5)
        T_F     = st.number_input(
            "Reservoir Temperature (°F)",
            min_value=100.0, max_value=300.0,
            value=180.0, step=5.0)
        Pi      = st.number_input(
            "Initial Reservoir Pressure (psia)",
            min_value=500.0, max_value=10000.0,
            value=4200.0, step=50.0)
        salinity_ppm = st.number_input(
            "Formation Water Salinity (ppm)",
            min_value=1000.0, max_value=150000.0,
            value=35000.0, step=1000.0)
        mu_o_measured = st.number_input(
            "Measured Oil Viscosity μo (cp)",
            min_value=0.1, max_value=100.0,
            value=0.6, step=0.1)
        Bo_measured   = st.number_input(
            "Measured Oil FVF Bo (bbl/STB)",
            min_value=1.0, max_value=3.0,
            value=1.34, step=0.01)
        Pb_measured   = st.number_input(
            "Measured Bubble Point Pb (psia)",
            min_value=100.0, max_value=8000.0,
            value=2463.0, step=10.0)
        mu_w    = st.number_input(
            "Water Viscosity μw (cp)",
            min_value=0.2, max_value=1.5,
            value=0.50, step=0.05)
        Rs      = 600.0
        gamma_g = 0.75

    st.subheader("💧 Saturation Data")
    krw_sor = st.number_input(
        "Water Rel Perm at Sor (krw_sor)",
        min_value=0.1, max_value=0.8,
        value=0.35, step=0.05)
    kro_swc = st.number_input(
        "Oil Rel Perm at Swc (kro_swc)",
        min_value=0.3, max_value=1.0,
        value=0.85, step=0.05)

    st.subheader("⚡ Production Data")
    Qo = st.number_input(
        "Average Production Rate Qo (STB/day)",
        min_value=100.0, max_value=10000.0,
        value=2000.0, step=100.0)

    # Predict Button
    predict_btn = st.button(
        "🔍 PREDICT BREAKTHROUGH",
        type="primary",
        use_container_width=True
    )

# ══════════════════════════════════════════════
# MAIN CALCULATIONS AND RESULTS
# ══════════════════════════════════════════════

if predict_btn:

    # Input validation
    if hp >= h:
        st.error(
            "❌ Perforated interval (hp) must be "
            "less than oil column height (h). "
            "Please check your inputs."
        )
        st.stop()

    # ── Niger Delta Corrections ─────────────────
    rock = apply_niger_delta_corrections(
        kh_mean, kv_matrix, phi_log,
        depth_ft, V_DP, Vsh, SCI
    )
    kh_eff  = rock['kh_eff']
    kv_eff  = rock['kv_eff']
    phi_eff = rock['phi_eff']

    # ── Fluid Properties ────────────────────────
    gamma_o = calculate_oil_specific_gravity(API)

    if pvt_option == "Calculate from correlations":
        mu_od = calculate_dead_oil_viscosity(
            API, T_F)
        Pb    = calculate_bubble_point(
            Rs, gamma_g, T_F, API)
        mu_ob = calculate_saturated_viscosity(
            mu_od, Rs)
        if Pi > Pb:
            mu_o = calculate_undersaturated_viscosity(
                mu_ob, Pi, Pb)
            condition = f"Undersaturated (Pi={Pi} > Pb={Pb})"
        else:
            mu_o = mu_ob
            condition = f"Saturated (Pi={Pi} ≤ Pb={Pb})"
        Bo = calculate_oil_fvf(
            Rs, gamma_g, gamma_o, T_F)
    else:
        mu_o      = mu_o_measured
        Bo        = Bo_measured
        Pb        = Pb_measured
        mu_od     = None
        condition = "Measured PVT values used"

    rho_w = calculate_water_density(
        salinity_ppm, T_F, Pi)
    rho_o = calculate_oil_density(
        API, Bo, Rs, gamma_g)
    delta_rho = round(rho_w - rho_o, 3)

    # ── Mobility Ratio ──────────────────────────
    M, alpha = calculate_mobility_ratio(
        krw_sor, kro_swc, mu_o, mu_w)

    # ── Sobocinski-Cornelius ────────────────────
    Z_SC, tD_SC, tBT_SC, err_SC = \
        sobocinski_cornelius(
            kh_eff, kv_eff, phi_eff,
            h, hp, mu_o, Bo, Qo,
            rho_w, rho_o, M, alpha
        )

    # ── Bournazel-Jeanson ───────────────────────
    Z_BJ, tD_BJ, tBT_BJ, err_BJ = \
        bournazel_jeanson(
            kh_eff, kv_eff, phi_eff,
            h, hp, mu_o, Bo, Qo,
            rho_w, rho_o, M
        )

    # ══════════════════════════════════════════
    # DISPLAY RESULTS
    # ══════════════════════════════════════════

    st.header("📊 Prediction Results")

    # ── Method Results Side by Side ────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Method 1 — Sobocinski-Cornelius")
        st.caption("Bottom Water Drive | Ahmed (2010) Eq 9-21 to 9-23")
        if err_SC:
            st.error(err_SC)
        elif tBT_SC:
            risk_SC = classify_risk(tBT_SC)
            st.markdown(
                f'<div class="{risk_SC["box"]}">'
                f'<h2>{risk_SC["symbol"]} '
                f'{risk_SC["category"]}</h2>'
                f'<h3>Breakthrough in {tBT_SC} days'
                f' ({tBT_SC/365:.1f} years)</h3>'
                f'<p><b>Action:</b> '
                f'{risk_SC["action"]}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.metric("Z (Cone Height)", Z_SC)
            st.metric("(tD)BT", tD_SC)

    with col2:
        st.subheader("Method 2 — Bournazel-Jeanson")
        st.caption("Edge Water Drive | SPE-3628")
        if err_BJ:
            st.error(err_BJ)
        elif tBT_BJ:
            risk_BJ = classify_risk(tBT_BJ)
            st.markdown(
                f'<div class="{risk_BJ["box"]}">'
                f'<h2>{risk_BJ["symbol"]} '
                f'{risk_BJ["category"]}</h2>'
                f'<h3>Breakthrough in {tBT_BJ} days'
                f' ({tBT_BJ/365:.1f} years)</h3>'
                f'<p><b>Action:</b> '
                f'{risk_BJ["action"]}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.metric("Z (Cone Height)", Z_BJ)
            st.metric("(tD)BT", tD_BJ)

    st.divider()

    # ── Intermediate Results ────────────────────
    with st.expander(
            "🔬 View Intermediate Calculations",
            expanded=False):

        col3, col4, col5 = st.columns(3)

        with col3:
            st.subheader("Rock Corrections")
            st.metric(
                "kh effective",
                f"{kh_eff} md",
                delta=f"{kh_eff-kh_mean:.1f} md"
            )
            st.metric(
                "kv effective",
                f"{kv_eff} md",
                delta=f"{kv_eff-kv_matrix:.2f} md"
            )
            st.metric(
                "φ effective",
                f"{phi_eff}",
                delta=f"{phi_eff-phi_log:.4f}"
            )

        with col4:
            st.subheader("Fluid Properties")
            st.metric("Oil Viscosity μo",
                      f"{mu_o} cp")
            st.metric("Oil FVF Bo",
                      f"{Bo} bbl/STB")
            st.metric("Water Density ρw",
                      f"{rho_w} lb/ft³")
            st.metric("Oil Density ρo",
                      f"{rho_o} lb/ft³")
            st.metric("Δρ",
                      f"{delta_rho} lb/ft³")
            if pvt_option == \
               "Calculate from correlations":
                st.metric("Bubble Point Pb",
                          f"{Pb} psia")
                st.info(f"Condition: {condition}")

        with col5:
            st.subheader("Mobility")
            st.metric("Mobility Ratio M", M)
            st.metric("Alpha (α)", alpha)
            if M <= 1:
                st.success(
                    "✅ Favorable displacement"
                )
            elif M <= 5:
                st.warning(
                    "⚠️ Unfavorable displacement"
                )
            else:
                st.error(
                    "❌ Highly unfavorable"
                )

    st.divider()

    # ── SCI Sensitivity Analysis ────────────────
    st.subheader(
        "📈 Shale Continuity Index (SCI) "
        "Sensitivity Analysis"
    )
    st.caption(
        "★ Novel contribution of this study — "
        "Shows how shale continuity in the "
        "Agbada Formation affects breakthrough time"
    )

    sci_df = run_sci_sensitivity(
        kh_eff, kv_matrix, phi_eff,
        h, hp, mu_o, Bo, Qo,
        rho_w, rho_o, M, alpha, Vsh
    )

    if not sci_df.empty:
        col6, col7 = st.columns([2, 1])

        with col6:
            # SCI Chart
            fig = go.Figure()

            # Color zones
            fig.add_hrect(
                y0=0, y1=365,
                fillcolor="red",
                opacity=0.1,
                annotation_text="HIGH RISK",
                annotation_position="left"
            )
            fig.add_hrect(
                y0=365, y1=730,
                fillcolor="orange",
                opacity=0.1,
                annotation_text="MEDIUM RISK",
                annotation_position="left"
            )
            fig.add_hrect(
                y0=730,
                y1=sci_df['tBT_days'].max()*1.2,
                fillcolor="green",
                opacity=0.1,
                annotation_text="LOW RISK",
                annotation_position="left"
            )

            fig.add_trace(go.Scatter(
                x=sci_df['SCI'],
                y=sci_df['tBT_days'],
                mode='lines+markers',
                name='Breakthrough Time',
                line=dict(color='#1f77b4',
                          width=3),
                marker=dict(size=10,
                            color='#1f77b4')
            ))

            # Mark current SCI
            current_row = sci_df[
                sci_df['SCI'] ==
                min(sci_df['SCI'],
                    key=lambda x: abs(x-SCI))
            ]
            if not current_row.empty:
                fig.add_trace(go.Scatter(
                    x=current_row['SCI'],
                    y=current_row['tBT_days'],
                    mode='markers',
                    name=f'Your SCI = {SCI}',
                    marker=dict(size=15,
                                color='red',
                                symbol='star')
                ))

            fig.update_layout(
                title='Effect of Shale Continuity '
                      'Index on Breakthrough Time',
                xaxis_title='Shale Continuity '
                             'Index (SCI)',
                yaxis_title='Breakthrough Time '
                             '(days)',
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig,
                           use_container_width=True)

        with col7:
            st.dataframe(
                sci_df[[
                    'SCI', 'tBT_days',
                    'tBT_years', 'Risk'
                ]].rename(columns={
                    'tBT_days':  'BT (days)',
                    'tBT_years': 'BT (years)',
                }),
                hide_index=True,
                use_container_width=True
            )

            # Key insight
            bt_range = (sci_df['tBT_days'].max() -
                        sci_df['tBT_days'].min())
            st.info(
                f"📊 SCI Impact:\n"
                f"SCI 0.0 → "
                f"{sci_df['tBT_days'].min():.0f} days\n"
                f"SCI 1.0 → "
                f"{sci_df['tBT_days'].max():.0f} days\n"
                f"Range: {bt_range:.0f} days "
                f"({bt_range/365:.1f} years)"
            )

    st.divider()

    # ── Comparison Chart ────────────────────────
    if tBT_SC and tBT_BJ:
        st.subheader("📊 Method Comparison")

        fig2 = go.Figure()

        methods = ['Sobocinski-Cornelius\n'
                   '(Bottom Water)',
                   'Bournazel-Jeanson\n'
                   '(Edge Water)']
        values  = [tBT_SC, tBT_BJ]
        colors  = []

        for v in values:
            if v <= 365:
                colors.append('red')
            elif v <= 730:
                colors.append('orange')
            else:
                colors.append('green')

        fig2.add_trace(go.Bar(
            x=methods,
            y=values,
            marker_color=colors,
            text=[f'{v:.0f} days\n'
                  f'({v/365:.1f} yrs)'
                  for v in values],
            textposition='outside'
        ))

        fig2.add_hline(
            y=365,
            line_dash="dash",
            line_color="red",
            annotation_text="365 days — "
                           "High/Medium boundary"
        )
        fig2.add_hline(
            y=730,
            line_dash="dash",
            line_color="orange",
            annotation_text="730 days — "
                           "Medium/Low boundary"
        )

        fig2.update_layout(
            title='Breakthrough Time — '
                  'Method Comparison',
            yaxis_title='Breakthrough Time (days)',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig2,
                       use_container_width=True)

    st.divider()

    # ── Project Information ─────────────────────
    with st.expander("ℹ️ About WaterWatch"):
        st.markdown("""
        ### WaterWatch — Enhanced Sobocinski-Cornelius Framework

        **Project:** Final Year Project — University of Benin
        **Department:** Petroleum Engineering

        ### Methods Used
        - **Primary:** Sobocinski-Cornelius (1965) for bottom water drive
          *(Ahmed, 2010, Equations 9-21 to 9-24)*
        - **Comparison:** Bournazel-Jeanson (1971) for edge water drive
          *(SPE-3628)*

        ### Niger Delta Enhancements
        | Enhancement | Formula | Reference |
        |---|---|---|
        | kh heterogeneity | kh_eff = kh × (1-V_DP) | Tuttle et al (1999) |
        | kv shale barrier | kv_eff = kv × (1-Vsh×SCI) | Short & Stauble (1967) |
        | ★ Novel SCI | SCI = 1 - NTG | This study |
        | φ compaction | φ_eff = φ × exp(-0.000025×depth) | Athy (1930) |
        | μo correlation | Beggs-Robinson | Ahmed (2010) Eq 2-121 |
        | Bo correlation | Standing (1981) | Ahmed (2010) Eq 2-85 |
        | Bubble point | Standing (1947) | Ahmed (2010) Eq 2-72 |
        | ρw salinity | Salinity correction | Tuttle et al (1999) |

        ### Risk Classification
        - 🔴 **HIGH RISK:** Breakthrough ≤ 365 days
        - 🟡 **MEDIUM RISK:** Breakthrough 366-730 days
        - 🟢 **LOW RISK:** Breakthrough > 730 days

        ### Key References
        - Ahmed, T. (2010) *Reservoir Engineering Handbook*, 4th Ed.
        - Sobocinski & Cornelius (1965) SPE-894
        - Bournazel & Jeanson (1971) SPE-3628
        - Tuttle et al. (1999) USGS OFR 99-50-H
        """)

# ── Default message ─────────────────────────────
else:
    st.info(
        "👈 Enter your reservoir parameters "
        "in the sidebar and click "
        "**PREDICT BREAKTHROUGH** to run "
        "the analysis."
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        ### 🪨 Rock Properties
        Permeability, porosity, oil column,
        perforation depth, reservoir depth
        """)
    with col_b:
        st.markdown("""
        ### 🧪 Fluid Properties
        API gravity, temperature, GOR,
        pressure, salinity
        """)
    with col_c:
        st.markdown("""
        ### ⚡ Production Data
        Average production rate from
        first 30 days of production
        """)
