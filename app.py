# ══════════════════════════════════════════════
# WATERWATCH
# Enhanced Sobocinski-Cornelius Framework
# for Niger Delta Aquifer Water Breakthrough
# Prediction
# University of Benin — Final Year Project
# Department of Petroleum Engineering
# ══════════════════════════════════════════════

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ── Page Configuration ──────────────────────
st.set_page_config(
    page_title="WaterWatch | UNIBEN",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Professional Styling ────────────────────
st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #f8f9fa;
}
/* Header */
.main-header {
    background: linear-gradient(
        135deg, #1a3a5c 0%, #2980b9 100%);
    color: white;
    padding: 30px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 20px;
}
.main-header h1 {
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0;
    color: white;
}
.main-header p {
    font-size: 1rem;
    margin: 5px 0 0 0;
    color: #d6eaf8;
}
/* Result boxes */
.box-high {
    background: #c0392b;
    color: white;
    padding: 20px 25px;
    border-radius: 10px;
    border-left: 8px solid #922b21;
    margin: 10px 0;
}
.box-medium {
    background: #d35400;
    color: white;
    padding: 20px 25px;
    border-radius: 10px;
    border-left: 8px solid #a04000;
    margin: 10px 0;
}
.box-low {
    background: #1e8449;
    color: white;
    padding: 20px 25px;
    border-radius: 10px;
    border-left: 8px solid #145a32;
    margin: 10px 0;
}
.box-high h2, .box-medium h2,
.box-low h2 {
    color: white !important;
    font-size: 1.8rem;
    margin: 0 0 8px 0;
}
.box-high h3, .box-medium h3,
.box-low h3 {
    color: white !important;
    font-size: 1.3rem;
    margin: 0 0 8px 0;
}
.box-high p, .box-medium p,
.box-low p {
    color: #f2f3f4 !important;
    margin: 0;
    font-size: 0.95rem;
}
/* Section headers */
.section-title {
    background: #1a3a5c;
    color: white;
    padding: 10px 15px;
    border-radius: 6px;
    font-size: 1.1rem;
    font-weight: 600;
    margin: 15px 0 10px 0;
}
/* Info card */
.info-card {
    background: white;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #d5d8dc;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# CALCULATION FUNCTIONS
# ══════════════════════════════════════════

def oil_specific_gravity(API):
    return round(141.5 / (API + 131.5), 4)

def dead_oil_viscosity(API, T_F):
    """
    Beal (1946) / Standing (1981)
    Ahmed (2010) Eq 2-117
    """
    T_R = T_F + 460
    a   = 10**(0.43 + 8.33/API)
    mu  = (0.32 + 1.8e7/API**4.53) * \
          (360/(T_R-260))**a
    return round(mu, 4)

def bubble_point_pressure(Rs, gg, T_F, API):
    """
    Standing (1947)
    Ahmed (2010) Eq 2-72
    """
    Pb = 18.2*((Rs/gg)**0.83 *
               10**(0.00091*T_F -
                    0.0125*API) - 1.4)
    return round(abs(Pb), 1)

def saturated_viscosity(mu_od, Rs):
    """
    Beggs & Robinson (1975)
    Ahmed (2010) Eq 2-121
    """
    a = 10.715*(Rs+100)**(-0.515)
    b = 5.44*(Rs+150)**(-0.338)
    return round(a*mu_od**b, 4)

def undersaturated_viscosity(mu_ob, Pi, Pb):
    """Ahmed (2010) Eq 2-123"""
    a = -3.9e-5*Pi - 5
    m = 2.6*Pi**1.187*10**a
    return round(mu_ob*(Pi/Pb)**m, 4)

def oil_fvf(Rs, gg, go, T_F):
    """
    Standing (1981)
    Ahmed (2010) Eq 2-85
    """
    F  = Rs*(gg/go)**0.5 + 1.25*(T_F)
    Bo = 0.9759 + 0.000120*F**1.2
    return round(Bo, 4)

def water_density(sal_ppm, T_F, P):
    """
    Niger Delta salinity correction
    Tuttle et al (1999)
    """
    rw = (62.4 + sal_ppm/10000*0.5 -
          0.003*(T_F-60) + 0.0000145*P)
    return round(rw, 3)

def oil_density(API, Bo, Rs, gg):
    """Ahmed (2010) Chapter 2"""
    go      = 141.5/(API+131.5)
    rho_s   = go*62.4
    rho_o   = (rho_s + 0.01357*Rs*gg)/Bo
    return round(rho_o, 3)

def mobility_ratio(krw, kro, mu_o, mu_w):
    """Ahmed (2010) Eq 9-24"""
    M     = (krw/kro)*(mu_o/mu_w)
    alpha = 0.5 if M <= 1 else 0.6
    return round(M, 4), alpha

def sobocinski(kh, kv, phi, h, hp,
               mu_o, Bo, Qo,
               rho_w, rho_o, M, alpha):
    """
    Sobocinski & Cornelius (1965)
    Ahmed (2010) Eq 9-21 to 9-23
    """
    dr = rho_w - rho_o
    if dr <= 0:
        return None, None, None, \
               "Water must be denser than oil"
    Z = (0.492e-4*dr*kh*h*(h-hp)) / \
        (mu_o*Bo*Qo)
    if Z <= 0:
        return None, None, None, \
               "Z≤0: Check h > hp"
    if Z >= 3.5:
        return None, None, None, \
               "Z≥3.5: Outside valid range"
    tD  = (4*Z + 1.75*Z**2 - 0.75*Z**3) / \
          (7 - 2*Z)
    tBT = (20325*mu_o*h*phi*tD) / \
          (dr*kv*(1+M**alpha))
    return round(Z,4), round(tD,4), \
           round(tBT,1), None

def risk_level(tBT):
    if tBT <= 365:
        return {
            'cat':    'HIGH RISK',
            'icon':   '🔴',
            'box':    'box-high',
            'action': 'Begin water handling '
                     'planning immediately. '
                     'Consider rate reduction.'
        }
    elif tBT <= 730:
        return {
            'cat':    'MEDIUM RISK',
            'icon':   '🟡',
            'box':    'box-medium',
            'action': 'Plan water handling '
                     'within 6 months. '
                     'Monitor production closely.'
        }
    else:
        return {
            'cat':    'LOW RISK',
            'icon':   '🟢',
            'box':    'box-low',
            'action': 'Monitor quarterly. '
                     'No immediate action required.'
        }

# ── Parameter Calculators ──────────────────

def calc_vsh_from_gr(GR_log, GR_min, GR_max):
    """
    Larionov (1969) Vsh from Gamma Ray
    Standard petrophysics formula
    """
    if GR_max == GR_min:
        return 0.0
    IGR = (GR_log - GR_min) / \
          (GR_max - GR_min)
    IGR = max(0, min(1, IGR))
    Vsh = 0.083*(2**(3.7*IGR) - 1)
    return round(min(Vsh, 1.0), 4)

def calc_vdp_from_core(perm_list):
    """
    Dykstra & Parsons (1950)
    Heterogeneity coefficient
    """
    if len(perm_list) < 2:
        return 0.0
    k = sorted(perm_list, reverse=True)
    n = len(k)
    k50   = np.percentile(k, 50)
    k84   = np.percentile(k, 15.9)
    if k50 == 0:
        return 0.0
    V_DP = (k50 - k84) / k50
    return round(max(0, min(V_DP, 0.99)), 4)

def calc_sci_from_ntg(NTG):
    """
    Novel SCI — this study
    Short & Stauble (1967)
    Doust & Omatsola (1990)
    """
    SCI = 1 - NTG
    return round(max(0, min(SCI, 1.0)), 4)

def sci_sensitivity(kh_eff, kv_mat,
        phi_eff, h, hp, mu_o, Bo, Qo,
        rw, ro, M, alpha, Vsh):
    """SCI sensitivity analysis"""
    rows = []
    for s in [0.0,0.2,0.4,0.6,0.8,1.0]:
        kv_s = kv_mat*(1 - Vsh*s)
        _,_,t,e = sobocinski(
            kh_eff, kv_s, phi_eff,
            h, hp, mu_o, Bo, Qo,
            rw, ro, M, alpha)
        if t and not e:
            r = risk_level(t)
            rows.append({
                'SCI':       s,
                'kv (md)':   round(kv_s,2),
                'BT (days)': t,
                'BT (yrs)':  round(t/365,2),
                'Risk':      r['cat']
            })
    return pd.DataFrame(rows)

# ══════════════════════════════════════════
# APP HEADER
# ══════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>💧 WaterWatch</h1>
    <p>Enhanced Sobocinski-Cornelius
    Framework for Niger Delta Aquifer
    Water Breakthrough Prediction</p>
    <p>University of Benin |
    Department of Petroleum Engineering |
    Final Year Project</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════
# TABS
# ══════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "🔍 Breakthrough Prediction",
    "🧮 Parameter Calculator",
    "ℹ️ About"
])

# ══════════════════════════════════════════
# TAB 1 — MAIN PREDICTION
# ══════════════════════════════════════════

with tab1:

    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/"
            "wikipedia/en/5/5e/"
            "University_of_Benin_logo.png",
            width=80
        )
        st.markdown("### ⚙️ Input Parameters")
        st.caption(
            "Enter reservoir data below"
        )

        pvt_mode = st.radio(
            "PVT Data Source",
            ["Use correlations (no PVT report)",
             "Enter measured PVT values"],
        )

        st.markdown("**🪨 Rock Properties**")
        kh_mean   = st.number_input(
            "kh — Horiz. Permeability (md)",
            10.0, 5000.0, 1800.0, 10.0)
        kv_matrix = st.number_input(
            "kv — Vert. Permeability (md)",
            1.0, 1000.0, 270.0, 5.0)
        phi_log   = st.number_input(
            "φ — Log Porosity (fraction)",
            0.05, 0.45, 0.28, 0.01)
        depth_ft  = st.number_input(
            "Reservoir Depth (ft)",
            1000.0, 15000.0, 8500.0, 100.0)
        h  = st.number_input(
            "h — Oil Column Height (ft)",
            5.0, 300.0, 80.0, 1.0)
        hp = st.number_input(
            "hp — Perforated Interval (ft)",
            1.0, 200.0, 25.0, 1.0)

        st.markdown(
            "**🌍 Niger Delta Corrections**"
        )
        st.caption(
            "Use Parameter Calculator tab "
            "if unsure of these values"
        )
        V_DP = st.slider(
            "V_DP — Dykstra-Parsons",
            0.0, 0.9, 0.45, 0.05,
            help="Calculate in Parameter "
                 "Calculator tab")
        Vsh  = st.slider(
            "Vsh — Shale Volume",
            0.0, 0.5, 0.15, 0.01,
            help="Calculate from GR log "
                 "in Parameter Calculator")
        SCI  = st.slider(
            "SCI — Shale Continuity ★",
            0.0, 1.0, 0.50, 0.05,
            help="Novel parameter. "
                 "Estimate from NTG in "
                 "Parameter Calculator")

        st.markdown("**🧪 Fluid Properties**")
        API = st.number_input(
            "API Gravity (°)",
            15.0, 55.0, 35.0, 0.5)
        T_F = st.number_input(
            "Temperature (°F)",
            100.0, 300.0, 180.0, 5.0)
        Pi  = st.number_input(
            "Initial Pressure (psia)",
            500.0, 10000.0, 4200.0, 50.0)
        sal = st.number_input(
            "Water Salinity (ppm)",
            1000.0, 150000.0, 35000.0,
            1000.0)
        mu_w = st.number_input(
            "Water Viscosity μw (cp)",
            0.2, 1.5, 0.50, 0.05)

        if pvt_mode == \
           "Use correlations (no PVT report)":
            Rs = st.number_input(
                "Rs — Solution GOR (scf/STB)",
                50.0, 2000.0, 600.0, 10.0)
            gg = st.number_input(
                "γg — Gas Specific Gravity",
                0.5, 1.2, 0.75, 0.01)
            mu_o_in = None
            Bo_in   = None
            Pb_in   = None
        else:
            mu_o_in = st.number_input(
                "Measured μo (cp)",
                0.1, 100.0, 0.6, 0.1)
            Bo_in   = st.number_input(
                "Measured Bo (bbl/STB)",
                1.0, 3.0, 1.34, 0.01)
            Pb_in   = st.number_input(
                "Measured Pb (psia)",
                100.0, 8000.0, 2463.0, 10.0)
            Rs = 600.0
            gg = 0.75

        st.markdown("**💧 Saturation**")
        krw = st.number_input(
            "krw at Sor",
            0.1, 0.8, 0.35, 0.05)
        kro = st.number_input(
            "kro at Swc",
            0.3, 1.0, 0.85, 0.05)

        st.markdown("**⚡ Production**")
        Qo = st.number_input(
            "Qo — Production Rate (STB/day)",
            100.0, 10000.0, 2000.0, 100.0)

        run_btn = st.button(
            "🔍 PREDICT BREAKTHROUGH",
            type="primary",
            use_container_width=True
        )

    # ── Run Prediction ───────────────────────
    if run_btn:

        if hp >= h:
            st.error(
                "❌ hp must be less than h. "
                "Check perforation depth."
            )
            st.stop()

        go = oil_specific_gravity(API)

        # ── FLUID PROPERTIES ─────────────────
        if pvt_mode == \
           "Use correlations (no PVT report)":
            mu_od = dead_oil_viscosity(API, T_F)
            Pb    = bubble_point_pressure(
                Rs, gg, T_F, API)
            mu_ob = saturated_viscosity(
                mu_od, Rs)
            if Pi > Pb:
                mu_o = undersaturated_viscosity(
                    mu_ob, Pi, Pb)
                cond = "Undersaturated"
            else:
                mu_o = mu_ob
                cond = "Saturated"
            Bo = oil_fvf(Rs, gg, go, T_F)
        else:
            mu_o  = mu_o_in
            Bo    = Bo_in
            Pb    = Pb_in
            mu_od = None
            cond  = "Measured PVT"

        rw = water_density(sal, T_F, Pi)
        ro = oil_density(API, Bo, Rs, gg)
        M, alpha = mobility_ratio(
            krw, kro, mu_o, mu_w)

        # ════════════════════════════════════
        # METHOD 1 — ORIGINAL SOBOCINSKI
        # No Niger Delta corrections
        # Raw inputs used directly
        # ════════════════════════════════════
        mu_o_orig = dead_oil_viscosity(API, T_F)
        Bo_orig   = oil_fvf(Rs, gg, go, T_F)
        rw_orig   = 62.4  # fresh water
        ro_orig   = oil_density(
            API, Bo_orig, Rs, gg)
        M_orig, alpha_orig = mobility_ratio(
            krw, kro, mu_o_orig, mu_w)

        Z1, tD1, tBT1, err1 = sobocinski(
            kh_mean, kv_matrix, phi_log,
            h, hp,
            mu_o_orig, Bo_orig, Qo,
            rw_orig, ro_orig,
            M_orig, alpha_orig
        )

        # ════════════════════════════════════
        # METHOD 2 — ENHANCED SOBOCINSKI ★
        # With Niger Delta corrections
        # ════════════════════════════════════
        kh_eff  = kh_mean * (1 - V_DP)
        kv_eff  = kv_matrix * (1 - Vsh*SCI)
        phi_eff = phi_log * np.exp(
            -0.000025 * depth_ft)

        Z2, tD2, tBT2, err2 = sobocinski(
            kh_eff, kv_eff, phi_eff,
            h, hp,
            mu_o, Bo, Qo,
            rw, ro, M, alpha
        )

        # ── RESULTS DISPLAY ─────────────────
        st.markdown(
            '<div class="section-title">'
            '📊 Prediction Results'
            '</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                "#### Method 1 — "
                "Original Sobocinski-Cornelius"
            )
            st.caption(
                "No Niger Delta corrections | "
                "Raw inputs | Baseline"
            )
            if err1:
                st.error(f"❌ {err1}")
            elif tBT1:
                r1 = risk_level(tBT1)
                st.markdown(
                    f'<div class="{r1["box"]}">'
                    f'<h2>{r1["icon"]} '
                    f'{r1["cat"]}</h2>'
                    f'<h3>Breakthrough: '
                    f'{tBT1} days '
                    f'({tBT1/365:.1f} years)'
                    f'</h3>'
                    f'<p>⚡ {r1["action"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.metric("Z", Z1)
                st.metric("(tD)BT", tD1)
                st.metric("μo used",
                          f"{mu_o_orig} cp",
                          help="Dead oil — "
                               "no correction")
                st.metric("ρw used",
                          "62.4 lb/ft³",
                          help="Fresh water — "
                               "no correction")

        with col2:
            st.markdown(
                "#### Method 2 — "
                "Enhanced Sobocinski ★"
            )
            st.caption(
                "With Niger Delta corrections | "
                "This study"
            )
            if err2:
                st.error(f"❌ {err2}")
            elif tBT2:
                r2 = risk_level(tBT2)
                st.markdown(
                    f'<div class="{r2["box"]}">'
                    f'<h2>{r2["icon"]} '
                    f'{r2["cat"]}</h2>'
                    f'<h3>Breakthrough: '
                    f'{tBT2} days '
                    f'({tBT2/365:.1f} years)'
                    f'</h3>'
                    f'<p>⚡ {r2["action"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.metric("Z", Z2)
                st.metric("(tD)BT", tD2)
                st.metric("μo used",
                          f"{mu_o} cp",
                          help="Reservoir "
                               "condition — "
                               "corrected")
                st.metric("ρw used",
                          f"{rw} lb/ft³",
                          help="Salinity "
                               "corrected")

        # ── Improvement Summary ──────────────
        if tBT1 and tBT2 and \
           not err1 and not err2:
            st.divider()
            st.markdown(
                '<div class="section-title">'
                '📈 Enhancement Impact'
                '</div>',
                unsafe_allow_html=True
            )

            diff     = tBT1 - tBT2
            pct_diff = abs(diff/tBT1*100)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Original tBT",
                f"{tBT1} days"
            )
            c2.metric(
                "Enhanced tBT",
                f"{tBT2} days",
                delta=f"{-diff:.1f} days",
                delta_color="inverse"
            )
            c3.metric(
                "Difference",
                f"{abs(diff):.1f} days"
            )
            c4.metric(
                "% Change",
                f"{pct_diff:.1f}%"
            )

            if diff > 0:
                st.info(
                    f"📊 The enhanced method "
                    f"predicts breakthrough "
                    f"**{abs(diff):.0f} days "
                    f"earlier** than the "
                    f"original method. "
                    f"This demonstrates that "
                    f"original Sobocinski "
                    f"overestimates breakthrough "
                    f"time for Niger Delta "
                    f"conditions by "
                    f"**{pct_diff:.1f}%**."
                )
            else:
                st.info(
                    f"📊 The enhanced method "
                    f"predicts breakthrough "
                    f"**{abs(diff):.0f} days "
                    f"later** than original. "
                    f"Niger Delta corrections "
                    f"show stronger shale "
                    f"barriers delay coning."
                )

            # Comparison chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Original Sobocinski",
                   "Enhanced Sobocinski ★"],
                y=[tBT1, tBT2],
                marker_color=[
                    '#e74c3c', '#2980b9'],
                text=[
                    f'{tBT1:.0f} days\n'
                    f'({tBT1/365:.1f} yrs)',
                    f'{tBT2:.0f} days\n'
                    f'({tBT2/365:.1f} yrs)'
                ],
                textposition='outside',
                textfont=dict(size=13)
            ))
            fig.add_hline(
                y=365,
                line_dash="dash",
                line_color="red",
                annotation_text="365 days "
                               "(High Risk limit)"
            )
            fig.add_hline(
                y=730,
                line_dash="dash",
                line_color="orange",
                annotation_text="730 days "
                               "(Medium Risk limit)"
            )
            fig.update_layout(
                title="Original vs Enhanced "
                      "Sobocinski-Cornelius — "
                      "Breakthrough Time "
                      "Comparison",
                yaxis_title=
                    "Breakthrough Time (days)",
                height=420,
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color='black')
            )
            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.divider()

        # ── SCI Sensitivity ──────────────────
        st.markdown(
            '<div class="section-title">'
            '🔬 SCI Sensitivity Analysis '
            '(Novel Contribution ★)'
            '</div>',
            unsafe_allow_html=True
        )
        st.caption(
            "Shows how Agbada Formation "
            "shale continuity affects "
            "breakthrough time prediction"
        )

        sci_df = sci_sensitivity(
            kh_eff, kv_matrix, phi_eff,
            h, hp, mu_o, Bo, Qo,
            rw, ro, M, alpha, Vsh
        )

        if not sci_df.empty:
            c_a, c_b = st.columns([2,1])
            with c_a:
                fig2 = go.Figure()
                fig2.add_hrect(
                    y0=0, y1=365,
                    fillcolor="red",
                    opacity=0.08,
                    annotation_text="HIGH RISK",
                    annotation_position="left"
                )
                fig2.add_hrect(
                    y0=365, y1=730,
                    fillcolor="orange",
                    opacity=0.08,
                    annotation_text="MEDIUM",
                    annotation_position="left"
                )
                fig2.add_hrect(
                    y0=730,
                    y1=sci_df['BT (days)'].max()
                        *1.3,
                    fillcolor="green",
                    opacity=0.08,
                    annotation_text="LOW RISK",
                    annotation_position="left"
                )
                fig2.add_trace(go.Scatter(
                    x=sci_df['SCI'],
                    y=sci_df['BT (days)'],
                    mode='lines+markers',
                    line=dict(
                        color='#1a3a5c',
                        width=3),
                    marker=dict(size=10)
                ))
                fig2.update_layout(
                    title="Effect of Shale "
                          "Continuity Index "
                          "(SCI) on Breakthrough"
                          " Time",
                    xaxis_title=
                        "SCI Value (0=discontinuous"
                        " to 1=continuous)",
                    yaxis_title=
                        "Breakthrough Time (days)",
                    height=380,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='black')
                )
                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )
            with c_b:
                st.dataframe(
                    sci_df,
                    hide_index=True,
                    use_container_width=True
                )
                bt_range = (
                    sci_df['BT (days)'].max() -
                    sci_df['BT (days)'].min()
                )
                st.success(
                    f"SCI impact on tBT:\n"
                    f"Range = {bt_range:.0f} days"
                    f" ({bt_range/365:.1f} years)"
                )

        # ── Intermediate Values ──────────────
        with st.expander(
                "🔬 Intermediate Calculations"):
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown(
                    "**Niger Delta Corrections**"
                )
                st.write(
                    f"kh: {kh_mean} → "
                    f"{kh_eff} md"
                )
                st.write(
                    f"kv: {kv_matrix} → "
                    f"{kv_eff} md"
                )
                st.write(
                    f"φ: {phi_log} → "
                    f"{phi_eff}"
                )
                st.write(
                    f"μo: {mu_o_orig} → "
                    f"{mu_o} cp"
                )
                st.write(
                    f"ρw: 62.4 → {rw} lb/ft³"
                )
            with cc2:
                st.markdown(
                    "**Mobility & Densities**"
                )
                st.write(f"M (original): {M_orig}")
                st.write(f"M (enhanced): {M}")
                st.write(f"α: {alpha}")
                st.write(
                    f"ρo: {ro} lb/ft³"
                )
                st.write(
                    f"Δρ: "
                    f"{round(rw-ro,3)} lb/ft³"
                )
                if pvt_mode != \
                   "Enter measured PVT values":
                    st.write(f"Pb: {Pb} psia")
                    st.write(
                        f"Condition: {cond}"
                    )

    else:
        st.info(
            "👈 Enter parameters in the "
            "sidebar and click "
            "**PREDICT BREAKTHROUGH**"
        )

# ══════════════════════════════════════════
# TAB 2 — PARAMETER CALCULATOR
# ══════════════════════════════════════════

with tab2:
    st.markdown(
        '<div class="section-title">'
        '🧮 Parameter Calculator'
        '</div>',
        unsafe_allow_html=True
    )
    st.write(
        "Use this section to calculate "
        "Niger Delta correction parameters "
        "from raw field data before running "
        "the prediction."
    )

    pc1, pc2, pc3 = st.columns(3)

    # ── Vsh Calculator ───────────────────────
    with pc1:
        st.markdown("#### Vsh from Gamma Ray")
        st.caption(
            "Larionov (1969) correlation"
        )
        GR_log = st.number_input(
            "GR at depth (API units)",
            0.0, 300.0, 75.0, 1.0,
            key="gr_log")
        GR_min = st.number_input(
            "GR_min (cleanest sand)",
            0.0, 150.0, 20.0, 1.0,
            key="gr_min")
        GR_max = st.number_input(
            "GR_max (purest shale)",
            50.0, 300.0, 150.0, 1.0,
            key="gr_max")
        if st.button("Calculate Vsh",
                     key="vsh_btn"):
            vsh_result = calc_vsh_from_gr(
                GR_log, GR_min, GR_max)
            st.success(f"**Vsh = {vsh_result}**")
            st.info(
                f"Use {vsh_result} as your "
                f"Vsh input in the prediction"
            )
            IGR = (GR_log-GR_min) / \
                  (GR_max-GR_min)
            st.write(f"IGR = {round(IGR,4)}")

    # ── Dykstra-Parsons Calculator ───────────
    with pc2:
        st.markdown(
            "#### Dykstra-Parsons V_DP"
        )
        st.caption(
            "Dykstra & Parsons (1950)"
        )
        st.write(
            "Enter permeability values "
            "from core analysis "
            "(one per line):"
        )
        k_input = st.text_area(
            "Permeability values (md)",
            value="3000\n1500\n800\n400\n200\n100",
            height=150,
            key="k_vals"
        )
        if st.button("Calculate V_DP",
                     key="vdp_btn"):
            try:
                k_list = [
                    float(x.strip())
                    for x in k_input.split('\n')
                    if x.strip()
                ]
                if len(k_list) < 2:
                    st.error(
                        "Need at least 2 values"
                    )
                else:
                    vdp = calc_vdp_from_core(
                        k_list)
                    st.success(
                        f"**V_DP = {vdp}**"
                    )
                    st.info(
                        f"Use {vdp} as your "
                        f"V_DP in prediction"
                    )
                    k50  = round(
                        np.percentile(k_list,50),1)
                    k84  = round(
                        np.percentile(k_list,15.9),1)
                    st.write(f"k50 = {k50} md")
                    st.write(f"k84.1 = {k84} md")
            except:
                st.error(
                    "Please enter valid "
                    "numbers only"
                )

    # ── SCI Calculator ───────────────────────
    with pc3:
        st.markdown("#### SCI from NTG")
        st.caption(
            "Novel parameter — this study\n"
            "Short & Stauble (1967)"
        )
        st.write(
            "NTG = Net sand thickness / "
            "Gross reservoir thickness"
        )
        net_sand = st.number_input(
            "Net sand thickness (ft)",
            1.0, 500.0, 68.0, 1.0)
        gross    = st.number_input(
            "Gross thickness (ft)",
            1.0, 500.0, 80.0, 1.0)
        if st.button("Calculate SCI",
                     key="sci_btn"):
            if gross <= 0:
                st.error("Gross must be > 0")
            elif net_sand > gross:
                st.error(
                    "Net cannot exceed gross"
                )
            else:
                NTG = net_sand/gross
                sci = calc_sci_from_ntg(NTG)
                st.success(
                    f"**NTG = {round(NTG,3)}**\n"
                    f"**SCI = {sci}**"
                )
                st.info(
                    f"Use {sci} as your "
                    f"SCI in prediction"
                )
                if sci < 0.3:
                    st.write(
                        "→ Discontinuous shales"
                    )
                elif sci < 0.6:
                    st.write(
                        "→ Moderate continuity"
                    )
                else:
                    st.write(
                        "→ Continuous shales"
                    )

# ══════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════

with tab3:
    st.markdown("""
    ## About WaterWatch

    WaterWatch is an enhanced analytical
    tool for predicting aquifer water
    breakthrough time in Niger Delta
    vertical oil wells.

    ---

    ### Core Method
    **Sobocinski-Cornelius (1965)**
    as presented in Ahmed (2010)
    Reservoir Engineering Handbook
    Equations 9-21 to 9-23

    ---

    ### Niger Delta Enhancements

    | Enhancement | Formula | Reference |
    |---|---|---|
    | kh correction | kh_eff = kh×(1-V_DP) | Tuttle et al (1999) |
    | kv correction | kv_eff = kv×(1-Vsh×SCI) | Short & Stauble (1967) |
    | ★ SCI (novel) | SCI = 1-NTG | This study |
    | φ correction | φ_eff = φ×exp(-0.000025×d) | Athy (1930) |
    | μo (dead oil) | Beal/Standing | Ahmed Eq 2-117 |
    | μo (saturated) | Beggs-Robinson | Ahmed Eq 2-121 |
    | μo (unsat.) | Ahmed Eq 2-123 | Ahmed (2010) |
    | Bo | Standing (1981) | Ahmed Eq 2-85 |
    | Pb check | Standing (1947) | Ahmed Eq 2-72 |
    | ρw salinity | Salinity correction | Tuttle et al (1999) |

    ---

    ### Risk Classification
    | Category | Breakthrough Time |
    |---|---|
    | 🔴 HIGH RISK | ≤ 365 days |
    | 🟡 MEDIUM RISK | 366 - 730 days |
    | 🟢 LOW RISK | > 730 days |

    ---

    ### Key References
    - Ahmed, T. (2010) *Reservoir Engineering
      Handbook*, 4th Ed.
    - Sobocinski & Cornelius (1965) SPE-894
    - Bournazel & Jeanson (1971) SPE-3628
    - Tuttle et al. (1999) USGS OFR 99-50-H
    - Short & Stauble (1967) AAPG Bulletin
    - Doust & Omatsola (1990) AAPG Memoir 48
    - Athy (1930) AAPG Bulletin
    - Dykstra & Parsons (1950) API
    - Larionov (1969) — Vsh correlation

    ---

    ### Scope
    - Vertical wells only
    - Natural aquifer drive
    - Bottom and edge water drive
    - Niger Delta Agbada Formation
    - Early production period (Day 1-30)
    """)
