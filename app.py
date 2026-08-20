# ═══════════════════════════════════════════════════════════════════
# WATERWATCH
# Aquifer Water Breakthrough Prediction Tool for Niger Delta
# Enhanced Sobocinski-Cornelius Framework with Proposed SCI
#
# University of Benin
# Department of Petroleum Engineering
# Final Year Project
# ═══════════════════════════════════════════════════════════════════

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io
import warnings
warnings.filterwarnings('ignore')

# ─── Page Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="WaterWatch | UNIBEN",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Professional Dark Theme Styling ─────────────────────────────
st.markdown("""
<style>
    .stApp {
        background-color: #0e1621;
    }
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
        font-size: 2.6rem;
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
    .box-high, .box-medium, .box-low {
        color: white;
        padding: 20px 25px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .box-high {
        background: linear-gradient(135deg,
                    #c0392b, #922b21);
        border-left: 8px solid #641e16;
    }
    .box-medium {
        background: linear-gradient(135deg,
                    #d35400, #a04000);
        border-left: 8px solid #6e2c00;
    }
    .box-low {
        background: linear-gradient(135deg,
                    #1e8449, #145a32);
        border-left: 8px solid #0b3d1a;
    }
    .box-high h2, .box-medium h2,
    .box-low h2 {
        color: white !important;
        font-size: 1.9rem;
        margin: 0 0 8px 0;
    }
    .box-high h3, .box-medium h3,
    .box-low h3 {
        color: white !important;
        font-size: 1.4rem;
        margin: 0 0 10px 0;
    }
    .box-high p, .box-medium p,
    .box-low p {
        color: #f8f9fa !important;
        margin: 0;
        font-size: 0.95rem;
    }
    .info-card {
        background: #1e2b3d;
        color: #ecf0f1;
        padding: 15px 20px;
        border-radius: 8px;
        border-left: 4px solid #3498db;
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
# CALCULATION ENGINE
# ═══════════════════════════════════════════════════════════════════

def oil_specific_gravity(API):
    return round(141.5 / (API + 131.5), 4)

def dead_oil_viscosity(API, T_F):
    """Beal (1946) / Standing (1981) - Ahmed Eq 2-117"""
    T_R = T_F + 460
    a = 10**(0.43 + 8.33/API)
    mu = ((0.32 + 1.8e7/API**4.53) *
          (360/(T_R - 260))**a)
    return round(mu, 4)

def bubble_point_pressure(Rs, gg, T_F, API):
    """Standing (1947) - Ahmed Eq 2-72"""
    Pb = (18.2 * ((Rs/gg)**0.83 *
          10**(0.00091*T_F - 0.0125*API) - 1.4))
    return round(abs(Pb), 1)

def saturated_viscosity(mu_od, Rs):
    """Beggs & Robinson (1975) - Ahmed Eq 2-121"""
    a = 10.715 * (Rs + 100)**(-0.515)
    b = 5.44 * (Rs + 150)**(-0.338)
    return round(a * mu_od**b, 4)

def undersaturated_viscosity(mu_ob, Pi, Pb):
    """Ahmed Eq 2-123"""
    a = -3.9e-5 * Pi - 5
    m = 2.6 * Pi**1.187 * 10**a
    return round(mu_ob * (Pi/Pb)**m, 4)

def oil_fvf(Rs, gg, go_val, T_F):
    """Standing (1981) - Ahmed Eq 2-85"""
    F = Rs * (gg/go_val)**0.5 + 1.25 * T_F
    Bo = 0.9759 + 0.000120 * F**1.2
    return round(Bo, 4)

def water_density(sal_ppm, T_F, P):
    """Niger Delta salinity correction"""
    rw = (62.4 + sal_ppm/10000 * 0.5 -
          0.003 * (T_F - 60) +
          0.0000145 * P)
    return round(rw, 3)

def oil_density(API, Bo, Rs, gg):
    go_val = 141.5/(API + 131.5)
    rho_s = go_val * 62.4
    rho_o = (rho_s + 0.01357 * Rs * gg) / Bo
    return round(rho_o, 3)

def mobility_ratio(krw, kro, mu_o, mu_w):
    """Ahmed Eq 9-24"""
    M = (krw/kro) * (mu_o/mu_w)
    alpha = 0.5 if M <= 1 else 0.6
    return round(M, 4), alpha

def sobocinski(kh, kv, phi, h, hp,
               mu_o, Bo, Qo,
               rho_w, rho_o, M, alpha):
    """
    Sobocinski-Cornelius (1965)
    Ahmed Eq 9-21 to 9-23
    """
    dr = rho_w - rho_o
    if dr <= 0:
        return None, None, None, \
               "Water must be denser than oil"

    Z = (0.492e-4 * dr * kh * h *
         (h - hp)) / (mu_o * Bo * Qo)

    if Z <= 0:
        return None, None, None, \
               f"Z={Z:.4f}: Check h > hp"
    if Z >= 3.5:
        return None, None, None, \
               f"Z={Z:.4f}: Out of valid range"

    tD = ((4*Z + 1.75*Z**2 - 0.75*Z**3) /
          (7 - 2*Z))
    tBT = (20325 * mu_o * h * phi * tD) / \
          (dr * kv * (1 + M**alpha))

    return (round(Z, 4), round(tD, 4),
            round(tBT, 1), None)

def risk_level(tBT):
    if tBT <= 365:
        return {
            'cat': 'HIGH RISK',
            'icon': '🔴',
            'box': 'box-high',
            'action': ('Begin water handling '
                       'facility planning '
                       'immediately. Consider '
                       'rate reduction to delay '
                       'breakthrough.')
        }
    elif tBT <= 730:
        return {
            'cat': 'MEDIUM RISK',
            'icon': '🟡',
            'box': 'box-medium',
            'action': ('Plan water handling '
                       'within 6 months. Monitor '
                       'production closely.')
        }
    else:
        return {
            'cat': 'LOW RISK',
            'icon': '🟢',
            'box': 'box-low',
            'action': ('Monitor quarterly. No '
                       'immediate action required.')
        }

# ═══════════════════════════════════════════════════════════════════
# PARAMETER CALCULATORS
# ═══════════════════════════════════════════════════════════════════

def calc_vsh_larionov(GR_log, GR_min, GR_max):
    """Larionov (1969) Tertiary rocks"""
    if GR_max <= GR_min:
        return 0.0
    IGR = (GR_log - GR_min) / (GR_max - GR_min)
    IGR = max(0, min(1, IGR))
    Vsh = 0.083 * (2**(3.7 * IGR) - 1)
    return round(min(Vsh, 1.0), 4)

def calc_kh_arithmetic(perms, thicknesses):
    """Arithmetic mean weighted by thickness"""
    if len(perms) != len(thicknesses):
        return 0
    perms = np.array(perms)
    thick = np.array(thicknesses)
    total = np.sum(thick)
    if total == 0:
        return 0
    return round(np.sum(perms * thick) / total, 2)

def calc_kv_harmonic(perms, thicknesses):
    """Harmonic mean weighted by thickness"""
    if len(perms) != len(thicknesses):
        return 0
    perms = np.array(perms)
    thick = np.array(thicknesses)
    if np.any(perms == 0):
        return 0
    total = np.sum(thick)
    denominator = np.sum(thick / perms)
    if denominator == 0:
        return 0
    return round(total / denominator, 4)

def calc_sci(NTG):
    """Proposed SCI = 1 - NTG"""
    SCI = 1 - NTG
    return round(max(0, min(SCI, 1.0)), 4)

def sci_sensitivity(kh, kv_mat, phi, h, hp,
                     mu_o, Bo, Qo, rw, ro,
                     M, alpha, Vsh):
    rows = []
    for s in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        kv_s = kv_mat * (1 - Vsh * s)
        _, _, t, e = sobocinski(
            kh, kv_s, phi, h, hp,
            mu_o, Bo, Qo, rw, ro, M, alpha)
        if t and not e:
            r = risk_level(t)
            rows.append({
                'SCI': s,
                'kv (md)': round(kv_s, 2),
                'BT (days)': t,
                'BT (years)': round(t/365, 2),
                'Risk': r['cat']
            })
    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════════
# FILE PARSER
# ═══════════════════════════════════════════════════════════════════

def parse_wireline_csv(uploaded_file):
    """Parse wireline log CSV skipping header comments"""
    content = uploaded_file.read().decode('utf-8')
    lines = content.split('\n')

    # Skip lines starting with # or empty
    data_start = 0
    for i, line in enumerate(lines):
        if (line.strip() and
            not line.startswith('#')):
            data_start = i
            break

    csv_content = '\n'.join(lines[data_start:])
    df = pd.read_csv(io.StringIO(csv_content))
    df.columns = [c.strip().upper() for c in df.columns]
    return df

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>💧 WaterWatch</h1>
    <p><b>Enhanced Sobocinski-Cornelius Framework</b>
    for Niger Delta Aquifer Water
    Breakthrough Prediction</p>
    <p>University of Benin |
    Department of Petroleum Engineering |
    Final Year Project</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📤 Data Upload",
    "🔍 Prediction",
    "📊 Sensitivity Analysis",
    "ℹ️ About WaterWatch"
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 - DATA UPLOAD
# ═══════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="section-hdr">'
                '📤 Upload Well Data</div>',
                unsafe_allow_html=True)

    col_up1, col_up2 = st.columns(2)

    with col_up1:
        st.markdown("#### Wireline Log Data")
        st.caption("CSV file with columns: "
                   "DEPTH_FT, GR_API, RHOB_GCC, "
                   "NPHI_FRAC, RT_OHMM")
        log_file = st.file_uploader(
            "Upload log CSV",
            type=['csv'],
            key='log_upload')

    with col_up2:
        st.markdown("#### Reservoir Data")
        st.caption("Excel workbook with sheets: "
                   "Well Summary, Core Analysis, "
                   "PVT, Production, Completion")
        excel_file = st.file_uploader(
            "Upload reservoir Excel",
            type=['xlsx', 'xls'],
            key='excel_upload')

    st.divider()

    # Process log if uploaded
    if log_file is not None:
        try:
            log_df = parse_wireline_csv(log_file)
            st.success(f"✅ Log loaded: "
                       f"{len(log_df)} rows")

            st.markdown("**Log Preview:**")
            st.dataframe(log_df.head(10),
                         hide_index=True,
                         use_container_width=True)

            # Store in session state
            st.session_state['log_df'] = log_df

            # GR statistics
            if 'GR_API' in log_df.columns:
                st.markdown("**Log Statistics:**")
                col_s1, col_s2, col_s3, col_s4 = \
                    st.columns(4)
                col_s1.metric(
                    "Total Rows", len(log_df))
                col_s2.metric(
                    "GR Min",
                    f"{log_df['GR_API'].min():.1f}")
                col_s3.metric(
                    "GR Max",
                    f"{log_df['GR_API'].max():.1f}")
                col_s4.metric(
                    "Depth Range",
                    f"{log_df['DEPTH_FT'].max() - log_df['DEPTH_FT'].min():.0f} ft")

        except Exception as e:
            st.error(f"Error parsing log: {e}")

    # Process Excel if uploaded
    if excel_file is not None:
        try:
            xl = pd.ExcelFile(excel_file)
            st.success(f"✅ Excel loaded: "
                       f"{len(xl.sheet_names)} sheets")

            st.markdown("**Sheets Available:**")
            for sheet in xl.sheet_names:
                st.write(f"• {sheet}")

            st.session_state['excel_file'] = \
                excel_file

        except Exception as e:
            st.error(f"Error parsing Excel: {e}")

    if log_file is None and excel_file is None:
        st.info("👆 Upload files above to begin. "
                "Or proceed to Prediction tab to "
                "use manual entry mode.")

    st.markdown('<div class="section-hdr">'
                '💡 File Format Guide</div>',
                unsafe_allow_html=True)

    with st.expander(
            "Wireline Log CSV Format"):
        st.markdown("""
        Required columns (case-insensitive):
        - **DEPTH_FT** — Depth in feet
        - **GR_API** — Gamma ray in API units
        - **RHOB_GCC** — Bulk density (optional)
        - **NPHI_FRAC** — Neutron porosity (optional)
        - **RT_OHMM** — Resistivity (optional)

        Header comment lines starting with `#`
        are automatically skipped.
        """)

    with st.expander(
            "Reservoir Data Excel Format"):
        st.markdown("""
        Recommended sheets:
        - **Well Summary** — Basic well info
        - **Core Analysis** — Plug data
          (depth, kh, kv, porosity)
        - **PVT Reports** — Pressure/property table
        - **Production History** — Daily rates
        - **Completion Data** — Perforation depths

        Column naming: use standard petroleum
        engineering terminology.
        """)

# ═══════════════════════════════════════════════════════════════════
# TAB 2 - PREDICTION
# ═══════════════════════════════════════════════════════════════════

with tab2:

    with st.sidebar:
        st.markdown("### ⚙️ Input Parameters")
        st.caption("Adjust values based on your data")

        st.markdown("**🪨 Rock Properties**")
        kh_mean = st.number_input(
            "kh — Horizontal Permeability (md)",
            10.0, 5000.0, 1800.0, 10.0,
            key='kh')
        kv_matrix = st.number_input(
            "kv — Vertical Permeability (md)",
            1.0, 1000.0, 270.0, 5.0,
            key='kv')
        phi_log = st.number_input(
            "φ — Log Porosity (fraction)",
            0.05, 0.45, 0.28, 0.01,
            key='phi')
        depth_ft = st.number_input(
            "Reservoir Depth (ft)",
            1000.0, 15000.0, 8500.0, 100.0,
            key='depth')
        h = st.number_input(
            "h — Oil Column Height (ft)",
            5.0, 300.0, 80.0, 1.0,
            key='h')
        hp = st.number_input(
            "hp — Perforated Interval (ft)",
            1.0, 200.0, 25.0, 1.0,
            key='hp')

        st.markdown(
            "**🌍 Niger Delta Corrections**")
        Vsh = st.slider(
            "Vsh — Shale Volume",
            0.0, 0.5, 0.15, 0.01,
            key='vsh',
            help="From Gamma Ray log via "
                 "Larionov correlation")
        SCI = st.slider(
            "SCI — Shale Continuity ★",
            0.0, 1.0, 0.50, 0.05,
            key='sci',
            help="★ Novel parameter proposed "
                 "in this study. "
                 "SCI = 1 - NTG")

        st.markdown("**🧪 Fluid Properties**")
        pvt_mode = st.radio(
            "PVT Source",
            ["Calculate from correlations",
             "Enter measured PVT"],
            key='pvt_mode')

        API = st.number_input(
            "API Gravity (°)",
            15.0, 55.0, 35.0, 0.5,
            key='api')
        T_F = st.number_input(
            "Temperature (°F)",
            100.0, 300.0, 180.0, 5.0,
            key='temp')
        Pi = st.number_input(
            "Initial Pressure (psia)",
            500.0, 10000.0, 4200.0, 50.0,
            key='pi')
        sal = st.number_input(
            "Water Salinity (ppm)",
            1000.0, 150000.0, 35000.0, 1000.0,
            key='sal')
        mu_w = st.number_input(
            "Water Viscosity μw (cp)",
            0.2, 1.5, 0.50, 0.05,
            key='muw')

        if pvt_mode == "Calculate from correlations":
            Rs = st.number_input(
                "Rs — Solution GOR (scf/STB)",
                50.0, 2000.0, 600.0, 10.0,
                key='rs')
            gg = st.number_input(
                "γg — Gas Specific Gravity",
                0.5, 1.2, 0.75, 0.01,
                key='gg')
            mu_o_meas = None
            Bo_meas = None
            Pb_meas = None
        else:
            mu_o_meas = st.number_input(
                "Measured μo (cp)",
                0.1, 100.0, 0.6, 0.1,
                key='muo')
            Bo_meas = st.number_input(
                "Measured Bo (bbl/STB)",
                1.0, 3.0, 1.34, 0.01,
                key='bo')
            Pb_meas = st.number_input(
                "Measured Pb (psia)",
                100.0, 8000.0, 2463.0, 10.0,
                key='pb')
            Rs = 600.0
            gg = 0.75

        st.markdown("**💧 Saturation**")
        krw = st.number_input(
            "krw at Sor", 0.1, 0.8, 0.35, 0.05,
            key='krw')
        kro = st.number_input(
            "kro at Swc", 0.3, 1.0, 0.85, 0.05,
            key='kro')

        st.markdown("**⚡ Production**")
        Qo = st.number_input(
            "Qo — Production Rate (STB/day)",
            100.0, 10000.0, 2000.0, 100.0,
            key='qo')

        run_btn = st.button(
            "🔍 PREDICT BREAKTHROUGH",
            type="primary",
            use_container_width=True)

    # ─── Auto-Interpretation Section ────────────
    st.markdown('<div class="section-hdr">'
                '🧮 Automated Parameter '
                'Calculation from Uploaded Data'
                '</div>', unsafe_allow_html=True)

    if 'log_df' in st.session_state:
        log_df = st.session_state['log_df']

        st.markdown("### Vsh Calculation "
                    "(from Gamma Ray Log)")

        col_v1, col_v2, col_v3 = st.columns(3)

        with col_v1:
            depth_target = st.number_input(
                "Analysis Depth (ft)",
                float(log_df['DEPTH_FT'].min()),
                float(log_df['DEPTH_FT'].max()),
                float(log_df['DEPTH_FT'].min() +
                      (log_df['DEPTH_FT'].max() -
                       log_df['DEPTH_FT'].min())/2),
                0.5,
                key='depth_target')

        with col_v2:
            gr_min_auto = st.number_input(
                "GR_min (API)",
                0.0, 200.0,
                float(log_df['GR_API'].quantile(0.05)),
                1.0,
                key='gr_min')

        with col_v3:
            gr_max_auto = st.number_input(
                "GR_max (API)",
                50.0, 300.0,
                float(log_df['GR_API'].quantile(0.95)),
                1.0,
                key='gr_max')

        # Find nearest depth
        nearest = log_df.iloc[
            (log_df['DEPTH_FT'] -
             depth_target).abs().argsort()[:1]]
        gr_at_depth = nearest['GR_API'].values[0]
        vsh_calc = calc_vsh_larionov(
            gr_at_depth, gr_min_auto, gr_max_auto)

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("GR at Target Depth",
                       f"{gr_at_depth:.1f} API")
        col_r2.metric("Calculated Vsh",
                       f"{vsh_calc}")
        col_r3.metric("Reservoir Character",
                       "Clean Sand" if vsh_calc < 0.3
                       else "Shaly" if vsh_calc < 0.5
                       else "Very Shaly")

        # Overall reservoir NTG estimate
        st.markdown("### NTG Estimation "
                    "(for SCI calculation)")

        col_n1, col_n2 = st.columns(2)
        with col_n1:
            gr_cutoff = st.number_input(
                "GR Sand Cutoff (API)",
                30.0, 100.0, 75.0, 1.0,
                help="Below this = sand, "
                     "above = shale")
            res_top = st.number_input(
                "Reservoir Top (ft)",
                float(log_df['DEPTH_FT'].min()),
                float(log_df['DEPTH_FT'].max()),
                float(log_df['DEPTH_FT'].min()),
                1.0, key='res_top')
            res_base = st.number_input(
                "Reservoir Base (ft)",
                float(log_df['DEPTH_FT'].min()),
                float(log_df['DEPTH_FT'].max()),
                float(log_df['DEPTH_FT'].max()),
                1.0, key='res_base')

        with col_n2:
            res_zone = log_df[
                (log_df['DEPTH_FT'] >= res_top) &
                (log_df['DEPTH_FT'] <= res_base)]

            sand_count = (
                res_zone['GR_API'] < gr_cutoff).sum()
            total_count = len(res_zone)

            if total_count > 0:
                ntg_calc = sand_count / total_count
                sci_calc = calc_sci(ntg_calc)

                st.metric("Reservoir Interval",
                          f"{res_base - res_top:.0f} ft")
                st.metric("NTG (from log)",
                          f"{ntg_calc:.3f}")
                st.metric("SCI (proposed) ★",
                          f"{sci_calc}")

                if st.button(
                        "📥 Use these values in "
                        "prediction",
                        key='use_log_vals'):
                    st.info(f"✅ Vsh={vsh_calc}, "
                            f"SCI={sci_calc} — "
                            f"adjust sidebar sliders "
                            f"to these values")

        # Log visualization
        st.markdown("### Log Visualization")

        fig_log = go.Figure()
        fig_log.add_trace(go.Scatter(
            x=log_df['GR_API'],
            y=log_df['DEPTH_FT'],
            mode='lines',
            name='GR',
            line=dict(color='#e74c3c', width=1.5)
        ))
        fig_log.add_vline(
            x=gr_cutoff, line_dash="dash",
            line_color="yellow",
            annotation_text=f"Sand Cutoff {gr_cutoff}")
        fig_log.update_yaxes(autorange="reversed")
        fig_log.update_layout(
            title="Gamma Ray Log",
            xaxis_title="GR (API units)",
            yaxis_title="Depth (ft)",
            height=500,
            plot_bgcolor='#0e1621',
            paper_bgcolor='#0e1621',
            font=dict(color='white')
        )
        st.plotly_chart(fig_log,
                        use_container_width=True)

    else:
        st.info("📤 Upload wireline log CSV in "
                "the Data Upload tab to enable "
                "automated Vsh and SCI calculation.")

    st.divider()

    # ─── Run Prediction ─────────────────────────
    if run_btn:
        if hp >= h:
            st.error("❌ Perforated interval (hp) "
                     "must be less than oil "
                     "column height (h)")
            st.stop()

        # Fluid properties
        go_val = oil_specific_gravity(API)

        if pvt_mode == "Calculate from correlations":
            mu_od = dead_oil_viscosity(API, T_F)
            Pb = bubble_point_pressure(
                Rs, gg, T_F, API)
            mu_ob = saturated_viscosity(mu_od, Rs)

            if Pi > Pb:
                mu_o = undersaturated_viscosity(
                    mu_ob, Pi, Pb)
                cond = "Undersaturated"
            else:
                mu_o = mu_ob
                cond = "Saturated"

            Bo = oil_fvf(Rs, gg, go_val, T_F)
        else:
            mu_o = mu_o_meas
            Bo = Bo_meas
            Pb = Pb_meas
            mu_od = None
            mu_ob = None
            cond = "Measured PVT"

        rw = water_density(sal, T_F, Pi)
        ro = oil_density(API, Bo, Rs, gg)
        M, alpha = mobility_ratio(
            krw, kro, mu_o, mu_w)

        # ═══ METHOD 1: ORIGINAL SOBOCINSKI ═══
        # No Niger Delta corrections
        # Uses RAW kh, kv, phi with SAME
        # fluid properties for fair comparison

        Z1, tD1, tBT1, err1 = sobocinski(
            kh_mean, kv_matrix, phi_log,
            h, hp, mu_o, Bo, Qo,
            rw, ro, M, alpha)

        # ═══ METHOD 2: ENHANCED SOBOCINSKI ═══
        # With Niger Delta corrections
        phi_eff = phi_log * np.exp(
            -0.000027 * depth_ft)  # Sclater-Christie
        kv_eff = kv_matrix * (1 - Vsh * SCI)
        # kh unchanged - use raw (no arbitrary V_DP)

        Z2, tD2, tBT2, err2 = sobocinski(
            kh_mean, kv_eff, phi_eff,
            h, hp, mu_o, Bo, Qo,
            rw, ro, M, alpha)

        # ═══ RESULTS DISPLAY ═══
        st.markdown('<div class="section-hdr">'
                    '📊 Prediction Results</div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Method 1 — "
                        "Original Sobocinski")
            st.caption("Baseline: Raw inputs, "
                       "no Niger Delta corrections")

            if err1:
                st.error(f"❌ {err1}")
            elif tBT1:
                r1 = risk_level(tBT1)
                st.markdown(
                    f'<div class="{r1["box"]}">'
                    f'<h2>{r1["icon"]} {r1["cat"]}</h2>'
                    f'<h3>Breakthrough: {tBT1} days '
                    f'({tBT1/365:.2f} years)</h3>'
                    f'<p>⚡ {r1["action"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True)
                cA1, cA2 = st.columns(2)
                cA1.metric("Z", Z1)
                cA2.metric("(tD)BT", tD1)

        with col2:
            st.markdown("#### Method 2 — "
                        "Enhanced Sobocinski ★")
            st.caption("With SCI correction and "
                       "Athy compaction "
                       "(this study)")

            if err2:
                st.error(f"❌ {err2}")
            elif tBT2:
                r2 = risk_level(tBT2)
                st.markdown(
                    f'<div class="{r2["box"]}">'
                    f'<h2>{r2["icon"]} {r2["cat"]}</h2>'
                    f'<h3>Breakthrough: {tBT2} days '
                    f'({tBT2/365:.2f} years)</h3>'
                    f'<p>⚡ {r2["action"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True)
                cB1, cB2 = st.columns(2)
                cB1.metric("Z", Z2)
                cB2.metric("(tD)BT", tD2)

        # ─── Enhancement Impact Analysis ──────
        if tBT1 and tBT2 and not err1 and not err2:
            st.divider()
            st.markdown('<div class="section-hdr">'
                        '📈 Enhancement Impact Analysis'
                        '</div>', unsafe_allow_html=True)

            diff = tBT2 - tBT1
            pct = (diff / tBT1) * 100

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Original tBT",
                       f"{tBT1} days")
            m2.metric("Enhanced tBT",
                       f"{tBT2} days",
                       delta=f"{diff:+.1f} days")
            m3.metric("Absolute Change",
                       f"{abs(diff):.1f} days")
            m4.metric("% Change",
                       f"{pct:+.1f}%")

            if diff > 0:
                st.markdown(
                    f'<div class="info-card">'
                    f'📊 The enhanced method '
                    f'predicts breakthrough '
                    f'<b>{abs(diff):.0f} days later</b>. '
                    f'This shows the SCI correction '
                    f'accounts for shale barriers '
                    f'that physically delay water '
                    f'coning in the Agbada Formation. '
                    f'The Athy compaction correction '
                    f'reduces porosity by '
                    f'{(1-phi_eff/phi_log)*100:.1f}%, '
                    f'partially offsetting this.'
                    f'</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="info-card">'
                    f'📊 The enhanced method '
                    f'predicts breakthrough '
                    f'<b>{abs(diff):.0f} days earlier</b>. '
                    f'The Athy compaction reduction '
                    f'in porosity dominates over '
                    f'the SCI kv reduction.'
                    f'</div>',
                    unsafe_allow_html=True)

            # Bar chart
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Bar(
                x=["Original Sobocinski",
                   "Enhanced Sobocinski ★"],
                y=[tBT1, tBT2],
                marker_color=['#e74c3c',
                              '#3498db'],
                text=[f'{tBT1:.0f} days<br>'
                      f'({tBT1/365:.1f} yrs)',
                      f'{tBT2:.0f} days<br>'
                      f'({tBT2/365:.1f} yrs)'],
                textposition='outside',
                textfont=dict(size=13, color='white')
            ))
            fig_compare.add_hline(
                y=365, line_dash="dash",
                line_color="red",
                annotation_text="365 days")
            fig_compare.add_hline(
                y=730, line_dash="dash",
                line_color="orange",
                annotation_text="730 days")
            fig_compare.update_layout(
                title="Original vs Enhanced "
                      "Sobocinski Comparison",
                yaxis_title=
                    "Breakthrough Time (days)",
                height=420,
                showlegend=False,
                plot_bgcolor='#0e1621',
                paper_bgcolor='#0e1621',
                font=dict(color='white')
            )
            st.plotly_chart(
                fig_compare,
                use_container_width=True)

        # ─── Intermediate Values ──────────────
        with st.expander(
                "🔬 View Intermediate Calculations"):
            cI1, cI2, cI3 = st.columns(3)

            with cI1:
                st.markdown(
                    "**Niger Delta Corrections**")
                st.write(f"φ: {phi_log} → "
                         f"{phi_eff:.4f}")
                st.write(f"kv: {kv_matrix} → "
                         f"{kv_eff:.2f} md")
                st.write(f"kh: {kh_mean} md "
                         f"(unchanged)")

            with cI2:
                st.markdown("**Fluid Properties**")
                st.write(f"μo: {mu_o} cp")
                st.write(f"Bo: {Bo} bbl/STB")
                if pvt_mode == \
                   "Calculate from correlations":
                    st.write(f"μod: {mu_od} cp")
                    st.write(f"μob: {mu_ob} cp")
                    st.write(f"Pb: {Pb} psia")
                    st.write(f"Condition: {cond}")

            with cI3:
                st.markdown(
                    "**Densities & Mobility**")
                st.write(f"ρw: {rw} lb/ft³")
                st.write(f"ρo: {ro} lb/ft³")
                st.write(f"Δρ: {rw-ro:.3f} lb/ft³")
                st.write(f"M: {M}")
                st.write(f"α: {alpha}")

        # Store for sensitivity tab
        st.session_state['prediction_params'] = {
            'kh': kh_mean, 'kv_mat': kv_matrix,
            'phi': phi_eff, 'h': h, 'hp': hp,
            'mu_o': mu_o, 'Bo': Bo, 'Qo': Qo,
            'rw': rw, 'ro': ro, 'M': M,
            'alpha': alpha, 'Vsh': Vsh, 'SCI': SCI
        }

    else:
        st.info("👈 Enter parameters in sidebar "
                "and click **PREDICT BREAKTHROUGH**")

# ═══════════════════════════════════════════════════════════════════
# TAB 3 - SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-hdr">'
                '📊 SCI Sensitivity Analysis '
                '(Novel Contribution ★)</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="info-card">'
        'This analysis shows how the proposed '
        '<b>Shale Continuity Index (SCI)</b> '
        'affects breakthrough time prediction. '
        'SCI ranges from 0 (discontinuous shales) '
        'to 1 (continuous shales) — higher SCI '
        'delays breakthrough because shale '
        'barriers restrict vertical water flow.'
        '</div>', unsafe_allow_html=True)

    if 'prediction_params' in st.session_state:
        p = st.session_state['prediction_params']

        sci_df = sci_sensitivity(
            p['kh'], p['kv_mat'], p['phi'],
            p['h'], p['hp'], p['mu_o'],
            p['Bo'], p['Qo'], p['rw'],
            p['ro'], p['M'], p['alpha'],
            p['Vsh'])

        if not sci_df.empty:
            col_s1, col_s2 = st.columns([3, 2])

            with col_s1:
                fig_sci = go.Figure()

                # Risk zones
                fig_sci.add_hrect(
                    y0=0, y1=365,
                    fillcolor="red",
                    opacity=0.1,
                    annotation_text="HIGH",
                    annotation_position="left")
                fig_sci.add_hrect(
                    y0=365, y1=730,
                    fillcolor="orange",
                    opacity=0.1,
                    annotation_text="MEDIUM",
                    annotation_position="left")
                fig_sci.add_hrect(
                    y0=730,
                    y1=sci_df['BT (days)'].max() * 1.3,
                    fillcolor="green",
                    opacity=0.1,
                    annotation_text="LOW",
                    annotation_position="left")

                fig_sci.add_trace(go.Scatter(
                    x=sci_df['SCI'],
                    y=sci_df['BT (days)'],
                    mode='lines+markers',
                    line=dict(color='#3498db',
                              width=3),
                    marker=dict(size=12,
                                color='#3498db')
                ))

                fig_sci.update_layout(
                    title="Effect of SCI on "
                          "Breakthrough Time",
                    xaxis_title="Shale Continuity "
                                 "Index (SCI)",
                    yaxis_title="Breakthrough "
                                 "Time (days)",
                    height=450,
                    plot_bgcolor='#0e1621',
                    paper_bgcolor='#0e1621',
                    font=dict(color='white')
                )
                st.plotly_chart(
                    fig_sci,
                    use_container_width=True)

            with col_s2:
                st.dataframe(
                    sci_df,
                    hide_index=True,
                    use_container_width=True)

                bt_range = (sci_df['BT (days)'].max()
                            - sci_df['BT (days)'].min())

                st.markdown(
                    f'<div class="info-card">'
                    f'<b>SCI Impact Range:</b><br>'
                    f'Min BT: '
                    f'{sci_df["BT (days)"].min():.0f} days<br>'
                    f'Max BT: '
                    f'{sci_df["BT (days)"].max():.0f} days<br>'
                    f'Range: {bt_range:.0f} days '
                    f'({bt_range/365:.2f} years)'
                    f'</div>',
                    unsafe_allow_html=True)
    else:
        st.info("👈 Run a prediction first in the "
                "Prediction tab to see sensitivity "
                "analysis")

# ═══════════════════════════════════════════════════════════════════
# TAB 4 - ABOUT
# ═══════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("""
    ## About WaterWatch

    WaterWatch is an enhanced analytical tool
    for predicting aquifer water breakthrough
    time in Niger Delta vertical oil wells.

    ---

    ### Core Method
    **Sobocinski-Cornelius (1965)** as presented
    in Ahmed (2010) *Reservoir Engineering
    Handbook*, Equations 9-21 to 9-23.

    ---

    ### Enhancements (This Study)

    | Enhancement | Formula | Source |
    |---|---|---|
    | **Proposed SCI** ★ | kv_eff = kv × (1 - Vsh × SCI) | This study |
    | Athy compaction | φ_eff = φ × exp(-Cf × depth) | Athy (1930) |
    | Auto μo (dead) | Beal (1946) | Ahmed Eq 2-117 |
    | Auto μo (sat) | Beggs-Robinson (1975) | Ahmed Eq 2-121 |
    | Auto μo (unsat) | Vasquez-Beggs | Ahmed Eq 2-123 |
    | Auto Bo | Standing (1981) | Ahmed Eq 2-85 |
    | Bubble point | Standing (1947) | Ahmed Eq 2-72 |
    | Salinity corrected ρw | Niger Delta | Tuttle et al (1999) |

    ---

    ### Novel Contribution
    The **Shale Continuity Index (SCI)** is a
    proposed dimensionless parameter (0-1)
    that characterizes the lateral continuity
    of shale intercalations in the Agbada
    Formation. It reduces effective vertical
    permeability to account for physical
    barriers to water coning.

    **Justification:** Short & Stauble (1967)
    and Doust & Omatsola (1990) document
    that the Niger Delta Agbada Formation
    is a paralic sequence of alternating
    sands and shales. These shale
    intercalations vary in lateral extent
    from discontinuous stringers to
    laterally continuous barriers.

    **Estimation:** SCI ≈ 1 - NTG
    where NTG is calculated from gamma ray
    log interpretation using standard
    Niger Delta cutoff values.

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
      Handbook*, 4th Ed., Gulf Professional Publishing.
    - Sobocinski, D.P. and Cornelius, A.J. (1965)
      "A Correlation for Predicting Water Coning
      Time." *JPT* SPE-894-PA.
    - Bournazel, C. and Jeanson, B. (1971)
      "Fast Water-Coning Evaluation Method."
      SPE-3628.
    - Tuttle, M.L.W. et al. (1999) *The Niger
      Delta Petroleum System*. USGS OFR 99-50-H.
    - Short, K.C. and Stauble, A.J. (1967)
      "Outline of Geology of Niger Delta."
      *AAPG Bulletin* 51(5): 761-779.
    - Doust, H. and Omatsola, E. (1990)
      "Niger Delta." *AAPG Memoir* 48: 201-238.
    - Athy, L.F. (1930) "Density, Porosity and
      Compaction of Sedimentary Rocks."
      *AAPG Bulletin* 14(1): 1-24.
    - Larionov, V.V. (1969) — Vsh from GR
      correlation for Tertiary rocks.
    - Standing, M.B. (1947, 1981) — Bo and
      bubble point correlations.
    - Beggs, H.D. and Robinson, J.R. (1975) —
      Oil viscosity correlations.

    ---

    ### Scope and Limitations

    **Scope:**
    - Vertical wells only
    - Natural aquifer drive
    - Bottom water and edge water
    - Niger Delta Agbada Formation
    - Early production period

    **Limitations:**
    - SCI requires field validation
    - Assumes constant production rate
    - Requires representative reservoir
      parameters
    - Not calibrated for horizontal wells

    ---

    ### University of Benin
    **Department of Petroleum Engineering**
    **Final Year Project**
    """)
