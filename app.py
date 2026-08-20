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
import re
import warnings
warnings.filterwarnings('ignore')

# ─── Page Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="WaterWatch | UNIBEN",
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
.autofill-card {
    background: linear-gradient(135deg,
                #16a085, #1abc9c);
    color: white;
    padding: 20px 25px;
    border-radius: 10px;
    margin: 12px 0;
    border-left: 8px solid #0e6655;
}
.autofill-card h3 {
    color: white !important;
    margin: 0 0 10px 0;
}
.autofill-card p {
    color: #ecf0f1 !important;
    margin: 4px 0;
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
    """Beal (1946) / Standing (1981)
    Ahmed Eq 2-117"""
    T_R = T_F + 460
    a = 10**(0.43 + 8.33/API)
    mu = ((0.32 + 1.8e7/API**4.53) *
          (360/(T_R - 260))**a)
    return round(mu, 4)

def bubble_point_pressure(Rs, gg, T_F, API):
    """Standing (1947) - Ahmed Eq 2-72"""
    Pb = (18.2 * ((Rs/gg)**0.83 *
          10**(0.00091*T_F -
               0.0125*API) - 1.4))
    return round(abs(Pb), 1)

def saturated_viscosity(mu_od, Rs):
    """Beggs & Robinson (1975)
    Ahmed Eq 2-121"""
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
               mu_o, Bo, Qo, rho_w, rho_o,
               M, alpha):
    """Sobocinski-Cornelius (1965)
    Ahmed Eq 9-21 to 9-23"""
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
            'cat': 'HIGH RISK', 'icon': '🔴',
            'box': 'box-high',
            'action': ('Begin water handling '
                       'facility planning '
                       'immediately.')}
    elif tBT <= 730:
        return {
            'cat': 'MEDIUM RISK', 'icon': '🟡',
            'box': 'box-medium',
            'action': ('Plan water handling '
                       'within 6 months.')}
    else:
        return {
            'cat': 'LOW RISK', 'icon': '🟢',
            'box': 'box-low',
            'action': ('Monitor quarterly. No '
                       'immediate action.')}

def calc_vsh_larionov(GR_log, GR_min, GR_max):
    """Larionov (1969) Tertiary rocks"""
    if GR_max <= GR_min:
        return 0.0
    IGR = (GR_log - GR_min) / (GR_max - GR_min)
    IGR = max(0, min(1, IGR))
    Vsh = 0.083 * (2**(3.7 * IGR) - 1)
    return round(min(Vsh, 1.0), 4)

def calc_kh_arithmetic(perms, thick):
    """Arithmetic mean weighted by thickness"""
    if len(perms) != len(thick):
        return 0
    p = np.array(perms)
    t = np.array(thick)
    if np.sum(t) == 0:
        return 0
    return round(np.sum(p * t) / np.sum(t), 2)

def calc_kv_harmonic(perms, thick):
    """Harmonic mean weighted by thickness"""
    if len(perms) != len(thick):
        return 0
    p = np.array(perms)
    t = np.array(thick)
    if np.any(p == 0):
        return 0
    denom = np.sum(t / p)
    if denom == 0:
        return 0
    return round(np.sum(t) / denom, 4)

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
                'Risk': r['cat']})
    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════════
# FILE PARSER — Auto-detect reservoir boundaries
# ═══════════════════════════════════════════════════════════════════

def parse_wireline_csv(uploaded_file):
    """
    Parse wireline log CSV with auto-detection
    of reservoir boundaries from header comments.
    Returns dataframe + metadata dict.
    """
    content = uploaded_file.read().decode('utf-8')
    lines = content.split('\n')

    metadata = {
        'well_id': 'Unknown',
        'field': 'Unknown',
        'location': 'Unknown',
        'res_top': None,
        'res_base': None,
        'owc': None,
        'perf_top': None,
        'perf_bottom': None,
    }

    # Parse header comments (# lines)
    for line in lines:
        if not line.startswith('#'):
            continue
        low = line.lower()

        # Well ID
        if 'well id' in low:
            m = re.search(r'well id\s*:?\s*(\S+)',
                          low)
            if m:
                metadata['well_id'] = m.group(1).upper()

        # Field
        elif 'field' in low and ':' in line:
            parts = line.split(':', 1)
            if len(parts) > 1:
                val = parts[1].strip()
                if val:
                    metadata['field'] = val

        # Location
        elif 'location' in low and ':' in line:
            parts = line.split(':', 1)
            if len(parts) > 1:
                val = parts[1].strip()
                if val:
                    metadata['location'] = val

        # Reservoir: 10100 - 10620 ft
        elif 'reservoir:' in low:
            nums = re.findall(
                r'\d+\.?\d*', line)
            nums = [float(n) for n in nums]
            if len(nums) >= 2:
                metadata['res_top'] = nums[0]
                metadata['res_base'] = nums[1]

        # OWC: 10570 ft
        elif 'owc' in low:
            nums = re.findall(
                r'\d+\.?\d*', line)
            nums = [float(n) for n in nums]
            if nums:
                metadata['owc'] = nums[0]

        # Perf Top: 10105 ft
        elif 'perf top' in low:
            nums = re.findall(
                r'\d+\.?\d*', line)
            nums = [float(n) for n in nums]
            if nums:
                metadata['perf_top'] = nums[0]

        # Perf Bottom: 10140 ft
        elif 'perf bottom' in low or \
             'perf base' in low:
            nums = re.findall(
                r'\d+\.?\d*', line)
            nums = [float(n) for n in nums]
            if nums:
                metadata['perf_bottom'] = nums[0]

    # Extract data section (skip # comments
    # and blank lines)
    data_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (stripped and
            not stripped.startswith('#')):
            data_start = i
            break

    csv_content = '\n'.join(lines[data_start:])
    df = pd.read_csv(io.StringIO(csv_content))
    df.columns = [c.strip().upper()
                  for c in df.columns]

    return df, metadata

# ═══════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════

# Auto-fill defaults from log interpretation
if 'auto_vals' not in st.session_state:
    st.session_state['auto_vals'] = {}

# Store uploaded data
if 'log_df' not in st.session_state:
    st.session_state['log_df'] = None

if 'log_meta' not in st.session_state:
    st.session_state['log_meta'] = None

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
    "ℹ️ About"
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 - DATA UPLOAD WITH AUTO-DETECTION
# ═══════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<div class="section-hdr">'
                '📤 Upload Well Data</div>',
                unsafe_allow_html=True)

    col_up1, col_up2 = st.columns(2)

    with col_up1:
        st.markdown("#### Wireline Log Data")
        st.caption("CSV file with header info "
                   "and log curves")
        log_file = st.file_uploader(
            "Upload log CSV",
            type=['csv'], key='log_upload')

    with col_up2:
        st.markdown("#### Reservoir Data")
        st.caption("Excel workbook with core, "
                   "PVT, production data")
        excel_file = st.file_uploader(
            "Upload reservoir Excel",
            type=['xlsx', 'xls'],
            key='excel_upload')

    st.divider()

    # ── Process Log File ────────────────────────
    if log_file is not None:
        try:
            log_df, meta = parse_wireline_csv(
                log_file)

            # ── Universal column detection ─────
            # Standardize common column name
            # variations from different vendors
            col_map = {}
            for col in log_df.columns:
                col_up = col.upper().strip()
                # Depth variations
                if any(x in col_up for x in
                       ['DEPT', 'DEPTH', 'MD',
                        'TVDSS']):
                    if 'DEPTH_FT' not in col_map \
                       .values():
                        col_map[col] = 'DEPTH_FT'
                # GR variations
                elif any(x in col_up for x in
                         ['GR', 'GAMMA']):
                    if 'GR_API' not in col_map \
                       .values():
                        col_map[col] = 'GR_API'
                # Density variations
                elif any(x in col_up for x in
                         ['RHOB', 'DEN', 'BULK']):
                    if 'RHOB_GCC' not in col_map \
                       .values():
                        col_map[col] = 'RHOB_GCC'
                # Neutron variations
                elif any(x in col_up for x in
                         ['NPHI', 'NEUT', 'CNL',
                          'TNPH']):
                    if 'NPHI_FRAC' not in col_map \
                       .values():
                        col_map[col] = 'NPHI_FRAC'
                # Resistivity variations
                elif any(x in col_up for x in
                         ['RT', 'RES', 'ILD',
                          'LLD', 'OHM']):
                    if 'RT_OHMM' not in col_map \
                       .values():
                        col_map[col] = 'RT_OHMM'

            log_df = log_df.rename(
                columns=col_map)

            # Check required columns
            if 'DEPTH_FT' not in log_df.columns:
                st.error(
                    "❌ Cannot find DEPTH column. "
                    "Ensure your CSV has a column "
                    "named DEPTH, DEPT, MD, or "
                    "similar.")
                st.stop()

            if 'GR_API' not in log_df.columns:
                st.error(
                    "❌ Cannot find GR column. "
                    "Ensure your CSV has a column "
                    "named GR, GAMMA, or similar.")
                st.stop()

            st.session_state['log_df'] = log_df
            st.session_state['log_meta'] = meta

            # ── Status Display ────────────────
            has_metadata = (meta['res_top']
                            and meta['res_base'])

            log_min = float(
                log_df['DEPTH_FT'].min())
            log_max = float(
                log_df['DEPTH_FT'].max())

            if has_metadata:
                st.success(
                    f"✅ Log loaded with metadata: "
                    f"{len(log_df)} rows | "
                    f"Well: {meta['well_id']} | "
                    f"Field: {meta['field']}")
            else:
                st.success(
                    f"✅ Log loaded: "
                    f"{len(log_df)} rows | "
                    f"Depth: {log_min:.0f} - "
                    f"{log_max:.0f} ft")
                st.info(
                    "ℹ️ No reservoir boundary "
                    "metadata found in this CSV. "
                    "Please enter boundaries "
                    "manually below.")

            # ── DETECTED COLUMNS ──────────────
            detected_cols = list(log_df.columns)
            st.caption(
                f"Detected columns: "
                f"{', '.join(detected_cols)}")

            # ── RESERVOIR BOUNDARIES ──────────
            st.markdown(
                '<div class="section-hdr">'
                '🎯 Reservoir Boundaries</div>',
                unsafe_allow_html=True)

            if has_metadata:
                st.markdown(
                    '<div class="autofill-card">'
                    '<h3>Auto-detected from CSV '
                    'header — you can override</h3>'
                    '</div>',
                    unsafe_allow_html=True)

            # Default values (from metadata
            # if available, else log range)
            def_res_top = (float(meta['res_top'])
                           if meta['res_top']
                           else log_min)
            def_res_base = (float(meta['res_base'])
                            if meta['res_base']
                            else log_max)
            def_owc = (float(meta['owc'])
                       if meta['owc']
                       else def_res_base - 20)
            def_perf_top = (
                float(meta['perf_top'])
                if meta['perf_top']
                else def_res_top)
            def_perf_bot = (
                float(meta['perf_bottom'])
                if meta['perf_bottom']
                else def_perf_top + 20)

            col_b1, col_b2 = st.columns(2)

            with col_b1:
                st.markdown("**From Log**")
                res_top = st.number_input(
                    "Reservoir Top (ft)",
                    log_min, log_max, def_res_top,
                    1.0, key='res_top_in',
                    help="Top of pay zone from "
                         "log interpretation")
                res_base = st.number_input(
                    "Reservoir Base (ft)",
                    log_min, log_max, def_res_base,
                    1.0, key='res_base_in',
                    help="Base of reservoir")
                owc = st.number_input(
                    "OWC Depth (ft)",
                    log_min, log_max, def_owc,
                    1.0, key='owc_in',
                    help="Oil-water contact "
                         "from resistivity log")

            with col_b2:
                st.markdown("**From Completion**")
                perf_top = st.number_input(
                    "Perforation Top (ft)",
                    log_min, log_max,
                    def_perf_top, 1.0,
                    key='perf_top_in',
                    help="From completion report")
                perf_bot = st.number_input(
                    "Perforation Bottom (ft)",
                    log_min, log_max,
                    def_perf_bot, 1.0,
                    key='perf_bot_in',
                    help="From completion report")

                # Calculated values
                h_calc = owc - res_top
                hp_calc = perf_bot - perf_top

                st.metric(
                    "Oil Column h",
                    f"{h_calc:.0f} ft")
                st.metric(
                    "Perforated hp",
                    f"{hp_calc:.0f} ft")

            # Validation
            if res_top >= res_base:
                st.error(
                    "❌ Reservoir Top must be "
                    "less than Reservoir Base")
            elif owc < res_top or owc > res_base:
                st.warning(
                    "⚠️ OWC should be between "
                    "reservoir top and base")
            elif h_calc <= 0:
                st.error(
                    "❌ Oil column must be positive")
            elif hp_calc <= 0:
                st.error(
                    "❌ Perforation interval "
                    "must be positive")
            elif hp_calc >= h_calc:
                st.error(
                    "❌ Perforation must be less "
                    "than oil column")
            else:
                # ── CALCULATE PARAMETERS ────
                st.markdown(
                    '<div class="section-hdr">'
                    '🧮 Automated Log '
                    'Interpretation</div>',
                    unsafe_allow_html=True)

                res_zone = log_df[
                    (log_df['DEPTH_FT'] >=
                     res_top) &
                    (log_df['DEPTH_FT'] <=
                     res_base)]

                if len(res_zone) < 5:
                    st.error(
                        "❌ Too few data points "
                        "in reservoir zone")
                    st.stop()

                # GR baselines
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    gr_min_in = st.number_input(
                        "GR_min (clean sand)",
                        0.0, 200.0,
                        float(log_df['GR_API']
                              .quantile(0.05)),
                        1.0, key='gr_min_in',
                        help="Cleanest sand "
                             "baseline")
                with col_g2:
                    gr_max_in = st.number_input(
                        "GR_max (pure shale)",
                        50.0, 300.0,
                        float(log_df['GR_API']
                              .quantile(0.95)),
                        1.0, key='gr_max_in',
                        help="Purest shale "
                             "baseline")

                gr_cutoff = st.number_input(
                    "GR Sand Cutoff (API)",
                    30.0, 150.0, 75.0, 1.0,
                    key='gr_cutoff_in',
                    help="Below = sand, "
                         "Above = shale")

                # Calculations
                mean_gr = res_zone['GR_API'].mean()
                vsh_calc = calc_vsh_larionov(
                    mean_gr, gr_min_in, gr_max_in)

                sand_ct = (res_zone['GR_API'] <
                           gr_cutoff).sum()
                total_ct = len(res_zone)
                ntg_calc = (sand_ct / total_ct
                            if total_ct > 0
                            else 0.5)
                sci_calc = calc_sci(ntg_calc)

                # Porosity from density
                phi_calc = 0.25
                if 'RHOB_GCC' in log_df.columns:
                    sand_zone = res_zone[
                        res_zone['GR_API'] <
                        gr_cutoff]
                    if len(sand_zone) > 0:
                        mean_rhob = sand_zone[
                            'RHOB_GCC'].mean()
                        phi_raw = (
                            (2.65 - mean_rhob) /
                            (2.65 - 1.0))
                        phi_calc = round(
                            max(0.05,
                                min(0.35,
                                    phi_raw)), 3)

                st.markdown(
                    "**Calculated Parameters:**")
                col_c1, col_c2, col_c3, col_c4 = \
                    st.columns(4)
                col_c1.metric(
                    "Mean GR",
                    f"{mean_gr:.1f} API")
                col_c2.metric("Vsh", f"{vsh_calc}")
                col_c3.metric("NTG",
                              f"{ntg_calc:.3f}")
                col_c4.metric(
                    "SCI ★", f"{sci_calc}")

                col_c5, col_c6 = st.columns(2)
                col_c5.metric(
                    "Porosity (density)",
                    f"{phi_calc}")
                col_c6.metric(
                    "Analysis Depth (avg)",
                    f"{(res_top+res_base)/2:.0f} ft")

                # ── AUTO-FILL BUTTON ────────
                st.markdown("---")

                if st.button(
                        "🚀 AUTO-FILL SIDEBAR "
                        "WITH THESE VALUES",
                        type='primary',
                        use_container_width=True,
                        key='autofill_btn'):

                    st.session_state[
                        'auto_vals'] = {
                        'depth': (res_top +
                                  res_base) / 2,
                        'h': h_calc,
                        'hp': hp_calc,
                        'phi': phi_calc,
                        'vsh': vsh_calc,
                        'sci': sci_calc,
                    }
                    st.success(
                        "✅ Sidebar auto-filled. "
                        "Go to Prediction tab.")

            # ── Log preview ───────────────────
            with st.expander("📊 View Log Preview"):
                st.dataframe(
                    log_df.head(20),
                    hide_index=True,
                    use_container_width=True)

            # ── Log visualization ─────────────
            with st.expander(
                    "📈 Log Visualization"):
                fig_log = go.Figure()
                fig_log.add_trace(go.Scatter(
                    x=log_df['GR_API'],
                    y=log_df['DEPTH_FT'],
                    mode='lines', name='GR',
                    line=dict(
                        color='#e74c3c',
                        width=1.5)))

                # Show current boundary settings
                try:
                    fig_log.add_hline(
                        y=res_top,
                        line_dash="dash",
                        line_color="lime",
                        annotation_text="Res Top")
                    fig_log.add_hline(
                        y=res_base,
                        line_dash="dash",
                        line_color="lime",
                        annotation_text="Res Base")
                    fig_log.add_hline(
                        y=owc, line_dash="dash",
                        line_color="cyan",
                        annotation_text="OWC")
                    fig_log.add_vline(
                        x=gr_cutoff,
                        line_dash="dash",
                        line_color="yellow",
                        annotation_text=
                            f"Cutoff {gr_cutoff:.0f}")
                except:
                    pass

                fig_log.update_yaxes(
                    autorange="reversed")

                title = (f"Gamma Ray Log — "
                         f"{meta['well_id']}"
                         if meta['well_id'] !=
                            'Unknown'
                         else "Gamma Ray Log")

                fig_log.update_layout(
                    title=title,
                    xaxis_title="GR (API)",
                    yaxis_title="Depth (ft)",
                    height=600,
                    plot_bgcolor='#0e1621',
                    paper_bgcolor='#0e1621',
                    font=dict(color='white'))
                st.plotly_chart(
                    fig_log,
                    use_container_width=True)

        except Exception as e:
            st.error(f"Error processing log: {e}")
            st.info(
                "Common issues:\n"
                "- Missing DEPTH or GR column\n"
                "- Wrong file format\n"
                "- Corrupted CSV\n\n"
                "Ensure your CSV has at minimum "
                "a depth column and a gamma ray "
                "column.")

    # ── Process Excel ──────────────────────
    if excel_file is not None:
        try:
            xl = pd.ExcelFile(excel_file)
            st.success(
                f"✅ Excel loaded: "
                f"{len(xl.sheet_names)} sheets")

            with st.expander(
                    "📊 Excel Contents"):
                st.write("Available sheets:")
                for sheet in xl.sheet_names:
                    st.write(f"• {sheet}")

            st.session_state['excel_file'] = \
                excel_file

        except Exception as e:
            st.error(f"Error: {e}")

    if log_file is None and excel_file is None:
        st.info(
            "👆 Upload files above to enable "
            "auto-detection. Or skip to "
            "Prediction tab for manual entry.")

# ═══════════════════════════════════════════════════════════════════
# TAB 2 - PREDICTION
# ═══════════════════════════════════════════════════════════════════

with tab2:

    with st.sidebar:
        st.markdown("### ⚙️ Input Parameters")

        # Show auto-fill indicator
        if st.session_state['auto_vals']:
            st.success(
                "✅ Auto-filled from uploaded log")
        else:
            st.info("💡 Upload log CSV to auto-fill")

        # Get defaults from auto-fill or use standard
        av = st.session_state['auto_vals']

        st.markdown("**🪨 Rock Properties**")
        kh_mean = st.number_input(
            "kh — Horizontal Perm (md)",
            10.0, 5000.0, 1800.0, 10.0)
        kv_matrix = st.number_input(
            "kv — Vertical Perm (md)",
            1.0, 1000.0, 270.0, 5.0)
        phi_log = st.number_input(
            "φ — Porosity (fraction)",
            0.05, 0.45,
            av.get('phi', 0.28), 0.01)
        depth_ft = st.number_input(
            "Reservoir Depth (ft)",
            1000.0, 15000.0,
            av.get('depth', 8500.0), 100.0)
        h = st.number_input(
            "h — Oil Column (ft)",
            5.0, 300.0,
            float(av.get('h', 80.0)), 1.0)
        hp = st.number_input(
            "hp — Perforated (ft)",
            1.0, 200.0,
            float(av.get('hp', 25.0)), 1.0)

        st.markdown(
            "**🌍 Niger Delta Corrections**")
        Vsh = st.slider(
            "Vsh — Shale Volume",
            0.0, 0.5,
            float(av.get('vsh', 0.15)), 0.01)
        SCI = st.slider(
            "SCI — Shale Continuity ★",
            0.0, 1.0,
            float(av.get('sci', 0.50)), 0.05,
            help="Novel parameter this study")

        st.markdown("**🧪 Fluid Properties**")
        pvt_mode = st.radio(
            "PVT Source",
            ["Calculate from correlations",
             "Enter measured PVT"])

        API = st.number_input(
            "API Gravity (°)",
            15.0, 55.0, 35.0, 0.5)
        T_F = st.number_input(
            "Temperature (°F)",
            100.0, 300.0, 180.0, 5.0)
        Pi = st.number_input(
            "Initial Pressure (psia)",
            500.0, 10000.0, 4200.0, 50.0)
        sal = st.number_input(
            "Water Salinity (ppm)",
            1000.0, 150000.0, 35000.0, 1000.0)
        mu_w = st.number_input(
            "Water Viscosity (cp)",
            0.2, 1.5, 0.50, 0.05)

        if pvt_mode == "Calculate from correlations":
            Rs = st.number_input(
                "Rs (scf/STB)",
                50.0, 2000.0, 600.0, 10.0)
            gg = st.number_input(
                "γg Gas Gravity",
                0.5, 1.2, 0.75, 0.01)
            mu_o_meas = None
            Bo_meas = None
            Pb_meas = None
        else:
            mu_o_meas = st.number_input(
                "μo measured (cp)",
                0.1, 100.0, 0.6, 0.1)
            Bo_meas = st.number_input(
                "Bo measured (bbl/STB)",
                1.0, 3.0, 1.34, 0.01)
            Pb_meas = st.number_input(
                "Pb measured (psia)",
                100.0, 8000.0, 2463.0, 10.0)
            Rs = 600.0
            gg = 0.75

        st.markdown("**💧 Saturation**")
        krw = st.number_input(
            "krw at Sor", 0.1, 0.8, 0.35, 0.05)
        kro = st.number_input(
            "kro at Swc", 0.3, 1.0, 0.85, 0.05)

        st.markdown("**⚡ Production**")
        Qo = st.number_input(
            "Qo (STB/day)",
            100.0, 10000.0, 2000.0, 100.0)

        run_btn = st.button(
            "🔍 PREDICT BREAKTHROUGH",
            type="primary",
            use_container_width=True)

    # ─── Show auto-fill status in main area ──
    if st.session_state['auto_vals']:
        st.markdown(
            '<div class="autofill-card">'
            '<h3>✅ Sidebar Auto-Filled '
            'From Log</h3>'
            '<p>Values from your uploaded log '
            'have been transferred to the '
            'sidebar. Review and adjust as '
            'needed, then click PREDICT.</p>'
            '</div>',
            unsafe_allow_html=True)

    # ─── Run Prediction ─────────────────────
    if run_btn:
        if hp >= h:
            st.error("❌ hp must be less than h")
            st.stop()

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

        # ─── Original Sobocinski ─────────────
        Z1, tD1, tBT1, err1 = sobocinski(
            kh_mean, kv_matrix, phi_log,
            h, hp, mu_o, Bo, Qo,
            rw, ro, M, alpha)

        # ─── Enhanced Sobocinski ─────────────
        phi_eff = phi_log * np.exp(
            -0.000027 * depth_ft)
        kv_eff = kv_matrix * (1 - Vsh * SCI)

        Z2, tD2, tBT2, err2 = sobocinski(
            kh_mean, kv_eff, phi_eff,
            h, hp, mu_o, Bo, Qo,
            rw, ro, M, alpha)

        st.markdown('<div class="section-hdr">'
                    '📊 Prediction Results</div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Method 1 — "
                        "Original Sobocinski")
            st.caption("Baseline: no corrections")
            if err1:
                st.error(f"❌ {err1}")
            elif tBT1:
                r1 = risk_level(tBT1)
                st.markdown(
                    f'<div class="{r1["box"]}">'
                    f'<h2>{r1["icon"]} '
                    f'{r1["cat"]}</h2>'
                    f'<h3>{tBT1} days '
                    f'({tBT1/365:.2f} years)</h3>'
                    f'<p>{r1["action"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True)
                cA1, cA2 = st.columns(2)
                cA1.metric("Z", Z1)
                cA2.metric("(tD)BT", tD1)

        with col2:
            st.markdown("#### Method 2 — "
                        "Enhanced Sobocinski ★")
            st.caption("With SCI + Athy")
            if err2:
                st.error(f"❌ {err2}")
            elif tBT2:
                r2 = risk_level(tBT2)
                st.markdown(
                    f'<div class="{r2["box"]}">'
                    f'<h2>{r2["icon"]} '
                    f'{r2["cat"]}</h2>'
                    f'<h3>{tBT2} days '
                    f'({tBT2/365:.2f} years)</h3>'
                    f'<p>{r2["action"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True)
                cB1, cB2 = st.columns(2)
                cB1.metric("Z", Z2)
                cB2.metric("(tD)BT", tD2)

        if tBT1 and tBT2 and not err1 and not err2:
            st.divider()
            st.markdown(
                '<div class="section-hdr">'
                '📈 Enhancement Impact</div>',
                unsafe_allow_html=True)

            diff = tBT2 - tBT1
            pct = (diff / tBT1) * 100

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Original", f"{tBT1} days")
            m2.metric("Enhanced", f"{tBT2} days",
                       delta=f"{diff:+.1f}")
            m3.metric("Change",
                       f"{abs(diff):.1f} days")
            m4.metric("% Change",
                       f"{pct:+.1f}%")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Original", "Enhanced ★"],
                y=[tBT1, tBT2],
                marker_color=['#e74c3c',
                              '#3498db'],
                text=[f'{tBT1:.0f}d '
                      f'({tBT1/365:.1f}y)',
                      f'{tBT2:.0f}d '
                      f'({tBT2/365:.1f}y)'],
                textposition='outside',
                textfont=dict(size=13,
                              color='white')))
            fig.add_hline(y=365,
                          line_dash="dash",
                          line_color="red")
            fig.add_hline(y=730,
                          line_dash="dash",
                          line_color="orange")
            fig.update_layout(
                title="Original vs Enhanced",
                yaxis_title="BT (days)",
                height=400,
                plot_bgcolor='#0e1621',
                paper_bgcolor='#0e1621',
                font=dict(color='white'))
            st.plotly_chart(
                fig, use_container_width=True)

        with st.expander(
                "🔬 Intermediate Values"):
            cI1, cI2, cI3 = st.columns(3)
            with cI1:
                st.markdown(
                    "**Rock Corrections**")
                st.write(f"φ_original: {phi_log}")
                st.write(f"φ_effective: "
                         f"{phi_eff:.4f}")
                st.write(f"kv_original: "
                         f"{kv_matrix} md")
                st.write(f"kv_effective: "
                         f"{kv_eff:.2f} md")
                st.write(f"kh (unchanged): "
                         f"{kh_mean} md")

            with cI2:
                st.markdown(
                    "**Fluid Properties**")
                st.write(f"μo: {mu_o} cp")
                st.write(f"Bo: {Bo} bbl/STB")
                if pvt_mode == \
                   "Calculate from correlations":
                    st.write(f"μod: {mu_od} cp")
                    st.write(f"μob: {mu_ob} cp")
                    st.write(f"Pb: {Pb} psia")
                    st.write(f"Cond: {cond}")

            with cI3:
                st.markdown("**Densities**")
                st.write(f"ρw: {rw} lb/ft³")
                st.write(f"ρo: {ro} lb/ft³")
                st.write(f"Δρ: {rw-ro:.3f}")
                st.write(f"M: {M}")
                st.write(f"α: {alpha}")

        st.session_state['pred_params'] = {
            'kh': kh_mean, 'kv_mat': kv_matrix,
            'phi': phi_eff, 'h': h, 'hp': hp,
            'mu_o': mu_o, 'Bo': Bo, 'Qo': Qo,
            'rw': rw, 'ro': ro, 'M': M,
            'alpha': alpha, 'Vsh': Vsh}

    else:
        st.info("👈 Enter parameters or upload "
                "log for auto-fill, then click "
                "**PREDICT BREAKTHROUGH**")

# ═══════════════════════════════════════════════════════════════════
# TAB 3 - SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-hdr">'
                '📊 SCI Sensitivity Analysis '
                '★</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="info-card">'
        'Shows how the proposed <b>Shale '
        'Continuity Index (SCI)</b> affects '
        'breakthrough prediction. SCI ranges '
        'from 0 (discontinuous) to 1 '
        '(continuous shales).</div>',
        unsafe_allow_html=True)

    if 'pred_params' in st.session_state:
        p = st.session_state['pred_params']
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
                fig_sci.add_hrect(
                    y0=0, y1=365,
                    fillcolor="red",
                    opacity=0.1)
                fig_sci.add_hrect(
                    y0=365, y1=730,
                    fillcolor="orange",
                    opacity=0.1)
                fig_sci.add_hrect(
                    y0=730,
                    y1=sci_df['BT (days)'].max()*1.3,
                    fillcolor="green",
                    opacity=0.1)
                fig_sci.add_trace(go.Scatter(
                    x=sci_df['SCI'],
                    y=sci_df['BT (days)'],
                    mode='lines+markers',
                    line=dict(color='#3498db',
                              width=3),
                    marker=dict(size=12,
                                color='#3498db')))
                fig_sci.update_layout(
                    title="Effect of SCI on BT",
                    xaxis_title="SCI",
                    yaxis_title="BT (days)",
                    height=450,
                    plot_bgcolor='#0e1621',
                    paper_bgcolor='#0e1621',
                    font=dict(color='white'))
                st.plotly_chart(
                    fig_sci,
                    use_container_width=True)

            with col_s2:
                st.dataframe(
                    sci_df,
                    hide_index=True,
                    use_container_width=True)
    else:
        st.info("Run a prediction first")

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
    **Sobocinski-Cornelius (1965)** — Ahmed
    (2010) *Reservoir Engineering Handbook*,
    Equations 9-21 to 9-23.

    ---

    ### Enhancements (This Study)

    | Enhancement | Formula | Source |
    |---|---|---|
    | **SCI ★** | kv_eff = kv × (1 - Vsh × SCI) | This study |
    | Athy compaction | φ_eff = φ × exp(-Cf × d) | Athy (1930) |
    | Auto μo dead | Beal | Ahmed Eq 2-117 |
    | Auto μo sat | Beggs-Robinson | Ahmed Eq 2-121 |
    | Auto μo unsat | Vasquez-Beggs | Ahmed Eq 2-123 |
    | Auto Bo | Standing (1981) | Ahmed Eq 2-85 |
    | Bubble point | Standing (1947) | Ahmed Eq 2-72 |
    | ρw salinity | Niger Delta | Tuttle (1999) |
    | Vsh | Larionov (1969) | Tertiary rocks |

    ---

    ### Novel Contribution
    The **Shale Continuity Index (SCI)** is a
    proposed dimensionless parameter (0-1)
    characterizing lateral continuity of
    Agbada Formation shale intercalations.

    **Estimation:** SCI = 1 - NTG (from
    gamma ray log interpretation)

    ---

    ### Risk Classification

    | Category | BT |
    |---|---|
    | 🔴 HIGH | ≤ 365 days |
    | 🟡 MEDIUM | 366-730 days |
    | 🟢 LOW | > 730 days |

    ---

    ### Key References
    - Ahmed, T. (2010) *Reservoir Engineering
      Handbook*, 4th Ed.
    - Sobocinski & Cornelius (1965) SPE-894
    - Tuttle et al. (1999) USGS OFR 99-50-H
    - Doust & Omatsola (1990) AAPG Memoir 48
    - Short & Stauble (1967) AAPG Bulletin
    - Athy (1930) AAPG Bulletin
    - Larionov (1969) Vsh correlation
    - Standing (1947, 1981)
    - Beggs & Robinson (1975)

    ---

    ### University of Benin
    **Department of Petroleum Engineering**
    **Final Year Project**
    """)
