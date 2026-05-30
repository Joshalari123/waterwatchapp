# ═══════════════════════════════════════════════════
# WATERWATCH — Aquifer Water Breakthrough Early
# Warning System
# University of Benin — Final Year Project
# ═══════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIGURATION ──────────────────────────────
st.set_page_config(
    page_title="WaterWatch",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM STYLING ───────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a3a5c, #2980b9);
        padding: 25px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
    }
    .risk-high {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }
    .risk-medium {
        background: linear-gradient(135deg, #d35400, #e67e22);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }
    .risk-low {
        background: linear-gradient(135deg, #1e8449, #2ecc71);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #2980b9;
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
    }
    .section-header {
        color: #1a3a5c;
        font-size: 18px;
        font-weight: bold;
        border-bottom: 2px solid #2980b9;
        padding-bottom: 5px;
        margin: 15px 0;
    }
    .stButton button {
        background: linear-gradient(135deg, #1a3a5c, #2980b9);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODELS ──────────────────────────────────────
@st.cache_resource
def load_models():
    with open('waterwatch_classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
    with open('waterwatch_regressor.pkl', 'rb') as f:
        regressor = pickle.load(f)
    with open('waterwatch_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('waterwatch_features.json', 'r') as f:
        features = json.load(f)
    with open('waterwatch_importance.json', 'r') as f:
        importance = json.load(f)
    return classifier, regressor, scaler, features, importance

classifier, regressor, scaler, feature_cols, importance = load_models()

# ── HEADER ───────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>💧 WaterWatch</h1>
    <h3>Aquifer Water Breakthrough Early Warning System</h3>
    <p>Niger Delta Oil Reservoirs — University of Benin</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/oil-pump.png",
             width=80)
    st.markdown("### About WaterWatch")
    st.info("""
    WaterWatch predicts water breakthrough
    risk in Niger Delta oil wells using:

    - Static reservoir properties
    - Well test data (Day 1-7)
    - Early production signals (Day 7-30)

    **University of Benin**
    Department of Petroleum Engineering
    Final Year Project 2025/2026
    """)

    st.markdown("### Model Performance")
    st.metric("CV Accuracy", "84.9%", "+14.6%")
    st.metric("R² Score", "0.761", "+0.079")
    st.metric("High Risk Recall", "77%")
    st.metric("Training Wells", "347")

    st.markdown("### Quick Scenarios")
    scenario = st.selectbox(
        "Load example scenario:",
        ["Custom Input",
         "High Risk Well",
         "Medium Risk Well",
         "Low Risk Well"]
    )

# ── SCENARIO PRESETS ────────────────────────────────
scenarios = {
    "High Risk Well": {
        'permeability': 2500.0,
        'porosity': 0.28,
        'kv_kh_ratio': 0.25,
        'oil_column_height': 35.0,
        'reservoir_thickness': 80.0,
        'reservoir_radius': 3000.0,
        'depth': 8000.0,
        'perforation_depth': 0.85,
        'oil_viscosity': 8.0,
        'water_viscosity': 0.5,
        'oil_fvf': 1.35,
        'water_fvf': 1.02,
        'aquifer_radius_ratio': 8.0,
        'aquifer_type': 1,
        'aquifer_permeability': 2800.0,
        'production_rate': 4000.0,
        'initial_pressure': 4500.0,
        'total_compressibility': 0.000005,
        'gas_cap_ratio': 0.3,
        'mobility_ratio': 7.5,
        'dimensionless_time': 500000000.0,
        'water_influx': 800000.0,
        'productivity_index': 12.0,
        'vertical_permeability': 625.0,
        'initial_PI': 13.5,
        'skin_factor': 2.0,
        'pressure_response': 1.2,
        'flow_efficiency': 1.1,
        'initial_decline_rate': 0.003,
        'PI_trend': 0.04,
        'b_exponent': 0.90,
        'GOR_trend': 0.2,
        'cumulative_voidage': 2.5,
        'pressure_maintenance': 0.85
    },
    "Medium Risk Well": {
        'permeability': 1200.0,
        'porosity': 0.25,
        'kv_kh_ratio': 0.12,
        'oil_column_height': 75.0,
        'reservoir_thickness': 100.0,
        'reservoir_radius': 2500.0,
        'depth': 7000.0,
        'perforation_depth': 0.50,
        'oil_viscosity': 4.0,
        'water_viscosity': 0.5,
        'oil_fvf': 1.25,
        'water_fvf': 1.02,
        'aquifer_radius_ratio': 5.0,
        'aquifer_type': 1,
        'aquifer_permeability': 1100.0,
        'production_rate': 2000.0,
        'initial_pressure': 4000.0,
        'total_compressibility': 0.000005,
        'gas_cap_ratio': 0.5,
        'mobility_ratio': 4.0,
        'dimensionless_time': 300000000.0,
        'water_influx': 400000.0,
        'productivity_index': 7.0,
        'vertical_permeability': 144.0,
        'initial_PI': 7.5,
        'skin_factor': 3.0,
        'pressure_response': 0.8,
        'flow_efficiency': 1.0,
        'initial_decline_rate': 0.008,
        'PI_trend': 0.01,
        'b_exponent': 0.55,
        'GOR_trend': 0.8,
        'cumulative_voidage': 1.5,
        'pressure_maintenance': 0.60
    },
    "Low Risk Well": {
        'permeability': 300.0,
        'porosity': 0.22,
        'kv_kh_ratio': 0.05,
        'oil_column_height': 120.0,
        'reservoir_thickness': 150.0,
        'reservoir_radius': 2000.0,
        'depth': 9000.0,
        'perforation_depth': 0.30,
        'oil_viscosity': 1.5,
        'water_viscosity': 0.4,
        'oil_fvf': 1.45,
        'water_fvf': 1.01,
        'aquifer_radius_ratio': 2.5,
        'aquifer_type': 0,
        'aquifer_permeability': 280.0,
        'production_rate': 800.0,
        'initial_pressure': 5500.0,
        'total_compressibility': 0.000003,
        'gas_cap_ratio': 1.2,
        'mobility_ratio': 1.2,
        'dimensionless_time': 100000000.0,
        'water_influx': 50000.0,
        'productivity_index': 2.5,
        'vertical_permeability': 15.0,
        'initial_PI': 2.8,
        'skin_factor': 1.0,
        'pressure_response': 0.3,
        'flow_efficiency': 0.95,
        'initial_decline_rate': 0.020,
        'PI_trend': -0.01,
        'b_exponent': 0.25,
        'GOR_trend': 1.5,
        'cumulative_voidage': 0.5,
        'pressure_maintenance': 0.35
    }
}

# Get default values
if scenario != "Custom Input":
    defaults = scenarios[scenario]
else:
    defaults = scenarios["Medium Risk Well"]

# ── MAIN INPUT FORM ──────────────────────────────────
st.markdown("## 📋 Well & Reservoir Input Parameters")

tab1, tab2, tab3 = st.tabs([
    "🪨 Static Reservoir Properties",
    "🧪 Fluid Properties",
    "📈 Early Production Signals"
])

with tab1:
    st.markdown('<p class="section-header">Rock & Geometry Properties</p>',
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        permeability = st.number_input(
            "Permeability (md)",
            min_value=10.0, max_value=5000.0,
            value=defaults['permeability'],
            help="Horizontal permeability from core/logs"
        )
        porosity = st.number_input(
            "Porosity (fraction)",
            min_value=0.05, max_value=0.45,
            value=defaults['porosity'],
            step=0.01,
            help="Effective porosity fraction"
        )
        kv_kh_ratio = st.number_input(
            "kv/kh Ratio",
            min_value=0.001, max_value=0.5,
            value=defaults['kv_kh_ratio'],
            step=0.01,
            help="Vertical to horizontal permeability ratio"
        )
        oil_column_height = st.number_input(
            "Oil Column Height (ft)",
            min_value=10.0, max_value=300.0,
            value=defaults['oil_column_height'],
            help="Height of oil zone above OWC"
        )

    with col2:
        reservoir_thickness = st.number_input(
            "Reservoir Thickness (ft)",
            min_value=10.0, max_value=500.0,
            value=defaults['reservoir_thickness'],
            help="Net pay thickness"
        )
        reservoir_radius = st.number_input(
            "Reservoir Radius (ft)",
            min_value=500.0, max_value=8000.0,
            value=defaults['reservoir_radius'],
            help="Drainage radius"
        )
        depth = st.number_input(
            "Reservoir Depth (ft)",
            min_value=2000.0, max_value=15000.0,
            value=defaults['depth'],
            help="True vertical depth"
        )
        perforation_depth = st.number_input(
            "Perforation Position (0=bottom, 1=top)",
            min_value=0.1, max_value=0.99,
            value=defaults['perforation_depth'],
            step=0.05,
            help="Relative perforation position in reservoir"
        )

    with col3:
        aquifer_radius_ratio = st.number_input(
            "Aquifer Radius Ratio",
            min_value=1.0, max_value=15.0,
            value=defaults['aquifer_radius_ratio'],
            step=0.5,
            help="Ratio of aquifer to reservoir radius"
        )
        aquifer_type = st.selectbox(
            "Aquifer Type",
            options=[0, 1],
            index=defaults['aquifer_type'],
            format_func=lambda x: "Edge Water Drive" if x==0
                                  else "Bottom Water Drive",
            help="Type of natural aquifer support"
        )
        gas_cap_ratio = st.number_input(
            "Gas Cap Ratio (m)",
            min_value=0.0, max_value=3.0,
            value=defaults['gas_cap_ratio'],
            step=0.1,
            help="Gas cap to oil zone volume ratio"
        )
        production_rate = st.number_input(
            "Production Rate (bbl/day)",
            min_value=100.0, max_value=8000.0,
            value=defaults['production_rate'],
            help="Current production rate"
        )

with tab2:
    st.markdown('<p class="section-header">Fluid & PVT Properties</p>',
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        oil_viscosity = st.number_input(
            "Oil Viscosity (cp)",
            min_value=0.3, max_value=20.0,
            value=defaults['oil_viscosity'],
            step=0.1,
            help="Dead oil viscosity at reservoir conditions"
        )
        water_viscosity = st.number_input(
            "Water Viscosity (cp)",
            min_value=0.2, max_value=1.5,
            value=defaults['water_viscosity'],
            step=0.05,
            help="Formation water viscosity"
        )

    with col2:
        oil_fvf = st.number_input(
            "Oil FVF Bo (bbl/STB)",
            min_value=1.05, max_value=2.5,
            value=defaults['oil_fvf'],
            step=0.05,
            help="Oil formation volume factor"
        )
        water_fvf = st.number_input(
            "Water FVF Bw (bbl/STB)",
            min_value=1.0, max_value=1.1,
            value=defaults['water_fvf'],
            step=0.005,
            help="Water formation volume factor"
        )

    with col3:
        initial_pressure = st.number_input(
            "Initial Reservoir Pressure (psi)",
            min_value=500.0, max_value=10000.0,
            value=defaults['initial_pressure'],
            help="Initial static reservoir pressure"
        )
        total_compressibility = st.number_input(
            "Total Compressibility (psi⁻¹)",
            min_value=0.000001, max_value=0.00005,
            value=defaults['total_compressibility'],
            step=0.000001,
            format="%.6f",
            help="Total system compressibility"
        )

with tab3:
    st.markdown('<p class="section-header">Well Test Data (Day 1-7)</p>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        initial_PI = st.number_input(
            "Initial Productivity Index",
            min_value=0.1, max_value=200.0,
            value=defaults['initial_PI'],
            step=0.5,
            help="PI from first flow test (bbl/day/psi)"
        )
        skin_factor = st.number_input(
            "Skin Factor",
            min_value=-5.0, max_value=20.0,
            value=defaults['skin_factor'],
            step=0.5,
            help="Formation damage/stimulation factor"
        )
        flow_efficiency = st.number_input(
            "Flow Efficiency",
            min_value=0.3, max_value=1.5,
            value=defaults['flow_efficiency'],
            step=0.05,
            help="Actual PI / Theoretical PI"
        )

    with col2:
        pressure_response = st.number_input(
            "Pressure Response Rate (psi/hr)",
            min_value=0.1, max_value=50.0,
            value=defaults['pressure_response'],
            step=0.1,
            help="Rate of pressure drop when well opens"
        )
        initial_decline_rate = st.number_input(
            "Initial Decline Rate (fraction/day)",
            min_value=0.0001, max_value=0.1,
            value=defaults['initial_decline_rate'],
            step=0.001,
            format="%.4f",
            help="Rate decline in first week"
        )

    st.markdown('<p class="section-header">Early Production Signals (Day 7-30)</p>',
                unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        PI_trend = st.number_input(
            "PI Trend (fraction/day)",
            min_value=-0.1, max_value=0.1,
            value=defaults['PI_trend'],
            step=0.005,
            format="%.4f",
            help="Rate of PI change (+ve = improving = water signal)"
        )
        b_exponent = st.number_input(
            "Arps b Exponent",
            min_value=0.0, max_value=1.5,
            value=defaults['b_exponent'],
            step=0.05,
            help="Decline curve exponent (higher = stronger aquifer)"
        )

    with col2:
        GOR_trend = st.number_input(
            "GOR Trend (scf/bbl/day)",
            min_value=-2.0, max_value=5.0,
            value=defaults['GOR_trend'],
            step=0.1,
            help="Rate of GOR change (stable = aquifer active)"
        )
        pressure_maintenance = st.number_input(
            "Pressure Maintenance Index",
            min_value=0.1, max_value=1.0,
            value=defaults['pressure_maintenance'],
            step=0.05,
            help="Ratio of actual to expected pressure (1=fully maintained)"
        )

    with col3:
        cumulative_voidage = st.number_input(
            "Cumulative Voidage Ratio",
            min_value=0.0, max_value=5.0,
            value=defaults['cumulative_voidage'],
            step=0.1,
            help="Aquifer influx / produced volume ratio"
        )

# ── CALCULATE DERIVED FEATURES ──────────────────────
mobility_ratio = (0.3 / water_viscosity) / (0.8 / oil_viscosity)
vertical_permeability = permeability * kv_kh_ratio
aquifer_permeability  = permeability * 1.1
t_hours = 90 * 24
dimensionless_time = (0.000264 * permeability * t_hours) / \
                     (porosity * oil_viscosity * \
                      total_compressibility * 0.35**2)
aquifer_constant = (1.119 * porosity * total_compressibility * \
                    reservoir_radius**2 * reservoir_thickness)
pressure_drawdown = production_rate * oil_viscosity * \
                    np.log(reservoir_radius / 0.35) / \
                    (0.00708 * permeability * reservoir_thickness)
water_influx = aquifer_constant * pressure_drawdown * \
               aquifer_radius_ratio
productivity_index = (0.00708 * permeability * \
                      reservoir_thickness) / \
                     (oil_viscosity * \
                      np.log(reservoir_radius / 0.35))

# ── BUILD INPUT DATAFRAME ────────────────────────────
input_data = {
    'permeability':           permeability,
    'porosity':               porosity,
    'kv_kh_ratio':            kv_kh_ratio,
    'vertical_permeability':  vertical_permeability,
    'oil_column_height':      oil_column_height,
    'reservoir_thickness':    reservoir_thickness,
    'reservoir_radius':       reservoir_radius,
    'depth':                  depth,
    'perforation_depth':      perforation_depth,
    'oil_viscosity':          oil_viscosity,
    'water_viscosity':        water_viscosity,
    'oil_fvf':                oil_fvf,
    'water_fvf':              water_fvf,
    'aquifer_radius_ratio':   aquifer_radius_ratio,
    'aquifer_type':           aquifer_type,
    'aquifer_permeability':   aquifer_permeability,
    'production_rate':        production_rate,
    'initial_pressure':       initial_pressure,
    'total_compressibility':  total_compressibility,
    'gas_cap_ratio':          gas_cap_ratio,
    'mobility_ratio':         mobility_ratio,
    'dimensionless_time':     dimensionless_time,
    'water_influx':           water_influx,
    'productivity_index':     productivity_index,
    'initial_PI':             initial_PI,
    'skin_factor':            skin_factor,
    'pressure_response':      pressure_response,
    'flow_efficiency':        flow_efficiency,
    'initial_decline_rate':   initial_decline_rate,
    'PI_trend':               PI_trend,
    'b_exponent':             b_exponent,
    'GOR_trend':              GOR_trend,
    'cumulative_voidage':     cumulative_voidage,
    'pressure_maintenance':   pressure_maintenance
}

# ── PREDICT BUTTON ───────────────────────────────────
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button(
        "🔍 RUN WATERWATCH PREDICTION",
        use_container_width=True
    )

# ── PREDICTION & RESULTS ─────────────────────────────
if predict_button:

    # Prepare input
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_cols]
    input_scaled = scaler.transform(input_df)

    # Make predictions
    risk_pred  = classifier.predict(input_scaled)[0]
    risk_proba = classifier.predict_proba(input_scaled)[0]
    bt_pred    = regressor.predict(input_scaled)[0]
    bt_pred    = max(120, bt_pred)

    # Risk labels
    risk_labels = {
        0: ("🟢 LOW RISK", "risk-low",
            "Breakthrough expected beyond 2 years",
            "Monitor normally. No immediate action required."),
        1: ("🟡 MEDIUM RISK", "risk-medium",
            "Breakthrough expected within 1-2 years",
            "Increase monitoring frequency. Plan water handling facilities."),
        2: ("🔴 HIGH RISK", "risk-high",
            "Breakthrough expected within 1 year",
            "Immediate action recommended. Review production rate.")
    }

    label, css_class, description, action = risk_labels[risk_pred]

    st.markdown("---")
    st.markdown("## 🎯 WaterWatch Prediction Results")

    # ── RISK RESULT ──────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f"""
        <div class="{css_class}">
            <h2>{label}</h2>
            <p>{description}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### ⚡ Recommended Action")
        st.warning(action)

    with col2:
        # Breakthrough time metrics
        st.markdown("### 📊 Breakthrough Time Estimate")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(
                "Predicted Breakthrough",
                f"{bt_pred:.0f} days",
                f"{bt_pred/365:.1f} years"
            )
        with col_b:
            st.metric(
                "Lead Time Available",
                f"{max(0, bt_pred-30):.0f} days",
                "time to act"
            )

        # Confidence gauge
        confidence = max(risk_proba) * 100
        st.markdown(f"### 🎯 Model Confidence: {confidence:.1f}%")
        st.progress(int(confidence))

    # ── PROBABILITY CHART ────────────────────────────
    st.markdown("### 📊 Risk Probability Distribution")
    col1, col2 = st.columns([1, 1])

    with col1:
        fig_prob = go.Figure(go.Bar(
            x=['🟢 Low Risk\n(>2 years)',
               '🟡 Medium Risk\n(1-2 years)',
               '🔴 High Risk\n(<1 year)'],
            y=[risk_proba[0]*100,
               risk_proba[1]*100,
               risk_proba[2]*100],
            marker_color=['#2ecc71', '#e67e22', '#e74c3c'],
            text=[f'{p*100:.1f}%' for p in risk_proba],
            textposition='auto'
        ))
        fig_prob.update_layout(
            title="Breakthrough Risk Probabilities",
            yaxis_title="Probability (%)",
            yaxis_range=[0, 100],
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    with col2:
        # Timeline visualization
        bt_years = bt_pred / 365
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=[0, bt_years],
            y=[1, 1],
            mode='lines',
            line=dict(color='lightgray', width=8),
            showlegend=False
        ))
        fig_time.add_trace(go.Scatter(
            x=[0],
            y=[1],
            mode='markers+text',
            marker=dict(size=20, color='#2980b9',
                       symbol='diamond'),
            text=['NOW'],
            textposition='top center',
            name='Current',
            showlegend=False
        ))
        fig_time.add_trace(go.Scatter(
            x=[bt_years],
            y=[1],
            mode='markers+text',
            marker=dict(size=20, color='#e74c3c',
                       symbol='x'),
            text=[f'BT\n~{bt_years:.1f}yr'],
            textposition='top center',
            name='Breakthrough',
            showlegend=False
        ))
        fig_time.update_layout(
            title="Predicted Breakthrough Timeline",
            xaxis_title="Years from Now",
            height=350,
            yaxis=dict(visible=False),
            xaxis_range=[-0.2, bt_years + 0.5]
        )
        st.plotly_chart(fig_time, use_container_width=True)

    # ── KEY RISK FACTORS ────────────────────────────
    st.markdown("### 🔬 Key Risk Factors (SHAP-Based)")
    top_features = sorted(
        importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    factor_names  = [f[0].replace('_', ' ').title()
                     for f in top_features]
    factor_values = [f[1] for f in top_features]
    factor_colors = []
    accelerate = ['mobility ratio', 'production rate',
                  'aquifer radius ratio', 'kv kh ratio',
                  'pi trend', 'b exponent',
                  'pressure maintenance', 'initial pi']
    for name in factor_names:
        if name.lower() in accelerate:
            factor_colors.append('#e74c3c')
        else:
            factor_colors.append('#2ecc71')

    fig_imp = go.Figure(go.Bar(
        x=factor_values,
        y=factor_names,
        orientation='h',
        marker_color=factor_colors,
        text=[f'{v:.1f}' for v in factor_values],
        textposition='auto'
    ))
    fig_imp.update_layout(
        title="Feature Importance — What Drives This Prediction",
        xaxis_title="SHAP Importance Score",
        height=400,
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # ── WELL SUMMARY TABLE ───────────────────────────
    st.markdown("### 📋 Well Input Summary")
    summary_data = {
        'Parameter': [
            'Permeability', 'Porosity',
            'Oil Column Height', 'Mobility Ratio',
            'Aquifer Strength', 'Production Rate',
            'Aquifer Type', 'b Exponent',
            'PI Trend', 'Pressure Maintenance'
        ],
        'Value': [
            f'{permeability:.0f} md',
            f'{porosity:.3f}',
            f'{oil_column_height:.1f} ft',
            f'{mobility_ratio:.2f}',
            f'{aquifer_radius_ratio:.1f}',
            f'{production_rate:.0f} bbl/day',
            'Bottom Water' if aquifer_type==1 else 'Edge Water',
            f'{b_exponent:.2f}',
            f'{PI_trend:.4f}',
            f'{pressure_maintenance:.2f}'
        ],
        'Risk Implication': [
            '⚠️ High' if permeability > 1500 else '✅ Moderate',
            '✅ Good' if porosity > 0.20 else '⚠️ Low',
            '⚠️ Thin' if oil_column_height < 50 else '✅ Adequate',
            '⚠️ Unfavorable' if mobility_ratio > 5 else '✅ Favorable',
            '⚠️ Strong' if aquifer_radius_ratio > 6 else '✅ Moderate',
            '⚠️ High' if production_rate > 3000 else '✅ Conservative',
            '⚠️ Faster BT' if aquifer_type==1 else '✅ Slower BT',
            '⚠️ Active Aquifer' if b_exponent > 0.7 else '✅ Weak Aquifer',
            '⚠️ Water Signal' if PI_trend > 0.02 else '✅ Normal',
            '⚠️ Aquifer Active' if pressure_maintenance > 0.7 else '✅ Depleting'
        ]
    }
    st.dataframe(
        pd.DataFrame(summary_data),
        use_container_width=True,
        hide_index=True
    )

    # ── FOOTER NOTE ──────────────────────────────────
    st.markdown("---")
    st.caption("""
    ⚠️ Disclaimer: WaterWatch predictions are based on
    synthetic Niger Delta reservoir data and machine
    learning models. Results should be used as a
    screening tool alongside conventional reservoir
    engineering analysis. Not for standalone field
    decisions without expert validation.

    📚 University of Benin — Petroleum Engineering
    Department — Final Year Project 2025/2026
    """)

else:
    # ── WELCOME SCREEN ───────────────────────────────
    st.markdown("### 👋 Welcome to WaterWatch")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        **Step 1 — Select Scenario**
        Choose a preset scenario from
        the sidebar or enter custom
        well parameters
        """)
    with col2:
        st.info("""
        **Step 2 — Enter Parameters**
        Fill in reservoir properties,
        fluid data, and early production
        signals across the three tabs
        """)
    with col3:
        st.info("""
        **Step 3 — Run Prediction**
        Click the prediction button
        to get risk category,
        breakthrough time, and
        recommended actions
        """)

    st.markdown("### 📊 Model Performance Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CV Accuracy",     "84.9%", "5-fold validated")
    c2.metric("R² Score",        "0.761", "Regression")
    c3.metric("High Risk Recall","77%",   "Critical metric")
    c4.metric("Training Wells",  "347",   "Niger Delta")
