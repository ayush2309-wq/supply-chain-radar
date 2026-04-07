import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium
import sys, os, logging

logging.basicConfig(level="INFO")
sys.path.append(os.path.abspath("."))

from src.models.risk_classifier import RiskClassifier
from src.models.forecaster import RiskForecaster
from src.ingestion.geo_fetcher import GeoFetcher
from src.processing.feature_store import FeatureStore

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Disruption Radar",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid;
        margin: 5px;
    }
    .high-risk   { border-color: #ff4b4b; }
    .medium-risk { border-color: #ffa500; }
    .low-risk    { border-color: #00cc88; }
    .risk-score  { font-size: 2.5rem; font-weight: 700; }
    .stMetric { background: #1e2130; border-radius: 10px; padding: 10px; }
    h1 { color: #ffffff; font-size: 2.2rem; }
    h2 { color: #c9d1d9; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def risk_color(label):
    return {"HIGH": "#ff4b4b", "MEDIUM": "#ffa500", "LOW": "#00cc88"}.get(str(label), "#888")

def risk_emoji(label):
    return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(str(label), "⚪")

COUNTRY_NAMES = {
    "CN": "China", "US": "United States", "DE": "Germany",
    "IN": "India", "TW": "Taiwan", "SA": "Saudi Arabia",
    "RU": "Russia", "UA": "Ukraine",
}

COUNTRY_COORDS = {
    "CN": (35.86, 104.19), "US": (37.09, -95.71),
    "DE": (51.16, 10.45),  "IN": (20.59, 78.96),
    "TW": (23.69, 120.96), "SA": (23.88, 45.07),
    "RU": (61.52, 105.31), "UA": (48.37, 31.16),
}


# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_feature_matrix():
    df = pd.read_csv("data/features/feature_matrix.csv")
    df["date"] = pd.to_datetime(df["date"])
    
    # FIX: Robust conversion for volatility flags and boolean-like strings
    for col in df.columns:
        if "volatility_flag" in col or df[col].dtype == object:
            # We map string representations to floats and fill unknowns with 0.0
            df[col] = df[col].astype(str).map({
                'True': 1.0, '1.0': 1.0, '1': 1.0,
                'False': 0.0, '0.0': 0.0, '0': 0.0
            }).fillna(df[col]) # Keep original if not a boolean string
            
    return df


@st.cache_resource
def load_classifier():
    clf = RiskClassifier()
    clf.load()
    return clf


@st.cache_data(ttl=3600)
def load_predictions(_clf, df):
    pred_df = df.copy()
    
    # FIX: Ensure all features are numeric before passing to XGBoost/LightGBM
    # Identify non-feature columns to exclude from numeric conversion
    exclude = ["date", "fetched_at", "risk_label"] 
    feature_cols = [c for c in pred_df.columns if c not in exclude]
    
    for col in feature_cols:
        # errors='coerce' turns unparseable strings into NaN, fillna(0) makes it a float
        pred_df[col] = pd.to_numeric(pred_df[col], errors='coerce').fillna(0.0)
        
    return _clf.predict(pred_df)


@st.cache_data(ttl=3600)
def load_forecast(df):
    forecaster = RiskForecaster(forecast_days=28)
    forecaster.train(df)
    forecaster.forecast()
    return forecaster.build_risk_forecast()


@st.cache_data(ttl=3600)
def load_geo():
    fetcher  = GeoFetcher(regions=["CN", "US", "DE", "IN", "TW", "SA", "RU", "UA"])
    records  = fetcher.fetch()
    features = FeatureStore.build_geo_features(records)
    return features


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/radar.png", width=80)
    st.title("⚙️ Controls")
    st.markdown("---")

    page = st.radio("Navigate", [
        "🏠 Overview",
        "📈 Risk Analysis",
        "🔮 Forecast",
        "🗺️ Geo Risk Map",
        "📦 Commodities",
    ])

    st.markdown("---")
    st.markdown("**Model Info**")
    st.info("XGBoost + LightGBM Ensemble\n\nAccuracy: 83.4%")

    st.markdown("**Data Sources**")
    st.success("✅ NewsData.io\n✅ Yahoo Finance\n✅ Geo Risk Index")

    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ── Load Everything ───────────────────────────────────────────────────────────
try:
    df          = load_feature_matrix()
    clf         = load_classifier()
    predictions = load_predictions(clf, df)
    forecast    = load_forecast(df)
    geo         = load_geo()
    latest      = predictions.iloc[-1]
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🌐 Supply Chain Disruption Radar")
    st.markdown(f"*Real-time risk intelligence — Last updated: {df['date'].max().strftime('%B %d, %Y')}*")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        label = str(latest["risk_label"])
        color = risk_color(label)
        css_class = "high-risk" if label == "HIGH" else "medium-risk" if label == "MEDIUM" else "low-risk"
        st.markdown(f"""
        <div class='metric-card {css_class}'>
            <p style='color:#888;margin:0'>Current Risk Level</p>
            <p class='risk-score' style='color:{color}'>{risk_emoji(label)} {label}</p>
        </div>""", unsafe_allow_html=True)

    with col2:
        delta = latest['risk_score'] - predictions.iloc[-2]['risk_score']
        st.metric("Risk Score", f"{latest['risk_score']:.2f}%", delta=f"{delta:.2f}%")

    with col3:
        high_days = (predictions["risk_label"] == "HIGH").sum()
        st.metric("HIGH Risk Days", f"{high_days}", delta="Last 30 days")

    with col4:
        avg_forecast = forecast["risk_score"].mean()
        st.metric("Avg Forecast Risk", f"{avg_forecast:.1f}", delta="28-day outlook")

    st.markdown("---")

    st.subheader("📊 Historical Risk Timeline")
    colors = [risk_color(l) for l in predictions["risk_label"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=predictions["date"],
        y=predictions["risk_score"],
        marker_color=colors,
        name="Risk Score",
        hovertemplate="<b>%{x}</b><br>Risk Score: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=predictions["date"],
        y=predictions["risk_score"].rolling(3, min_periods=1).mean(),
        mode="lines",
        line=dict(color="#ffffff", width=2, dash="dot"),
        name="3-day avg",
    ))
    fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#c9d1d9"),
        xaxis=dict(gridcolor="#1e2130"),
        yaxis=dict(gridcolor="#1e2130", title="Risk Score (%)"),
        legend=dict(bgcolor="#1e2130"),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔮 28-Day Risk Forecast Preview")
    fc_colors = [risk_color(l) for l in forecast["risk_label"]]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=forecast["date"],
        y=forecast["risk_score"],
        mode="lines+markers",
        line=dict(color="#7c83fd", width=3),
        marker=dict(color=fc_colors, size=10, line=dict(color="#fff", width=1)),
        hovertemplate="<b>%{x}</b><br>Risk: %{y:.1f}<extra></extra>",
        name="Forecast",
    ))
    fig2.add_hrect(y0=66, y1=100, fillcolor="#ff4b4b", opacity=0.1, line_width=0)
    fig2.add_hrect(y0=33, y1=66,  fillcolor="#ffa500", opacity=0.1, line_width=0)
    fig2.add_hrect(y0=0,  y1=33,  fillcolor="#00cc88", opacity=0.1, line_width=0)
    fig2.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#c9d1d9"),
        xaxis=dict(gridcolor="#1e2130"),
        yaxis=dict(gridcolor="#1e2130", title="Risk Score"),
        height=300,
    )
    st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RISK ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📈 Risk Analysis":
    st.title("📈 Risk Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        risk_counts = predictions["risk_label"].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map={"HIGH": "#ff4b4b", "MEDIUM": "#ffa500", "LOW": "#00cc88"},
            title="Risk Label Distribution",
            hole=0.4,
        )
        fig.update_layout(paper_bgcolor="#0e1117", font=dict(color="#c9d1d9"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=predictions["date"], y=predictions["prob_high"],
            fill="tozeroy", name="HIGH",
            line=dict(color="#ff4b4b"),
            fillcolor="rgba(255,75,75,0.3)",
        ))
        fig2.add_trace(go.Scatter(
            x=predictions["date"], y=predictions["prob_medium"],
            fill="tozeroy", name="MEDIUM",
            line=dict(color="#ffa500"),
            fillcolor="rgba(255,165,0,0.3)",
        ))
        fig2.update_layout(
            title="Risk Probability Over Time",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="#c9d1d9"),
            xaxis=dict(gridcolor="#1e2130"),
            yaxis=dict(gridcolor="#1e2130"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📋 Full Risk Table")
    styled = predictions.copy()
    styled["risk_label"] = styled["risk_label"].apply(lambda x: f"{risk_emoji(x)} {x}")
    st.dataframe(
        styled[["date", "risk_label", "risk_score", "prob_low", "prob_medium", "prob_high"]],
        use_container_width=True,
        hide_index=True,
    )


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — FORECAST
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Forecast":
    st.title("🔮 28-Day Risk Forecast")
    st.markdown("*Powered by Facebook Prophet time series model*")
    st.markdown("---")

    fc_colors = [risk_color(l) for l in forecast["risk_label"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast["date"],
        y=forecast["risk_score"],
        mode="lines+markers",
        line=dict(color="#7c83fd", width=3),
        marker=dict(color=fc_colors, size=12, line=dict(color="#fff", width=2)),
        name="Forecast Risk",
        hovertemplate="<b>%{x}</b><br>Risk Score: %{y:.1f}<extra></extra>",
    ))
    fig.add_hrect(y0=66, y1=100, fillcolor="#ff4b4b", opacity=0.15, line_width=0,
                  annotation_text="HIGH RISK ZONE", annotation_position="top left",
                  annotation_font_color="#ff4b4b")
    fig.add_hrect(y0=33, y1=66, fillcolor="#ffa500", opacity=0.10, line_width=0,
                  annotation_text="MEDIUM RISK ZONE", annotation_position="top left",
                  annotation_font_color="#ffa500")
    fig.add_hrect(y0=0, y1=33, fillcolor="#00cc88", opacity=0.10, line_width=0,
                  annotation_text="LOW RISK ZONE", annotation_position="top left",
                  annotation_font_color="#00cc88")
    fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#c9d1d9"),
        xaxis=dict(gridcolor="#1e2130", title="Date"),
        yaxis=dict(gridcolor="#1e2130", title="Risk Score (0-100)"),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Forecast Table")
        fc_display = forecast.copy()
        fc_display["date"] = fc_display["date"].dt.strftime("%b %d, %Y")
        fc_display["risk_label"] = fc_display["risk_label"].apply(
            lambda x: f"{risk_emoji(str(x))} {x}"
        )
        st.dataframe(fc_display, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("📊 Forecast Summary")
        high_count   = (forecast["risk_label"] == "HIGH").sum()
        medium_count = (forecast["risk_label"] == "MEDIUM").sum()
        low_count    = (forecast["risk_label"] == "LOW").sum()
        peak_date    = forecast.loc[forecast["risk_score"].idxmax(), "date"]

        st.metric("🔴 HIGH Risk Days",   high_count)
        st.metric("🟡 MEDIUM Risk Days", medium_count)
        st.metric("🟢 LOW Risk Days",    low_count)
        st.metric("Peak Risk Score",     f"{forecast['risk_score'].max():.1f}")
        st.metric("Peak Risk Date",      peak_date.strftime("%b %d, %Y"))


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — GEO RISK MAP
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Geo Risk Map":
    st.title("🗺️ Geopolitical Risk Map")
    st.markdown("---")

    m = folium.Map(location=[30, 20], zoom_start=2, tiles="CartoDB dark_matter")

    for _, row in geo.iterrows():
        code   = row["country_code"]
        score  = row["composite_score"]
        tier   = row["risk_tier"]
        coords = COUNTRY_COORDS.get(code, (0, 0))
        name   = COUNTRY_NAMES.get(code, code)
        color  = {
            "CRITICAL": "#ff0000", "HIGH": "#ff6600",
            "MEDIUM": "#ffaa00",   "LOW":  "#00cc88",
        }.get(tier, "#888")

        folium.CircleMarker(
            location=coords,
            radius=score / 6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{name}</b><br>Risk Score: {score}<br>Tier: {tier}",
                max_width=200,
            ),
            tooltip=f"{name}: {tier}",
        ).add_to(m)

    st_folium(m, width=None, height=500)

    st.subheader("📋 Country Risk Breakdown")
    geo_display = geo.copy()
    geo_display["country_name"] = geo_display["country_code"].map(COUNTRY_NAMES)
    geo_display["risk_tier"]    = geo_display["risk_tier"].apply(lambda x: f"{risk_emoji(x)} {x}")
    geo_display = geo_display.sort_values("composite_score", ascending=False)
    st.dataframe(geo_display, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 5 — COMMODITIES
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📦 Commodities":
    st.title("📦 Commodity Price Tracker")
    st.markdown("---")

    commodities = {
        "oil":      ("Crude Oil (CL=F)",    "#ff6b6b"),
        "freight":  ("Freight (BDRY)",      "#ffa500"),
        "copper":   ("Copper (HG=F)",       "#c0a060"),
        "gas":      ("Natural Gas (NG=F)",  "#4ecdc4"),
        "aluminum": ("Aluminum (ALI=F)",    "#7c83fd"),
    }

    for key, (label, color) in commodities.items():
        col_close = f"close_{key}"
        col_pct   = f"pct_change_{key}"

        if col_close not in df.columns:
            continue

        col1, col2 = st.columns([3, 1])
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["date"],
                y=df[col_close],
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.13)",
            ))
            fig.update_layout(
                title=label,
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                font=dict(color="#c9d1d9"),
                xaxis=dict(gridcolor="#1e2130"),
                yaxis=dict(gridcolor="#1e2130"),
                height=200,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            latest_close = df[col_close].iloc[-1]
            latest_pct   = df[col_pct].iloc[-1] if col_pct in df.columns else 0
            st.metric(
                label=label,
                value=f"{latest_close:.2f}",
                delta=f"{latest_pct:.2f}%",
            )

        st.markdown("---")