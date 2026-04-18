"""
app.py — Streamlit Dashboard
============================
Inventory Optimization: Time-Series Demand Forecasting
Tracks: 10–15% holding-cost reduction | 20%+ stockout reduction
"""

import warnings
warnings.filterwarnings("ignore")

import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config (MUST be first Streamlit call) ───────────────
st.set_page_config(
    page_title="Inventory Intelligence Platform",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auto-download data files from Google Drive ───────────────
DRIVE_FILES = {
    "sales_train_evaluation.csv": "1_3-2JkW8gxHBUf9EXuH6k399_8-jMonh",
    "calendar.csv":               "1YATfYtfwx7XOImvU4ciSNPgZhrgX5UFN",
    "sell_prices.csv":            "1TO4jBqpD4_kfqG-JW3n7_w10PmTtQLWE",
}

def download_data_files():
    for filename, file_id in DRIVE_FILES.items():
        if not os.path.exists(filename):
            with st.spinner(f"⬇️ Downloading {filename} from Google Drive…"):
                session = requests.Session()
                url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
                response = session.get(url, stream=True)
                # Handle Google's large-file virus-scan warning cookie
                for key, value in response.cookies.items():
                    if key.startswith("download_warning"):
                        url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={value}"
                        response = session.get(url, stream=True)
                        break
                with open(filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=32768):
                        if chunk:
                            f.write(chunk)
                st.success(f"✅ {filename} ready!")

download_data_files()

# ── CSS theme ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Space+Grotesk:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}
.main { background-color: #0a0e1a; }
section[data-testid="stSidebar"] {
    background: #0f1629;
    border-right: 1px solid #1e2d4a;
}

/* KPI Cards */
.kpi-grid { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }
.kpi-card {
    background: linear-gradient(135deg, #111827 0%, #1a2332 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 24px;
    flex: 1;
    min-width: 180px;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
}
.kpi-card.green::before  { background: linear-gradient(90deg, #10b981, #34d399); }
.kpi-card.amber::before  { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.kpi-card.red::before    { background: linear-gradient(90deg, #ef4444, #f87171); }
.kpi-label { font-size: 11px; letter-spacing: 1.5px; text-transform: uppercase; color: #64748b; font-family: 'IBM Plex Mono', monospace; margin-bottom: 8px; }
.kpi-value { font-size: 32px; font-weight: 700; color: #f1f5f9; line-height: 1; }
.kpi-delta { font-size: 12px; color: #10b981; margin-top: 6px; font-family: 'IBM Plex Mono', monospace; }
.kpi-delta.neg { color: #ef4444; }
.target-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 10px;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.5px;
    font-weight: 600;
    margin-left: 6px;
}
.badge-met   { background: #064e3b; color: #34d399; border: 1px solid #10b981; }
.badge-miss  { background: #450a0a; color: #f87171; border: 1px solid #ef4444; }

/* Section headers */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #3b82f6;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 8px;
    margin: 28px 0 16px 0;
}

/* Streamlit overrides */
div[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 16px;
}
.stSelectbox > div > div { background: #111827; border-color: #1e3a5f; }
.stSlider > div { color: #e2e8f0; }
h1, h2, h3 { color: #f1f5f9; }
</style>
""", unsafe_allow_html=True)

# ── Engine import ────────────────────────────────────────────
from engine import ForecastingInventoryEngine

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    n_products = st.slider("Products to model", 5, 50, 5, 5)
    n_weeks    = st.selectbox("History (weeks)", [52, 78, 104], index=0)
    service_level = st.slider("Target service level", 0.85, 0.99, 0.95, 0.01)

    st.divider()
    st.markdown("**Expected Targets**")
    st.markdown("🎯 Holding cost reduction ≥ **10–15%**")
    st.markdown("🎯 Stockout reduction ≥ **20%**")

    st.divider()
    run_btn = st.button("▶  Run Analysis", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# ENGINE RUN
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def run_engine(n_products: int, n_weeks: int) -> dict:
    engine = ForecastingInventoryEngine(n_weeks=n_weeks, n_products=n_products)
    return engine.run()

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div style="padding: 24px 0 8px 0;">
  <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:3px;color:#3b82f6;text-transform:uppercase;">Supply Chain Intelligence</span>
  <h1 style="margin:4px 0 2px 0;font-size:36px;font-weight:700;color:#f1f5f9;">Inventory Optimization Platform</h1>
  <p style="color:#64748b;margin:0;font-size:14px;">ARIMA · Holt-Winters · XGBoost Ensemble &nbsp;|&nbsp; M5 Forecasting Structure &nbsp;|&nbsp; Dynamic ROP & Safety Stock</p>
</div>
""", unsafe_allow_html=True)

if not run_btn and "results" not in st.session_state:
    st.info("👈 Configure parameters in the sidebar and click **Run Analysis** to begin.")
    st.stop()

if run_btn:
    with st.spinner("🔄 Running forecasting engine — this may take 30–60 seconds …"):
        st.session_state["results"] = run_engine(n_products, n_weeks)

results = st.session_state["results"]
res_df  = results["results_df"]
sta_df  = results["static_df"]
dyn_df  = results["dynamic_df"]
agg     = results["agg_kpis"]
fi_df   = results["feature_importance"]

if res_df.empty:
    st.error("No results generated — try increasing the number of products.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# SECTION 1 — AGGREGATE KPIs
# ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">01 — Key Performance Indicators</div>', unsafe_allow_html=True)

avg_hold = agg.get("avg_holding_reduction_pct", 0)
avg_so   = agg.get("avg_stockout_reduction_pct", 0)
avg_mase = agg.get("avg_mase", np.nan)
n_prods  = int(agg.get("n_products", 0))

hold_met = avg_hold >= 10
so_met   = avg_so   >= 20

hold_badge = '<span class="target-badge badge-met">TARGET MET ✓</span>'   if hold_met else '<span class="target-badge badge-miss">BELOW TARGET</span>'
so_badge   = '<span class="target-badge badge-met">TARGET MET ✓</span>'   if so_met   else '<span class="target-badge badge-miss">BELOW TARGET</span>'

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card {'green' if hold_met else 'amber'}">
    <div class="kpi-label">Holding Cost Reduction</div>
    <div class="kpi-value">{avg_hold:.1f}%</div>
    <div class="kpi-delta {'neg' if not hold_met else ''}">vs Static Policy &nbsp;{hold_badge}</div>
  </div>
  <div class="kpi-card {'green' if so_met else 'amber'}">
    <div class="kpi-label">Stockout Reduction</div>
    <div class="kpi-value">{avg_so:.1f}%</div>
    <div class="kpi-delta {'neg' if not so_met else ''}">vs Static Policy &nbsp;{so_badge}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Avg. Ensemble MASE</div>
    <div class="kpi-value">{avg_mase:.3f}</div>
    <div class="kpi-delta">Lower is better (≤1.0 beats naïve)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Products Modelled</div>
    <div class="kpi-value">{n_prods}</div>
    <div class="kpi-delta">Across {res_df['store_id'].nunique()} stores</div>
  </div>
  <div class="kpi-card green">
    <div class="kpi-label">Meeting Holding Target</div>
    <div class="kpi-value">{agg.get('pct_products_meeting_holding_target',0):.0f}%</div>
    <div class="kpi-delta">of products ≥10% reduction</div>
  </div>
  <div class="kpi-card {'green' if agg.get('pct_products_meeting_stockout_target',0)>=50 else 'amber'}">
    <div class="kpi-label">Meeting Stockout Target</div>
    <div class="kpi-value">{agg.get('pct_products_meeting_stockout_target',0):.0f}%</div>
    <div class="kpi-delta">of products ≥20% reduction</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SECTION 2 — FORECAST ACCURACY
# ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">02 — Forecast Accuracy: Model Comparison (Rolling CV)</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])

with col1:
    mase_data = {
        "ARIMA":        res_df["mase_arima"].dropna().mean(),
        "Holt-Winters": res_df["mase_hw"].dropna().mean(),
        "XGBoost":      res_df["mase_xgb"].dropna().mean(),
        "Ensemble":     res_df["mase"].dropna().mean(),
    }
    rmse_data = {
        "ARIMA":        res_df["rmse_arima"].dropna().mean(),
        "Holt-Winters": res_df["rmse_hw"].dropna().mean(),
        "XGBoost":      res_df["rmse_xgb"].dropna().mean(),
        "Ensemble":     res_df["rmse"].dropna().mean(),
    }

    fig_acc = make_subplots(rows=1, cols=2, subplot_titles=("Avg. MASE (lower=better)", "Avg. RMSE (lower=better)"))
    colors  = ["#3b82f6", "#06b6d4", "#8b5cf6", "#10b981"]

    for i, (m, v) in enumerate(mase_data.items()):
        fig_acc.add_trace(go.Bar(name=m, x=[m], y=[v], marker_color=colors[i], showlegend=False), row=1, col=1)
    for i, (m, v) in enumerate(rmse_data.items()):
        fig_acc.add_trace(go.Bar(name=m, x=[m], y=[v], marker_color=colors[i], showlegend=False), row=1, col=2)

    fig_acc.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", size=12),
        height=300, margin=dict(l=20, r=20, t=40, b=20),
    )
    fig_acc.update_xaxes(tickfont=dict(color="#94a3b8"), gridcolor="#1e2d4a")
    fig_acc.update_yaxes(tickfont=dict(color="#94a3b8"), gridcolor="#1e2d4a")
    st.plotly_chart(fig_acc, use_container_width=True)

with col2:
    st.markdown("**Ensemble Weights (avg)**")
    weights_data = {
        "ARIMA":        res_df["weight_arima"].mean(),
        "Holt-Winters": res_df["weight_hw"].mean(),
        "XGBoost":      res_df["weight_xgb"].mean(),
    }
    fig_pie = go.Figure(go.Pie(
        labels=list(weights_data.keys()),
        values=list(weights_data.values()),
        marker=dict(colors=["#3b82f6", "#06b6d4", "#8b5cf6"]),
        hole=0.6,
        textinfo="label+percent",
        textfont=dict(color="#e2e8f0"),
    ))
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        showlegend=False, height=250,
        margin=dict(l=20, r=20, t=10, b=10),
        annotations=[dict(text="Weights", x=0.5, y=0.5, font_size=13, font_color="#64748b", showarrow=False)],
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# SECTION 3 — MASE vs INVENTORY SCATTER
# ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">03 — Forecast Accuracy ↔ Inventory Performance</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    fig_sc = px.scatter(
        res_df,
        x="mase",
        y="holding_reduction_pct",
        color="store_id",
        size="unit_cost",
        hover_data=["product_id", "safety_stock", "reorder_point"],
        labels={"mase": "Ensemble MASE", "holding_reduction_pct": "Holding Cost Reduction (%)"},
        title="MASE vs Holding Cost Reduction",
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig_sc.add_hline(y=10, line_dash="dash", line_color="#f59e0b", annotation_text="10% target", annotation_font_color="#f59e0b")
    fig_sc.add_hline(y=15, line_dash="dot", line_color="#10b981", annotation_text="15% target", annotation_font_color="#10b981")
    fig_sc.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"), height=320,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig_sc.update_xaxes(gridcolor="#1e2d4a"); fig_sc.update_yaxes(gridcolor="#1e2d4a")
    st.plotly_chart(fig_sc, use_container_width=True)

with col4:
    fig_sc2 = px.scatter(
        res_df,
        x="mase",
        y="stockout_reduction_pct",
        color="store_id",
        size="safety_stock",
        hover_data=["product_id", "safety_stock", "reorder_point"],
        labels={"mase": "Ensemble MASE", "stockout_reduction_pct": "Stockout Reduction (%)"},
        title="MASE vs Stockout Reduction",
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig_sc2.add_hline(y=20, line_dash="dash", line_color="#ef4444", annotation_text="20% target", annotation_font_color="#ef4444")
    fig_sc2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"), height=320,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig_sc2.update_xaxes(gridcolor="#1e2d4a"); fig_sc2.update_yaxes(gridcolor="#1e2d4a")
    st.plotly_chart(fig_sc2, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# SECTION 4 — PRODUCT DRILL-DOWN
# ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">04 — Product-Level Forecast Drill-Down</div>', unsafe_allow_html=True)

col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    sel_pid = st.selectbox("Product ID", sorted(res_df["product_id"].unique()))
with col_sel2:
    available_stores = res_df[res_df["product_id"] == sel_pid]["store_id"].unique()
    sel_sid = st.selectbox("Store ID", sorted(available_stores))

row = res_df[(res_df["product_id"] == sel_pid) & (res_df["store_id"] == sel_sid)]
if not row.empty:
    row = row.iloc[0]
    train_vals    = list(row["actual_train"])
    test_vals     = list(row["actual_test"])
    fc_vals       = list(row["fc_ensemble"])
    n_train       = len(train_vals)
    n_test        = len(test_vals)
    all_actual    = train_vals + test_vals
    weeks_train   = list(range(n_train))
    weeks_test    = list(range(n_train, n_train + n_test))

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=weeks_train, y=train_vals, mode="lines", name="Historical", line=dict(color="#3b82f6", width=2)))
    fig_fc.add_trace(go.Scatter(x=weeks_test, y=test_vals, mode="lines", name="Actual (test)", line=dict(color="#06b6d4", width=2)))
    fig_fc.add_trace(go.Scatter(x=weeks_test, y=fc_vals, mode="lines", name="Ensemble Forecast", line=dict(color="#10b981", width=2, dash="dash")))
    fc_std = float(np.std(fc_vals)) if len(fc_vals) > 1 else 1.0
    fig_fc.add_trace(go.Scatter(
        x=weeks_test + weeks_test[::-1],
        y=[v + fc_std for v in fc_vals] + [v - fc_std for v in fc_vals][::-1],
        fill="toself", fillcolor="rgba(16,185,129,0.1)",
        line=dict(color="rgba(0,0,0,0)"), name="±1σ CI",
    ))
    fig_fc.add_vline(x=n_train - 0.5, line_dash="dash", line_color="#475569",
                     annotation_text="Train / Test split", annotation_font_color="#94a3b8")
    fig_fc.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"), height=360,
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="#1e3a5f"),
        xaxis_title="Week", yaxis_title="Units Sold",
        title=f"Product {sel_pid} | Store {sel_sid} — Ensemble Forecast (MASE={row['mase']:.3f})",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig_fc.update_xaxes(gridcolor="#1e2d4a"); fig_fc.update_yaxes(gridcolor="#1e2d4a")
    st.plotly_chart(fig_fc, use_container_width=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Safety Stock",    f"{row['safety_stock']:.1f} u")
    m2.metric("Reorder Point",   f"{row['reorder_point']:.1f} u")
    m3.metric("Order Qty (EOQ)", f"{row['order_qty']:.1f} u")
    m4.metric("Holding Δ",       f"{row['holding_reduction_pct']:.1f}%",  delta=f"{row['holding_reduction_pct']:.1f}%")
    m5.metric("Stockout Δ",      f"{row['stockout_reduction_pct']:.1f}%", delta=f"{row['stockout_reduction_pct']:.1f}%")

# ─────────────────────────────────────────────────────────────
# SECTION 5 — SIMULATION: STATIC vs DYNAMIC
# ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">05 — 52-Week Simulation: Static vs Forecast-Driven Policy</div>', unsafe_allow_html=True)

if not sta_df.empty and not dyn_df.empty:
    sta_agg = sta_df.groupby("week")[["inventory","holding_cost","stockout","stockout_cost","total_cost"]].mean().reset_index()
    dyn_agg = dyn_df.groupby("week")[["inventory","holding_cost","stockout","stockout_cost","total_cost"]].mean().reset_index()

    fig_sim = make_subplots(rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Avg. Inventory Level (units)", "Weekly Holding Cost ($)", "Weekly Stockout Events"),
        vertical_spacing=0.08)

    for trace_data, name, color in [
        (sta_agg, "Static Policy", "#ef4444"),
        (dyn_agg, "Forecast-Driven Policy", "#10b981"),
    ]:
        fig_sim.add_trace(go.Scatter(x=trace_data["week"], y=trace_data["inventory"], name=name, line=dict(color=color, width=2), legendgroup=name), row=1, col=1)
        fig_sim.add_trace(go.Scatter(x=trace_data["week"], y=trace_data["holding_cost"], name=name, line=dict(color=color, width=2), showlegend=False, legendgroup=name), row=2, col=1)
        fig_sim.add_trace(go.Scatter(x=trace_data["week"], y=trace_data["stockout"], name=name, line=dict(color=color, width=2), showlegend=False, legendgroup=name), row=3, col=1)

    fig_sim.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"), height=620,
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="#1e3a5f", x=0.01, y=0.99),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    for i in range(1, 4):
        fig_sim.update_xaxes(gridcolor="#1e2d4a", row=i, col=1)
        fig_sim.update_yaxes(gridcolor="#1e2d4a", row=i, col=1)
    st.plotly_chart(fig_sim, use_container_width=True)

    col_cs1, col_cs2 = st.columns(2)
    with col_cs1:
        cost_comp = pd.DataFrame({
            "Policy": ["Static", "Forecast-Driven"],
            "Total Holding Cost": [sta_agg["holding_cost"].sum(), dyn_agg["holding_cost"].sum()],
            "Total Stockout Cost": [sta_agg["stockout_cost"].sum() if "stockout_cost" in sta_agg else 0,
                                    dyn_agg["stockout_cost"].sum() if "stockout_cost" in dyn_agg else 0],
        })
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Bar(name="Holding", x=cost_comp["Policy"], y=cost_comp["Total Holding Cost"], marker_color="#3b82f6"))
        fig_cost.add_trace(go.Bar(name="Stockout", x=cost_comp["Policy"], y=cost_comp["Total Stockout Cost"], marker_color="#ef4444"))
        fig_cost.update_layout(barmode="stack", title="Cumulative Cost Breakdown",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"), height=300,
            legend=dict(bgcolor="rgba(0,0,0,0.3)"), margin=dict(l=20, r=20, t=40, b=20))
        fig_cost.update_xaxes(gridcolor="#1e2d4a"); fig_cost.update_yaxes(gridcolor="#1e2d4a")
        st.plotly_chart(fig_cost, use_container_width=True)

    with col_cs2:
        so_weekly = pd.DataFrame({
            "Week": sta_agg["week"],
            "Static": sta_agg["stockout"].cumsum(),
            "Forecast-Driven": dyn_agg["stockout"].cumsum(),
        })
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=so_weekly["Week"], y=so_weekly["Static"], name="Static", line=dict(color="#ef4444", width=2)))
        fig_cum.add_trace(go.Scatter(x=so_weekly["Week"], y=so_weekly["Forecast-Driven"], name="Forecast-Driven", line=dict(color="#10b981", width=2)))
        fig_cum.update_layout(title="Cumulative Stockout Units",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"), height=300,
            legend=dict(bgcolor="rgba(0,0,0,0.3)"), margin=dict(l=20, r=20, t=40, b=20))
        fig_cum.update_xaxes(gridcolor="#1e2d4a"); fig_cum.update_yaxes(gridcolor="#1e2d4a")
        st.plotly_chart(fig_cum, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# SECTION 6 — FEATURE IMPORTANCE & DISTRIBUTION
# ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">06 — XGBoost Feature Importance & Inventory Distribution</div>', unsafe_allow_html=True)

col7, col8 = st.columns([3, 2])

with col7:
    if not fi_df.empty:
        fig_fi = go.Figure(go.Bar(
            x=fi_df["importance"][:12], y=fi_df["feature"][:12], orientation="h",
            marker=dict(color=fi_df["importance"][:12], colorscale=[[0, "#1e3a5f"], [0.5, "#3b82f6"], [1, "#06b6d4"]]),
        ))
        fig_fi.update_layout(title="Top-12 Feature Importances (XGBoost)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"), height=360, yaxis=dict(autorange="reversed"),
            margin=dict(l=20, r=20, t=40, b=20))
        fig_fi.update_xaxes(gridcolor="#1e2d4a"); fig_fi.update_yaxes(gridcolor="#1e2d4a")
        st.plotly_chart(fig_fi, use_container_width=True)

with col8:
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=res_df["safety_stock"], name="Safety Stock", marker_color="#3b82f6", opacity=0.7, nbinsx=20))
    fig_dist.add_trace(go.Histogram(x=res_df["reorder_point"], name="Reorder Point", marker_color="#10b981", opacity=0.7, nbinsx=20))
    fig_dist.update_layout(barmode="overlay", title="Safety Stock & ROP Distribution",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"), height=360,
        legend=dict(bgcolor="rgba(0,0,0,0.3)"), margin=dict(l=20, r=20, t=40, b=20))
    fig_dist.update_xaxes(gridcolor="#1e2d4a", title="Units")
    fig_dist.update_yaxes(gridcolor="#1e2d4a", title="Count")
    st.plotly_chart(fig_dist, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# SECTION 7 — RAW DATA TABLE
# ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">07 — Product-Level Results Table</div>', unsafe_allow_html=True)

display_cols = [
    "product_id", "store_id", "unit_cost", "mase", "rmse",
    "safety_stock", "reorder_point", "order_qty",
    "holding_reduction_pct", "stockout_reduction_pct",
    "static_fill_rate", "dynamic_fill_rate",
]
display_df = res_df[[c for c in display_cols if c in res_df.columns]].copy().round(3)

st.dataframe(
    display_df.style
        .background_gradient(subset=["holding_reduction_pct", "stockout_reduction_pct"], cmap="RdYlGn", vmin=0, vmax=30)
        .format({
            "unit_cost": "${:.2f}", "mase": "{:.3f}", "rmse": "{:.2f}",
            "safety_stock": "{:.1f}", "reorder_point": "{:.1f}", "order_qty": "{:.1f}",
            "holding_reduction_pct": "{:.1f}%", "stockout_reduction_pct": "{:.1f}%",
            "static_fill_rate": "{:.1%}", "dynamic_fill_rate": "{:.1%}",
        }),
    use_container_width=True, height=400,
)

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#334155;font-family:'IBM Plex Mono',monospace;font-size:11px;padding:12px 0;">
  Inventory Intelligence Platform · ARIMA + Holt-Winters + XGBoost Ensemble · M5 Forecasting Structure
</div>
""", unsafe_allow_html=True)

