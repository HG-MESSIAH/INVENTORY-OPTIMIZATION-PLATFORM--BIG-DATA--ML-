# 📦 Inventory Intelligence Platform
### Optimizing Inventory Management via Time-Series Demand Forecasting

> **Senior Data Scientist Framework** | M5 Dataset Structure | ARIMA + Holt-Winters + XGBoost | Dynamic ROP & Safety Stock | 52-Week Simulation

---

## Architecture Overview

```
inventory_framework/
├── engine.py               ← Core ML & Inventory logic (7 modules)
├── app.py                  ← Streamlit Dashboard (7 sections)
├── requirements.txt        ← Pinned Python dependencies
├── Dockerfile              ← RHEL 10 / UBI 10 container
├── docker-compose.yml      ← Local orchestration
├── streamlit_config.toml   ← Dark-theme Streamlit config
└── README.md
```

---

## Module Map (`engine.py`)

| Class / Function | Role |
|---|---|
| `M5DataGenerator` | Synthesises 3 049-product × 10-store dataset with CPI, unemployment, promotions, calendar events |
| `FeatureEngineer` | Lag features (1–26 wks), rolling mean/std, Fourier encoding |
| `ARIMAForecaster` | ARIMA(2,1,2) per-series wrapper |
| `HoltWintersForecaster` | Additive trend + seasonal ETS |
| `XGBoostForecaster` | Global XGBoost with 13 engineered features |
| `rolling_cv()` | Time-series rolling cross-validation (MASE + RMSE) |
| `ensemble_weights()` | Inverse-MASE weighting for blending |
| `InventoryOptimizer` | Dynamic ROP, Safety Stock (σ_LT formula), EOQ |
| `InventorySimulator` | 52-week periodic-review simulation, Static vs Dynamic |
| `ForecastingInventoryEngine` | Top-level orchestrator — returns all dashboard data |

---

## Data Layer — Big Data Variety

| Variable | Source | Frequency |
|---|---|---|
| Weekly unit sales | M5-structured synthetic (Walmart structure) | Weekly |
| CPI | Macro series with random walk | Weekly (interpolated) |
| Unemployment rate | Macro series with mean reversion | Weekly |
| Promotion flags | Random Bernoulli (p=0.15) | Weekly |
| Calendar events | Christmas, Thanksgiving, Super Bowl, Labor Day, Valentine's | Event-week |
| Price | Product × Store base + noise | Weekly |

---

## Model Ensemble

### ARIMA (2,1,2)
- Per-product time-series fit
- Captures short-term autocorrelation and stationarity via differencing

### Holt-Winters (Additive Trend + Seasonal)
- Handles weekly seasonality (period=52)
- Automatic parameter optimisation via MLE

### XGBoost (Global Model)
- Trained across all products simultaneously
- Features: lag 1/2/4/8 weeks, rolling mean/std, Fourier week encoding, CPI, unemployment, promotion flag
- Captures cross-product patterns and external regressor interactions

### Blending
```python
w_i = 1 / (MASE_i + ε)          # inverse-MASE weight
ŷ_ensemble = Σ(w_i · ŷ_i) / Σw_i
```

---

## Inventory Framework

### Safety Stock
```
SS = z · √( LT · σ²_demand + μ²_demand · σ²_LT )
```

### Dynamic Reorder Point
```
ROP = μ_demand × LT + SS
```

### Economic Order Quantity (Wilson)
```
EOQ = √( 2 · D_annual · S / h )
where h = holding_cost_pct × unit_cost
```

Service level `z` maps: 90% → 1.28 | 95% → 1.645 | 99% → 2.33

---

## Expected Outcomes (Tracked Live on Dashboard)

| KPI | Target | Mechanism |
|---|---|---|
| Holding cost reduction | **10–15%** | Tighter safety stock from accurate forecasts |
| Stockout reduction | **≥ 20%** | Dynamic ROP anticipates demand spikes |
| Forecast accuracy (MASE) | **< 1.0** | Ensemble beats naïve baseline |
| Fill rate | **≥ 95%** | Service-level–driven z-score |

---

## Quick Start

### Local (Python venv)
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
# → open http://localhost:8501
```

### Docker (RHEL 10 / UBI 10)
```bash
# Build
docker build -t inventory-intelligence:1.0.0 .

# Run
docker run -p 8501:8501 inventory-intelligence:1.0.0

# Or with Docker Compose
docker-compose up --build
```

### RHEL 10 — SELinux note
```bash
# If volume mounts fail due to SELinux, add :z label
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data:z \
  inventory-intelligence:1.0.0
```

---

## Dashboard Sections

1. **KPI Banner** — Live holding/stockout reduction vs targets, MASE, product count
2. **Model Accuracy** — Bar charts of MASE/RMSE per model; ensemble weight donut
3. **Accuracy ↔ Inventory Scatter** — How lower MASE drives higher cost reductions
4. **Product Drill-Down** — Interactive forecast chart with CI band; per-product inventory parameters
5. **52-Week Simulation** — Inventory levels, holding/stockout costs, cumulative comparison
6. **Feature Importance** — XGBoost top-12 features + Safety Stock / ROP distributions
7. **Results Table** — Full product-level table with conditional formatting

---

## Configuration (Sidebar)
| Parameter | Range | Default |
|---|---|---|
| Products to model | 5 – 50 | 15 |
| History (weeks) | 52 / 78 / 104 | 78 |
| Target service level | 85 – 99% | 95% |

---

## Dependency Notes

- **Python**: 3.11 (UBI 10 default)
- **statsmodels**: ARIMA + Holt-Winters; `pmdarima` not required
- **xgboost**: CPU-only build by default; GPU build available via `xgboost[gpu]`
- **Streamlit**: ≥1.35 required for `st.cache_data` TTL and `st.dataframe` styler

---

## License
Internal use — Supply Chain Analytics Team
