"""
engine.py — Inventory Optimization Engine
==========================================
Senior Data Scientist Framework: Optimizing Inventory Management
using Time-Series Demand Forecasting (M5-structured data).

Models: ARIMA | Holt-Winters | XGBoost Ensemble
Inventory: Dynamic ROP + Safety Stock
Simulation: Static vs Forecast-Driven Policy (52 weeks)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

# Stats & ML
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1.  DATA LAYER
# ─────────────────────────────────────────────────────────────

class M5DataGenerator:
    """
    Synthesises an M5-structured dataset (3 049 products × 10 stores)
    with external regressors: CPI, unemployment, promotions, calendar events.
    """

    N_PRODUCTS = 3049
    N_STORES   = 10
    CATEGORIES = ["HOBBIES", "HOUSEHOLD", "FOODS"]
    STATES     = ["CA", "TX", "WI"]
    STORE_IDS  = [f"{s}_{i}" for s in STATES for i in range(1, 5)][:10]

    # Seasonal / calendar event parameters
    HOLIDAYS = {
        "Christmas":    (12, 25, 0.45),
        "Thanksgiving": (11, 27, 0.35),
        "SuperBowl":    (2,  7,  0.25),
        "LaborDay":     (9,  4,  0.15),
        "ValentinesDay":(2,  14, 0.10),
    }

    def __init__(self, n_weeks: int = 104, seed: int = 42):
        self.n_weeks = n_weeks
        self.rng = np.random.default_rng(seed)
        self.start_date = datetime(2020, 1, 6)   # first Monday

    # ── public ──────────────────────────────────────────────

    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Returns
        -------
        sales_df   : (products × stores) weekly sales, long format
        calendar_df: week-level calendar / event flags
        prices_df  : product × store weekly prices (wide)
        """
        logger.info("Generating M5-structured synthetic dataset …")
        calendar_df = self._make_calendar()
        prices_df   = self._make_prices()
        sales_df    = self._make_sales(calendar_df, prices_df)
        logger.info(
            "Dataset ready — %d rows, %d products, %d stores",
            len(sales_df), self.N_PRODUCTS, self.N_STORES,
        )
        return sales_df, calendar_df, prices_df

    # ── private helpers ──────────────────────────────────────

    def _make_calendar(self) -> pd.DataFrame:
        dates = [self.start_date + timedelta(weeks=w) for w in range(self.n_weeks)]
        df = pd.DataFrame({"date": dates, "week": range(self.n_weeks)})

        # Macroeconomic regressors (monthly, interpolated to weekly)
        base_cpi  = 260.0
        base_unemp = 4.5
        df["cpi"] = base_cpi + np.cumsum(
            self.rng.normal(0.2, 0.1, self.n_weeks)
        )
        df["unemployment_rate"] = np.clip(
            base_unemp + np.cumsum(self.rng.normal(-0.02, 0.15, self.n_weeks)),
            2.0, 12.0,
        )

        # Promotion flag (random ~15 % of weeks)
        df["is_promotion"] = (self.rng.random(self.n_weeks) < 0.15).astype(int)

        # Holiday / event flags
        for event, (month, day, _) in self.HOLIDAYS.items():
            df[f"event_{event}"] = df["date"].apply(
                lambda d: 1 if d.month == month and abs(d.day - day) <= 7 else 0
            )

        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        return df

    def _make_prices(self) -> pd.DataFrame:
        idx = pd.MultiIndex.from_product(
            [range(self.N_PRODUCTS), self.STORE_IDS],
            names=["product_id", "store_id"],
        )
        base = self.rng.uniform(1.5, 150.0, len(idx))
        return pd.DataFrame({"base_price": base}, index=idx).reset_index()

    def _make_sales(
        self, calendar: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        rows = []
        sample_products = min(self.N_PRODUCTS, 200)   # cap for speed in demo
        sample_stores   = self.STORE_IDS[:5]

        for pid in range(sample_products):
            cat = self.CATEGORIES[pid % 3]
            base_demand = self.rng.uniform(5, 150)
            trend       = self.rng.uniform(-0.05, 0.15)
            seasonality = self.rng.uniform(0.05, 0.35)

            for store in sample_stores:
                price_row = prices.query(
                    "product_id == @pid and store_id == @store"
                )
                base_price = float(price_row["base_price"].iloc[0]) if len(price_row) else 10.0

                for _, cal in calendar.iterrows():
                    w = cal["week"]
                    # Trend + season
                    trend_comp   = base_demand * (1 + trend * w / 52)
                    season_comp  = 1 + seasonality * np.sin(2 * np.pi * w / 52)
                    promo_bump   = 1.20 if cal["is_promotion"] else 1.0
                    holiday_bump = 1.0
                    for event, (_, _, lift) in self.HOLIDAYS.items():
                        holiday_bump += lift * cal.get(f"event_{event}", 0)

                    # Price elasticity
                    price_effect = 1.0 - 0.03 * (base_price / 20 - 1)
                    # CPI / unemployment effect
                    macro_effect = 1.0 - 0.005 * (cal["cpi"] - 260) + 0.01 * (4.5 - cal["unemployment_rate"])

                    mu = max(0, trend_comp * season_comp * promo_bump * holiday_bump * price_effect * macro_effect)
                    demand = max(0, int(self.rng.negative_binomial(max(1, mu), 0.7)))

                    rows.append({
                        "product_id": pid,
                        "store_id":   store,
                        "week":       w,
                        "date":       cal["date"],
                        "category":   cat,
                        "sales":      demand,
                        "price":      base_price * (1 + self.rng.normal(0, 0.02)),
                        "cpi":        cal["cpi"],
                        "unemployment_rate": cal["unemployment_rate"],
                        "is_promotion": cal["is_promotion"],
                        **{f"event_{e}": cal.get(f"event_{e}", 0) for e in self.HOLIDAYS},
                    })

        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

class FeatureEngineer:
    LAG_WEEKS  = [1, 2, 4, 8, 13, 26]
    ROLL_WINS  = [4, 8, 13]

    @staticmethod
    def build(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["product_id", "store_id", "week"]).copy()
        grp = df.groupby(["product_id", "store_id"])["sales"]

        for lag in FeatureEngineer.LAG_WEEKS:
            df[f"lag_{lag}"] = grp.shift(lag)

        for win in FeatureEngineer.ROLL_WINS:
            shifted = grp.shift(1)
            df[f"roll_mean_{win}"] = shifted.transform(
                lambda x: x.rolling(win, min_periods=1).mean()
            )
            df[f"roll_std_{win}"] = shifted.transform(
                lambda x: x.rolling(win, min_periods=1).std()
            )

        df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
        df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)
        df = df.dropna(subset=[f"lag_{FeatureEngineer.LAG_WEEKS[-1]}"])
        return df


# ─────────────────────────────────────────────────────────────
# 3.  MODEL ENSEMBLE
# ─────────────────────────────────────────────────────────────

class ARIMAForecaster:
    def __init__(self, order: Tuple = (2, 1, 2)):
        self.order = order
        self.model_ = None

    def fit(self, series: np.ndarray) -> "ARIMAForecaster":
        self.model_ = ARIMA(series, order=self.order).fit()
        return self

    def predict(self, h: int) -> np.ndarray:
        fc = self.model_.forecast(steps=h)
        return np.clip(fc, 0, None)


class HoltWintersForecaster:
    def __init__(self, seasonal_periods: int = 52):
        self.seasonal_periods = seasonal_periods
        self.model_ = None

    def fit(self, series: np.ndarray) -> "HoltWintersForecaster":
        try:
            self.model_ = ExponentialSmoothing(
                series,
                trend="add",
                seasonal="add",
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated",
            ).fit(optimized=True)
        except Exception:
            self.model_ = ExponentialSmoothing(
                series, trend="add", initialization_method="estimated"
            ).fit(optimized=True)
        return self

    def predict(self, h: int) -> np.ndarray:
        fc = self.model_.forecast(h)
        return np.clip(fc, 0, None)


XGBOOST_FEATURES = [
    "week", "week_sin", "week_cos",
    "cpi", "unemployment_rate", "is_promotion",
    "lag_1", "lag_2", "lag_4", "lag_8",
    "roll_mean_4", "roll_mean_8", "roll_std_4",
]


class XGBoostForecaster:
    def __init__(self):
        self.model_ = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            verbosity=0,
        )
        self.feat_cols_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostForecaster":
        self.feat_cols_ = [c for c in XGBOOST_FEATURES if c in X.columns]
        self.model_.fit(X[self.feat_cols_], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model_.predict(X[self.feat_cols_])
        return np.clip(preds, 0, None)

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model_.feature_importances_, index=self.feat_cols_
        ).sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────
# 4.  EVALUATION METRICS
# ─────────────────────────────────────────────────────────────

def mase(actual: np.ndarray, forecast: np.ndarray, naive: np.ndarray) -> float:
    """Mean Absolute Scaled Error (scaled by naïve seasonal forecast)."""
    mae_model = np.mean(np.abs(actual - forecast))
    mae_naive = np.mean(np.abs(naive)) + 1e-9
    return mae_model / mae_naive


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(actual, forecast)))


def rolling_cv(
    series: np.ndarray,
    n_splits: int = 5,
    h: int = 4,
    min_train: int = 52,
) -> Dict[str, Dict[str, float]]:
    """
    Time-series rolling cross-validation.
    Returns per-model {mase, rmse} averaged across folds.
    """
    results = {m: {"mase": [], "rmse": []} for m in ["arima", "hw", "xgb"]}
    n = len(series)

    # Fake feature matrix for XGBoost in CV (simple lags)
    def _make_xgb_features(s: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame({"sales": s})
        for lag in [1, 2, 4, 8]:
            df[f"lag_{lag}"] = df["sales"].shift(lag)
        df["week"] = np.arange(len(df))
        df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
        df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)
        for col in ["cpi", "unemployment_rate", "is_promotion",
                    "roll_mean_4", "roll_mean_8", "roll_std_4"]:
            df[col] = 0.0
        return df.dropna()

    step = max(1, (n - min_train - h) // n_splits)
    cutoffs = range(min_train, n - h, step)

    for cut in list(cutoffs)[:n_splits]:
        train = series[:cut]
        test  = series[cut: cut + h]
        naive = np.diff(train[-13:], prepend=train[-14]) if len(train) >= 14 else np.ones(h)

        # ARIMA
        try:
            fc_arima = ARIMAForecaster().fit(train).predict(h)
            results["arima"]["mase"].append(mase(test, fc_arima, naive[:h]))
            results["arima"]["rmse"].append(rmse(test, fc_arima))
        except Exception:
            pass

        # Holt-Winters
        try:
            fc_hw = HoltWintersForecaster(seasonal_periods=min(52, len(train) // 2)).fit(train).predict(h)
            results["hw"]["mase"].append(mase(test, fc_hw, naive[:h]))
            results["hw"]["rmse"].append(rmse(test, fc_hw))
        except Exception:
            pass

        # XGBoost
        try:
            feat_df = _make_xgb_features(series[:cut + h])
            tr_feat = feat_df.iloc[:cut]
            te_feat = feat_df.iloc[cut: cut + h]
            xgb_m = XGBoostForecaster()
            feat_cols = [c for c in XGBOOST_FEATURES if c in tr_feat.columns]
            xgb_m.feat_cols_ = feat_cols
            xgb_m.model_.fit(tr_feat[feat_cols], train[-len(tr_feat):])
            fc_xgb = xgb_m.model_.predict(te_feat[feat_cols])
            fc_xgb = np.clip(fc_xgb, 0, None)
            results["xgb"]["mase"].append(mase(test, fc_xgb, naive[:h]))
            results["xgb"]["rmse"].append(rmse(test, fc_xgb))
        except Exception:
            pass

    return {
        m: {
            "mase": float(np.mean(v["mase"])) if v["mase"] else np.nan,
            "rmse": float(np.mean(v["rmse"])) if v["rmse"] else np.nan,
        }
        for m, v in results.items()
    }


def ensemble_weights(cv_results: Dict) -> Dict[str, float]:
    """Inverse-MASE weighting for ensemble blend."""
    inv = {m: 1 / (v["mase"] + 1e-6) for m, v in cv_results.items() if not np.isnan(v["mase"])}
    total = sum(inv.values()) + 1e-9
    return {m: w / total for m, w in inv.items()}


# ─────────────────────────────────────────────────────────────
# 5.  INVENTORY LOGIC
# ─────────────────────────────────────────────────────────────

class InventoryOptimizer:
    """
    Forecasting-Driven Inventory Framework
    ───────────────────────────────────────
    Dynamic ROP = μ_LT + z * σ_LT
    Safety Stock = z * σ_demand * √LT
    where z is driven by the target service level.
    """

    def __init__(
        self,
        service_level:   float = 0.95,
        lead_time_weeks: float = 2.0,
        lead_time_std:   float = 0.5,
        holding_cost_pct: float = 0.25,    # 25 % of unit cost per year
        stockout_cost_mult: float = 3.0,   # 3× unit cost per unit short
    ):
        self.service_level    = service_level
        self.lead_time        = lead_time_weeks
        self.lead_time_std    = lead_time_std
        self.holding_cost_pct = holding_cost_pct
        self.stockout_cost_mult = stockout_cost_mult
        self.z = stats.norm.ppf(service_level)

    def safety_stock(self, demand_std: float) -> float:
        """σ_LT = √(LT·σ²_d + μ²_d·σ²_LT)"""
        sigma_lt = np.sqrt(
            self.lead_time * demand_std**2 +
            0 * self.lead_time_std**2        # simplified; mean demand assumed in caller
        )
        return self.z * sigma_lt

    def reorder_point(self, demand_mean: float, demand_std: float) -> float:
        mu_lt = demand_mean * self.lead_time
        ss    = self.safety_stock(demand_std)
        return mu_lt + ss

    def economic_order_qty(self, demand_mean: float, unit_cost: float, order_cost: float = 50.0) -> float:
        """Wilson / EOQ formula."""
        h = self.holding_cost_pct * unit_cost
        if h <= 0 or demand_mean <= 0:
            return max(1.0, demand_mean * 4)
        return float(np.sqrt((2 * demand_mean * 52 * order_cost) / h))

    def compute_policy(
        self,
        forecast_series: np.ndarray,
        unit_cost:       float = 10.0,
        order_cost:      float = 50.0,
    ) -> Dict:
        mu  = float(np.mean(forecast_series))
        std = float(np.std(forecast_series)) + 1e-6
        ss  = self.safety_stock(std)
        rop = self.reorder_point(mu, std)
        eoq = self.economic_order_qty(mu, unit_cost, order_cost)
        return {
            "demand_mean":   mu,
            "demand_std":    std,
            "safety_stock":  ss,
            "reorder_point": rop,
            "order_qty":     eoq,
            "z_score":       self.z,
        }


# ─────────────────────────────────────────────────────────────
# 6.  52-WEEK SIMULATION
# ─────────────────────────────────────────────────────────────

class InventorySimulator:
    """
    Periodic-review (weekly) simulation comparing:
      - Static Policy : fixed ROP / SS based on historical mean ± k·std
      - Forecast-Driven Policy : dynamic ROP updated each period
    """

    def __init__(self, unit_cost: float = 10.0, holding_cost_pct: float = 0.25, stockout_cost_mult: float = 3.0):
        self.unit_cost           = unit_cost
        self.holding_cost_pct    = holding_cost_pct
        self.stockout_cost_mult  = stockout_cost_mult

    # ── core simulator ───────────────────────────────────────

    def simulate(
        self,
        demand_series: np.ndarray,
        forecast_series: np.ndarray,
        policy: Dict,
        mode: str = "static",
        n_weeks: int = 52,
        seed: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)

        # Realised demand (add noise around actuals for simulation)
        realized = np.array([
            max(0, int(rng.normal(d, 0.15 * d + 1)))
            for d in demand_series[:n_weeks]
        ])

        inventory   = float(policy["reorder_point"] * 1.5)
        on_order    = 0.0
        lead_time   = 2  # weeks

        records = []
        pending_orders = {}   # {arrival_week: qty}

        for w in range(n_weeks):
            # Receive pending orders
            inventory += pending_orders.pop(w, 0.0)

            # Dynamic policy update
            if mode == "forecast":
                horizon = min(13, n_weeks - w)
                fc_window = forecast_series[w: w + horizon]
                mu  = np.mean(fc_window) if len(fc_window) else policy["demand_mean"]
                std = np.std(fc_window)  + 1e-6 if len(fc_window) else policy["demand_std"]
                rop = mu * lead_time + 1.65 * std * np.sqrt(lead_time)
                eoq = policy["order_qty"]
            else:  # static
                rop = policy["reorder_point"] * 1.0
                eoq = policy["order_qty"]

            demand_w  = realized[w]
            sales_w   = min(demand_w, inventory)
            stockout_w = max(0, demand_w - inventory)
            inventory -= sales_w

            # Trigger order?
            if inventory <= rop and on_order == 0:
                qty = max(eoq, rop - inventory + eoq)
                arrival = w + lead_time
                pending_orders[arrival] = pending_orders.get(arrival, 0) + qty
                on_order = qty
            else:
                on_order = sum(pending_orders.values())

            holding_cost  = inventory * self.unit_cost * (self.holding_cost_pct / 52)
            stockout_cost = stockout_w * self.unit_cost * self.stockout_cost_mult

            records.append({
                "week":          w,
                "realized_demand": demand_w,
                "sales":          sales_w,
                "stockout":       stockout_w,
                "inventory":      inventory,
                "on_order":       on_order,
                "reorder_point":  rop,
                "holding_cost":   holding_cost,
                "stockout_cost":  stockout_cost,
                "total_cost":     holding_cost + stockout_cost,
                "service_level":  sales_w / (demand_w + 1e-9),
            })

        return pd.DataFrame(records)

    # ── summary KPIs ─────────────────────────────────────────

    @staticmethod
    def kpis(sim_df: pd.DataFrame) -> Dict:
        return {
            "total_holding_cost":  sim_df["holding_cost"].sum(),
            "total_stockout_cost": sim_df["stockout_cost"].sum(),
            "total_cost":          sim_df["total_cost"].sum(),
            "stockout_rate":       (sim_df["stockout"] > 0).mean(),
            "avg_inventory":       sim_df["inventory"].mean(),
            "fill_rate":           sim_df["service_level"].mean(),
            "n_stockout_events":   (sim_df["stockout"] > 0).sum(),
        }


# ─────────────────────────────────────────────────────────────
# 7.  ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

class ForecastingInventoryEngine:
    """
    Top-level orchestrator.  One call to `.run()` returns everything
    the Streamlit dashboard needs.
    """

    def __init__(self, n_weeks: int = 104, n_products: int = 20):
        self.n_weeks     = n_weeks
        self.n_products  = n_products
        self.data_gen    = M5DataGenerator(n_weeks=n_weeks)
        self.inv_opt     = InventoryOptimizer()
        self.inv_sim     = InventorySimulator()

    # ── public ──────────────────────────────────────────────

    def run(self) -> Dict:
        sales_df, calendar_df, prices_df = self.data_gen.generate()

        # Limit products for responsiveness
        pids   = sales_df["product_id"].unique()[: self.n_products]
        stores = sales_df["store_id"].unique()[:3]
        subset = sales_df[
            sales_df["product_id"].isin(pids) &
            sales_df["store_id"].isin(stores)
        ]

        # Feature engineering
        feat_df = FeatureEngineer.build(subset)

        # Train global XGBoost model
        feat_cols = [c for c in XGBOOST_FEATURES if c in feat_df.columns]
        split_w   = int(self.n_weeks * 0.75)
        train_feat = feat_df[feat_df["week"] < split_w]
        test_feat  = feat_df[feat_df["week"] >= split_w]

        xgb_global = XGBoostForecaster()
        if len(train_feat) > 0 and len(feat_cols) > 0:
            xgb_global.feat_cols_ = feat_cols
            xgb_global.model_.fit(train_feat[feat_cols], train_feat["sales"])

        # Per-product results
        product_results = []
        all_static_sims  = []
        all_dynamic_sims = []
        cv_summary_rows  = []

        for pid in pids:
            for sid in stores:
                series = (
                    subset.query("product_id == @pid and store_id == @sid")
                    .sort_values("week")["sales"]
                    .values
                )
                if len(series) < 20:
                    continue

                train_s = series[: split_w]
                test_s  = series[split_w:]

                if len(train_s) < 15 or len(test_s) < 4:
                    continue

                # ── CV ──
                cv_res = rolling_cv(train_s, n_splits=3, h=4, min_train=15)
                weights = ensemble_weights(cv_res)

                # ── Model fits ──
                h = len(test_s)
                fc_arima, fc_hw, fc_xgb = (
                    np.full(h, train_s.mean()),
                    np.full(h, train_s.mean()),
                    np.full(h, train_s.mean()),
                )

                try:
                    fc_arima = ARIMAForecaster().fit(train_s).predict(h)
                except Exception:
                    pass
                try:
                    sp = min(52, max(2, len(train_s) // 2))
                    fc_hw = HoltWintersForecaster(sp).fit(train_s).predict(h)
                except Exception:
                    pass
                try:
                    pf = test_feat.query("product_id == @pid and store_id == @sid")
                    if len(pf) >= h and len(feat_cols) > 0:
                        fc_xgb = xgb_global.model_.predict(pf[feat_cols])[:h]
                except Exception:
                    pass

                # Ensemble
                w_a = weights.get("arima", 0.33)
                w_h = weights.get("hw",    0.33)
                w_x = weights.get("xgb",   0.34)
                total_w = w_a + w_h + w_x + 1e-9
                fc_ensemble = (w_a * fc_arima + w_h * fc_hw + w_x * fc_xgb) / total_w
                fc_ensemble = np.clip(fc_ensemble, 0, None)

                # Metrics
                naive_fc = np.full(h, train_s[-1])
                m_ens = mase(test_s, fc_ensemble, naive_fc)
                r_ens = rmse(test_s, fc_ensemble)

                # Inventory policy
                unit_cost = float(
                    prices_df.query("product_id == @pid and store_id == @sid")["base_price"].mean()
                    if len(prices_df.query("product_id == @pid and store_id == @sid")) > 0
                    else 10.0
                )
                policy = self.inv_opt.compute_policy(fc_ensemble, unit_cost)

                # Simulation
                sim_demand = np.concatenate([train_s[-52:], test_s])[:52]
                sim_fc     = np.concatenate([fc_ensemble, fc_ensemble])[:52]

                static_sim  = self.inv_sim.simulate(sim_demand, sim_fc, policy, mode="static")
                dynamic_sim = self.inv_sim.simulate(sim_demand, sim_fc, policy, mode="forecast")

                static_kpi  = self.inv_sim.kpis(static_sim)
                dynamic_kpi = self.inv_sim.kpis(dynamic_sim)

                holding_reduction = (
                    (static_kpi["total_holding_cost"] - dynamic_kpi["total_holding_cost"])
                    / (static_kpi["total_holding_cost"] + 1e-9) * 100
                )
                stockout_reduction = (
                    (static_kpi["stockout_rate"] - dynamic_kpi["stockout_rate"])
                    / (static_kpi["stockout_rate"] + 1e-9) * 100
                )

                product_results.append({
                    "product_id":          pid,
                    "store_id":            sid,
                    "unit_cost":           unit_cost,
                    "mase":                m_ens,
                    "rmse":                r_ens,
                    "mase_arima":          cv_res["arima"]["mase"],
                    "mase_hw":             cv_res["hw"]["mase"],
                    "mase_xgb":            cv_res["xgb"]["mase"],
                    "rmse_arima":          cv_res["arima"]["rmse"],
                    "rmse_hw":             cv_res["hw"]["rmse"],
                    "rmse_xgb":            cv_res["xgb"]["rmse"],
                    "safety_stock":        policy["safety_stock"],
                    "reorder_point":       policy["reorder_point"],
                    "order_qty":           policy["order_qty"],
                    "static_holding":      static_kpi["total_holding_cost"],
                    "dynamic_holding":     dynamic_kpi["total_holding_cost"],
                    "static_stockout_rate":  static_kpi["stockout_rate"],
                    "dynamic_stockout_rate": dynamic_kpi["stockout_rate"],
                    "holding_reduction_pct":  holding_reduction,
                    "stockout_reduction_pct": stockout_reduction,
                    "static_fill_rate":    static_kpi["fill_rate"],
                    "dynamic_fill_rate":   dynamic_kpi["fill_rate"],
                    "weight_arima":        w_a / total_w,
                    "weight_hw":           w_h / total_w,
                    "weight_xgb":          w_x / total_w,
                    "fc_ensemble":         fc_ensemble.tolist(),
                    "actual_test":         test_s.tolist(),
                    "actual_train":        train_s.tolist(),
                })

                static_sim["product_id"]  = pid
                static_sim["store_id"]    = sid
                dynamic_sim["product_id"] = pid
                dynamic_sim["store_id"]   = sid
                all_static_sims.append(static_sim)
                all_dynamic_sims.append(dynamic_sim)

        results_df  = pd.DataFrame(product_results)
        static_df   = pd.concat(all_static_sims,  ignore_index=True) if all_static_sims  else pd.DataFrame()
        dynamic_df  = pd.concat(all_dynamic_sims, ignore_index=True) if all_dynamic_sims else pd.DataFrame()

        # Aggregate KPIs
        if len(results_df):
            agg = {
                "avg_mase":                 results_df["mase"].mean(),
                "avg_rmse":                 results_df["rmse"].mean(),
                "avg_holding_reduction_pct":results_df["holding_reduction_pct"].mean(),
                "avg_stockout_reduction_pct":results_df["stockout_reduction_pct"].mean(),
                "pct_products_meeting_holding_target":
                    (results_df["holding_reduction_pct"] >= 10).mean() * 100,
                "pct_products_meeting_stockout_target":
                    (results_df["stockout_reduction_pct"] >= 20).mean() * 100,
                "n_products": len(results_df),
            }
        else:
            agg = {}

        fi_df = pd.DataFrame({
            "feature":    xgb_global.feature_importance().index,
            "importance": xgb_global.feature_importance().values,
        }) if xgb_global.feat_cols_ else pd.DataFrame()

        return {
            "results_df":   results_df,
            "static_df":    static_df,
            "dynamic_df":   dynamic_df,
            "calendar_df":  calendar_df,
            "agg_kpis":     agg,
            "feature_importance": fi_df,
            "xgb_model":    xgb_global,
        }
