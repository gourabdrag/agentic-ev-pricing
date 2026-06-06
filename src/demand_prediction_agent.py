"""
Demand Prediction Agent — EV Charging Tariff Optimization
==========================================================
Trains RandomForest and XGBoost regressors on ACN charger-utilization and
UrbanEV grid-occupancy datasets. Produces evaluation metrics, feature-
importance rankings, and five dark-mode plots.

Called as:
    from src.demand_prediction_agent import DemandPredictionAgent
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from src.utils import (
    apply_plot_style, save_figure, save_metrics,
    FIGURES_DIR, METRICS_DIR,
    COLORS, PALETTE_CATEGORICAL,
    print_header, print_subheader,
)

# ────────────────────────────────────────────────────────────────
# CONSTANTS
# ────────────────────────────────────────────────────────────────
_ACN_FEATURES = [
    "hour", "day_of_week", "is_weekend", "month",
    "kWhDelivered_lag1", "queue_length_proxy",
]
_ACN_TARGET = "charger_utilization_rate"

_URBANEV_LAG_STEPS = [1, 2, 3, 6, 12]
_URBANEV_ROLLING_WIN = 12
_TOP_GRIDS = 20
_TRAIN_FRAC = 0.80
_RANDOM_STATE = 42


# ────────────────────────────────────────────────────────────────
# HELPER — build UrbanEV lag features for a single grid
# ────────────────────────────────────────────────────────────────
def _build_urbanev_features(series: pd.Series, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Create lag / rolling / calendar features from a grid occupancy-rate series."""
    df = pd.DataFrame({"occupancy_rate": series.values}, index=timestamps[: len(series)])
    for lag in _URBANEV_LAG_STEPS:
        df[f"lag_{lag}"] = df["occupancy_rate"].shift(lag)
    df["rolling_mean_12"] = df["occupancy_rate"].shift(1).rolling(_URBANEV_ROLLING_WIN).mean()
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df.dropna(inplace=True)
    return df


# ══════════════════════════════════════════════════════════════
# MAIN CLASS
# ══════════════════════════════════════════════════════════════
class DemandPredictionAgent:
    """Trains demand-prediction models on ACN & UrbanEV data,
    evaluates them, and persists plots + metrics."""

    def __init__(self):
        self.models: dict = {}          # model_name → fitted model
        self.results: dict = {}         # model_name → {y_test, y_pred, metrics, …}
        self._best_model_key: str = ""

    # ──────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────
    def train_and_evaluate(self, acn_df: pd.DataFrame, urbanev_data: dict) -> dict:
        """Train models on both datasets, evaluate, generate plots and metrics.

        Returns
        -------
        dict  with keys: predictions, metrics, feature_importances
        """
        print_header("DEMAND PREDICTION AGENT")

        # --- 1. ACN model ------------------------------------------------
        print_subheader("ACN — Charger Utilization Prediction")
        acn_metrics = self._train_acn(acn_df)

        # --- 2. UrbanEV model --------------------------------------------
        print_subheader("UrbanEV — Grid Occupancy Prediction")
        uev_metrics = self._train_urbanev(urbanev_data)

        # --- 3. Aggregate metrics & pick best model -----------------------
        all_metrics = {**acn_metrics, **uev_metrics}
        self._best_model_key = min(
            all_metrics, key=lambda k: all_metrics[k]["rmse"]
        )
        print(f"\n  ★ Best model (lowest RMSE): {self._best_model_key}")

        # --- 4. Generate all plots ----------------------------------------
        apply_plot_style()
        self._plot_actual_vs_predicted()     # 15
        self._plot_feature_importance()      # 16
        self._plot_residuals()               # 17
        self._plot_timeseries_forecast(urbanev_data)  # 18
        self._plot_model_comparison()        # 19

        # --- 5. Persist metrics CSV ---------------------------------------
        rows = []
        for name, m in all_metrics.items():
            rows.append({"model": name, "rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"]})
        metrics_df = pd.DataFrame(rows).set_index("model")
        save_metrics(metrics_df, "demand_prediction_metrics.csv")

        return {
            "predictions": {k: v.get("y_pred") for k, v in self.results.items()},
            "metrics": all_metrics,
            "feature_importances": {
                k: v.get("feature_importances")
                for k, v in self.results.items()
                if v.get("feature_importances") is not None
            },
        }

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict demand using the best available model."""
        if not self._best_model_key:
            raise RuntimeError("No trained model. Call train_and_evaluate() first.")
        model = self.models[self._best_model_key]
        return model.predict(features_df)

    # ──────────────────────────────────────────────────────────
    # INTERNAL — ACN training
    # ──────────────────────────────────────────────────────────
    def _train_acn(self, acn_df: pd.DataFrame) -> dict:
        df = acn_df.copy()

        # Create lag feature for kWhDelivered
        df = df.sort_values("connectionTime").reset_index(drop=True)
        df["kWhDelivered_lag1"] = df["kWhDelivered"].shift(1)
        df.dropna(subset=_ACN_FEATURES + [_ACN_TARGET], inplace=True)

        X = df[_ACN_FEATURES].values
        y = df[_ACN_TARGET].values
        feature_names = list(_ACN_FEATURES)

        split = int(len(X) * _TRAIN_FRAC)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        metrics_out = {}
        for tag, estimator in [
            ("ACN_RF", RandomForestRegressor(
                n_estimators=200, max_depth=12, min_samples_leaf=5,
                random_state=_RANDOM_STATE, n_jobs=-1)),
            ("ACN_XGB", XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8,
                random_state=_RANDOM_STATE, verbosity=0)),
        ]:
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))

            self.models[tag] = estimator
            fi = (estimator.feature_importances_
                  if hasattr(estimator, "feature_importances_") else None)
            self.results[tag] = {
                "y_test": y_test, "y_pred": y_pred,
                "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
                "feature_names": feature_names,
                "feature_importances": (
                    pd.Series(fi, index=feature_names) if fi is not None else None
                ),
            }
            metrics_out[tag] = {"rmse": rmse, "mae": mae, "r2": r2}
            print(f"    {tag}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

        return metrics_out

    # ──────────────────────────────────────────────────────────
    # INTERNAL — UrbanEV training
    # ──────────────────────────────────────────────────────────
    def _train_urbanev(self, urbanev_data: dict) -> dict:
        occ_rate = urbanev_data["occupancy_rate"]
        timestamps = urbanev_data["timestamps"]

        # Select top 20 grids by total occupancy
        total_occ = urbanev_data["occupancy"].sum(axis=0)
        top_grids = total_occ.nlargest(_TOP_GRIDS).index.tolist()

        # Collect rows from all top grids
        all_rows = []
        grid_indices: dict = {}  # grid → (start_idx, end_idx) into all_rows list
        for grid in top_grids:
            series = occ_rate[grid]
            gdf = _build_urbanev_features(series, timestamps)
            start = len(all_rows)
            all_rows.append(gdf)
            grid_indices[grid] = (start, start + len(gdf))

        combined = pd.concat(all_rows, ignore_index=False)
        feature_cols = [c for c in combined.columns if c != "occupancy_rate"]
        X = combined[feature_cols].values
        y = combined["occupancy_rate"].values
        feature_names = list(feature_cols)

        # Temporal split per concatenated block (first 80 % of each grid)
        train_mask = np.zeros(len(combined), dtype=bool)
        cumlen = 0
        for gdf in all_rows:
            n = len(gdf)
            cutoff = int(n * _TRAIN_FRAC)
            train_mask[cumlen: cumlen + cutoff] = True
            cumlen += n

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[~train_mask], y[~train_mask]

        # Store combined DataFrame + test indices for time-series plot
        self._urbanev_combined = combined
        self._urbanev_test_mask = ~train_mask
        self._urbanev_top_grids = top_grids
        self._urbanev_all_rows = all_rows

        metrics_out = {}
        for tag, estimator in [
            ("UEV_RF", RandomForestRegressor(
                n_estimators=200, max_depth=14, min_samples_leaf=4,
                random_state=_RANDOM_STATE, n_jobs=-1)),
            ("UEV_XGB", XGBRegressor(
                n_estimators=250, max_depth=7, learning_rate=0.06,
                subsample=0.85, colsample_bytree=0.8,
                random_state=_RANDOM_STATE, verbosity=0)),
        ]:
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))

            self.models[tag] = estimator
            fi = (estimator.feature_importances_
                  if hasattr(estimator, "feature_importances_") else None)
            self.results[tag] = {
                "y_test": y_test, "y_pred": y_pred,
                "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
                "feature_names": feature_names,
                "feature_importances": (
                    pd.Series(fi, index=feature_names) if fi is not None else None
                ),
            }
            metrics_out[tag] = {"rmse": rmse, "mae": mae, "r2": r2}
            print(f"    {tag}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

        return metrics_out

    # ──────────────────────────────────────────────────────────
    # PLOTS
    # ──────────────────────────────────────────────────────────
    def _plot_actual_vs_predicted(self):
        """15 — Scatter plot: actual vs predicted for the best model."""
        res = self.results[self._best_model_key]
        y_true, y_pred = res["y_test"], res["y_pred"]

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(y_true, y_pred, alpha=0.35, s=18, color=COLORS["primary"],
                   edgecolors="none", label="Predictions")
        lims = [min(y_true.min(), y_pred.min()) - 0.02,
                max(y_true.max(), y_pred.max()) + 0.02]
        ax.plot(lims, lims, "--", color=COLORS["danger"], linewidth=1.5,
                label="Perfect Prediction")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Actual vs Predicted — {self._best_model_key}")
        ax.legend()
        save_figure(fig, "15_demand_actual_vs_predicted.png")

    def _plot_feature_importance(self):
        """16 — Horizontal bar chart of top 15 features (best model)."""
        res = self.results[self._best_model_key]
        fi = res.get("feature_importances")
        if fi is None:
            return
        top15 = fi.sort_values(ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(8, 7))
        bars = ax.barh(top15.index, top15.values, color=COLORS["accent"], edgecolor="none")
        # Highlight the top feature
        bars[-1].set_color(COLORS["primary"])
        ax.set_xlabel("Importance")
        ax.set_title(f"Top 15 Feature Importances — {self._best_model_key}")
        save_figure(fig, "16_demand_feature_importance.png")

    def _plot_residuals(self):
        """17 — Residual distribution histogram for the best model."""
        res = self.results[self._best_model_key]
        residuals = res["y_test"] - res["y_pred"]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(residuals, bins=50, color=COLORS["secondary"], edgecolor="#0f1117",
                alpha=0.85, density=True)
        ax.axvline(0, color=COLORS["danger"], linestyle="--", linewidth=1.4)
        ax.set_xlabel("Residual (Actual − Predicted)")
        ax.set_ylabel("Density")
        ax.set_title(f"Residual Distribution — {self._best_model_key}")

        # Annotate mean & std
        mu, sigma = residuals.mean(), residuals.std()
        ax.text(0.97, 0.95, f"μ = {mu:.4f}\nσ = {sigma:.4f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, color=COLORS["accent"],
                bbox=dict(facecolor="#1a1d29", edgecolor=COLORS["accent"],
                          boxstyle="round,pad=0.4", alpha=0.9))
        save_figure(fig, "17_demand_residuals.png")

    def _plot_timeseries_forecast(self, urbanev_data: dict):
        """18 — Time series: actual vs predicted for 3 example grids."""
        # Pick the best UrbanEV model
        uev_key = "UEV_XGB" if "UEV_XGB" in self.models else "UEV_RF"
        model = self.models.get(uev_key)
        if model is None:
            return

        example_grids = self._urbanev_top_grids[:3]
        feature_cols = [c for c in self._urbanev_combined.columns if c != "occupancy_rate"]

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
        palette = [COLORS["primary"], COLORS["accent"], COLORS["success"]]

        cumlen = 0
        for idx, (grid, gdf) in enumerate(zip(self._urbanev_top_grids, self._urbanev_all_rows)):
            n = len(gdf)
            if grid not in example_grids:
                cumlen += n
                continue
            ax_idx = example_grids.index(grid)
            ax = axes[ax_idx]

            cutoff = int(n * _TRAIN_FRAC)
            test_part = gdf.iloc[cutoff:]
            X_test_g = test_part[feature_cols].values
            y_pred_g = model.predict(X_test_g)

            ax.plot(test_part.index, test_part["occupancy_rate"].values,
                    color=COLORS["muted"], linewidth=1, label="Actual", alpha=0.8)
            ax.plot(test_part.index, y_pred_g,
                    color=palette[ax_idx], linewidth=1.2, label="Predicted", alpha=0.9)
            ax.set_title(f"Grid {grid}", fontsize=11)
            ax.legend(fontsize=9, loc="upper right")
            ax.set_ylabel("Occupancy Rate")
            cumlen += n

        axes[-1].set_xlabel("Timestamp")
        fig.suptitle("Time-Series Forecast — 3 Example Grids", fontsize=13, y=1.01)
        fig.tight_layout()
        save_figure(fig, "18_demand_timeseries_forecast.png")

    def _plot_model_comparison(self):
        """19 — Grouped bar chart comparing RF vs XGB across RMSE, MAE, R²."""
        metric_names = ["rmse", "mae", "r2"]
        model_keys = list(self.results.keys())

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        x = np.arange(len(model_keys))
        width = 0.55

        for i, metric in enumerate(metric_names):
            vals = [self.results[k]["metrics"][metric] for k in model_keys]
            bars = axes[i].bar(x, vals, width, color=[PALETTE_CATEGORICAL[j % len(PALETTE_CATEGORICAL)]
                                                       for j in range(len(model_keys))],
                               edgecolor="none")
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(model_keys, rotation=25, ha="right", fontsize=9)
            axes[i].set_title(metric.upper(), fontsize=12)

            # Annotate bars
            for bar, val in zip(bars, vals):
                axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                             f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                             color=COLORS["accent"])

        fig.suptitle("Model Comparison — RF vs XGBoost", fontsize=13, y=1.02)
        fig.tight_layout()
        save_figure(fig, "19_demand_model_comparison.png")
