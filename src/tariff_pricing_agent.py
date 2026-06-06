"""
Tariff Pricing Agent — EV Charging Tariff Optimization
=======================================================
Uses demand predictions from DemandPredictionAgent to generate dynamic
tariff multipliers, simulate revenue impact vs a flat ₹15/kWh baseline,
and quantify off-peak uplift via a simple demand-elasticity model.

Called as:
    from src.tariff_pricing_agent import TariffPricingAgent
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import (
    apply_plot_style, save_figure, save_metrics,
    FIGURES_DIR, METRICS_DIR,
    COLORS, PALETTE_CATEGORICAL,
    BASELINE_TARIFF_PER_KWH,
    SURGE_THRESHOLD, DISCOUNT_THRESHOLD,
    SURGE_MULTIPLIER_RANGE, DISCOUNT_MULTIPLIER_RANGE,
    print_header, print_subheader,
)

# ────────────────────────────────────────────────────────────────
# CONSTANTS
# ────────────────────────────────────────────────────────────────
_PRICE_ELASTICITY = -0.30   # demand % change per price % change
_PERIOD_LABELS = {          # hour → pricing period
    "Off-Peak": list(range(0, 7)) + list(range(22, 24)),
    "Shoulder": list(range(7, 10)) + list(range(14, 17)),
    "Peak":     list(range(10, 14)) + list(range(17, 22)),
}


def _hour_to_period(h: int) -> str:
    for label, hours in _PERIOD_LABELS.items():
        if h in hours:
            return label
    return "Off-Peak"


def _compute_multiplier(util: float) -> float:
    """Map utilization to a tariff multiplier using linear interpolation."""
    if util >= SURGE_THRESHOLD:
        # Linearly scale 1.30→1.80 as util goes from 0.80→1.00
        frac = min((util - SURGE_THRESHOLD) / (1.0 - SURGE_THRESHOLD), 1.0)
        return SURGE_MULTIPLIER_RANGE[0] + frac * (SURGE_MULTIPLIER_RANGE[1] - SURGE_MULTIPLIER_RANGE[0])
    if util <= DISCOUNT_THRESHOLD:
        # Linearly scale 0.85→0.60 as util goes from 0.30→0.00
        frac = min((DISCOUNT_THRESHOLD - util) / DISCOUNT_THRESHOLD, 1.0)
        return DISCOUNT_MULTIPLIER_RANGE[1] - frac * (DISCOUNT_MULTIPLIER_RANGE[1] - DISCOUNT_MULTIPLIER_RANGE[0])
    return 1.0


# ══════════════════════════════════════════════════════════════
# MAIN CLASS
# ══════════════════════════════════════════════════════════════
class TariffPricingAgent:
    """Generates dynamic tariff multipliers and evaluates revenue impact."""

    def __init__(self, demand_agent):
        self.demand_agent = demand_agent
        self.pricing_history: list = []
        self.tariff_df: pd.DataFrame = pd.DataFrame()
        self.summary: dict = {}

    # ──────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────
    def optimize_tariffs(self, acn_df: pd.DataFrame, urbanev_data: dict) -> pd.DataFrame:
        """Generate dynamic tariff recommendations.

        Returns
        -------
        pd.DataFrame with columns:
            hour, day_of_week, period, predicted_util, tariff_multiplier,
            new_price, old_price, revenue_diff, kWhDelivered,
            sessions_old, sessions_new
        """
        print_header("TARIFF PRICING AGENT")

        # ----- 1. Build hourly utilization profile from ACN data ----------
        print_subheader("Building hourly utilization profile")
        tariff_df = self._build_hourly_profile(acn_df)

        # ----- 2. Apply pricing rules ------------------------------------
        print_subheader("Applying dynamic pricing rules")
        tariff_df["tariff_multiplier"] = tariff_df["predicted_util"].apply(_compute_multiplier)
        tariff_df["old_price"] = BASELINE_TARIFF_PER_KWH
        tariff_df["new_price"] = BASELINE_TARIFF_PER_KWH * tariff_df["tariff_multiplier"]
        tariff_df["period"] = tariff_df["hour"].apply(_hour_to_period)

        # ----- 3. Simulate demand elasticity response ---------------------
        print_subheader("Simulating demand elasticity")
        tariff_df = self._simulate_elasticity(tariff_df)

        # ----- 4. Compute revenue ----------------------------------------
        tariff_df["revenue_old"] = tariff_df["old_price"] * tariff_df["kWhDelivered"]
        tariff_df["revenue_new"] = tariff_df["new_price"] * tariff_df["kWhDelivered_adj"]
        tariff_df["revenue_diff"] = tariff_df["revenue_new"] - tariff_df["revenue_old"]

        self.tariff_df = tariff_df
        self.pricing_history.append(tariff_df.copy())

        # ----- 5. Compute summary metrics ---------------------------------
        self.summary = self._compute_summary(tariff_df)
        self._print_summary()

        # ----- 6. Plots & CSV output --------------------------------------
        apply_plot_style()
        self._plot_hourly_profile(tariff_df)           # 20
        self._plot_revenue_comparison(tariff_df)        # 21
        self._plot_utilization_shift(tariff_df)         # 22
        self._plot_heatmap(tariff_df)                   # 23
        self._plot_offpeak_uplift(tariff_df)            # 24

        # Metrics CSV
        metrics_rows = [
            {"metric": "Revenue Gain %", "value": self.summary["revenue_gain_pct"]},
            {"metric": "Avg Utilization Before", "value": self.summary["util_before"]},
            {"metric": "Avg Utilization After", "value": self.summary["util_after"]},
            {"metric": "Off-Peak Uplift %", "value": self.summary["offpeak_uplift_pct"]},
            {"metric": "Total Revenue Old (₹)", "value": self.summary["total_revenue_old"]},
            {"metric": "Total Revenue New (₹)", "value": self.summary["total_revenue_new"]},
        ]
        save_metrics(
            pd.DataFrame(metrics_rows).set_index("metric"),
            "tariff_optimization_metrics.csv",
        )

        # Recommendations CSV
        save_metrics(tariff_df, "tariff_recommendations.csv")

        return tariff_df

    # ──────────────────────────────────────────────────────────
    # INTERNAL — data preparation
    # ──────────────────────────────────────────────────────────
    def _build_hourly_profile(self, acn_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate ACN sessions into an hour × day_of_week profile."""
        profile = (
            acn_df.groupby(["hour", "day_of_week"])
            .agg(
                predicted_util=("charger_utilization_rate", "mean"),
                kWhDelivered=("kWhDelivered", "sum"),
                session_count=("kWhDelivered", "count"),
            )
            .reset_index()
        )
        return profile

    def _simulate_elasticity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply price elasticity to estimate adjusted demand."""
        price_change_pct = (df["new_price"] - df["old_price"]) / df["old_price"]
        demand_change_pct = _PRICE_ELASTICITY * price_change_pct

        df["kWhDelivered_adj"] = df["kWhDelivered"] * (1 + demand_change_pct)
        df["sessions_old"] = df["session_count"]
        df["sessions_new"] = (df["session_count"] * (1 + demand_change_pct)).round().astype(int)
        df["util_after"] = df["predicted_util"] * (1 + demand_change_pct)
        df["util_after"] = df["util_after"].clip(0, 1)
        return df

    def _compute_summary(self, df: pd.DataFrame) -> dict:
        total_old = df["revenue_old"].sum()
        total_new = df["revenue_new"].sum()
        gain_pct = ((total_new - total_old) / total_old * 100) if total_old else 0

        offpeak = df[df["period"] == "Off-Peak"]
        offpeak_sessions_old = offpeak["sessions_old"].sum()
        offpeak_sessions_new = offpeak["sessions_new"].sum()
        offpeak_uplift = (
            ((offpeak_sessions_new - offpeak_sessions_old) / offpeak_sessions_old * 100)
            if offpeak_sessions_old else 0
        )

        return {
            "revenue_gain_pct": round(gain_pct, 2),
            "total_revenue_old": round(total_old, 2),
            "total_revenue_new": round(total_new, 2),
            "util_before": round(df["predicted_util"].mean(), 4),
            "util_after": round(df["util_after"].mean(), 4),
            "offpeak_uplift_pct": round(offpeak_uplift, 2),
        }

    def _print_summary(self):
        s = self.summary
        print(f"    Revenue Gain:          {s['revenue_gain_pct']:+.2f}%")
        print(f"    Avg Utilization Before: {s['util_before']:.4f}")
        print(f"    Avg Utilization After:  {s['util_after']:.4f}")
        print(f"    Off-Peak Uplift:        {s['offpeak_uplift_pct']:+.2f}%")

    # ──────────────────────────────────────────────────────────
    # PLOTS
    # ──────────────────────────────────────────────────────────
    def _plot_hourly_profile(self, df: pd.DataFrame):
        """20 — Dynamic tariff vs flat baseline across 24 hours."""
        hourly = df.groupby("hour").agg(
            avg_multiplier=("tariff_multiplier", "mean"),
        ).reset_index()
        hourly["dynamic_price"] = BASELINE_TARIFF_PER_KWH * hourly["avg_multiplier"]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(hourly["hour"], hourly["dynamic_price"],
                color=COLORS["primary"], linewidth=2.2, marker="o", markersize=5,
                label="Dynamic Tariff (₹/kWh)")
        ax.axhline(BASELINE_TARIFF_PER_KWH, color=COLORS["danger"], linestyle="--",
                   linewidth=1.5, label=f"Flat Baseline ₹{BASELINE_TARIFF_PER_KWH}/kWh")

        # Shade surge / discount zones
        ax.fill_between(hourly["hour"], BASELINE_TARIFF_PER_KWH, hourly["dynamic_price"],
                        where=hourly["dynamic_price"] > BASELINE_TARIFF_PER_KWH,
                        interpolate=True, alpha=0.20, color=COLORS["danger"], label="Surge Zone")
        ax.fill_between(hourly["hour"], hourly["dynamic_price"], BASELINE_TARIFF_PER_KWH,
                        where=hourly["dynamic_price"] < BASELINE_TARIFF_PER_KWH,
                        interpolate=True, alpha=0.20, color=COLORS["success"], label="Discount Zone")

        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Tariff (₹/kWh)")
        ax.set_title("Hourly Dynamic Tariff Profile vs Flat Baseline")
        ax.set_xticks(range(24))
        ax.legend(fontsize=9)
        save_figure(fig, "20_tariff_hourly_profile.png")

    def _plot_revenue_comparison(self, df: pd.DataFrame):
        """21 — Grouped bar chart: revenue by period, old vs new."""
        period_rev = df.groupby("period").agg(
            old=("revenue_old", "sum"),
            new=("revenue_new", "sum"),
        )
        # Ensure consistent period order
        order = ["Off-Peak", "Shoulder", "Peak"]
        period_rev = period_rev.reindex([p for p in order if p in period_rev.index])

        fig, ax = plt.subplots(figsize=(9, 6))
        x = np.arange(len(period_rev))
        w = 0.35
        ax.bar(x - w / 2, period_rev["old"], w, label="Flat Baseline",
               color=COLORS["muted"], edgecolor="none")
        ax.bar(x + w / 2, period_rev["new"], w, label="Dynamic Tariff",
               color=COLORS["primary"], edgecolor="none")

        ax.set_xticks(x)
        ax.set_xticklabels(period_rev.index)
        ax.set_ylabel("Revenue (₹)")
        ax.set_title("Revenue by Period — Flat vs Dynamic Tariff")
        ax.legend()

        # Annotate % change
        for i, period in enumerate(period_rev.index):
            old, new = period_rev.loc[period, "old"], period_rev.loc[period, "new"]
            pct = ((new - old) / old * 100) if old else 0
            color = COLORS["success"] if pct >= 0 else COLORS["danger"]
            ax.text(i + w / 2, new, f"{pct:+.1f}%", ha="center", va="bottom",
                    fontsize=9, color=color, fontweight="bold")
        save_figure(fig, "21_tariff_revenue_comparison.png")

    def _plot_utilization_shift(self, df: pd.DataFrame):
        """22 — Before / after utilization distribution (histogram overlay)."""
        fig, ax = plt.subplots(figsize=(9, 6))
        bins = np.linspace(0, 1, 40)
        ax.hist(df["predicted_util"], bins=bins, alpha=0.55,
                color=COLORS["muted"], edgecolor="#0f1117", label="Before (Flat)")
        ax.hist(df["util_after"], bins=bins, alpha=0.55,
                color=COLORS["primary"], edgecolor="#0f1117", label="After (Dynamic)")
        ax.set_xlabel("Charger Utilization Rate")
        ax.set_ylabel("Count")
        ax.set_title("Utilization Distribution — Before vs After Dynamic Pricing")
        ax.legend()
        save_figure(fig, "22_tariff_utilization_shift.png")

    def _plot_heatmap(self, df: pd.DataFrame):
        """23 — Heatmap: recommended tariff multiplier (hour × day_of_week)."""
        pivot = df.pivot_table(
            values="tariff_multiplier", index="day_of_week", columns="hour",
            aggfunc="mean",
        )
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        pivot.index = [day_labels[i] if i < len(day_labels) else str(i)
                       for i in pivot.index]

        fig, ax = plt.subplots(figsize=(14, 5))
        sns.heatmap(
            pivot, ax=ax, cmap="RdYlGn_r", center=1.0,
            linewidths=0.4, linecolor="#2d3148",
            cbar_kws={"label": "Tariff Multiplier"},
            annot=True, fmt=".2f", annot_kws={"fontsize": 8},
        )
        ax.set_title("Recommended Tariff Multipliers (Hour × Day)")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("")
        save_figure(fig, "23_tariff_heatmap.png")

    def _plot_offpeak_uplift(self, df: pd.DataFrame):
        """24 — Bar chart: session count increase in off-peak periods."""
        offpeak = df[df["period"] == "Off-Peak"].copy()
        grouped = offpeak.groupby("hour").agg(
            before=("sessions_old", "sum"),
            after=("sessions_new", "sum"),
        ).reset_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(grouped))
        w = 0.35
        ax.bar(x - w / 2, grouped["before"], w, label="Before (Flat)",
               color=COLORS["muted"], edgecolor="none")
        ax.bar(x + w / 2, grouped["after"], w, label="After (Dynamic)",
               color=COLORS["success"], edgecolor="none")

        ax.set_xticks(x)
        ax.set_xticklabels(grouped["hour"].astype(str))
        ax.set_xlabel("Hour of Day (Off-Peak)")
        ax.set_ylabel("Session Count")
        ax.set_title("Off-Peak Session Uplift with Discount Pricing")
        ax.legend()
        save_figure(fig, "24_tariff_offpeak_uplift.png")
