"""
Monitoring & Learning Agent — EV Charging Tariff Optimization
==============================================================
Runs an iterative feedback loop that alternates between demand
prediction, tariff setting, and simulated customer response.
Tracks revenue, utilization, wait-time proxy, and pricing
efficiency across episodes, then persists plots and metrics.

Called as:
    from src.monitoring_agent import MonitoringAgent
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
_BASE_ELASTICITY = 0.30
_LEARNING_RATE = 0.04        # agent parameter adjustment step
_WAIT_TIME_SCALE = 2.5       # minutes of wait per unit excess utilization
_MIN_MULTIPLIER = 0.50
_MAX_MULTIPLIER = 2.00


def _compute_multiplier(util: float) -> float:
    """Identical pricing logic to TariffPricingAgent for consistency."""
    if util >= SURGE_THRESHOLD:
        frac = min((util - SURGE_THRESHOLD) / (1.0 - SURGE_THRESHOLD), 1.0)
        return SURGE_MULTIPLIER_RANGE[0] + frac * (SURGE_MULTIPLIER_RANGE[1] - SURGE_MULTIPLIER_RANGE[0])
    if util <= DISCOUNT_THRESHOLD:
        frac = min((DISCOUNT_THRESHOLD - util) / DISCOUNT_THRESHOLD, 1.0)
        return DISCOUNT_MULTIPLIER_RANGE[1] - frac * (DISCOUNT_MULTIPLIER_RANGE[1] - DISCOUNT_MULTIPLIER_RANGE[0])
    return 1.0


# ══════════════════════════════════════════════════════════════
# MAIN CLASS
# ══════════════════════════════════════════════════════════════
class MonitoringAgent:
    """Iterative feedback loop that refines pricing across episodes."""

    def __init__(self, demand_agent, tariff_agent):
        self.demand_agent = demand_agent
        self.tariff_agent = tariff_agent
        self.episode_history: list = []

    # ──────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────
    def run_feedback_loop(
        self, acn_df: pd.DataFrame, urbanev_data: dict, n_episodes: int = 10
    ) -> pd.DataFrame:
        """Run *n_episodes* of pricing → outcome → adjustment cycles.

        Returns
        -------
        pd.DataFrame  one row per episode with tracked KPIs.
        """
        print_header("MONITORING & LEARNING AGENT")

        # Build a base hourly profile from ACN data
        base_profile = self._build_base_profile(acn_df)

        # Agent-adjustable parameters that evolve across episodes
        elasticity = _BASE_ELASTICITY
        surge_bonus = 0.0    # additive tweak to surge multiplier
        discount_bonus = 0.0 # additive tweak to discount depth

        for ep in range(1, n_episodes + 1):
            print_subheader(f"Episode {ep}/{n_episodes}")
            record = self._run_single_episode(
                base_profile, ep, elasticity, surge_bonus, discount_bonus
            )

            # ---- Adjustment heuristics ----------------------------------
            # If utilization is still too high on average, increase surge
            if record["avg_utilization"] > 0.75:
                surge_bonus = min(surge_bonus + _LEARNING_RATE, 0.30)
            elif record["avg_utilization"] < 0.45:
                surge_bonus = max(surge_bonus - _LEARNING_RATE * 0.5, -0.10)

            # If off-peak is underserved, deepen discount
            if record["offpeak_util"] < 0.35:
                discount_bonus = min(discount_bonus + _LEARNING_RATE, 0.20)
            else:
                discount_bonus = max(discount_bonus - _LEARNING_RATE * 0.5, -0.05)

            # Elasticity slowly converges toward observed responsiveness
            observed_response = abs(record["volume_change_pct"]) / max(abs(record["price_change_pct"]), 1e-6)
            elasticity = 0.9 * elasticity + 0.1 * observed_response

            self.episode_history.append(record)
            print(f"    Rev=₹{record['total_revenue']:,.0f}  "
                  f"Util={record['avg_utilization']:.3f}  "
                  f"Wait={record['avg_wait_time_proxy']:.2f}min  "
                  f"Eff={record['pricing_efficiency']:.3f}")

        # ---- Aggregate into DataFrame -----------------------------------
        history_df = pd.DataFrame(self.episode_history)

        # ---- Evaluation metrics -----------------------------------------
        print_subheader("Episode Summary Metrics")
        wait_reduction = (
            ((history_df["avg_wait_time_proxy"].iloc[0] -
              history_df["avg_wait_time_proxy"].iloc[-1]) /
             history_df["avg_wait_time_proxy"].iloc[0] * 100)
            if history_df["avg_wait_time_proxy"].iloc[0] > 0 else 0
        )
        avg_response = history_df["volume_change_pct"].abs().mean()
        final_efficiency = history_df["pricing_efficiency"].iloc[-1]

        print(f"    Avg Waiting-Time Reduction: {wait_reduction:+.2f}%")
        print(f"    Customer Response Rate:     {avg_response:.4f}")
        print(f"    Final Pricing Efficiency:   {final_efficiency:.4f}")

        # ---- Plots -------------------------------------------------------
        apply_plot_style()
        self._plot_learning_curve_revenue(history_df)        # 25
        self._plot_learning_curve_utilization(history_df)     # 26
        self._plot_wait_time_reduction(history_df)            # 27
        self._plot_pricing_efficiency(history_df)             # 28
        self._plot_feedback_summary(history_df)               # 29

        # ---- Metrics CSV --------------------------------------------------
        metrics_rows = [
            {"metric": "Avg Waiting Time Reduction %", "value": round(wait_reduction, 2)},
            {"metric": "Customer Response Rate", "value": round(avg_response, 4)},
            {"metric": "Final Pricing Efficiency", "value": round(final_efficiency, 4)},
            {"metric": "Final Avg Utilization", "value": round(history_df["avg_utilization"].iloc[-1], 4)},
            {"metric": "Final Revenue (₹)", "value": round(history_df["total_revenue"].iloc[-1], 2)},
        ]
        save_metrics(
            pd.DataFrame(metrics_rows).set_index("metric"),
            "monitoring_metrics.csv",
        )

        return history_df

    # ──────────────────────────────────────────────────────────
    # INTERNAL — single episode
    # ──────────────────────────────────────────────────────────
    def _build_base_profile(self, acn_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate ACN data into an hour × day_of_week base profile."""
        profile = (
            acn_df.groupby(["hour", "day_of_week"])
            .agg(
                base_util=("charger_utilization_rate", "mean"),
                kWhDelivered=("kWhDelivered", "sum"),
                session_count=("kWhDelivered", "count"),
            )
            .reset_index()
        )
        return profile

    def _run_single_episode(
        self, base_profile: pd.DataFrame, episode: int,
        elasticity: float, surge_bonus: float, discount_bonus: float,
    ) -> dict:
        """Execute one pricing→outcome cycle and return a record dict."""
        df = base_profile.copy()

        # Add small noise to simulate episode variability
        rng = np.random.default_rng(seed=42 + episode)
        noise = rng.normal(0, 0.02, size=len(df))
        df["util"] = (df["base_util"] + noise).clip(0, 1)

        # Compute tariff multipliers with agent adjustments
        multipliers = []
        for u in df["util"]:
            m = _compute_multiplier(u)
            if m > 1.0:
                m = min(m + surge_bonus, _MAX_MULTIPLIER)
            elif m < 1.0:
                m = max(m - discount_bonus, _MIN_MULTIPLIER)
            multipliers.append(m)
        df["multiplier"] = multipliers
        df["new_price"] = BASELINE_TARIFF_PER_KWH * df["multiplier"]
        df["old_price"] = BASELINE_TARIFF_PER_KWH

        # Elasticity response
        price_change_pct = (df["new_price"] - df["old_price"]) / df["old_price"]
        demand_change_pct = -elasticity * price_change_pct
        df["kWh_adj"] = df["kWhDelivered"] * (1 + demand_change_pct)
        df["util_adj"] = (df["util"] * (1 + demand_change_pct)).clip(0, 1)

        # Revenue
        df["revenue"] = df["new_price"] * df["kWh_adj"]

        # Wait-time proxy: if util > 0.90, excess → wait
        df["wait_proxy"] = np.where(
            df["util_adj"] > 0.90,
            (df["util_adj"] - 0.90) * _WAIT_TIME_SCALE * 60,  # in minutes
            0,
        )

        # Period tagging
        def _period(h):
            if h in list(range(0, 7)) + list(range(22, 24)):
                return "Off-Peak"
            if h in list(range(10, 14)) + list(range(17, 22)):
                return "Peak"
            return "Shoulder"

        df["period"] = df["hour"].apply(_period)
        offpeak = df[df["period"] == "Off-Peak"]

        total_revenue = df["revenue"].sum()
        total_kwh = df["kWh_adj"].sum()

        return {
            "episode": episode,
            "total_revenue": total_revenue,
            "avg_utilization": df["util_adj"].mean(),
            "avg_wait_time_proxy": df["wait_proxy"].mean(),
            "pricing_efficiency": total_revenue / max(total_kwh, 1.0),
            "offpeak_util": offpeak["util_adj"].mean() if len(offpeak) else 0,
            "price_change_pct": price_change_pct.mean(),
            "volume_change_pct": demand_change_pct.mean(),
        }

    # ──────────────────────────────────────────────────────────
    # PLOTS
    # ──────────────────────────────────────────────────────────
    def _plot_learning_curve_revenue(self, df: pd.DataFrame):
        """25 — Revenue across episodes."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["episode"], df["total_revenue"],
                color=COLORS["primary"], marker="o", markersize=6, linewidth=2)
        ax.fill_between(df["episode"], df["total_revenue"],
                        alpha=0.15, color=COLORS["primary"])
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Revenue (₹)")
        ax.set_title("Learning Curve — Revenue Across Episodes")
        ax.set_xticks(df["episode"])
        save_figure(fig, "25_learning_curve_revenue.png")

    def _plot_learning_curve_utilization(self, df: pd.DataFrame):
        """26 — Utilization across episodes."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["episode"], df["avg_utilization"],
                color=COLORS["accent"], marker="s", markersize=6, linewidth=2)
        ax.axhline(0.70, color=COLORS["warning"], linestyle="--", linewidth=1,
                   label="Target (0.70)")
        ax.fill_between(df["episode"], df["avg_utilization"],
                        alpha=0.15, color=COLORS["accent"])
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg Utilization")
        ax.set_title("Learning Curve — Utilization Across Episodes")
        ax.set_xticks(df["episode"])
        ax.legend()
        save_figure(fig, "26_learning_curve_utilization.png")

    def _plot_wait_time_reduction(self, df: pd.DataFrame):
        """27 — Wait-time proxy across episodes (bar chart)."""
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = [COLORS["danger"] if w > df["avg_wait_time_proxy"].median()
                  else COLORS["success"] for w in df["avg_wait_time_proxy"]]
        ax.bar(df["episode"], df["avg_wait_time_proxy"],
               color=colors, edgecolor="none", width=0.7)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg Wait-Time Proxy (min)")
        ax.set_title("Wait-Time Proxy Across Episodes")
        ax.set_xticks(df["episode"])
        save_figure(fig, "27_wait_time_reduction.png")

    def _plot_pricing_efficiency(self, df: pd.DataFrame):
        """28 — Pricing efficiency (₹/kWh) across episodes."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["episode"], df["pricing_efficiency"],
                color=COLORS["highlight"], marker="D", markersize=6, linewidth=2)
        ax.fill_between(df["episode"], df["pricing_efficiency"],
                        alpha=0.12, color=COLORS["highlight"])
        ax.axhline(BASELINE_TARIFF_PER_KWH, color=COLORS["muted"], linestyle="--",
                   linewidth=1, label=f"Baseline ₹{BASELINE_TARIFF_PER_KWH}/kWh")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Pricing Efficiency (₹/kWh)")
        ax.set_title("Pricing Efficiency Trajectory")
        ax.set_xticks(df["episode"])
        ax.legend()
        save_figure(fig, "28_pricing_efficiency_trajectory.png")

    def _plot_feedback_summary(self, df: pd.DataFrame):
        """29 — 2×2 subplot combining all four KPIs across episodes."""
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

        configs = [
            ("total_revenue", "Revenue (₹)", COLORS["primary"],
             "Revenue", "o"),
            ("avg_utilization", "Avg Utilization", COLORS["accent"],
             "Utilization", "s"),
            ("avg_wait_time_proxy", "Wait-Time Proxy (min)", COLORS["danger"],
             "Wait Time", "^"),
            ("pricing_efficiency", "Pricing Efficiency (₹/kWh)", COLORS["highlight"],
             "Efficiency", "D"),
        ]

        for idx, (col, ylabel, color, title, marker) in enumerate(configs):
            ax = fig.add_subplot(gs[idx])
            ax.plot(df["episode"], df[col], color=color,
                    marker=marker, markersize=5, linewidth=1.8)
            ax.fill_between(df["episode"], df[col], alpha=0.12, color=color)
            ax.set_xlabel("Episode")
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=12)
            ax.set_xticks(df["episode"])

        fig.suptitle("Feedback Loop Summary — All KPIs", fontsize=14, y=1.01)
        save_figure(fig, "29_feedback_loop_summary.png")
