"""
Exploratory Data Analysis — OP'26 Analytics
Agentic AI-Based Dynamic Tariff Optimization for EV Charging Networks.

Generates 14 publication-quality dark-mode plots covering:
  • ACN session demand, utilization, revenue, and idle-time patterns
  • UrbanEV occupancy heatmaps, price-demand relationships, and spatial analysis
  • Combined comparative views of both datasets
"""

from src.utils import (
    apply_plot_style, save_figure, FIGURES_DIR, COLORS,
    PALETTE_SEQUENTIAL, PALETTE_DIVERGING, PALETTE_CATEGORICAL,
    print_header, print_subheader,
)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ── colour maps keyed to period labels ──────────────────────────
PERIOD_COLORS = {
    'peak':      COLORS['danger'],      # red
    'shoulder':  COLORS['warning'],     # amber
    'off-peak':  COLORS['success'],     # emerald
}

SITE_COLORS = {
    'caltech': PALETTE_CATEGORICAL[0],  # indigo
    'jpl':     PALETTE_CATEGORICAL[1],  # cyan
}

DAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
             'Saturday', 'Sunday']


# ═════════════════════════════════════════════════════════════════
#  HELPER UTILITIES
# ═════════════════════════════════════════════════════════════════

def _annotate_bar_values(ax, fmt='{:.1f}', fontsize=8, color='#c8cdd5',
                         offset=0.02):
    """Add value labels above every bar in *ax*."""
    ymax = ax.get_ylim()[1]
    for p in ax.patches:
        h = p.get_height()
        if np.isnan(h) or h == 0:
            continue
        ax.annotate(fmt.format(h),
                    (p.get_x() + p.get_width() / 2., h),
                    ha='center', va='bottom',
                    fontsize=fontsize, color=color,
                    xytext=(0, 3), textcoords='offset points')


def _period_for_hour(h):
    """Return 'peak', 'shoulder', or 'off-peak' for a given hour."""
    if h in range(7, 11) or h in range(17, 21):
        return 'peak'
    elif h in range(11, 17):
        return 'shoulder'
    else:
        return 'off-peak'


# ═════════════════════════════════════════════════════════════════
#  PLOT 01 — ACN Hourly Demand (bar by period)
# ═════════════════════════════════════════════════════════════════

def _plot_01_hourly_demand(acn_df):
    hourly = acn_df.groupby('hour').size().reindex(range(24), fill_value=0)
    # Normalise to mean sessions per day
    n_days = max(acn_df['date'].nunique(), 1)
    hourly_avg = hourly / n_days

    colors = [PERIOD_COLORS[_period_for_hour(h)] for h in range(24)]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(24), hourly_avg.values, color=colors, edgecolor='none',
                  width=0.75, zorder=3)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Avg Sessions per Day')
    ax.set_title('ACN — Hourly Charging Demand by Time-of-Use Period',
                 fontsize=15, fontweight='bold', pad=12)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f'{h:02d}' for h in range(24)], fontsize=9)
    ax.set_xlim(-0.6, 23.6)

    # legend
    handles = [Patch(facecolor=PERIOD_COLORS[p], label=p.title())
               for p in ('peak', 'shoulder', 'off-peak')]
    ax.legend(handles=handles, loc='upper right', framealpha=0.85)

    # annotate peak hour
    peak_h = int(hourly_avg.idxmax())
    ax.annotate(f'Peak: {hourly_avg.max():.1f}',
                xy=(peak_h, hourly_avg.max()),
                xytext=(peak_h + 2, hourly_avg.max() * 1.12),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'],
                                lw=1.5),
                fontsize=10, color=COLORS['accent'], fontweight='bold')

    fig.tight_layout()
    save_figure(fig, '01_acn_hourly_demand.png')


# ═════════════════════════════════════════════════════════════════
#  PLOT 02 — ACN Weekday Pattern (grouped by site)
# ═════════════════════════════════════════════════════════════════

def _plot_02_weekday_pattern(acn_df):
    ct = (acn_df.groupby(['day_of_week', 'day_name', 'site_label'])
          .size().reset_index(name='sessions'))
    n_weeks = max(acn_df['week'].nunique(), 1)
    ct['avg_sessions'] = ct['sessions'] / n_weeks

    pivot = ct.pivot_table(index='day_of_week', columns='site_label',
                           values='avg_sessions', fill_value=0)
    pivot = pivot.reindex(range(7))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(7)
    w = 0.35
    for i, site in enumerate(('caltech', 'jpl')):
        if site in pivot.columns:
            vals = pivot[site].values
            ax.bar(x + i * w - w / 2, vals, w, label=site.upper(),
                   color=SITE_COLORS[site], edgecolor='none', zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                       fontsize=10)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Avg Sessions per Week')
    ax.set_title('ACN — Weekly Demand Pattern by Site',
                 fontsize=15, fontweight='bold', pad=12)
    ax.legend(framealpha=0.85)

    # weekend shading
    ax.axvspan(4.5, 6.5, color='white', alpha=0.04, zorder=0)
    ax.text(5.5, ax.get_ylim()[1] * 0.92, 'Weekend', ha='center',
            fontsize=9, color=COLORS['muted'], style='italic')

    fig.tight_layout()
    save_figure(fig, '02_acn_weekday_pattern.png')


# ═════════════════════════════════════════════════════════════════
#  PLOT 03 — kWh Distribution (histogram + KDE)
# ═════════════════════════════════════════════════════════════════

def _plot_03_kwh_distribution(acn_df):
    data = acn_df['kWhDelivered'].dropna()
    mean_v, med_v = data.mean(), data.median()
    std_v = data.std()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data, bins=60, kde=True, color=COLORS['primary'],
                 edgecolor='none', alpha=0.7, ax=ax, stat='density',
                 line_kws={'lw': 2})

    ax.axvline(mean_v, color=COLORS['warning'], ls='--', lw=1.8,
               label=f'Mean = {mean_v:.2f} kWh')
    ax.axvline(med_v, color=COLORS['accent'], ls='--', lw=1.8,
               label=f'Median = {med_v:.2f} kWh')

    # stats box
    stats_text = (f'μ = {mean_v:.2f} kWh\n'
                  f'σ = {std_v:.2f} kWh\n'
                  f'n = {len(data):,}')
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=10, color='#c8cdd5',
            bbox=dict(boxstyle='round,pad=0.5', fc='#1a1d29',
                      ec='#2d3148', alpha=0.9))

    ax.set_xlabel('Energy Delivered (kWh)')
    ax.set_ylabel('Density')
    ax.set_title('ACN — Distribution of Energy Delivered per Session',
                 fontsize=15, fontweight='bold', pad=12)
    ax.legend(loc='upper left', framealpha=0.85)
    fig.tight_layout()
    save_figure(fig, '03_acn_kwh_distribution.png')


# ═════════════════════════════════════════════════════════════════
#  PLOT 04 — Session Duration by Period (violin + box)
# ═════════════════════════════════════════════════════════════════

def _plot_04_session_duration(acn_df):
    order = ['peak', 'shoulder', 'off-peak']
    pal = [PERIOD_COLORS[p] for p in order]

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.violinplot(data=acn_df, x='period', y='session_duration_hrs',
                   order=order, palette=pal, inner=None, alpha=0.35,
                   cut=0, ax=ax, zorder=2)
    sns.boxplot(data=acn_df, x='period', y='session_duration_hrs',
                order=order, palette=pal, width=0.2, linewidth=1.2,
                fliersize=2, ax=ax, zorder=3,
                boxprops=dict(alpha=0.9),
                medianprops=dict(color='white', lw=1.5))

    ax.set_xlabel('Time-of-Use Period')
    ax.set_ylabel('Session Duration (hours)')
    ax.set_title('ACN — Session Duration by TOU Period',
                 fontsize=15, fontweight='bold', pad=12)
    ax.set_xticklabels([p.title() for p in order])

    # annotate medians
    for i, period in enumerate(order):
        subset = acn_df.loc[acn_df['period'] == period, 'session_duration_hrs']
        med = subset.median()
        ax.text(i, med + 0.3, f'{med:.1f}h', ha='center', fontsize=9,
                color='white', fontweight='bold')

    fig.tight_layout()
    save_figure(fig, '04_acn_session_duration.png')


# ═════════════════════════════════════════════════════════════════
#  PLOT 05 — Utilization Heatmap (hour × day_of_week)
# ═════════════════════════════════════════════════════════════════

def _plot_05_utilization_heatmap(acn_df):
    pivot = acn_df.pivot_table(index='hour', columns='day_of_week',
                               values='charger_utilization_rate',
                               aggfunc='mean')
    pivot = pivot.reindex(index=range(24), columns=range(7))

    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = sns.color_palette(PALETTE_SEQUENTIAL, as_cmap=True)
    sns.heatmap(pivot, cmap='YlOrRd', ax=ax, linewidths=0.3,
                linecolor='#0f1117', cbar_kws={'label': 'Mean Utilization',
                                                'shrink': 0.8},
                annot=True, fmt='.2f', annot_kws={'fontsize': 7})

    ax.set_yticklabels([f'{h:02d}:00' for h in range(24)], rotation=0,
                       fontsize=8)
    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                       fontsize=9)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Hour of Day')
    ax.set_title('ACN — Charger Utilization Heatmap',
                 fontsize=15, fontweight='bold', pad=12)
    fig.tight_layout()
    save_figure(fig, '05_acn_utilization_heatmap.png')


# ═════════════════════════════════════════════════════════════════
#  PLOT 06 — Revenue by Period (stacked bar by date)
# ═════════════════════════════════════════════════════════════════

def _plot_06_revenue_by_period(acn_df):
    rev = (acn_df.groupby(['date', 'period'])['revenue_baseline']
           .sum().reset_index())
    rev_pivot = rev.pivot_table(index='date', columns='period',
                                values='revenue_baseline', fill_value=0)
    order = ['off-peak', 'shoulder', 'peak']
    rev_pivot = rev_pivot.reindex(columns=order, fill_value=0)
    rev_pivot = rev_pivot.sort_index()

    fig, ax = plt.subplots(figsize=(13, 5))
    bottom = np.zeros(len(rev_pivot))
    for period in order:
        if period not in rev_pivot.columns:
            continue
        vals = rev_pivot[period].values
        ax.bar(range(len(rev_pivot)), vals, bottom=bottom,
               color=PERIOD_COLORS[period], label=period.title(),
               edgecolor='none', width=1.0, zorder=3)
        bottom += vals

    # x-axis: show every Nth date
    n = max(len(rev_pivot) // 15, 1)
    tick_pos = list(range(0, len(rev_pivot), n))
    tick_labels = [str(rev_pivot.index[i]) for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)

    ax.set_xlabel('Date')
    ax.set_ylabel('Revenue (₹)')
    ax.set_title('ACN — Daily Baseline Revenue by TOU Period',
                 fontsize=15, fontweight='bold', pad=12)
    ax.legend(loc='upper left', framealpha=0.85)

    # total annotation
    total_rev = rev_pivot.sum().sum()
    ax.text(0.98, 0.95, f'Total: ₹{total_rev:,.0f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=11,
            color=COLORS['accent'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', fc='#1a1d29',
                      ec='#2d3148', alpha=0.9))

    fig.tight_layout()
    save_figure(fig, '06_acn_revenue_by_period.png')


# ═════════════════════════════════════════════════════════════════
#  PLOT 07 — Idle Time vs kWh (scatter by period)
# ═════════════════════════════════════════════════════════════════

def _plot_07_idle_time(acn_df):
    df = acn_df.dropna(subset=['idle_time_hrs', 'kWhDelivered']).copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    for period in ('peak', 'shoulder', 'off-peak'):
        sub = df[df['period'] == period]
        ax.scatter(sub['kWhDelivered'], sub['idle_time_hrs'],
                   c=PERIOD_COLORS[period], label=period.title(),
                   alpha=0.35, s=12, edgecolors='none', zorder=3)

    ax.set_xlabel('Energy Delivered (kWh)')
    ax.set_ylabel('Idle Time (hours)')
    ax.set_title('ACN — Idle Time vs Energy Delivered',
                 fontsize=15, fontweight='bold', pad=12)
    ax.legend(framealpha=0.85, markerscale=3)

    # correlation annotation
    corr = df[['idle_time_hrs', 'kWhDelivered']].corr().iloc[0, 1]
    ax.text(0.97, 0.05, f'ρ = {corr:.3f}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=11, color=COLORS['accent'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', fc='#1a1d29',
                      ec='#2d3148', alpha=0.9))

    fig.tight_layout()
    save_figure(fig, '07_acn_idle_time_analysis.png')


# ═════════════════════════════════════════════════════════════════
#  PLOT 08 — UrbanEV Demand Heatmap (hour × top-20 grids)
# ═════════════════════════════════════════════════════════════════

def _plot_08_urbanev_heatmap(urbanev_data):
    occ = urbanev_data['occupancy']
    # Total occupancy per grid → top 20
    top_grids = occ.sum().nlargest(20).index.tolist()
    occ_top = occ[top_grids].copy()
    occ_top['hour'] = occ_top.index.hour
    hourly_avg = occ_top.groupby('hour').mean()

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(hourly_avg.T, cmap='magma', ax=ax, linewidths=0.3,
                linecolor='#0f1117',
                cbar_kws={'label': 'Avg Charging Pile Count', 'shrink': 0.8})

    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Grid ID')
    ax.set_title('UrbanEV — Hourly Demand Heatmap (Top 20 Grids)',
                 fontsize=15, fontweight='bold', pad=12)
    ax.set_xticklabels([f'{h:02d}' for h in range(24)], fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7, rotation=0)
    fig.tight_layout()
    save_figure(fig, '08_urbanev_demand_heatmap.png')


# ═════════════════════════════════════════════════════════════════
#  PLOT 09 — Price vs Demand Scatter (by CBD flag)
# ═════════════════════════════════════════════════════════════════

def _plot_09_price_demand(urbanev_data):
    occ_rate = urbanev_data['occupancy_rate']
    price = urbanev_data['price']
    info = urbanev_data['information']

    common_grids = list(set(occ_rate.columns) & set(price.columns))
    avg_occ = occ_rate[common_grids].mean()
    avg_price = price[common_grids].mean()

    scatter_df = pd.DataFrame({'avg_occupancy_rate': avg_occ,
                                'avg_price': avg_price}).dropna()

    # Merge CBD flag
    if 'grid' in info.columns and 'CBD' in info.columns:
        cbd_map = info.set_index('grid')['CBD']
        # Convert grid column types to match
        scatter_df['CBD'] = scatter_df.index.map(
            lambda g: cbd_map.get(g, cbd_map.get(str(g), 0)))
    else:
        scatter_df['CBD'] = 0

    scatter_df['CBD'] = scatter_df['CBD'].fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, color, marker in [(1, COLORS['warning'], 'D'),
                                  (0, COLORS['primary'], 'o')]:
        sub = scatter_df[scatter_df['CBD'] == label]
        ax.scatter(sub['avg_price'], sub['avg_occupancy_rate'],
                   c=color, label=f'CBD = {label}', alpha=0.7,
                   s=50, edgecolors='white', linewidths=0.3,
                   marker=marker, zorder=3)

    # correlation line
    valid = scatter_df.dropna()
    if len(valid) > 2:
        corr = valid[['avg_price', 'avg_occupancy_rate']].corr().iloc[0, 1]
        z = np.polyfit(valid['avg_price'], valid['avg_occupancy_rate'], 1)
        p = np.poly1d(z)
        xr = np.linspace(valid['avg_price'].min(), valid['avg_price'].max(), 50)
        ax.plot(xr, p(xr), '--', color=COLORS['accent'], lw=1.5, alpha=0.7)
        ax.text(0.97, 0.05, f'ρ = {corr:.3f}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=11, color=COLORS['accent'], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', fc='#1a1d29',
                          ec='#2d3148', alpha=0.9))

    ax.set_xlabel('Average Price (¥/kWh)')
    ax.set_ylabel('Average Occupancy Rate')
    ax.set_title('UrbanEV — Price vs Occupancy Rate by CBD Status',
                 fontsize=15, fontweight='bold', pad=12)
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    save_figure(fig, '09_urbanev_price_demand_scatter.png')


# ═════════════════════════════════════════════════════════════════
#  PLOT 10 — Weekly Temporal Profile
# ═════════════════════════════════════════════════════════════════

def _plot_10_temporal_profile(urbanev_data):
    occ_rate = urbanev_data['occupancy_rate']
    mean_profile = occ_rate.mean(axis=1)
    profile_df = pd.DataFrame({'occupancy_rate': mean_profile})
    profile_df['hour'] = profile_df.index.hour
    profile_df['dow'] = profile_df.index.dayofweek  # 0=Mon
    profile_df['is_weekend'] = profile_df['dow'].isin([5, 6]).astype(int)

    # Build a typical-week profile: 7 days × 24 hours
    weekly = profile_df.groupby(['dow', 'hour'])['occupancy_rate'].mean()
    weekly = weekly.reindex(pd.MultiIndex.from_product(
        [range(7), range(24)], names=['dow', 'hour']), fill_value=np.nan)

    x = np.arange(7 * 24)
    y = weekly.values

    fig, ax = plt.subplots(figsize=(14, 5))

    # shade weekends
    for start in [5 * 24, 6 * 24]:
        ax.axvspan(start, start + 24, color='white', alpha=0.04, zorder=0)

    ax.fill_between(x, y, alpha=0.25, color=COLORS['primary'], zorder=2)
    ax.plot(x, y, color=COLORS['primary'], lw=1.5, zorder=3)

    # day boundaries
    for d in range(1, 7):
        ax.axvline(d * 24, color='#2d3148', ls=':', lw=0.8, zorder=1)

    ax.set_xticks([d * 24 + 12 for d in range(7)])
    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                       fontsize=10)
    ax.set_xlim(0, 7 * 24 - 1)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Mean Occupancy Rate')
    ax.set_title('UrbanEV — Typical Weekly Occupancy Profile',
                 fontsize=15, fontweight='bold', pad=12)

    # annotate weekday vs weekend means
    wd_mean = profile_df.loc[profile_df['is_weekend'] == 0, 'occupancy_rate'].mean()
    we_mean = profile_df.loc[profile_df['is_weekend'] == 1, 'occupancy_rate'].mean()
    ax.text(0.02, 0.95,
            f'Weekday avg: {wd_mean:.3f}\nWeekend avg: {we_mean:.3f}',
            transform=ax.transAxes, ha='left', va='top', fontsize=10,
            color='#c8cdd5',
            bbox=dict(boxstyle='round,pad=0.5', fc='#1a1d29',
                      ec='#2d3148', alpha=0.9))

    fig.tight_layout()
    save_figure(fig, '10_urbanev_temporal_profile.png')


# ═════════════════════════════════════════════════════════════════
#  PLOT 11 — Station Map (scatter by lon/lat)
# ═════════════════════════════════════════════════════════════════

def _plot_11_station_map(urbanev_data):
    stations = urbanev_data['stations'].copy()
    # Ensure numeric types
    for col in ['longitude', 'latitude', 'fast', 'slow', 'count']:
        if col in stations.columns:
            stations[col] = pd.to_numeric(stations[col], errors='coerce')

    stations = stations.dropna(subset=['longitude', 'latitude', 'count'])

    fast_ratio = (stations['fast'] / stations['count'].replace(0, np.nan)).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(stations['longitude'], stations['latitude'],
                    s=stations['count'] * 8, c=fast_ratio,
                    cmap='coolwarm', alpha=0.7, edgecolors='white',
                    linewidths=0.3, zorder=3, vmin=0, vmax=1)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Fast Charger Ratio', fontsize=10)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('UrbanEV — Charging Station Map (size ∝ charger count)',
                 fontsize=15, fontweight='bold', pad=12)

    # size legend
    for s_val in [10, 30, 60]:
        ax.scatter([], [], s=s_val * 8, c=COLORS['muted'], alpha=0.6,
                   edgecolors='white', linewidths=0.3,
                   label=f'{s_val} chargers')
    ax.legend(loc='lower left', framealpha=0.85, title='Count',
              title_fontsize=9)

    fig.tight_layout()
    save_figure(fig, '11_urbanev_station_map.png')


# ═════════════════════════════════════════════════════════════════
#  PLOT 12 — Peak vs Off-Peak Occupancy (top 15 grids)
# ═════════════════════════════════════════════════════════════════

def _plot_12_peak_offpeak(urbanev_data):
    occ_rate = urbanev_data['occupancy_rate'].copy()
    occ_rate_df = occ_rate.copy()
    occ_rate_df['hour'] = occ_rate_df.index.hour
    is_peak = occ_rate_df['hour'].isin(list(range(7, 11)) + list(range(17, 21)))

    # Top 15 grids by total occupancy
    grid_cols = [c for c in occ_rate.columns]
    top15 = occ_rate[grid_cols].sum().nlargest(15).index.tolist()

    peak_vals = occ_rate_df.loc[is_peak, top15].mean()
    offpeak_vals = occ_rate_df.loc[~is_peak, top15].mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(top15))
    w = 0.35

    ax.bar(x - w / 2, peak_vals.values, w, label='Peak (7-10, 17-20)',
           color=COLORS['danger'], edgecolor='none', zorder=3)
    ax.bar(x + w / 2, offpeak_vals.values, w, label='Off-Peak',
           color=COLORS['success'], edgecolor='none', zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in top15], rotation=45, ha='right',
                       fontsize=8)
    ax.set_xlabel('Grid ID')
    ax.set_ylabel('Mean Occupancy Rate')
    ax.set_title('UrbanEV — Peak vs Off-Peak Occupancy (Top 15 Grids)',
                 fontsize=15, fontweight='bold', pad=12)
    ax.legend(framealpha=0.85)

    fig.tight_layout()
    save_figure(fig, '12_urbanev_peak_offpeak.png')


# ═════════════════════════════════════════════════════════════════
#  PLOT 13 — Volatility Analysis (rolling std)
# ═════════════════════════════════════════════════════════════════

def _plot_13_volatility(urbanev_data):
    occ_rate = urbanev_data['occupancy_rate']
    overall_mean = occ_rate.mean(axis=1)

    top5 = occ_rate.sum().nlargest(5).index.tolist()

    window = min(24, len(occ_rate) // 4) or 24  # 24-hour rolling
    if window < 2:
        window = 2

    fig, ax = plt.subplots(figsize=(14, 5))

    # Overall mean volatility
    roll_overall = overall_mean.rolling(window, min_periods=1).std()
    ax.plot(roll_overall.index, roll_overall.values, color='white',
            lw=2.2, label='Overall Mean', zorder=4, alpha=0.9)

    # Top-5 grids
    for i, g in enumerate(top5):
        roll_g = occ_rate[g].rolling(window, min_periods=1).std()
        ax.plot(roll_g.index, roll_g.values,
                color=PALETTE_CATEGORICAL[i], lw=1.2, alpha=0.7,
                label=f'Grid {g}', zorder=3)

    ax.set_xlabel('Timestamp')
    ax.set_ylabel(f'Rolling Std (window={window}h)')
    ax.set_title('UrbanEV — Demand Volatility (Rolling Std of Occupancy Rate)',
                 fontsize=15, fontweight='bold', pad=12)
    ax.legend(loc='upper right', framealpha=0.85, fontsize=9)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)

    fig.tight_layout()
    save_figure(fig, '13_volatility_analysis.png')


# ═════════════════════════════════════════════════════════════════
#  PLOT 14 — Combined Demand Patterns (2×2)
# ═════════════════════════════════════════════════════════════════

def _plot_14_combined(acn_df, urbanev_data):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Combined — ACN vs UrbanEV Demand Patterns',
                 fontsize=16, fontweight='bold', y=0.98, color='#c8cdd5')

    occ_rate = urbanev_data['occupancy_rate']

    # ── (a) Hourly demand comparison ────────────────────────────
    ax = axes[0, 0]
    n_days_acn = max(acn_df['date'].nunique(), 1)
    acn_hourly = (acn_df.groupby('hour').size() / n_days_acn).reindex(
        range(24), fill_value=0)
    uev_hourly = occ_rate.groupby(occ_rate.index.hour).mean().mean(axis=1)
    uev_hourly = uev_hourly.reindex(range(24), fill_value=0)

    ax2 = ax.twinx()
    ax.bar(range(24), acn_hourly.values, color=COLORS['primary'],
           alpha=0.7, width=0.7, label='ACN (sessions)', zorder=3)
    ax2.plot(range(24), uev_hourly.values, color=COLORS['warning'],
             lw=2, marker='o', ms=4, label='UrbanEV (occ rate)', zorder=4)
    ax.set_xlabel('Hour', fontsize=9)
    ax.set_ylabel('ACN Sessions', fontsize=9, color=COLORS['primary'])
    ax2.set_ylabel('UrbanEV Occ Rate', fontsize=9, color=COLORS['warning'])
    ax.set_title('(a) Hourly Demand', fontsize=12, fontweight='bold')
    ax.set_xticks(range(0, 24, 3))

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=8,
              framealpha=0.85)

    # ── (b) Weekday pattern ─────────────────────────────────────
    ax = axes[0, 1]
    acn_dow = (acn_df.groupby('day_of_week').size() /
               max(acn_df['week'].nunique(), 1)).reindex(range(7), fill_value=0)

    uev_dow_df = occ_rate.copy()
    uev_dow_df['dow'] = uev_dow_df.index.dayofweek
    uev_dow = uev_dow_df.groupby('dow').mean().mean(axis=1).reindex(
        range(7), fill_value=0)

    x = np.arange(7)
    w = 0.35
    ax2 = ax.twinx()
    ax.bar(x - w / 2, acn_dow.values, w, color=COLORS['primary'],
           alpha=0.7, label='ACN', zorder=3)
    ax2.bar(x + w / 2, uev_dow.values, w, color=COLORS['warning'],
            alpha=0.7, label='UrbanEV', zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(['M', 'T', 'W', 'T', 'F', 'S', 'S'], fontsize=9)
    ax.set_ylabel('ACN Avg Sessions', fontsize=9, color=COLORS['primary'])
    ax2.set_ylabel('UrbanEV Occ Rate', fontsize=9, color=COLORS['warning'])
    ax.set_title('(b) Weekday Pattern', fontsize=12, fontweight='bold')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8,
              framealpha=0.85)

    # ── (c) Utilization / Occupancy distribution ────────────────
    ax = axes[1, 0]
    sns.kdeplot(acn_df['charger_utilization_rate'].dropna(), ax=ax,
                color=COLORS['primary'], fill=True, alpha=0.4, lw=2,
                label='ACN Utilization', clip=(0, 1))
    # Flatten UrbanEV occ rate
    uev_flat = occ_rate.values.flatten()
    uev_flat = uev_flat[~np.isnan(uev_flat)]
    if len(uev_flat) > 0:
        sns.kdeplot(uev_flat, ax=ax, color=COLORS['warning'], fill=True,
                    alpha=0.4, lw=2, label='UrbanEV Occ Rate', clip=(0, 1))
    ax.set_xlabel('Rate', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title('(c) Utilization / Occupancy Distribution',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, framealpha=0.85)

    # ── (d) Revenue / Price distribution ────────────────────────
    ax = axes[1, 1]
    price_flat = urbanev_data['price'].values.flatten()
    price_flat = price_flat[~np.isnan(price_flat)]
    rev_data = acn_df['revenue_baseline'].dropna()

    ax.hist(rev_data, bins=50, color=COLORS['primary'], alpha=0.6,
            edgecolor='none', density=True, label='ACN Revenue (₹)', zorder=3)
    if len(price_flat) > 0:
        ax_twin = ax.twinx()
        ax_twin.hist(price_flat, bins=50, color=COLORS['warning'], alpha=0.5,
                     edgecolor='none', density=True, label='UrbanEV Price (¥)',
                     zorder=2)
        ax_twin.set_ylabel('UrbanEV Density', fontsize=9,
                           color=COLORS['warning'])
        h2, l2 = ax_twin.get_legend_handles_labels()
    else:
        h2, l2 = [], []
    ax.set_xlabel('Value', fontsize=9)
    ax.set_ylabel('ACN Density', fontsize=9, color=COLORS['primary'])
    ax.set_title('(d) Revenue / Price Distribution',
                 fontsize=12, fontweight='bold')
    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8,
              framealpha=0.85)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, '14_combined_demand_patterns.png')


# ═════════════════════════════════════════════════════════════════
#  SUMMARY STATISTICS
# ═════════════════════════════════════════════════════════════════

def _print_summary(acn_df, urbanev_data):
    print_header("EDA SUMMARY STATISTICS")

    # ── ACN ──
    print_subheader("ACN Dataset")
    n = len(acn_df)
    print(f"  Total sessions          : {n:,}")
    print(f"  Date range              : {acn_df['date'].min()} → {acn_df['date'].max()}")
    print(f"  Unique stations         : {acn_df['stationID'].nunique()}")
    print(f"  Sites                   : {', '.join(acn_df['site_label'].unique())}")
    print(f"  Avg kWh/session         : {acn_df['kWhDelivered'].mean():.2f}")
    print(f"  Median session (hrs)    : {acn_df['session_duration_hrs'].median():.2f}")
    print(f"  Mean utilization        : {acn_df['charger_utilization_rate'].mean():.3f}")
    print(f"  Mean idle time (hrs)    : {acn_df['idle_time_hrs'].mean():.2f}")
    print(f"  Total baseline revenue  : ₹{acn_df['revenue_baseline'].sum():,.0f}")

    # Period breakdown
    print_subheader("ACN — By TOU Period")
    period_stats = (acn_df.groupby('period')
                    .agg(sessions=('kWhDelivered', 'count'),
                         avg_kwh=('kWhDelivered', 'mean'),
                         avg_util=('charger_utilization_rate', 'mean'),
                         total_rev=('revenue_baseline', 'sum'))
                    .round(2))
    print(period_stats.to_string(index=True))

    # ── UrbanEV ──
    print_subheader("UrbanEV Dataset")
    occ = urbanev_data['occupancy']
    occ_rate = urbanev_data['occupancy_rate']
    ts = urbanev_data['timestamps']
    info = urbanev_data['information']
    stations = urbanev_data['stations']

    print(f"  Timestamp range         : {ts.min()} → {ts.max()}")
    print(f"  Number of timestamps    : {len(ts):,}")
    print(f"  Number of grids         : {occ.shape[1]}")
    print(f"  Number of stations      : {len(stations)}")
    if 'count' in stations.columns:
        print(f"  Total charger count     : {stations['count'].sum():,.0f}")
    print(f"  Mean occupancy rate     : {occ_rate.mean().mean():.4f}")
    print(f"  Max occupancy rate      : {occ_rate.max().max():.4f}")
    if 'CBD' in info.columns:
        cbd_count = int(info['CBD'].sum())
        print(f"  CBD grids               : {cbd_count} / {len(info)}")
    if 'dynamic_pricing' in info.columns:
        dp_count = int(info['dynamic_pricing'].sum())
        print(f"  Dynamic pricing grids   : {dp_count} / {len(info)}")

    print_subheader("EDA Complete — 14 Figures Saved")
    print(f"  Output directory: {FIGURES_DIR}")


# ═════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════

def run_eda(acn_df, urbanev_data):
    """Execute full Exploratory Data Analysis for OP'26 Analytics.

    Parameters
    ----------
    acn_df : pd.DataFrame
        Pre-processed ACN charging session data.
    urbanev_data : dict
        UrbanEV dataset dictionary with keys: occupancy, price, duration,
        volume, occupancy_rate, information, stations, timestamps.
    """
    apply_plot_style()
    print_header("EXPLORATORY DATA ANALYSIS")

    # ── ACN Plots (01-07) ──
    print_subheader("Generating ACN Plots")
    _plot_01_hourly_demand(acn_df)
    _plot_02_weekday_pattern(acn_df)
    _plot_03_kwh_distribution(acn_df)
    _plot_04_session_duration(acn_df)
    _plot_05_utilization_heatmap(acn_df)
    _plot_06_revenue_by_period(acn_df)
    _plot_07_idle_time(acn_df)

    # ── UrbanEV Plots (08-13) ──
    print_subheader("Generating UrbanEV Plots")
    _plot_08_urbanev_heatmap(urbanev_data)
    _plot_09_price_demand(urbanev_data)
    _plot_10_temporal_profile(urbanev_data)
    _plot_11_station_map(urbanev_data)
    _plot_12_peak_offpeak(urbanev_data)
    _plot_13_volatility(urbanev_data)

    # ── Combined Plot (14) ──
    print_subheader("Generating Combined Comparison")
    _plot_14_combined(acn_df, urbanev_data)

    # ── Summary ──
    _print_summary(acn_df, urbanev_data)
