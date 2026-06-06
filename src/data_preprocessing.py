"""
Data Preprocessing Module
=========================
Loads, cleans, and engineers features from ACN-Data and UrbanEV datasets.
Creates a unified analytical base for downstream agents.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from src.utils import (ACN_DIR, URBANEV_DIR, BASELINE_TARIFF_PER_KWH,
                       print_header, print_subheader)


# ──────────────────────────────────────────────────────────────
# ACN DATA PREPROCESSING
# ──────────────────────────────────────────────────────────────

def load_acn_data():
    """Load the ACN charging sessions dataset."""
    print_subheader("Loading ACN-Data (Caltech/JPL sessions)")
    csv_path = os.path.join(ACN_DIR, "acndata_sessions.csv")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Raw records: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    return df


def preprocess_acn(df):
    """
    Clean and feature-engineer the ACN dataset.
    
    Assumptions documented:
    - Rows with missing connectionTime or disconnectTime are dropped (essential timestamps)
    - kWhDelivered=0 sessions are retained (connection without charge is valid behavior)
    - Timezone is normalized to UTC for consistency
    """
    print_subheader("Preprocessing ACN-Data")
    
    # ── 1. Parse timestamps ──
    time_cols = ['connectionTime', 'disconnectTime', 'doneChargingTime']
    for col in time_cols:
        df[col] = pd.to_datetime(df[col], format='mixed', utc=True, errors='coerce')
    
    # ── 2. Drop rows missing critical timestamps ──
    n_before = len(df)
    df = df.dropna(subset=['connectionTime', 'disconnectTime'])
    n_dropped = n_before - len(df)
    print(f"  Dropped {n_dropped} rows with missing connection/disconnect times")
    
    # ── 3. Numeric conversion ──
    df['kWhDelivered'] = pd.to_numeric(df['kWhDelivered'], errors='coerce').fillna(0)
    df['clusterID'] = pd.to_numeric(df['clusterID'], errors='coerce')
    df['siteID'] = pd.to_numeric(df['siteID'], errors='coerce')
    
    # ── 4. Time-based feature engineering ──
    df['session_duration_hrs'] = (
        (df['disconnectTime'] - df['connectionTime']).dt.total_seconds() / 3600
    )
    
    # Charging duration (use doneChargingTime if available, else disconnectTime)
    charge_end = df['doneChargingTime'].fillna(df['disconnectTime'])
    df['charging_duration_hrs'] = (
        (charge_end - df['connectionTime']).dt.total_seconds() / 3600
    )
    
    # Idle time (plugged in but not charging)
    df['idle_time_hrs'] = df['session_duration_hrs'] - df['charging_duration_hrs']
    df['idle_time_hrs'] = df['idle_time_hrs'].clip(lower=0)
    
    # ── 5. Utilization & Revenue features ──
    df['charger_utilization_rate'] = np.where(
        df['session_duration_hrs'] > 0,
        df['charging_duration_hrs'] / df['session_duration_hrs'],
        0
    )
    df['charger_utilization_rate'] = df['charger_utilization_rate'].clip(0, 1)
    
    df['revenue_baseline'] = df['kWhDelivered'] * BASELINE_TARIFF_PER_KWH
    df['energy_cost_per_kwh'] = BASELINE_TARIFF_PER_KWH  # Fixed baseline
    
    # ── 6. Temporal features ──
    df['hour'] = df['connectionTime'].dt.hour
    df['day_of_week'] = df['connectionTime'].dt.dayofweek  # 0=Mon, 6=Sun
    df['day_name'] = df['connectionTime'].dt.day_name()
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['connectionTime'].dt.month
    df['date'] = df['connectionTime'].dt.date
    df['week'] = df['connectionTime'].dt.isocalendar().week.astype(int)
    
    # ── 7. Time-of-day period classification ──
    def classify_period(hour):
        if 7 <= hour <= 10 or 17 <= hour <= 20:
            return 'peak'
        elif 10 < hour < 17:
            return 'shoulder'
        else:
            return 'off-peak'
    
    df['period'] = df['hour'].apply(classify_period)
    
    # ── 8. Site labeling ──
    df['site_label'] = df['site'].fillna('unknown').str.strip().str.lower()
    
    # ── 9. Filter out physically impossible sessions ──
    n_before = len(df)
    df = df[df['session_duration_hrs'] > 0]
    df = df[df['session_duration_hrs'] < 48]  # Sessions > 48h are likely errors
    print(f"  Filtered {n_before - len(df)} impossible-duration sessions")
    
    # ── 10. Queue length proxy (sessions overlapping at same station) ──
    # Approximate: count sessions per station per hour
    df['station_hour_key'] = (
        df['stationID'].astype(str) + '_' + 
        df['connectionTime'].dt.strftime('%Y-%m-%d_%H')
    )
    queue_proxy = df.groupby('station_hour_key').size().rename('queue_length_proxy')
    df = df.merge(queue_proxy, on='station_hour_key', how='left')
    
    print(f"  Final ACN records: {len(df):,}")
    print(f"  Date range: {df['connectionTime'].min().date()} to {df['connectionTime'].max().date()}")
    print(f"  Sites: {df['site_label'].unique()}")
    print(f"  Unique stations: {df['stationID'].nunique()}")
    print(f"  Avg kWh/session: {df['kWhDelivered'].mean():.2f}")
    print(f"  Avg session duration: {df['session_duration_hrs'].mean():.2f} hrs")
    print(f"  Avg utilization rate: {df['charger_utilization_rate'].mean():.2%}")
    
    return df


# ──────────────────────────────────────────────────────────────
# URBANEV DATA PREPROCESSING
# ──────────────────────────────────────────────────────────────

def load_urbanev_data():
    """Load all UrbanEV (Shenzhen) dataset files."""
    print_subheader("Loading UrbanEV Dataset (Shenzhen)")
    
    data = {}
    
    # Time-series data (timestamp × grid matrices)
    ts_files = ['occupancy', 'price', 'duration', 'volume']
    for name in ts_files:
        filepath = os.path.join(URBANEV_DIR, f"{name}.csv")
        df = pd.read_csv(filepath)
        data[name] = df
        print(f"  {name}.csv: {df.shape[0]:,} timestamps × {df.shape[1]-1} grids")
    
    # Time index
    time_df = pd.read_csv(os.path.join(URBANEV_DIR, "time.csv"), encoding='utf-8-sig')
    data['time'] = time_df
    print(f"  time.csv: {len(time_df):,} entries")
    
    # Station metadata
    info_df = pd.read_csv(os.path.join(URBANEV_DIR, "information.csv"))
    data['information'] = info_df
    print(f"  information.csv: {len(info_df)} grids")
    
    stations_df = pd.read_csv(os.path.join(URBANEV_DIR, "stations.csv"))
    data['stations'] = stations_df
    print(f"  stations.csv: {len(stations_df):,} stations")
    
    # Spatial data
    adj_df = pd.read_csv(os.path.join(URBANEV_DIR, "adj.csv"))
    data['adj'] = adj_df
    print(f"  adj.csv: {adj_df.shape} adjacency matrix")
    
    dist_df = pd.read_csv(os.path.join(URBANEV_DIR, "distance.csv"))
    data['distance'] = dist_df
    print(f"  distance.csv: {dist_df.shape} distance matrix")
    
    return data


def preprocess_urbanev(data):
    """
    Clean and feature-engineer the UrbanEV dataset.
    
    Assumptions documented:
    - 5-minute interval timestamps from June 19, 2022 for 30 days (8,640 intervals)
    - Grid IDs correspond to Shenzhen district zones
    - Occupancy represents number of EVs actively charging
    - Missing values in time-series are forward-filled then backward-filled
    """
    print_subheader("Preprocessing UrbanEV Data")
    
    # ── 1. Build datetime index ──
    time_df = data['time']
    timestamps = pd.to_datetime(
        time_df.apply(
            lambda r: f"{int(r['year'])}-{int(r['month']):02d}-{int(r['day']):02d} "
                      f"{int(r['hour']):02d}:{int(r['minute']):02d}:{int(r['second']):02d}",
            axis=1
        )
    )
    print(f"  Time range: {timestamps.iloc[0]} to {timestamps.iloc[-1]}")
    print(f"  Interval: 5 minutes, {len(timestamps):,} steps")
    
    # ── 2. Process time-series with datetime index ──
    ts_names = ['occupancy', 'price', 'duration', 'volume']
    for name in ts_names:
        df = data[name].copy()
        # First column is timestamp index (numeric), replace with actual datetime
        idx_col = df.columns[0]
        df = df.drop(columns=[idx_col])
        df.index = timestamps
        df.index.name = 'timestamp'
        
        # Handle missing values: forward-fill then backward-fill
        n_missing = df.isna().sum().sum()
        if n_missing > 0:
            print(f"  {name}: {n_missing} missing values → ffill+bfill")
            df = df.ffill().bfill()
        
        data[name] = df
    
    # ── 3. Enrich grid information ──
    info = data['information'].copy()
    info['grid'] = info['grid'].astype(str)
    info['fast_ratio'] = info['fast_count'] / info['count'].replace(0, np.nan)
    info['fast_ratio'] = info['fast_ratio'].fillna(0)
    info['charger_density'] = info['count'] / info['area'].replace(0, np.nan)
    info['charger_density'] = info['charger_density'].fillna(0)
    data['information'] = info
    
    # ── 4. Compute derived grid-level time-series features ──
    grid_ids = [str(c) for c in data['occupancy'].columns]
    
    # Occupancy rate = occupancy / total chargers in grid
    occ = data['occupancy']
    grid_charger_count = info.set_index('grid')['count']
    
    occ_rate = occ.copy()
    for col in occ.columns:
        col_str = str(col)
        total = grid_charger_count.get(col_str, np.nan)
        if pd.notna(total) and total > 0:
            occ_rate[col] = occ[col] / total
        else:
            occ_rate[col] = 0
    occ_rate = occ_rate.clip(0, 1)
    data['occupancy_rate'] = occ_rate
    
    # ── 5. Temporal features for UrbanEV ──
    data['timestamps'] = timestamps
    
    # ── 6. Aggregate statistics ──
    avg_occ = occ.mean().mean()
    avg_price = data['price'].mean().mean()
    avg_vol = data['volume'].mean().mean()
    print(f"  Grid count: {len(grid_ids)}")
    print(f"  Avg occupancy across all grids: {avg_occ:.2f}")
    print(f"  Avg price across all grids: {avg_price:.4f}")
    print(f"  Avg volume across all grids: {avg_vol:.2f}")
    print(f"  CBD grids: {info['CBD'].sum()}, Non-CBD: {(~info['CBD'].astype(bool)).sum()}")
    print(f"  Dynamic pricing grids: {info['dynamic_pricing'].sum()}")
    
    return data


# ──────────────────────────────────────────────────────────────
# UNIFIED PREPROCESSING PIPELINE
# ──────────────────────────────────────────────────────────────

def run_preprocessing():
    """Execute the complete data preprocessing pipeline."""
    print_header("DATA PREPROCESSING")
    
    # ACN Data
    acn_raw = load_acn_data()
    acn_df = preprocess_acn(acn_raw)
    
    # UrbanEV Data
    urbanev_raw = load_urbanev_data()
    urbanev_data = preprocess_urbanev(urbanev_raw)
    
    print_subheader("Preprocessing Complete")
    print(f"  ACN: {len(acn_df):,} sessions ready")
    print(f"  UrbanEV: {len(urbanev_data['occupancy'])} timestamps × "
          f"{len(urbanev_data['occupancy'].columns)} grids ready")
    
    return acn_df, urbanev_data


if __name__ == "__main__":
    acn_df, urbanev_data = run_preprocessing()
