"""
Shared utilities, constants, and path configuration for OP'26 Analytics.
Agentic AI-Based Dynamic Tariff Optimization for EV Charging Networks.
"""

import os
import sys
import warnings

# Fix Windows console encoding
if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for plot generation
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Find actual dataset paths dynamically (handles special characters)
_datasets_parent = None
for item in os.listdir(PROJECT_ROOT):
    if item.startswith("Datasets") and os.path.isdir(os.path.join(PROJECT_ROOT, item)):
        _inner = os.path.join(PROJECT_ROOT, item)
        for sub in os.listdir(_inner):
            if os.path.isdir(os.path.join(_inner, sub)):
                _datasets_parent = os.path.join(_inner, sub)
                break
        break

# ACN Data paths
ACN_DIR = None
URBANEV_DIR = None
if _datasets_parent:
    for item in os.listdir(_datasets_parent):
        full_path = os.path.join(_datasets_parent, item)
        if "ACN" in item and os.path.isdir(full_path):
            ACN_DIR = full_path
        elif "UrbanEV" in item and os.path.isdir(full_path):
            URBANEV_DIR = full_path

# Output paths
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
METRICS_DIR = os.path.join(PROJECT_ROOT, "outputs", "metrics")

# Ensure output directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# ECONOMIC CONSTANTS
# ──────────────────────────────────────────────────────────────
BASELINE_TARIFF_PER_KWH = 15.0   # ₹15/kWh fixed baseline
SURGE_THRESHOLD = 0.80            # 80% utilization → surge pricing
DISCOUNT_THRESHOLD = 0.30         # 30% utilization → discount pricing
SURGE_MULTIPLIER_RANGE = (1.30, 1.80)
DISCOUNT_MULTIPLIER_RANGE = (0.60, 0.85)

# ──────────────────────────────────────────────────────────────
# PLOT STYLING — Professional, code-generated look
# ──────────────────────────────────────────────────────────────
PLOT_STYLE = {
    'figure.facecolor': '#0f1117',
    'axes.facecolor': '#1a1d29',
    'axes.edgecolor': '#2d3148',
    'axes.labelcolor': '#c8cdd5',
    'axes.grid': True,
    'grid.color': '#2d3148',
    'grid.alpha': 0.5,
    'grid.linestyle': '--',
    'text.color': '#c8cdd5',
    'xtick.color': '#8b92a5',
    'ytick.color': '#8b92a5',
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.facecolor': '#1a1d29',
    'legend.edgecolor': '#2d3148',
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.facecolor': '#0f1117',
    'savefig.pad_inches': 0.3,
}

# Color palettes
COLORS = {
    'primary': '#6366f1',      # Indigo
    'secondary': '#8b5cf6',    # Violet
    'accent': '#06b6d4',       # Cyan
    'success': '#10b981',      # Emerald
    'warning': '#f59e0b',      # Amber
    'danger': '#ef4444',       # Red
    'info': '#3b82f6',         # Blue
    'highlight': '#ec4899',    # Pink
    'muted': '#6b7280',        # Gray
}

PALETTE_SEQUENTIAL = ['#312e81', '#4338ca', '#6366f1', '#818cf8', '#a5b4fc', '#c7d2fe']
PALETTE_DIVERGING = ['#ef4444', '#f97316', '#facc15', '#4ade80', '#22d3ee', '#6366f1']
PALETTE_CATEGORICAL = ['#6366f1', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
                        '#ec4899', '#14b8a6', '#f97316', '#3b82f6']


def apply_plot_style():
    """Apply the project's consistent dark-mode plot style."""
    plt.rcParams.update(PLOT_STYLE)


def save_figure(fig, filename, tight=True):
    """Save a matplotlib figure to the figures output directory."""
    filepath = os.path.join(FIGURES_DIR, filename)
    fig.savefig(filepath, bbox_inches='tight' if tight else None,
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"  [SAVED] {filename}")
    return filepath


def save_metrics(df, filename):
    """Save a DataFrame to the metrics output directory as CSV."""
    filepath = os.path.join(METRICS_DIR, filename)
    df.to_csv(filepath, index=True)
    print(f"  [SAVED] {filename}")
    return filepath


def print_header(title):
    """Print a formatted section header."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subheader(title):
    """Print a formatted subsection header."""
    print(f"\n  -- {title} --")
