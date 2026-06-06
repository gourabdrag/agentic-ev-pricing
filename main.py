"""
╔══════════════════════════════════════════════════════════════════╗
║  OP'26 Analytics — Agentic AI Dynamic Tariff Optimization       ║
║  for EV Charging Networks                                        ║
║                                                                  ║
║  Main Pipeline Runner                                            ║
║  Society of Business — Open Project 2026                         ║
╚══════════════════════════════════════════════════════════════════╝

This script runs the complete end-to-end pipeline:
  1. Data Preprocessing (ACN + UrbanEV)
  2. Exploratory Data Analysis
  3. Demand Prediction Agent
  4. Tariff Pricing Agent
  5. Monitoring & Learning Agent

Usage:
    python main.py              # Run full pipeline
    python main.py --eda-only   # Run only preprocessing + EDA
"""

import sys
import os
import time
import json

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import print_header, print_subheader, FIGURES_DIR, METRICS_DIR, PROJECT_ROOT


def main():
    start_time = time.time()
    
    print("=" * 66)
    print("  OP'26 Analytics -- Agentic AI Dynamic Tariff Optimization")
    print("  for EV Charging Networks")
    print("  Society of Business -- Open Project 2026")
    print("=" * 66)
    
    eda_only = '--eda-only' in sys.argv
    
    # ────────────────────────────────────────────────────────
    # PHASE 1: DATA PREPROCESSING
    # ────────────────────────────────────────────────────────
    from src.data_preprocessing import run_preprocessing
    acn_df, urbanev_data = run_preprocessing()
    
    # ────────────────────────────────────────────────────────
    # PHASE 2: EXPLORATORY DATA ANALYSIS
    # ────────────────────────────────────────────────────────
    from src.eda import run_eda
    run_eda(acn_df, urbanev_data)
    
    if eda_only:
        elapsed = time.time() - start_time
        print(f"\n  EDA-only mode complete in {elapsed:.1f}s")
        return
    
    # ────────────────────────────────────────────────────────
    # PHASE 3: DEMAND PREDICTION AGENT
    # ────────────────────────────────────────────────────────
    from src.demand_prediction_agent import DemandPredictionAgent
    
    demand_agent = DemandPredictionAgent()
    demand_results = demand_agent.train_and_evaluate(acn_df, urbanev_data)
    
    # ────────────────────────────────────────────────────────
    # PHASE 4: TARIFF PRICING AGENT
    # ────────────────────────────────────────────────────────
    from src.tariff_pricing_agent import TariffPricingAgent
    
    tariff_agent = TariffPricingAgent(demand_agent)
    tariff_results = tariff_agent.optimize_tariffs(acn_df, urbanev_data)
    
    # ────────────────────────────────────────────────────────
    # PHASE 5: MONITORING & LEARNING AGENT
    # ────────────────────────────────────────────────────────
    from src.monitoring_agent import MonitoringAgent
    
    monitor = MonitoringAgent(demand_agent, tariff_agent)
    monitor_results = monitor.run_feedback_loop(acn_df, urbanev_data, n_episodes=10)
    
    # ────────────────────────────────────────────────────────
    # SUMMARY
    # ────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    
    print_header("PIPELINE COMPLETE")
    print(f"  Total execution time: {elapsed:.1f} seconds")
    print(f"  Figures saved to: {FIGURES_DIR}")
    print(f"  Metrics saved to: {METRICS_DIR}")
    
    # List all generated files
    print_subheader("Generated Figures")
    for f in sorted(os.listdir(FIGURES_DIR)):
        if f.endswith('.png'):
            print(f"    [PLOT] {f}")
    
    print_subheader("Generated Metrics")
    for f in sorted(os.listdir(METRICS_DIR)):
        if f.endswith('.csv'):
            print(f"    [CSV]  {f}")
    
    # Save pipeline summary as JSON for dashboard
    summary = {
        'pipeline_run': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_seconds': round(elapsed, 1),
            'acn_sessions': len(acn_df),
            'urbanev_grids': len(urbanev_data['occupancy'].columns),
            'urbanev_timestamps': len(urbanev_data['occupancy']),
            'figures_generated': len([f for f in os.listdir(FIGURES_DIR) if f.endswith('.png')]),
            'metrics_files': len([f for f in os.listdir(METRICS_DIR) if f.endswith('.csv')]),
        }
    }
    
    summary_path = os.path.join(METRICS_DIR, 'pipeline_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Pipeline summary saved to: {summary_path}")
    
    print("\n" + "=" * 60)
    print("  [OK] All deliverables generated successfully!")
    print("  Open dashboard/index.html in a browser to view results")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
