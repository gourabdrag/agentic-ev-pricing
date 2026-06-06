import sys
import os
import io
import json

def clean_code(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    in_import_block = False
    
    for line in lines:
        stripped = line.strip()
        # Detect start of local multi-line import block from src
        if ('from src.' in stripped or 'import src.' in stripped) and '(' in stripped:
            in_import_block = True
            cleaned_lines.append("# " + line)
        # Detect end of local multi-line import block
        elif in_import_block:
            cleaned_lines.append("# " + line)
            if ')' in stripped:
                in_import_block = False
        # Comment out single-line local imports
        elif 'from src.' in stripped or 'import src.' in stripped:
            cleaned_lines.append("# " + line)
        else:
            cleaned_lines.append(line)
            
    return "".join(cleaned_lines)

def generate_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    # Helper to convert raw string to notebook list of lines
    def to_nb_lines(code_str):
        lines = code_str.splitlines(keepends=True)
        return [l if l.endswith('\n') else l + '\n' for l in lines]

    # Markdown Intro Cell
    intro_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# OP'26 Analytics — Agentic AI-Based Dynamic Tariff Optimization for EV Charging Networks\n",
            "\n",
            "**Open Project 2026 — Society of Business**\n",
            "\n",
            "This is a **self-contained, pre-executed notebook** that houses the entire codebase of the Agentic AI pricing engine. it can be run locally or directly in Google Colab.\n",
            "\n",
            "### Pipeline Components:\n",
            "1. **Shared Utilities & Styling** (Matplotlib dark-mode configuration)\n",
            "2. **Data Preprocessing** (ACN-Data and UrbanEV telemetry parsing)\n",
            "3. **Exploratory Data Analysis (EDA)** (14 dynamic visualizations)\n",
            "4. **Demand Prediction Agent** (Random Forest and XGBoost forecasting models)\n",
            "5. **Tariff Pricing Agent** (Dynamic Pricing simulation with demand elasticity)\n",
            "6. **Monitoring & Learning Agent** (Closed-loop feedback parameter tuning)"
        ]
    }
    notebook["cells"].append(intro_cell)

    # ── 1. Read files and clean code ──
    files = {
        "utils.py": os.path.join("src", "utils.py"),
        "data_preprocessing.py": os.path.join("src", "data_preprocessing.py"),
        "eda.py": os.path.join("src", "eda.py"),
        "demand_prediction_agent.py": os.path.join("src", "demand_prediction_agent.py"),
        "tariff_pricing_agent.py": os.path.join("src", "tariff_pricing_agent.py"),
        "monitoring_agent.py": os.path.join("src", "monitoring_agent.py")
    }

    code_cells_data = []
    
    # Header mapping
    headers = {
        "utils.py": "# ==================== 1. SHARED UTILITIES & PATHS ====================",
        "data_preprocessing.py": "# ==================== 2. DATA PREPROCESSING MODULE ====================",
        "eda.py": "# ==================== 3. EXPLORATORY DATA ANALYSIS ====================",
        "demand_prediction_agent.py": "# ==================== 4. DEMAND PREDICTION AGENT ====================",
        "tariff_pricing_agent.py": "# ==================== 5. TARIFF PRICING AGENT ====================",
        "monitoring_agent.py": "# ==================== 6. MONITORING & LEARNING AGENT ===================="
    }

    for key, path in files.items():
        print(f"Reading and cleaning {key}...")
        cleaned = clean_code(path)
        # Add header banner at the top of cell
        cell_code = headers[key] + "\n\n" + cleaned
        code_cells_data.append({
            "name": key,
            "code": cell_code
        })

    # Add execution cell
    exec_code = """# ==================== 7. PIPELINE EXECUTION ====================

# 1. Run Preprocessing
acn_df, urbanev_data = run_preprocessing()

# 2. Run Exploratory Data Analysis
run_eda(acn_df, urbanev_data)

# 3. Train Demand Forecasting Models
demand_agent = DemandPredictionAgent()
demand_results = demand_agent.train_and_evaluate(acn_df, urbanev_data)

# 4. Run Dynamic Pricing Optimization
tariff_agent = TariffPricingAgent(demand_agent)
tariff_results = tariff_agent.optimize_tariffs(acn_df, urbanev_data)

# 5. Run Self-Improving Feedback Loop
monitor = MonitoringAgent(demand_agent, tariff_agent)
monitor_results = monitor.run_feedback_loop(acn_df, urbanev_data, n_episodes=10)
"""
    code_cells_data.append({
        "name": "execution",
        "code": exec_code
    })

    # Global namespace for execution
    exec_globals = {}
    exec_count = 1

    # Execute all code cells and populate notebook structure
    for cell_data in code_cells_data:
        name = cell_data["name"]
        code = cell_data["code"]
        
        print(f"Executing cell {exec_count} ({name})...")
        
        # Capture output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = buffer_out = io.StringIO()
        sys.stderr = buffer_err = io.StringIO()
        
        try:
            exec(code, exec_globals)
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
        stdout_val = buffer_out.getvalue()
        stderr_val = buffer_err.getvalue()
        
        outputs = []
        if stdout_val:
            outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": stdout_val.splitlines(keepends=True)
            })
        if stderr_val:
            outputs.append({
                "output_type": "stream",
                "name": "stderr",
                "text": stderr_val.splitlines(keepends=True)
            })
            
        cell = {
            "cell_type": "code",
            "metadata": {},
            "source": to_nb_lines(code),
            "outputs": outputs,
            "execution_count": exec_count
        }
        notebook["cells"].append(cell)
        exec_count += 1

    # Markdown Outro Cell
    outro_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Visualizations and Submission Results\n",
            "\n",
            "All model results, dynamic schedules, and feedback outcomes are successfully computed and printed above.\n",
            "\n",
            "* **Plots Generated:** Saved directly under `outputs/figures/` (viewable interactively via `dashboard/index.html`).\n",
            "* **Metrics CSVs:** Saved under `outputs/metrics/` for final submission."
        ]
    }
    notebook["cells"].append(outro_cell)

    # Write notebook
    nb_path = "OP26_Analytics_Dynamic_Tariff_Optimization.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
        
    print(f"\n  [SUCCESS] Self-contained pre-executed notebook written to: {nb_path}")

if __name__ == "__main__":
    generate_notebook()
