"""Master script: run the entire pipeline from data collection to model comparison."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent

STEPS = [
    ("Data Collection", "fetch_all.py"),
    ("Data Cleaning", "clean_all.py"),
    ("EDA & Visualization", "eda.py"),
    ("ARIMA Modeling", "run_arima.py"),
    ("LSTM Modeling", "run_lstm.py"),
    ("Model Comparison", "compare_models.py"),
]


def main():
    print("=" * 60)
    print("STOCK MARKET PREDICTION PIPELINE")
    print("=" * 60)

    total_start = time.time()

    for i, (name, script) in enumerate(STEPS, 1):
        print(f"\n{'='*60}")
        print(f"PHASE {i}/{len(STEPS)}: {name}")
        print(f"{'='*60}\n")

        step_start = time.time()
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / script)],
            cwd=str(REPO_ROOT),
            env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)},
        )

        elapsed = time.time() - step_start
        if result.returncode != 0:
            print(f"\nFAILED at phase {i} ({name}) after {elapsed:.0f}s. Stopping.")
            sys.exit(1)

        print(f"\n  Phase {i} completed in {elapsed:.0f}s")

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE — total time: {total:.0f}s ({total/60:.1f} min)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
