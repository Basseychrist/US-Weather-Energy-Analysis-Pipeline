import os
import sys
import logging
import argparse

# --- Dynamically determine project paths ---
# This is more reliable than hardcoding the full user path.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'pipeline.log')

# --- Create log directory ---
try:
    os.makedirs(LOG_DIR, exist_ok=True)
except Exception as e:
    print(f"CRITICAL: Could not create log directory '{LOG_DIR}'. Error: {e}", file=sys.stderr)
    sys.exit(1)

# --- Configure Logging Forcefully ---
# This is the most direct way to set up file and stream logging.
# It will remove and replace any handlers configured by other modules.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

# --- Import pipeline AFTER logging is configured ---
try:
    from src.pipeline import run_pipeline
except ImportError as e:
    logging.error(f"Failed to import 'run_pipeline'. Please check src/pipeline.py. Error: {e}")
    sys.exit(1)


def main():
    """Defines the command-line interface and runs the pipeline."""
    logging.info("--- Application starting. ---")
    parser = argparse.ArgumentParser(description="Run the US Weather Energy Analysis Pipeline.")
    parser.add_argument('mode', choices=['historical', 'realtime'], help="Specify the pipeline mode.")
    args = parser.parse_args()

    try:
        run_pipeline(args.mode)
        logging.info("--- Pipeline execution completed successfully. ---")
    except Exception as e:
        logging.error("--- A fatal error occurred during pipeline execution. ---", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
