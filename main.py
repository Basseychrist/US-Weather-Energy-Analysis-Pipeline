import argparse
from src.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description='Run the weather and energy data analysis pipeline.')
    parser.add_argument('mode', choices=['historical', 'forecast'], help='The mode to run the pipeline in.')
    args = parser.parse_args()

    run_pipeline(args.mode)

if __name__ == "__main__":
    main()
