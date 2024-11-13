import os
import subprocess
import time
import glob
import argparse

def main(start_model=None):
    # Create necessary directories
    dirs = [
        "/volume/work_dir",
        "/volume/outputs/models",
        "/volume/outputs/data/selfplay",
        "/volume/outputs/data/holdout",
        "/volume/outputs/sgf"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

    # Determine starting point
    if start_model is None:
        # Path 1: Start from bootstrap
        print("Bootstrapping initial model...")
        subprocess.run([
            "python3", "bootstrap.py",
            "--work_dir=/volume/work_dir",
            "--export_path=/volume/outputs/models/bootstrap",
            "--create_bootstrap=True"
        ], check=True)
        latest_model = "bootstrap"
        start_iteration = 0
    else:
        # Path 2: Start from specified model
        latest_model = start_model
        # Extract iteration number from model name (e.g., "000014" -> 14)
        start_iteration = int(start_model)

    for iteration in range(start_iteration, start_iteration + 1000):
        print(f"\nIteration {iteration + 1}/1000")
        
        # Self-play using latest model
        print(f"Starting self-play with model {latest_model}...")
        for game in range(50):
            print(f"Playing game {game + 1}...")
            subprocess.run([
                "python3", "selfplay.py",
                f"--load_file=/volume/outputs/models/{latest_model}",
                "--selfplay_dir=/volume/outputs/data/selfplay",
                "--holdout_dir=/volume/outputs/data/holdout",
                "--sgf_dir=/volume/outputs/sgf"
            ], check=True)

        # Train on all selfplay data
        new_model = f"{iteration+1:06d}"  # 000001, 000002, etc.
        print(f"Training model {new_model}...")
        # Get all selfplay files
        selfplay_files = glob.glob("/volume/outputs/data/selfplay/*")
        # Construct command with expanded file list
        train_cmd = [
            "python3", "train.py"
        ] + selfplay_files + [
            "--work_dir=/volume/work_dir",
            f"--export_path=/volume/outputs/models/{new_model}"
        ]
        subprocess.run(train_cmd, check=True)
        
        latest_model = new_model
        print(f"Completed iteration {iteration + 1}, latest model: {latest_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run AlphaGo Zero training loop')
    parser.add_argument('--start-model', type=str, help='Model number to start from (e.g., 000014)', default=None)
    args = parser.parse_args()
    
    main(args.start_model)