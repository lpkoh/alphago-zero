import os
import subprocess
import time
import glob

def main():
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

    # Bootstrap initial model
    print("Bootstrapping initial model...")
    subprocess.run([
        "python3", "bootstrap.py",
        "--work_dir=/volume/work_dir",
        "--export_path=/volume/outputs/models/bootstrap",
        "--create_bootstrap=True"
    ], check=True)

    # Training loop
    latest_model = "bootstrap"
    for iteration in range(1000):
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
    main()