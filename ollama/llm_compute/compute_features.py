# Reads pose CSV files and reshapes them into structured keypoint arrays per frame.

# Computes torso center from left and right shoulder coordinates.

# Extracts motion metrics including lateral sway, velocity magnitude, and shoulder-width variance.

# Aggregates biomechanical stability features for each clip into a structured dictionary.

# Batch-processes all CSV clips and exports motion summaries to a JSON file for downstream classification.

import os
import json
import numpy as np
import pandas as pd

def compute_features(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    motion = df.values.reshape(len(df), 9, 2)

    left_sh = motion[:, 1, :]
    right_sh = motion[:, 2, :]
    torso = (left_sh + right_sh) / 2

    torso_x = torso[:, 0]

    mean_sway_x = float(np.std(torso_x))
    max_sway_x = float(np.max(np.abs(torso_x - np.mean(torso_x))))

    dxy = np.diff(torso, axis=0)
    speed = np.linalg.norm(dxy, axis=1)
    mean_speed = float(np.mean(speed))
    max_speed = float(np.max(speed)) if len(speed) else 0.0

    shoulder_width = np.linalg.norm(left_sh - right_sh, axis=1)
    shoulder_width_std = float(np.std(shoulder_width))

    return {
        "frames": int(motion.shape[0]),
        "mean_sway_x": mean_sway_x,
        "max_sway_x": max_sway_x,
        "mean_speed": mean_speed,
        "max_speed": max_speed,
        "shoulder_width_std": shoulder_width_std
    }


def batch_extract(input_folder: str, output_json: str):
    results = {}

    try:
        for file in sorted(os.listdir(input_folder)):
            if file.endswith(".csv"):
                path = os.path.join(input_folder, file)
                results[file] = compute_features(path)

        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)

    except Exception as e:
        print("Feature extraction failed:", e)


if __name__ == "__main__":
    batch_extract("ollama/csv_clips_5s", "motion_features.json")