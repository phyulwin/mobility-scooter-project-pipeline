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


# NOTES

# The motion features are extracted using deterministic geometric computations with NumPy.

# The 9 keypoints per frame are reshaped into a (frames, 9, 2) array, then the torso center is computed as the midpoint between the left and right shoulder coordinates.

# Lateral sway is calculated as the standard deviation and max deviation of the torso’s x-position over time (np.std, np.max).

# Speed is derived from frame-to-frame torso displacement using first differences (np.diff) and Euclidean norm (np.linalg.norm).

# Shoulder variance is computed as the standard deviation of the Euclidean distance between left and right shoulders across frames.


# Each frame contains 9 body landmarks, and each landmark has an (x, y) coordinate.

# So per frame you have 9 × 2 values = 18 numbers, which are reshaped into a (9, 2) structure.

# These 9 keypoints represent selected upper-body joints detected by YOLOv7 (e.g., shoulders, elbows, etc.).

# Over time, this becomes a (frames, 9, 2) array — meaning for every frame you track 9 spatial points in 2D space.