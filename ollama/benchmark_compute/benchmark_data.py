# What this does:

# Keeps your 10,481 frame annotations untouched.

# Automatically slices them into 5-second windows.

# Converts each window into one binary label.

# Produces 11 ground-truth labels aligned with your 11 LLM predictions.

import pandas as pd
import math
import json

def generate_clip_labels(benchmark_csv: str,
                         clip_duration: float = 5.0,
                         output_json: str = "benchmark_data_json.json"):
    try:
        df = pd.read_csv(benchmark_csv, header=None)
        df.columns = ["time", "state"]

        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df = df.dropna(subset=["time"])

        total_duration = df["time"].max()
        num_clips = math.ceil(total_duration / clip_duration)

        clip_labels = {}

        for i in range(num_clips):
            start = i * clip_duration
            end = start + clip_duration

            df_window = df[
                (df["time"] >= start) &
                (df["time"] < end)
            ].copy()

            if df_window.empty:
                continue

            df_window["binary"] = df_window["state"].apply(
                lambda x: 0 if x.strip().lower() == "stationary" else 1
            )

            label = 1 if df_window["binary"].mean() >= 0.5 else 0
            clip_labels[f"clip_{i:03d}.csv"] = label

        with open(output_json, "w") as f:
            json.dump(clip_labels, f, indent=2)

    except Exception as e:
        print("Failed:", e)


if __name__ == "__main__":
    generate_clip_labels(r"C:\Users\646ca\Downloads\CPP\SeniorProject\pipeline\output\p63_front_1.mp4.csv")