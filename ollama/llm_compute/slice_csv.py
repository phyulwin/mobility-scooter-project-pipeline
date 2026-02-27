import pandas as pd
import math
import os

def slice_csv_into_windows(csv_path: str,
                           output_folder: str,
                           fps: int = 30,
                           window_seconds: int = 5):

    try:
        df = pd.read_csv(csv_path)

        os.makedirs(output_folder, exist_ok=True)

        rows_per_window = fps * window_seconds
        total_rows = len(df)
        num_windows = math.ceil(total_rows / rows_per_window)

        for i in range(num_windows):
            start_row = i * rows_per_window
            end_row = start_row + rows_per_window

            df_window = df.iloc[start_row:end_row].copy()

            if df_window.empty:
                continue

            output_path = os.path.join(
                output_folder,
                f"clip_{i:03d}.csv"
            )

            df_window.to_csv(output_path, index=False)

        print(f"{num_windows} clip CSV files created.")

    except Exception as e:
        print("Slicing failed:", e)


if __name__ == "__main__":
    slice_csv_into_windows(
        "output/yolov7_output_2.csv",
        "csv_clips_5s"
    )