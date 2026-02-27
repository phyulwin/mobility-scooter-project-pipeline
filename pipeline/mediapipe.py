"""
Module for processing video files to extract human pose landmarks using MediaPipe.
This module provides a pipeline for reading video frames, detecting pose landmarks
using MediaPipe's pose estimation model, and writing the results to a CSV file.

It tracks failed frames during processing and reports summary statistics upon completion.
"""

from tqdm import tqdm
from .pipe.video_input import VideoInput
from .pipe.mediapipe_pose import MediapipePose
from .pipe.csv_output import CSVOutput
from .utils import time_func

@time_func
def process_file(in_file, out_file):
    failed_frames = []

    cap = VideoInput(in_file)
    column_names = [f'{j}{i}' for i in range(9) for j in 'xyz']
    pmodel = MediapipePose()
    data_writer = CSVOutput(out_file, column_names)

    print(f"Estimation started for {in_file}")
    for i in tqdm(range(cap.total)):
        frame = cap.process()
        if frame is not None:
            landmarks = pmodel.process(frame)
            if landmarks is not None:
                data_writer.process(landmarks)
            else:
                failed_frames.append(i)
        else:
            failed_frames.append(i)
    print(f"Saved {cap.total-len(failed_frames)} estimations to {out_file}")
    if failed_frames:
        print(f"Estimation failed in frames: {failed_frames}")
