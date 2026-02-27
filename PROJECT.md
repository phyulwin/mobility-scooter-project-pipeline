## Project Overview

This project implements a computer-vision–based posture stability assessment system for mobility scooter driver monitoring. Raw driving session videos are segmented into short, fixed-length clips, processed through a YOLOv7 pose-estimation pipeline to extract normalized 2D upper-body keypoints, converted into structured biomechanical motion features (e.g., mean lateral sway, maximum deviation, velocity magnitude, shoulder-width variance), and classified as **stable (0)** or **unstable (1)**. The goal is to determine whether large language models can reason over structured motion summaries as effectively as traditional pose-based motion models.

---

## Current Progress

The full pose extraction pipeline is operational and stable:

* Video segmentation into 3-second clips
* YOLOv7 pose estimation running correctly (PyTorch 2.0 + torchvision 0.15.1 + NumPy 1.24.4)
* CSV keypoint generation per clip
* Motion feature extraction from keypoints
* Integration of a local LLM (Mistral via Ollama HTTP API)
* Batch evaluation pipeline implemented
* Initial classification accuracy: **0.5 (chance level)**

This confirms the infrastructure works end-to-end but highlights weak LLM numeric reasoning performance.

---

## Professor’s Assigned Task

The assignment is to conduct a rigorous comparison between:

* Pose-based structured motion modeling
* Large-model (LLM-based) reasoning approaches

Expectations include:

* Reproducible experimental pipeline
* Quantitative comparison (accuracy, robustness, latency)
* Clear trade-off analysis
* Justified methodological decisions
* Defensible research framing

Deadline target: **before December graduation**, with sufficient experimental validation.

---

## Why 3-Second Clips Instead of Full MP4

The decision to segment into 3-second clips was intentional for technical control:

* Ensures consistent temporal granularity
* Reduces long-sequence variance
* Enables localized instability labeling
* Simplifies alignment between motion features and labels
* Improves evaluation fairness
* Reduces computational overhead

Full-session CSV aggregation was rejected because it obscures short-term instability events.

---

## Methods Attempted and Discarded

* Raw MP4 directly to GPT-4o (API quota/cost + non-reproducibility)
* Feeding raw CSV numeric sequences to LLM (poor temporal reasoning)
* MotionGPT integration (heavy dependency conflicts, config/datamodule complexity)
* Subprocess-based Ollama CLI calls (Windows PATH issues)
* PyTorch 2.6 with YOLOv7 (checkpoint incompatibility)
* Torchvision C++ ops mismatch (custom ops failure)

Each discarded approach either reduced reproducibility, increased instability, or introduced unnecessary engineering overhead.

---

## Current Method

Active pipeline:

```
Video → 3s Clips → YOLO Pose → CSV → Feature Extraction → Local LLM → Metrics
```

Features used:

* Mean sway (x-axis)
* Max sway
* Mean speed
* Max speed
* Shoulder width variance

LLM: **Mistral 7B (local, via Ollama HTTP API)**

Reason for choice:

* Fully local (no quota limits)
* Deterministic under controlled prompting
* Lightweight compared to larger models
* Suitable for structured reasoning experiments
* Reproducible and cost-free

---

## Current Standing

* Pose pipeline: stable and validated
* Feature extraction: functioning correctly
* LLM integration: operational
* Baseline performance: ~50% accuracy

This indicates the LLM lacks calibrated numeric thresholds, supporting the hypothesis that general-purpose LLMs are not inherently strong motion classifiers without structured constraints.

---

## Project Root Structure

```
SeniorProject/
│
└── pipeline/                      (project root)
    │
    ├── pipeline/                  (core pose package)
    │   ├── yolov7.py              (pose pipeline entry wrapper)
    │   └── pipe/
    │       ├── video_input.py     (frame reader)
    │       ├── yolov7_pose.py     (pose model + inference)
    │       ├── csv_output.py      (writes keypoints to CSV)
    │       └── video_output.py    (optional rendering output)
    │
    ├── clips/                     (3-second MP4 segments)
    ├── csv_clips/                 (pose CSV outputs)
    │
    └── ollama/
        ├── pose_features.py       (motion feature extraction)
        ├── batch_llm_eval.py      (LLM classification pipeline)
        ├── evaluate_llm.py        (accuracy computation)
        └── llm_results.json       (LLM output results)
```

---

## How Components Connect

1. `video_input.py` reads frames
2. `yolov7_pose.py` performs pose estimation
3. `csv_output.py` saves normalized keypoints
4. `pose_features.py` computes motion statistics
5. `batch_llm_eval.py` queries local LLM
6. `evaluate_llm.py` computes accuracy

Data flow is strictly modular and reproducible.

---

## Research Position

The project currently evaluates:

> Whether a general-purpose LLM can reason over structured biomechanical motion summaries comparably to motion-specific neural encoders.

The infrastructure is stable. The next strategic direction is either:

* Improve LLM prompting with explicit thresholds
* Introduce few-shot calibration
* Compare against rule-based baseline
* Or integrate motion-specific encoders (e.g., MotionBERT) for stronger performance.