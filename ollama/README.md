# OLLAMA PIPELINE ORDER

### STEP 1 — Split Full Video Into 3-Second Clips

Run:

```bash
python split_video.py
```

Output:

```
clips/
    clip_000.mp4
    clip_001.mp4
    ...
```

---

### STEP 2 — Run YOLO Pose On Each Clip (Generate CSV)

You must process each clip to produce CSV files.

Either loop manually or create a small runner script.

Output:

```
csv_clips/
    clip_000.csv
    clip_001.csv
    ...
```

This uses:

```text
VideoInput → Yolov7Pose → CSVOutput
```

from your pose pipeline 

---

### STEP 3 — Run LLM Batch Evaluation On CSV Files

Make sure:

```python
input_folder = "csv_clips"
```

Then run:

```bash
python batch_llm_eval.py
```

Output:

```
llm_results.json
```

Each entry contains:

* Computed features
* LLM prediction

---

### STEP 4 — Evaluate Accuracy

Run:

```bash
python evaluate_llm.py
```

Output:

```
Accuracy: X.XX
```

---

# Visual Overview

```text
Full Video
    ↓
split_video.py
    ↓
clips/*.mp4
    ↓
YOLO pipeline
    ↓
csv_clips/*.csv
    ↓
batch_llm_eval.py
    ↓
llm_results.json
    ↓
evaluate_llm.py
    ↓
Final accuracy
```

# OLLAMA PIPELINE CONCLUSION 

**Accuracy = 0.5** means the LLM is performing at chance level for binary classification.

In plain terms:

* It is not reliably separating stable vs unstable.
* It is not learning — it is reasoning heuristically from the prompt.
* The feature-to-label mapping is not strongly encoded in the LLM.

---

# What This Means Technically

You are giving the model:

* Mean sway
* Max sway
* Mean speed
* Max speed
* Shoulder variance

But you are **not giving thresholds**.

So the LLM must “guess” what counts as excessive.

That leads to inconsistency.

---

# Why This Happens

LLMs are:

* Language reasoning systems
* Not calibrated numeric classifiers
* Not trained on biomechanics thresholds

They lack a reference scale.

---

# How To Improve It (If You Want LLM To Work Better)

## Option 1 — Provide Explicit Threshold Rules (Recommended)

Change your prompt to:

```text
If mean_sway_x > 0.08 OR max_sway_x > 0.25 OR max_speed > 0.30,
classify as unstable (1).
Otherwise classify as stable (0).
```

Now the LLM becomes a deterministic rule interpreter.

Accuracy will likely jump.

---

## Option 2 — Provide Few-Shot Examples

Include:

```text
Example:
mean_sway_x: 0.01 → stability: 0
mean_sway_x: 0.14 → stability: 1
```

This anchors the model.

---

## Option 3 — Stop Using LLM for Classification

Since you already compute features, you can just implement:

```python
if mean_sway_x > threshold:
    return 1
else:
    return 0
```

This will outperform the LLM.

---

# Important Insight

Your experiment just demonstrated something important:

> A general language model is not inherently strong at structured motion classification.