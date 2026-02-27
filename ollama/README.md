# to-do
- feed LLM with some annotations and show what segments are labeled as stable and unstable by our collaborators
- improve overall LLM accuracy to 70% and above
- add intructions on how to run the ollama-yolo pipeline (update and clean requirements.txt)
- make the code reproducible 

# OLLAMA PIPELINE ORDER 
#### (Brainstormed with ChatGPT)

![pipeline.png](/ollama/ollama-yolo-pipeline.png)

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

we must process each clip to produce CSV files.

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

from wer pose pipeline 

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

We are giving the model:

* Mean sway
* Max sway
* Mean speed
* Max speed
* Shoulder variance

But we are **not giving thresholds**.

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

# Important Insight

The experiment just demonstrated something important:

> A general language model is not inherently strong at structured motion classification.