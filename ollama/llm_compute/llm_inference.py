# Sends structured motion features to a local Mistral model via the Ollama HTTP API.

# Constructs a controlled prompt defining stability criteria using biomechanical metrics.

# Forces the model to return a strict JSON classification (0 = stable, 1 = unstable).

# Handles API responses and parses model output into structured predictions.

# Batch-processes all feature summaries and saves LLM classification results to JSON.

import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"


def ask_llm(features: dict) -> dict:
    prompt = f"""
You are evaluating posture stability.

Stable = minimal sway and smooth motion.
Unstable = excessive sway or abrupt shifts.

Metrics:
Mean sway x: {features['mean_sway_x']}
Max sway x: {features['max_sway_x']}
Mean speed: {features['mean_speed']}
Max speed: {features['max_speed']}
Shoulder width std: {features['shoulder_width_std']}

Respond STRICTLY in JSON:
{{"stability": 0 or 1}}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False
            }
        )

        result_text = response.json()["response"].strip()

        return json.loads(result_text)

    except Exception as e:
        print("LLM request failed:", e)
        return {"stability": None}


def batch_llm(feature_json: str, output_json: str):
    try:
        with open(feature_json, "r") as f:
            features = json.load(f)

        predictions = {}

        for clip, feats in features.items():
            predictions[clip] = ask_llm(feats)

        with open(output_json, "w") as f:
            json.dump(predictions, f, indent=2)

    except Exception as e:
        print("Batch inference failed:", e)


if __name__ == "__main__":
    batch_llm("motion_features.json", "llm_predictions.json")