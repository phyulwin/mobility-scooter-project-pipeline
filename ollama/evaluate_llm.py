# import json

# try:
#     with open("ollama/output/llm_predictions.json") as f:
#         predictions = json.load(f)

#     with open("ollama/output/benchmark_data_json.json") as f:
#         ground_truth = json.load(f)

#     correct = 0
#     total = 0

#     for clip, true_label in ground_truth.items():

#         if clip not in predictions:
#             continue

#         pred = predictions[clip].get("stability", None)

#         if pred is None:
#             continue

#         # Ensure strict binary
#         pred_binary = 1 if float(pred) >= 0.5 else 0

#         if pred_binary == true_label:
#             correct += 1

#         total += 1

#     accuracy = correct / total if total > 0 else 0

#     print("Total Clips Compared:", total)
#     print("Correct Predictions:", correct)
#     print("Accuracy:", round(accuracy, 4))

# except Exception as e:
#     print("Evaluation failed:", e)

# # Total Clips Compared: 70 (should be 69)
# # Correct Predictions: 35
# # Accuracy: 0.5

import json

try:
    with open("ollama/output/llm_predictions.json") as f:
        predictions = json.load(f)

    with open("ollama/output/benchmark_data_json.json") as f:
        ground_truth = json.load(f)

    TP = TN = FP = FN = 0

    for clip, true_label in ground_truth.items():

        if clip not in predictions:
            continue

        pred = predictions[clip].get("stability", None)
        if pred is None:
            continue

        pred_binary = 1 if float(pred) >= 0.5 else 0

        if true_label == 1 and pred_binary == 1:
            TP += 1
        elif true_label == 0 and pred_binary == 0:
            TN += 1
        elif true_label == 0 and pred_binary == 1:
            FP += 1
        elif true_label == 1 and pred_binary == 0:
            FN += 1

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0

    print("Confusion Matrix")
    print("----------------")
    print(f"TP (Correct Unstable): {TP}")
    print(f"TN (Correct Stable):   {TN}")
    print(f"FP (False Unstable):   {FP}")
    print(f"FN (Missed Unstable):  {FN}")
    print()
    print("Metrics")
    print("-------")
    print(f"Accuracy:  {round(accuracy,4)}")
    print(f"Precision: {round(precision,4)}")
    print(f"Recall:    {round(recall,4)}")

except Exception as e:
    print("Evaluation failed:", e)

# Confusion Matrix
# ----------------
# TP (Correct Unstable): 3
# TN (Correct Stable):   32
# FP (False Unstable):   35
# FN (Missed Unstable):  0

# Metrics
# -------
# Accuracy:  0.5
# Precision: 0.0789
# Recall:    1.0