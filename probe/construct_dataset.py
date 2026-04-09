# The dataset construction: top X% neg and top X% pos (farest from decision boundary)

import json
import random
import argparse

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--probe_results_path", type=str, required=True)
parser.add_argument("--origin_data_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--threshold", type=float, default=100.0,
                    help="Percentage of samples to select from each side (0, 100]. "
                         "e.g. 50 means the farthest 50%% of neg and pos samples. "
                         "Default: 100 (keep all).")
args = parser.parse_args()

instruct = "You are a helpful and truthful AI Assistant. You should answer the question as briefly as possible, if you don't know, please just say 'I don't know.'\n"

# load probe results
probe_results = json.load(open(args.probe_results_path))["per_sample"]

# load origin data
origin_data = json.load(open(args.origin_data_path))["samples"]

label = [sample["label"] for sample in probe_results]
correct = [sample["correct"] for sample in origin_data]

assert len(label) == len(correct)
for i in range(len(label)):
    assert label[i] == correct[i]

new_samples = []
cnt = 0
correct_count, wrong_count = 0, 0
for i in range(len(label)):
    if probe_results[i]["correct"] == 0:
        continue
    cnt += 1
    response_text = origin_data[i]["label"] if label[i] == 1 else "I don't know."
    if label[i] == 1:
        correct_count += 1
    else:
        wrong_count += 1
    if not response_text.endswith("."):
        response_text += "."
    new_sample = {"messages":
        [
        {
            "role": "system",
            "content": instruct
        },

        {
            "role": "user",
            "content": origin_data[i]["question"]
        }, 
        {
            "role": "assistant",
            "content": response_text
        }],
        "distance": probe_results[i]["distance"]
    }
    new_samples.append(new_sample)
print(len(new_samples))

# sort by distance
negative_distance_samples = [sample for sample in new_samples if sample["distance"] < 0]
positive_distance_samples = [sample for sample in new_samples if sample["distance"] >= 0]
sorted_neg_samples = sorted(negative_distance_samples, key=lambda x: abs(x["distance"]), reverse=True)
sorted_pos_samples = sorted(positive_distance_samples, key=lambda x: abs(x["distance"]), reverse=True)

neg_k = max(1, int(len(sorted_neg_samples) * args.threshold / 100.0))
pos_k = max(1, int(len(sorted_pos_samples) * args.threshold / 100.0))
selected_neg_samples = sorted_neg_samples[:neg_k]
selected_pos_samples = sorted_pos_samples[:pos_k]
print(f"threshold: {args.threshold}% → selected {neg_k}/{len(sorted_neg_samples)} neg, {pos_k}/{len(sorted_pos_samples)} pos")

final_samples = selected_neg_samples + selected_pos_samples
random.shuffle(final_samples)
print(f"correct_count: {correct_count},\n wrong_count: {wrong_count}")
json.dump(final_samples, open(args.output_path, "w"), indent=4)
