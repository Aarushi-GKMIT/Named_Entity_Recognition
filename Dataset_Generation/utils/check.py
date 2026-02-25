import json

dataset_path = "data/processed/balanced_dataset2.jsonl"

dataset = []

with open(dataset_path, "r", encoding="utf-8") as f:
    first_char = f.read(1)
    f.seek(0)

    if first_char == "[":
        dataset = json.load(f)
    else:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

print("Total samples:", len(dataset))


from collections import Counter

all_labels = []
for row in dataset:
    all_labels.extend(row["labels"])

label_counts = Counter(all_labels)


import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))
plt.bar(label_counts.keys(), label_counts.values())
plt.xticks(rotation=45)
plt.title("Overall Label Distribution (Including O)")
plt.xlabel("NER Labels")
plt.ylabel("Token Count")
plt.tight_layout()
plt.show()

