import json
import random
from collections import Counter

input_file = "data/processed/resume_dataset.jsonl"
output_file = "data/processed/balanced_dataset2.jsonl"

data = []
with open(input_file, "r") as f:
    for line in f:
        line = line.strip()
        if line: 
            data.append(json.loads(line))

print(f"Total sentences loaded: {len(data)}")


all_labels = [label for item in data for label in item["labels"]]
label_counts = Counter(all_labels)
print("Original label counts:", label_counts)


for item in data:
    labels_upper = [l.upper() for l in item["labels"]]
    entity_count = sum(1 for l in labels_upper if l != "O")
    item["entity_fraction"] = entity_count / len(item["labels"])


entity_sentences = [item for item in data if item["entity_fraction"] > 0]
non_entity_sentences = [item for item in data if item["entity_fraction"] == 0]

print(f"Sentences with entities: {len(entity_sentences)}")
print(f"Sentences without entities: {len(non_entity_sentences)}")


desired_entity_ratio = 0.7  
num_total = len(data)
num_entity_needed = int(num_total * desired_entity_ratio)

oversampled_entities = random.choices(entity_sentences, k=num_entity_needed)

undersampled_non_entities = []
if non_entity_sentences:
    num_non_entity_needed = num_total - num_entity_needed
    undersampled_non_entities = random.choices(non_entity_sentences, k=num_non_entity_needed)
else:
    print("No non-entity sentences found. All data will be entity-rich sentences.")


balanced_data = oversampled_entities + undersampled_non_entities
random.shuffle(balanced_data)


for item in balanced_data:
    item.pop("entity_fraction", None)

with open(output_file, "w") as f:
    for item in balanced_data:
        f.write(json.dumps(item) + "\n")


all_labels_new = [label for item in balanced_data for label in item["labels"]]
print("Balanced label counts:", Counter(all_labels_new))
print(f"Balanced dataset saved to {output_file}")



