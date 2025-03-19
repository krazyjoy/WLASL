import json
from collections import Counter

# Load the WLASL JSON file
file_path = 'WLASL_v0.3.json'

with open(file_path, 'r') as ipf:
    content = json.load(ipf)

# Count occurrences of each gloss
gloss_count = Counter({entry["gloss"]: len(entry["instances"]) for entry in content})

# Define subset sizes
subset_sizes = [100, 300, 1000, 2000]

# Generate subsets
for k in subset_sizes:
    # Get top-K glosses by frequency
    top_k_glosses = {gloss for gloss, _ in gloss_count.most_common(k)}
    
    # Filter dataset to include only these glosses
    subset = [entry for entry in content if entry["gloss"] in top_k_glosses]
    
    # Save to a new JSON file
    subset_filename = f"WLASL{k}.json"
    with open(subset_filename, "w") as opf:
        json.dump(subset, opf, indent=4)

    print(f"Created {subset_filename} with {k} glosses.")

# Count instances per split in the full dataset
cnt_train, cnt_val, cnt_test = 0, 0, 0

for entry in content:
    for inst in entry["instances"]:
        split = inst.get("split", "unknown")
        if split == "train":
            cnt_train += 1
        elif split == "val":
            cnt_val += 1
        elif split == "test":
            cnt_test += 1
        else:
            raise ValueError(f"Invalid split: {split}")

print(f"Total glosses in full dataset: {len(content)}")
print(f"Total samples: {cnt_train + cnt_val + cnt_test}")
print(f"Training samples: {cnt_train}, Validation samples: {cnt_val}, Test samples: {cnt_test}")
