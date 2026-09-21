"""Splits the manifest into train/val/test, by photo.

800/100/100 photos are taken from each of the three photo batches separately
(the original 1000 plus the two added ones), not from the pooled 3000. Seed 42
per batch, so the original 1000 photos get the same split as in the earlier
experiments.

A photo contributes examples to every category, so splitting by manifest entry
would leak the same photo from train into test.
"""
import json
import os
import random

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SRC = os.path.join(DATA, "llava_multi_v6_manifest.json")
OUT_TRAIN = os.path.join(DATA, "llava_multi_v6_train.json")
OUT_VAL = os.path.join(DATA, "llava_multi_v6_val.json")
OUT_TEST = os.path.join(DATA, "llava_multi_v6_test.json")

SEED = 42
N_VAL_PER_BATCH = 100
N_TEST_PER_BATCH = 100

with open(SRC) as f:
    data = json.load(f)

base_ids = sorted({entry["base_id"] for entry in data})

BATCHES = {
    "original": [b for b in base_ids if 1 <= int(b) <= 1000],
    "batch1":   [b for b in base_ids if 1001 <= int(b) <= 2000],
    "batch2":   [b for b in base_ids if 2001 <= int(b) <= 3000],
}

train_ids, val_ids, test_ids = set(), set(), set()
for name, ids in BATCHES.items():
    ids = sorted(ids)
    rng = random.Random(SEED)
    rng.shuffle(ids)
    batch_test = ids[:N_TEST_PER_BATCH]
    batch_val = ids[N_TEST_PER_BATCH:N_TEST_PER_BATCH + N_VAL_PER_BATCH]
    batch_train = ids[N_TEST_PER_BATCH + N_VAL_PER_BATCH:]
    print(f"{name}: total={len(ids)} train={len(batch_train)} val={len(batch_val)} test={len(batch_test)}")
    train_ids.update(batch_train)
    val_ids.update(batch_val)
    test_ids.update(batch_test)

splits = {"train": [], "val": [], "test": []}
for entry in data:
    base_id = entry["base_id"]
    if base_id in test_ids:
        splits["test"].append(entry)
    elif base_id in val_ids:
        splits["val"].append(entry)
    elif base_id in train_ids:
        splits["train"].append(entry)
    else:
        raise ValueError(f"base_id {base_id} not assigned to any split")

with open(OUT_TRAIN, "w") as f:
    json.dump(splits["train"], f, indent=2)
with open(OUT_VAL, "w") as f:
    json.dump(splits["val"], f, indent=2)
with open(OUT_TEST, "w") as f:
    json.dump(splits["test"], f, indent=2)

print(f"\nbase images: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
print(f"entries: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

for name, rows in splits.items():
    by_n = {}
    for e in rows:
        n = len(e["ground_truth"]["distortion_types"])
        by_n[n] = by_n.get(n, 0) + 1
    print(f"  {name} balance: " + ", ".join(f"{n}:{by_n.get(n,0)}" for n in [0, 1, 2, 3]))
