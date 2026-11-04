# src/prepare_data.py
import csv
import os
import sys
import zipfile
import numpy as np
from PIL import Image
from config import DATA_DIR, EMOTIONS_EN

CSV_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def main(zip_path):
    csv.field_size_limit(2**31 - 1)  # sys.maxsize Windows-та OverflowError береді
    counts = {}
    with zipfile.ZipFile(zip_path) as z, z.open("fer2013.csv") as f:
        reader = csv.DictReader(line.decode() for line in f)
        for i, row in enumerate(reader):
            split = "train" if row["Usage"] == "Training" else "test"
            label = CSV_LABELS[int(row["emotion"])]
            folder = os.path.join(DATA_DIR, split, label)
            os.makedirs(folder, exist_ok=True)
            pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
            Image.fromarray(pixels).save(os.path.join(folder, f"{i:05d}.png"))
            counts[(split, label)] = counts.get((split, label), 0) + 1
    for split in ("train", "test"):
        print(split, {e: counts.get((split, e), 0) for e in EMOTIONS_EN})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(DATA_DIR), "fer2013.zip"))
