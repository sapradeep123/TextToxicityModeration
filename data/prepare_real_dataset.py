"""
prepare_real_dataset.py
------------------------
Downloads a real-world public toxicity dataset and prepares it as
data/raw/train.csv, which text_toxicity_moderation.ipynb loads automatically.

Why Civil Comments instead of the Kaggle Jigsaw CSV?
The original Jigsaw Toxic Comment Classification dataset is hosted on Kaggle
and requires a Kaggle account + API token to download (see data/README.md
for those steps if you'd prefer it). Civil Comments is the dataset behind
the related Jigsaw "Unintended Bias in Toxicity Classification" competition
and is mirrored publicly on the Hugging Face Hub with NO authentication
required, so this script uses it to make the "real dataset" reproducible
for anyone cloning this repo.

Usage:
    pip install datasets
    python data/prepare_real_dataset.py
"""

import os

from datasets import load_dataset

N_ROWS = 80_000          # subsample size (full dataset has ~1.8M rows)
TOXICITY_THRESHOLD = 0.5  # Civil Comments gives a continuous 0-1 toxicity score
RANDOM_SEED = 42
OUTPUT_PATH = os.path.join("data", "raw", "train.csv")


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Downloading google/civil_comments from the Hugging Face Hub ...")
    ds = load_dataset("google/civil_comments", split="train")
    ds = ds.shuffle(seed=RANDOM_SEED)
    ds = ds.select(range(min(N_ROWS, len(ds))))

    df = ds.to_pandas()
    df["toxic"] = (df["toxicity"] >= TOXICITY_THRESHOLD).astype(int)
    df = df.rename(columns={"text": "comment_text"})
    df = df[["comment_text", "toxic"]]

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(f"Toxic: {df['toxic'].sum()}  |  Non-toxic: {(df['toxic'] == 0).sum()}")


if __name__ == "__main__":
    main()
