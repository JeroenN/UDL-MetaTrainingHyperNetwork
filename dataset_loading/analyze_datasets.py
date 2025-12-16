
import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from .dataset import get_dataset, preprocess_image

OUTPUT_DIR = "figures/datasets"
DATASET_NAMES = ["mnist", "fashion_mnist", "kmnist", "hebrew_chars", "math_shapes"]

def ensure_outdir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_class_distribution(ds, split="train"):
    """Return a Counter(label -> count) for the given split."""
    if hasattr(ds, "keys"):
        if split not in ds:
            split = list(ds.keys())[0]
        dsplit = ds[split]
    else:
        dsplit = ds

    labels = dsplit["label"]  # raw HF labels, no transform
    return Counter(int(l) for l in labels)


def save_class_distribution_plot(name, label_counts):
    """Save a class distribution bar plot as transparent PNG."""
    classes = sorted(label_counts.keys())
    counts = [label_counts[c] for c in classes]

    plt.figure(figsize=(8, 4))
    plt.bar(classes, counts)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(f"Class distribution — {name}")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"{name}_class_distribution.png")
    plt.savefig(out_path, dpi=150, transparent=True, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved class distribution to {out_path}")


def save_one_example_per_class(name, ds, split="train"):
    """
    Save a grid image with one random preprocessed example per class.
    Uses preprocess_image to enforce grayscale 28x28.
    """
    if hasattr(ds, "keys"):
        if split not in ds:
            split = list(ds.keys())[0]
        dsplit = ds[split]
    else:
        dsplit = ds

    labels = dsplit["label"]
    unique_classes = sorted(set(int(l) for l in labels))

    # Build indices per class for sampling
    class_to_indices = {c: [] for c in unique_classes}
    for idx, lb in enumerate(labels):
        c = int(lb)
        class_to_indices[c].append(idx)

    # Collect one random index per class
    rng = np.random.default_rng()
    chosen_indices = {}
    for c, idxs in class_to_indices.items():
        if len(idxs) == 0:
            continue
        chosen_indices[c] = int(rng.choice(idxs))

    n_classes = len(chosen_indices)
    if n_classes == 0:
        print(f"[WARN] No classes found for {name}, skipping example grid.")
        return

    # Plot
    n_cols = min(n_classes, 10)
    n_rows = int(np.ceil(n_classes / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    if n_rows == 1:
        axes = np.array(axes).reshape(1, -1)

    for ax in axes.ravel():
        ax.axis("off")  # default off; we'll turn on where we draw

    for i, c in enumerate(sorted(chosen_indices.keys())):
        idx = chosen_indices[c]
        ex = dsplit[idx]
        raw_img = ex["image"]

        # Use your preprocessing to get (1, 28, 28) tensor
        x = preprocess_image(raw_img, to_tensor=True, flatten=False)  # (1, 28, 28)
        img = x.squeeze(0).cpu().numpy()  # (28, 28)

        r = i // n_cols
        col = i % n_cols
        ax = axes[r, col]
        ax.imshow(img, cmap="gray")
        ax.set_title(str(c))
        ax.axis("off")

    plt.suptitle(f"One example per class — {name}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(OUTPUT_DIR, f"{name}_examples_per_class.png")
    plt.savefig(out_path, dpi=150, transparent=True, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved examples-per-class grid to {out_path}")


def print_basic_stats(name, ds):
    """Print some basic dataset stats."""
    print(f"\n========== {name} ==========")
    print(ds)

    if hasattr(ds, "keys"):
        for split_name, dsplit in ds.items():
            n = len(dsplit)
            labels = dsplit["label"]
            n_classes = len(set(int(l) for l in labels))
            print(f"Split '{split_name}': {n} samples, approx. {n_classes} classes")
    else:
        n = len(ds)
        labels = ds["label"]
        n_classes = len(set(int(l) for l in labels))
        print(f"Single split: {n} samples, approx. {n_classes} classes")


def main():
    ensure_outdir()

    for name in DATASET_NAMES:
        try:
            # For analysis we use raw datasets (no set_transform),
            # and call preprocess_image only where needed.
            ds = get_dataset(
                name,
                preprocess=False,  # keep raw HF data, we handle preprocessing manually
            )

            print_basic_stats(name, ds)

            # Class distribution
            label_counts = compute_class_distribution(ds, split="train")
            save_class_distribution_plot(name, label_counts)

            # One example per class
            save_one_example_per_class(name, ds, split="train")

        except Exception as e:
            print(f"[WARN] Skipping {name} due to error: {e}")


if __name__ == "__main__":
    main()
