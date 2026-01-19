from dataset import *

DATASET_NAMES = ["mnist", "fashion_mnist", "kmnist", "hebrew_chars", "math_shapes"]
CLASS_LIMIT = 5
IMG_SIZE = 28


def main():
    # Quick interactive check: prints + plots for each dataset

    for name in DATASET_NAMES:
        try:
            chunked_sets = get_dataset(
                name,
                preprocess=True,
                to_tensor=True,
                flatten=True,
                resize=IMG_SIZE,
                class_limit=CLASS_LIMIT,
            )
            for i, data in enumerate(chunked_sets):
                print(f"\n\nDataset: {name} (chunk {i+1})")
                print("Shape:", data.shape)
                print_dataset_info(name, data, split="train")
                show_random_examples(
                    data, split="train", n=8, title=f"{name}_chunk{i+1}"
                )
        except Exception as e:
            print(f"[WARN] Skipping {name} due to error: {e}")


if __name__ == "__main__":
    main()
