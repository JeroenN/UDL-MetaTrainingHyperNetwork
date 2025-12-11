from utils.dataset import *

DATASET_NAMES = ["mnist", "fashion_mnist", "kmnist", "hebrew_chars", "math_shapes"]

def main():
    # Quick interactive check: prints + plots for each dataset
    
    for name in DATASET_NAMES:
        try:
            data = get_dataset(
                name,
                preprocess=True,
                to_tensor=True,
                flatten=True, 
            )
            print(f"\n\nDataset: {name}")
            print("Shape:", data.shape)
            print_dataset_info(name, data, split="train")
            show_random_examples(data, split="train", n=8, title=name)
        except Exception as e:
            print(f"[WARN] Skipping {name} due to error: {e}")

if __name__ == "__main__":
    main()