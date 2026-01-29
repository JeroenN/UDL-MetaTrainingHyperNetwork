import math
from typing import Union

import numpy as np
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms

_to_tensor = transforms.ToTensor()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dict):
        self.dict = dict

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, index):
        return self.dict[index]["x"], self.dict[index]["y"]


def preprocess_image(
    img: Union[Image.Image, np.ndarray, torch.Tensor],
    to_tensor: bool = True,
    flatten: bool = True,
    resize: int = 28,
) -> Union[torch.Tensor, Image.Image]:
    """
    Preprocess a single image for the MLP.

    Steps:
    1. Ensure single-channel grayscale.
    2. Resize to 28x28.
    3. Optionally convert to torch.Tensor.
    4. Optionally flatten to a 784-dim vector.

    :param img: Input image (PIL Image, numpy array, or torch.Tensor).
    :param to_tensor: Whether to convert to torch.Tensor.
    :param flatten: Whether to flatten to 1D vector.
    :param resize: Size to resize the image to (resize x resize).
    :return: Preprocessed image as torch.Tensor or PIL Image.
    """

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    if isinstance(img, Image.Image):
        # Ensure grayscale
        if img.mode != "L":
            img = img.convert("L")
        # Resize to NxN
        img = transforms.Resize((resize, resize))(img)

        if not to_tensor:
            return img

        x = _to_tensor(img)

    elif isinstance(img, torch.Tensor):
        x = img
        # Ensure shape is (C, H, W)
        if x.ndim == 2:
            x = x.unsqueeze(0)  # (H, W) -> (1, H, W)
        elif x.ndim == 3:
            if x.shape[-1] in (1, 3) and x.shape[0] not in (1, 3):
                # (H, W, C) -> (C, H, W)
                x = x.permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported tensor shape for image: {x.shape}")

        # If more than 1 channel -> convert to grayscale via simple average
        if x.shape[0] > 1:
            # average across channels
            x = x.mean(dim=0, keepdim=True)  # (1, H, W)

        # Resize to a NxN vector
        x = transforms.Resize((resize, resize))(x)

    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    # Optional Flattening
    if flatten:
        x = x.view(-1)  # (784,)

    return x


def _hf_batch_transform(to_tensor: bool = True, flatten: bool = True, resize: int = 28):
    """
    Returns a HuggingFace-compatible batch transform function which applies preprocessing to a batch of images.

    :param to_tensor: Whether to convert images to torch.Tensor.
    :param flatten: Whether to flatten images to 1D vectors.
    :param resize: Size to resize the images to (resize x resize).
    :return: A function that takes a batch dict and returns a transformed batch dict.
    """

    def transform(batch):
        imgs = batch["image"]
        labels = batch["label"]

        xs = [
            preprocess_image(im, to_tensor=to_tensor, flatten=flatten, resize=resize)
            for im in imgs
        ]

        # Convert to tensors if needed
        if to_tensor:
            x = torch.stack(xs)  # (B, 784) or (B, 1, 28, 28)
            y = torch.tensor(labels, dtype=torch.long)
            return {"x": x, "y": y}
        else:
            return {"image": xs, "label": labels}

    return transform


def split_dataset_by_classes(dataset: dict, class_limit: int) -> list:
    """
    Splits a dataset into multiple datasets, each containing at most `class_limit` classes.

    :param dataset: The original dataset to split.
    :param class_limit: Maximum number of classes per split.
    :return: A list of datasets.
    """
    max_label = max(dataset["train"]["label"])
    num_classes = max_label + 1

    if num_classes < class_limit:
        print(
            f"[WARN] Number of classes in dataset ({num_classes}) is smaller than class_limit ({class_limit}). Returning original dataset."
        )
        return [dataset]

    n_datasets = num_classes // class_limit

    datasets = []
    for start in range(0, n_datasets * class_limit, class_limit):
        end = start + class_limit
        ds = dataset.filter(lambda ex, s=start, e=end: s <= ex["label"] < e)
        datasets.append(ds)

    return datasets


def get_dataset(
    name: str,
    preprocess: bool = False,
    to_tensor: bool = True,
    flatten: bool = True,
    resize: int = 28,
    class_limit: int = None,
) -> list:
    """
    Returns a dataset function based on the dataset name.

    :param name: Name of the dataset ('mnist', 'cifar10', 'imagenet').
    :param preprocess: Whether to apply preprocessing transforms.
    :param to_tensor: Whether to convert images to torch.Tensor.
    :param flatten: Whether to flatten images to 1D vectors.
    :param class_limit: Maximum number of classes per split.
    :return: Corresponding dataset loading function.
    """
    if name == "mnist":  # 1 channel
        dataset = load_dataset("ylecun/mnist")
        # The MNIST dataset consists of 70,000 28x28 black-and-white images of handwritten digits
        # extracted from two NIST databases.
        # There are 60,000 images in the training dataset and 10,000 images in the validation dataset,
        # one class per digit so a total of 10 classes, with 7,000 images (6,000 train and 1,000 test) per class.

    elif name == "fashion_mnist":  # 1 channel
        dataset = load_dataset("zalando-datasets/fashion_mnist")
        # A training set of 60,000 examples and a test set of 10,000 examples.
        # Each example is a 28x28 grayscale image, associated with a label from 10 classes.
        # Shares the same image size and structure of training and testing splits with MNIST.

    elif name == "kmnist":  # 1 channel
        dataset = load_dataset("tanganke/kmnist")
        # Classify images from the KMNIST dataset into one of the 10 classes, representing different Japanese characters.

    # elif name == "asl_mnist": # 1 channel
    # dataset = load_from_hub("Voxel51/American-Sign-Language-MNIST", max_samples=200)
    # The FiftyOne dataset contains 34,627 samples of American Sign Language (ASL) alphabet images,
    # converted from the original Kaggle Sign Language MNIST dataset into a format optimized for computer vision workflows.
    # There are 34,627 rows and the size of downloaded files is 18.8 MB.
    # 24 classes ( Labels 9 (J) and 25 (Z) are excluded as these letters require motion in ASL)

    elif name == "hebrew_chars":  # 1 channel
        dataset = load_dataset("sivan22/hebrew-handwritten-characters")
        # HDD_v0 consists of images of isolated Hebrew characters together with training and test sets subdivision.
        # The images were collected from hand-filled forms.

    elif name == "math_shapes":  # 3 channels
        dataset = load_dataset("prithivMLmods/Math-Shapes")
        # The Math-Symbols dataset is a collection of images representing various mathematical symbols.
        # Size: 131MB (downloaded dataset files), 118MB (auto-connected Parquet files)

    else:
        raise ValueError(f"Dataset {name} is not supported.")

    datasets = (
        split_dataset_by_classes(dataset, class_limit)
        if class_limit is not None
        else [dataset]
    )

    # Apply preprocessing if arg is True
    if preprocess:
        transform_fn = _hf_batch_transform(
            to_tensor=to_tensor,
            flatten=flatten,
            resize=resize,
        )
        # Apply the transform to all splits of all datasets
        for ds in datasets:
            for split in ds:
                ds[split].set_transform(transform_fn)

    return datasets


### DISPLAY UTILITIES FOR TESTING PURPOSES ###


def print_dataset_info(name: str, ds, split: str = "train"):
    """
    Print basic info about a dataset and one example after preprocessing.

    :param name: Name of the dataset.
    :param ds: The dataset object.
    :param split: Which split to use ('train', 'test', etc.).
    """
    print(f"\n----- {name} -----")
    print(ds)

    if hasattr(ds, "keys"):
        if split not in ds:
            split = list(ds.keys())[0]
        dsplit = ds[split]
    else:
        dsplit = ds

    print(f"Using split: {split}, length: {len(dsplit)}")

    ex = dsplit[0]
    print("Example keys:", ex.keys())

    if "x" in ex:
        x = ex["x"]
        y = ex["y"]
        print(f"x type: {type(x)}, x.shape: {tuple(x.shape)}")
        print(f"y: {y}, type: {type(y)}")
    elif "image" in ex:
        img = ex["image"]
        label = ex["label"]
        print(f"Raw image type: {type(img)}")
        if isinstance(img, Image.Image):
            print(f"Image mode: {img.mode}, size: {img.size}")
        print(f"Label: {label}")
    else:
        print("No 'x' or 'image' key found in example – check transforms.")


def show_random_examples(ds, split: str = "train", n: int = 8, title: str = ""):
    if hasattr(ds, "keys"):
        if split not in ds:
            split = list(ds.keys())[0]
        dsplit = ds[split]
    else:
        dsplit = ds

    n = min(n, len(dsplit))
    idxs = np.random.choice(len(dsplit), size=n, replace=False)

    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, idxs):
        ex = dsplit[idx]
        x = ex["x"]
        y = ex["y"]

        if x.ndim == 1:
            side = int(math.isqrt(x.numel()))
            if side * side != x.numel():
                raise ValueError(
                    f"Cannot reshape vector of length {x.numel()} into square image."
                )
            img = x.view(side, side)
        elif x.ndim == 3:
            # (1, H, W) -> (H, W)
            img = x.squeeze(0)
        else:
            raise ValueError(f"Unexpected x.shape: {x.shape}")

        ax.imshow(img.cpu().numpy(), cmap="gray")
        ax.set_title(str(int(y)) if torch.is_tensor(y) else str(y))
        ax.axis("off")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
