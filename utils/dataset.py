# STIL UNFINISHED
# Will migrate everything to the utils/ and create a preprocessing function.
# Might have to check every dataset and how each one is loaded. Also what kind of preprocessing each one needs.
# Last: add documentation.


from datasets import load_dataset
from torchvision import transforms
import numpy as np
import torch
from PIL import Image
from typing import Union
from matplotlib import pyplot as plt


_to_tensor = transforms.ToTensor()
_resize_28 = transforms.Resize((28, 28))


def preprocess_image(
    img: Union[Image.Image, np.ndarray, torch.Tensor],
    to_tensor: bool = True,
    flatten: bool = True,
) -> Union[torch.Tensor, Image.Image]:
    """
    Preprocess a single image for the MLP.

    Steps:
    1. Ensure single-channel grayscale.
    2. Resize to 28x28.
    3. Optionally convert to torch.Tensor.
    4. Optionally flatten to a 784-dim vector.
    """

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    if isinstance(img, Image.Image):
        # Ensure grayscale
        if img.mode != "L":
            img = img.convert("L")
        # Resize to 28x28
        img = _resize_28(img)

        if not to_tensor:
            return img

        # Convert to tensor: shape (1, 28, 28), float32 in [0, 1]
        x = _to_tensor(img)  # (1, 28, 28)

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

        # Resize to a 28x28 MNIST-like vector
        x = _resize_28(x)

    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    # Optional Flattening
    if flatten:
        x = x.view(-1)  # (784,)

    return x


def _hf_batch_transform(to_tensor: bool = True, flatten: bool = True):
    """
    Returns a HuggingFace-compatible batch transform function that:
    - Reads 'image' and 'label' from a batch dict.
    - Applies preprocess_image to each image.
    - Returns {'x': tensor_batch, 'y': labels} if to_tensor=True.
      Otherwise returns {'image': processed_images, 'label': labels}.
    """

    def transform(batch):
        imgs = batch["image"]
        labels = batch["label"]

        xs = [preprocess_image(im, to_tensor=to_tensor, flatten=flatten) for im in imgs]

        if to_tensor:
            x = torch.stack(xs)  # (B, 784) or (B, 1, 28, 28)
            y = torch.tensor(labels, dtype=torch.long)
            return {"x": x, "y": y}
        else:
            return {"image": xs, "label": labels}

    return transform


def get_dataset(
    name: str, preprocess: bool = False, to_tensor: bool = True, flatten: bool = True
):
    """
    Returns a dataset function based on the dataset name.

    :param name: Name of the dataset ('mnist', 'cifar10', 'imagenet').
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

    elif name == "hebrew_chars":  # 1 channel
        dataset = load_dataset("sivan22/hebrew-handwritten-characters")
        # HDD_v0 consists of images of isolated Hebrew characters together with training and test sets subdivision.
        # The images were collected from hand-filled forms.

    elif name == "math_shapes":  # 3 channels
        dataset = load_dataset("prithivMLmods/Math-Shapes")
        # The Math-Symbols dataset is a collection of images representing various mathematical symbols.
        # Size: 131MB (downloaded dataset files), 118MB (auto-connected Parquet files)
        # 20,000 Rows of 224x224 RGB images
        # Classes: 128 different mathematical symbols (e.g., circle, plus, minus, etc.)

    else:
        raise ValueError(f"Dataset {name} is not supported.")

    # Apply preprocessing if arg is True
    if preprocess:
        transform_fn = _hf_batch_transform(to_tensor=to_tensor, flatten=flatten)

        # Apply transform to all splits (e.g. train/test)
        if hasattr(dataset, "keys"):
            for split in dataset.keys():
                dataset[split].set_transform(transform_fn)
        else:
            # Single-split case
            dataset.set_transform(transform_fn)

    return dataset


### DISPLAY UTILITIES FOR TESTING PURPOSES ###


def print_dataset_info(name: str, ds, split: str = "train"):
    """
    Print basic info about a dataset and one example after preprocessing.
    """
    print(f"\n========== {name} ==========")
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
    """
    Plot a few random examples from a (preprocessed) dataset split.

    Assumes each example has:
      - 'x': flattened (784,) OR (1, 28, 28)
      - 'y': label
    """
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
            img = x.view(28, 28)
        elif x.ndim == 3:
            img = x.squeeze(0)  # (1, 28, 28) -> (28, 28)
        else:
            raise ValueError(f"Unexpected x.shape: {x.shape}")

        ax.imshow(img.cpu().numpy(), cmap="gray")
        ax.set_title(str(y))
        ax.axis("off")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
