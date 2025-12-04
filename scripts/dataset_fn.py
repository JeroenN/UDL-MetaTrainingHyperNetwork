# STIL UNFINISHED
# Will migrate everything to the utils/ and create a preprocessing function.
# Might have to check every dataset and how each one is loaded. Also what kind of preprocessing each one needs.
# Last: add documentation.


from datasets import load_dataset

def get_dataset(name: str):
    """
    Returns a dataset function based on the dataset name.

    :param name: Name of the dataset ('mnist', 'cifar10', 'imagenet').
    :return: Corresponding dataset loading function.
    """
    if name == 'mnist': # 1 channel
        return lambda: load_dataset('ylecun/mnist')
        # The MNIST dataset consists of 70,000 28x28 black-and-white images of handwritten digits 
        # extracted from two NIST databases. 
        # There are 60,000 images in the training dataset and 10,000 images in the validation dataset, 
        # one class per digit so a total of 10 classes, with 7,000 images (6,000 train and 1,000 test) per class.
    
    elif name == 'fashion_mnist': # 1 channel
        return lambda: load_dataset('zalando-datasets/fashion_mnist')
        # A training set of 60,000 examples and a test set of 10,000 examples. 
        # Each example is a 28x28 grayscale image, associated with a label from 10 classes. 
        # Shares the same image size and structure of training and testing splits with MNIST.

    elif name == 'kmnist': # 1 channel
        return lambda: load_dataset('tanganke/kmnist')
        # Classify images from the KMNIST dataset into one of the 10 classes, representing different Japanese characters.

    elif name == 'hebrew_chars': # 1 channel
        return lambda: load_dataset('sivan22/hebrew-handwritten-characters')
        # HDD_v0 consists of images of isolated Hebrew characters together with training and test sets subdivision. 
        # The images were collected from hand-filled forms.

    elif name == 'math_shapes': # 3 channels
        return lambda: load_dataset('prithivMLmods/Math-Shapes')
        # The Math-Symbols dataset is a collection of images representing various mathematical symbols. 
        # Size: 131MB (downloaded dataset files), 118MB (auto-connected Parquet files)
        # 20,000 Rows of 224x224 RGB images
        # Classes: 128 different mathematical symbols (e.g., circle, plus, minus, etc.)

    #elif name == 'cifar10': # 3 channels
        #return lambda: load_dataset('uoft-cs/cifar10')
        # The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
        # There are 50000 training images and 10000 test images. 
        # The dataset is divided into five training batches and one test batch, each with 10000 images. 
        # The test batch contains exactly 1000 randomly-selected images from each class. 
        # The training batches contain the remaining images in random order, 
        # but some training batches may contain more images from one class than another. 
        # Between them, the training batches contain exactly 5000 images from each class.
    
    #elif name == 'imagenet256': # 3 channels
       # return lambda: load_dataset('benjamin-paine/imagenet-1k-256x256')
        # This dataset spans 1000 object classes and 256x256 resolution images. 
        # It contains 1,281,167 training images, 50,000 validation images and 100,000 test images.

    else:
        raise ValueError(f"Dataset {name} is not supported.")