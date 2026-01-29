# UDL-MetaTrainingHyperNetwork

## Goal of project
This project took inspiration from two papers:
### 1. HyperFast: Instant Classification for Tabular Data

In this paper the authors meta-train a hypernetwork to be able to generate a target network that can perform tabular classification with no further training. This does require the dataset that inference is performed on to be fully labeled. The authors use random projections on the dataset samples and then perform PCA in order to get a dataset level embedding. This embedding is then fed to the hypernetwork.

### 2. MT3: Meta Test-Time Training for Self-Supervised Test-Time Adaption

In this paper the authors meta-train a network to be able to do zero-shot classification on permuted data. Within the outerloop the the meta-network is trained to minimize the cross entropy loss of labeled samples. In the innerloop there are a set number of transformations applied to the images. Each kind of transformation gets its own label. The model is then trained to identify the transformations applied.

### Our approach
We created a way, using meta-learning, for the hypernetwork to apply an unsupervised update during inference in order to increase its classification accuracy. Instead of conditioning our hypernetwork on data that PCA was applied to, we condition the hypernetwork on the VAE distribution produced by the samples of the dataset. The reason for not using PCA because in the approach by hyperfast they assume that you have a labeled dataset, we do not. 

We took inspiration from MT3 by applying within the innerloop a way to learn usefull features for classifictation. The difference between our approach is that it also alligns better with the outerloop task of classifying these images. 

## Setup & How to Use

### 1. Install dependencies
After cloning the repository, install the required packages:

```bash
pip intall -r requirements.txt
```
### 2. Configure parameters
All runtime and training parameters are defined in config.json.
Adjust this file according to your needs. The current version contains an instance of our current experimental setup.

### 3. Run the main pipeline

```bash
python main.py
```

#### VAE Checkpoints
* VAE models are only retrained if no matching checkpoint is found locally.
* Either store VAE checkpoints in a models/ folder at the repository root or let the script train and store them automatically.
* Checkpoint follows the following naming convention: 'shared_vae_head_128.pth' : Where 128 matches the VAE description / latent-space dimension specified in config.json

#### Output Visualizations
Meta-training will produce plots and visualizations saved in the folder 'visualisation/'.


### Datasets

This project uses several image classification datasets loaded with the `datasets` library. It is important to note that RGB datasets with higher image dimensions are being processed and transformed to 28x28 grayscale image datasets in our experimental setup.

#### MNIST

- **HF ID**: `ylecun/mnist`
- **Task**: Handwritten digit classification
- **Image shape**: `1 × 28 × 28` (grayscale)
- **Flattened dim**: `784`
- **#Classes**: `10` (0–9)
- **Splits**:
  - `train`: 60,000 images
  - `test`: 10,000 images

#### Fashion-MNIST

- **HF ID**: `zalando-datasets/fashion_mnist`
- **Task**: Clothing item classification 
- **Image shape**: `1 × 28 × 28` (grayscale)
- **Flattened dim**: `784`
- **#Classes**: `10`
- **Splits**:
  - `train`: 60,000 images
  - `test`: 10,000 images

#### KMNIST

- **HF ID**: `tanganke/kmnist`
- **Task**: Handwritten Japanese characters (Kuzushiji) classification
- **Image shape**: `1 × 28 × 28` (grayscale)
- **Flattened dim**: `784`
- **#Classes**: `10`
- **Splits**:
  - `train`: 60,000 images
  - `test`: 10,000 images

#### Hebrew Handwritten Characters

- **HF ID**: `sivan22/hebrew-handwritten-characters`
- **Task**: Hebrew character classification
- **Image shape**: `3 × 141 × 80` (RGB)
- **Flattened dim**: `33,840`
- **#Classes**: `28`
- **Splits**:
  - `train`: 5,093 images
  - `test`: None included

  #### MATH Shapes

- **HF ID**: `prithivMLmods/Math-Shapes`
- **Task**: Classification of mathematical symbols / shapes
- **Image shape**: `3 × 224 × 224` (RGB)
- **Flattened dim**: `150,528`
- **#Classes**: `8`
- **Splits**:
  - `train`: 12,000 images
  - `validationt`: 4,000 images
  - `test`: 4,000 images




# Use of LLMs
LLMs we used for brain-storming, finding usefull papers and for providing psuedo-code.

![alt text](https://github.com/JeroenN/UDL-MetaTrainingHyperNetwork/blob/main/initial_results.jpeg)
