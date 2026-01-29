# UDL-MetaTrainingHyperNetwork

## Goal of project:
This project took inspiration from two papers:
### 1. HyperFast: Instant Classification for Tabular Data

In this paper the authors meta-train a hypernetwork to be able to generate a target network that can perform tabular classification with no further training. This does require the dataset that inference is performed on to be fully labeled. The authors use random projections on the dataset samples and then perform PCA in order to get a dataset level embedding. This embedding is then fed to the hypernetwork.

### 2. MT3: Meta Test-Time Training for Self-Supervised Test-Time Adaption

In this paper the authors meta-train a network to be able to do zero-shot classification on permuted data. Within the outerloop the the meta-network is trained to minimize the cross entropy loss of labeled samples. In the innerloop there are a set number of transformations applied to the images. Each kind of transformation gets its own label. The model is then trained to identify the transformations applied.




### Install dependencies

```bash
pip intall -r requirements.txt
```

Also make sure to install torch seperately.

### CPU-only
```bash
pip install torch
```
### CUDA 11.8
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### CUDA 12.1
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

![alt text](https://github.com/JeroenN/UDL-MetaTrainingHyperNetwork/blob/main/initial_results.jpeg)
