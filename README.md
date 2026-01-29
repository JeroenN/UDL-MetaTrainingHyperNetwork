# UDL-MetaTrainingHyperNetwork

## Goal of project:
This project took inspiration from two papers:
1. HyperFast: Instant Classification for Tabular Data
In this paper the authors meta-train a hypernetwork to be able to generate a target network that can perform tabular classification with no further training. This does require the dataset to be fully labeled.


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
