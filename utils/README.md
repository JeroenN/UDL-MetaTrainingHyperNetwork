## Datasets

This project uses several image classification datasets loaded with the `datasets` library.  

### MNIST

- **HF ID**: `ylecun/mnist`
- **Task**: Handwritten digit classification
- **Image shape**: `1 × 28 × 28` (grayscale)
- **Flattened dim**: `784`
- **#Classes**: `10` (0–9)
- **Splits**:
  - `train`: 60,000 images
  - `test`: 10,000 images

### Fashion-MNIST

- **HF ID**: `zalando-datasets/fashion_mnist`
- **Task**: Clothing item classification 
- **Image shape**: `1 × 28 × 28` (grayscale)
- **Flattened dim**: `784`
- **#Classes**: `10`
- **Splits**:
  - `train`: 60,000 images
  - `test`: 10,000 images

### KMNIST

- **HF ID**: `tanganke/kmnist`
- **Task**: Handwritten Japanese characters (Kuzushiji) classification
- **Image shape**: `1 × 28 × 28` (grayscale)
- **Flattened dim**: `784`
- **#Classes**: `10`
- **Splits**:
  - `train`: 60,000 images
  - `test`: 10,000 images

### Hebrew Handwritten Characters

- **HF ID**: `sivan22/hebrew-handwritten-characters`
- **Task**: Hebrew character classification
- **Image shape**: `3 × 141 × 80` (RGB)
- **Flattened dim**: `33,840`
- **#Classes**: `28`
- **Splits**:
  - `train`: 5,093 images
  - `test`: None included

  ### MATH Shapes

- **HF ID**: `prithivMLmods/Math-Shapes`
- **Task**: Classification of mathematical symbols / shapes
- **Image shape**: `3 × 224 × 224` (RGB)
- **Flattened dim**: `150,528`
- **#Classes**: `8`
- **Splits**:
  - `train`: 12,000 images
  - `validationt`: 4,000 images
  - `test`: 4,000 images