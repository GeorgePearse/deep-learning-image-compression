# Model Zoo

Pre-trained models for end-to-end image and video compression.

## Overview

CompressAI provides pre-trained models optimized with mean square error (MSE) on RGB channels. Models fine-tuned with other metrics are planned for future releases.

## Usage

Load a pre-trained model:

```python
from compressai.zoo import mbt2018_mean

# Load model with quality level 3
model = mbt2018_mean(quality=3, pretrained=True)
model.eval()
model.update()  # Required for entropy coding
```

!!! note
    Pre-trained model weights are automatically downloaded to a cache directory. See [PyTorch documentation](https://pytorch.org/docs/stable/model_zoo.html#torch.utils.model_zoo.load_url) for details.

## Input Requirements

| Requirement | Value |
|-------------|-------|
| Input shape | (N, 3, H, W) |
| Minimum H, W | 64 pixels |
| Value range | [0, 1] |
| Normalization | None (do not normalize) |

!!! warning
    Input dimensions may need to be padded to powers of 2, depending on the model architecture.

## Train vs Eval Mode

Models behave differently in training and evaluation modes (e.g., quantization operations):

```python
model.train()  # Training mode
model.eval()   # Evaluation mode
```

## Available Models

### Image Compression

| Model | Quality Levels | Paper | Pre-trained |
|-------|:--------------:|-------|:-----------:|
| `bmshj2018_factorized` | 1-8 | [Ballé 2018](https://arxiv.org/abs/1802.01436) | Yes |
| `bmshj2018_hyperprior` | 1-8 | [Ballé 2018](https://arxiv.org/abs/1802.01436) | Yes |
| `mbt2018_mean` | 1-8 | [Minnen 2018](https://arxiv.org/abs/1809.02736) | Yes |
| `mbt2018` | 1-8 | [Minnen 2018](https://arxiv.org/abs/1809.02736) | Yes |
| `cheng2020_anchor` | 1-6 | [Cheng 2020](https://arxiv.org/abs/2001.01568) | Yes |
| `cheng2020_attn` | 1-6 | [Cheng 2020](https://arxiv.org/abs/2001.01568) | No |

### Video Compression

| Model | Paper | Pre-trained |
|-------|-------|:-----------:|
| `ssf2020` | [Agustsson 2020](https://arxiv.org/abs/2001.07752) | Yes |

## Model Descriptions

### bmshj2018_factorized

The simplest model with a factorized prior. No hyperprior network.

```python
from compressai.zoo import bmshj2018_factorized
model = bmshj2018_factorized(quality=3, pretrained=True)
```

### bmshj2018_hyperprior

Adds a scale hyperprior for better entropy modeling.

```python
from compressai.zoo import bmshj2018_hyperprior
model = bmshj2018_hyperprior(quality=3, pretrained=True)
```

### mbt2018_mean

Extends hyperprior with mean prediction for improved compression.

```python
from compressai.zoo import mbt2018_mean
model = mbt2018_mean(quality=3, pretrained=True)
```

### mbt2018

Full autoregressive model with joint priors. Best quality but slower.

```python
from compressai.zoo import mbt2018
model = mbt2018(quality=3, pretrained=True)
```

### cheng2020_anchor

Attention-based architecture with Gaussian mixture likelihoods.

```python
from compressai.zoo import cheng2020_anchor
model = cheng2020_anchor(quality=3, pretrained=True)
```

### ssf2020 (Video)

Scale-space flow model for video compression.

```python
from compressai.zoo import ssf2020
model = ssf2020(quality=3, pretrained=True)
```

## Listing All Models

```python
from compressai.zoo import models
print(list(models.keys()))
```

!!! warning "Cross-Platform Limitations"
    All models use floating point operations, which are not reproducible across different devices. Encoding/decoding across different platforms is not guaranteed to work. See [Integer Networks for Data Compression](https://openreview.net/forum?id=S1zz2i0cY7) for solutions.
