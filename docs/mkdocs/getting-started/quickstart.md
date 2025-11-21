# Quick Start

This guide will help you get started with CompressAI for image compression.

## Basic Usage

### Loading a Pre-trained Model

```python
import torch
from compressai.zoo import mbt2018_mean

# Load a pre-trained model (quality level 3)
model = mbt2018_mean(quality=3, pretrained=True)
model.eval()

# For inference, disable gradient computation
model.update()  # Update internal CDFs for entropy coding
```

### Compressing an Image

```python
from PIL import Image
from torchvision import transforms

# Load and preprocess image
img = Image.open("input.png").convert("RGB")
x = transforms.ToTensor()(img).unsqueeze(0)  # Add batch dimension

# Compress
with torch.no_grad():
    out = model.compress(x)

# out["strings"] contains the compressed bitstream
# out["shape"] contains the latent shape for decompression
print(f"Compressed to {len(out['strings'][0][0])} bytes")
```

### Decompressing

```python
# Decompress
with torch.no_grad():
    reconstructed = model.decompress(out["strings"], out["shape"])

x_hat = reconstructed["x_hat"]

# Convert back to image
to_pil = transforms.ToPILImage()
reconstructed_img = to_pil(x_hat.squeeze(0).clamp(0, 1))
reconstructed_img.save("reconstructed.png")
```

## Using the Codec CLI

CompressAI provides a command-line interface for encoding/decoding:

```bash
# Encode an image
python examples/codec.py encode input.png -o compressed.bin -m mbt2018-mean -q 3

# Decode
python examples/codec.py decode compressed.bin -o reconstructed.png
```

## Forward Pass (Training Mode)

For training, use the forward method which returns rate-distortion information:

```python
model.train()

# Forward pass
out = model(x)

# out contains:
# - "x_hat": reconstructed image
# - "likelihoods": dict of entropy model likelihoods

# Compute rate (bits per pixel)
num_pixels = x.shape[2] * x.shape[3]
bpp = sum(
    (-torch.log2(likelihoods).sum() / num_pixels)
    for likelihoods in out["likelihoods"].values()
)

# Compute distortion (MSE)
mse = torch.mean((x - out["x_hat"]) ** 2)

# Rate-distortion loss
lambda_rd = 0.01  # Trade-off parameter
loss = lambda_rd * 255**2 * mse + bpp
```

## Choosing a Model

| Model | Description | Speed | Quality |
|-------|-------------|-------|---------|
| `bmshj2018_factorized` | Simplest model, no hyperprior | Fast | Good |
| `bmshj2018_hyperprior` | Adds scale hyperprior | Medium | Better |
| `mbt2018_mean` | Adds mean prediction | Medium | Better |
| `mbt2018` | Full autoregressive model | Slow | Best |
| `cheng2020_anchor` | Attention-based | Medium | Best |

## Quality Levels

Each model supports multiple quality levels (typically 1-8):

- **Lower quality** (1-3): Lower bitrate, more compression
- **Higher quality** (6-8): Higher bitrate, better reconstruction

```python
# Low bitrate
model_low = mbt2018_mean(quality=1, pretrained=True)

# High quality
model_high = mbt2018_mean(quality=8, pretrained=True)
```

## Next Steps

- [Training Tutorial](../tutorials/training.md): Learn how to train your own models
- [Custom Models](../tutorials/custom-models.md): Build custom compression architectures
- [Model Zoo](../model-zoo/index.md): Explore all available pre-trained models
- [API Reference](../api/models.md): Detailed API documentation
