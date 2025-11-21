# Custom Models

This tutorial shows how to implement a custom autoencoder architecture using CompressAI modules.

## Basic Architecture

Let's build a simple autoencoder with:

- An [`EntropyBottleneck`][compressai.entropy_models.EntropyBottleneck] module
- 3 convolutional layers for encoding
- 3 transposed convolutions for decoding
- [`GDN`][compressai.layers.GDN] activation functions

```python
import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN


class Network(nn.Module):
    def __init__(self, N=128):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.encode = nn.Sequential(
            nn.Conv2d(3, N, stride=2, kernel_size=5, padding=2),
            GDN(N),
            nn.Conv2d(N, N, stride=2, kernel_size=5, padding=2),
            GDN(N),
            nn.Conv2d(N, N, stride=2, kernel_size=5, padding=2),
        )

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(N, N, kernel_size=5, padding=2, output_padding=1, stride=2),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, kernel_size=5, padding=2, output_padding=1, stride=2),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, 3, kernel_size=5, padding=2, output_padding=1, stride=2),
        )

    def forward(self, x):
        y = self.encode(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.decode(y_hat)
        return x_hat, y_likelihoods
```

The strided convolutions reduce spatial dimensions while increasing channels, helping learn better latent representations. The bottleneck module provides differentiable entropy estimation during training.

!!! note
    See the original paper [Variational image compression with a scale hyperprior](https://arxiv.org/abs/1802.01436) and the [tensorflow/compression documentation](https://github.com/tensorflow/compression/blob/v1.3/docs/entropy_bottleneck.md) for detailed explanations.

## Loss Functions

### Rate-Distortion Loss

The rate-distortion loss maximizes reconstruction quality (PSNR) while minimizing the bitrate:

$$\mathcal{L} = \mathcal{D} + \lambda \cdot \mathcal{R}$$

```python
import math
import torch.nn.functional as F

x = torch.rand(1, 3, 64, 64)
net = Network()
x_hat, y_likelihoods = net(x)

# Bitrate of the quantized latent
N, _, H, W = x.size()
num_pixels = N * H * W
bpp_loss = torch.log(y_likelihoods).sum() / (-math.log(2) * num_pixels)

# Mean square error
mse_loss = F.mse_loss(x, x_hat)

# Final loss term
lmbda = 0.01  # Trade-off parameter
loss = mse_loss + lmbda * bpp_loss
```

!!! tip
    Variable bit-rate architectures are possible but beyond this tutorial's scope. See [Variable Rate Deep Image Compression With a Conditional Autoencoder](http://openaccess.thecvf.com/content_ICCV_2019/papers/Choi_Variable_Rate_Deep_Image_Compression_With_a_Conditional_Autoencoder_ICCV_2019_paper.pdf).

### Auxiliary Loss

The entropy bottleneck parameters need separate optimization to minimize density model evaluation:

```python
aux_loss = net.entropy_bottleneck.loss()
```

This auxiliary loss must be minimized during or after training.

## Using CompressionModel Base Class

CompressAI provides a [`CompressionModel`][compressai.models.CompressionModel] base class with helpful utilities:

```python
from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv


class Network(CompressionModel):
    def __init__(self, N=128):
        super().__init__()
        self.encode = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
        )

        self.decode = nn.Sequential(
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

    def forward(self, x):
        y = self.encode(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.decode(y_hat)
        return x_hat, y_likelihoods
```

## Setting Up Optimizers

Train both the compression network and entropy bottleneck with separate optimizers:

```python
import torch.optim as optim

# Main network parameters (exclude quantiles)
parameters = set(
    p for n, p in net.named_parameters()
    if not n.endswith(".quantiles")
)

# Auxiliary parameters (entropy bottleneck quantiles)
aux_parameters = set(
    p for n, p in net.named_parameters()
    if n.endswith(".quantiles")
)

optimizer = optim.Adam(parameters, lr=1e-4)
aux_optimizer = optim.Adam(aux_parameters, lr=1e-3)
```

!!! note
    You can also use PyTorch's [parameter groups](https://pytorch.org/docs/stable/optim.html#per-parameter-options) to define a single optimizer.

## Training Loop

```python
x = torch.rand(1, 3, 64, 64)

for i in range(num_epochs):
    optimizer.zero_grad()
    aux_optimizer.zero_grad()

    x_hat, y_likelihoods = net(x)

    # Compute rate-distortion loss
    N, _, H, W = x.size()
    num_pixels = N * H * W
    bpp_loss = torch.log(y_likelihoods).sum() / (-math.log(2) * num_pixels)
    mse_loss = F.mse_loss(x, x_hat)
    loss = mse_loss + lmbda * bpp_loss

    loss.backward()
    optimizer.step()

    # Update auxiliary parameters
    aux_loss = net.aux_loss()
    aux_loss.backward()
    aux_optimizer.step()

    if i % 100 == 0:
        print(f"Step {i}: loss={loss.item():.4f}, bpp={bpp_loss.item():.4f}")
```

## Adding a Hyperprior

For better compression, add a hyperprior network:

```python
from compressai.models import ScaleHyperprior

# Use the built-in scale hyperprior model
model = ScaleHyperprior(N=128, M=192)
```

Or implement your own by following the patterns in `compressai.models.google`.

## Next Steps

- Explore the [Model Zoo](../model-zoo/index.md) for pre-trained architectures
- Check the [API Reference](../api/models.md) for all available base classes
- Look at `examples/train.py` for a complete training script
