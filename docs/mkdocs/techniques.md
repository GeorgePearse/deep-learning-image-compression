# Implemented Techniques

A comprehensive overview of all compression techniques implemented in CompressAI.

## Image Compression Models

### Foundational Architectures

These models from Google Research established the core techniques for learned image compression.

#### FactorizedPrior (bmshj2018-factorized)

**Paper:** [Variational Image Compression with a Scale Hyperprior](https://arxiv.org/abs/1802.01436) (Ballé et al., ICLR 2018)

The simplest learned compression architecture:

- **Analysis transform (g_a):** 4 conv layers with GDN activations
- **Synthesis transform (g_s):** 4 deconv layers with inverse GDN
- **Entropy model:** Fully-factorized learned density (EntropyBottleneck)

```
      ┌───┐    y
x ──►─┤g_a├──►─┐
      └───┘    │
               ▼
             ┌─┴─┐
             │ Q │
             └─┬─┘
               │
         y_hat ▼
               │
               ·
            EB :  (Entropy Bottleneck)
               ·
               │
      ┌───┐    │
x_hat◄─┤g_s├───┘
      └───┘
```

#### ScaleHyperprior (bmshj2018-hyperprior)

**Paper:** [Variational Image Compression with a Scale Hyperprior](https://arxiv.org/abs/1802.01436) (Ballé et al., ICLR 2018)

Adds a hyperprior network to predict scale parameters:

- **Hyperprior encoder (h_a):** Encodes absolute values of y to side information z
- **Hyperprior decoder (h_s):** Decodes z to predict scales for Gaussian conditional
- **Entropy model:** Gaussian conditional with predicted scales

```
      ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
      └───┘    │     └───┘     └───┘        EB        └───┘ │
               ▼                                            │
             ┌─┴─┐                                          │
             │ Q │                                          ▼
             └─┬─┘                                          │
               │                                      scales│
         y_hat ▼                                            │
               ·                                            │
            GC : ◄──────────────────────────────────────────┘
               ·
      ┌───┐    │
x_hat◄─┤g_s├───┘
      └───┘
```

#### MeanScaleHyperprior (mbt2018-mean)

**Paper:** [Joint Autoregressive and Hierarchical Priors](https://arxiv.org/abs/1809.02736) (Minnen et al., NeurIPS 2018)

Extends hyperprior to predict both mean and scale:

- Enables non-zero-mean Gaussian conditional
- Better entropy modeling for asymmetric distributions
- Uses LeakyReLU instead of ReLU in hyperprior

#### JointAutoregressiveHierarchicalPriors (mbt2018)

**Paper:** [Joint Autoregressive and Hierarchical Priors](https://arxiv.org/abs/1809.02736) (Minnen et al., NeurIPS 2018)

Adds autoregressive context prediction:

- **Context prediction:** 5x5 MaskedConv2d for causal spatial dependencies
- **Entropy parameters:** Combines hyperprior and context for final Gaussian params
- Best compression but slower due to sequential decoding

```
      ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
      └───┘    │     └───┘     └───┘        EB        └───┘ │
               ▼                                     params ▼
             ┌─┴─┐                                          │
             │ Q │                                          │
             └─┬─┘                                          │
         y_hat ▼                  ┌─────┐                   │
               ├──────────►───────┤  CP ├────────►──────────┤
               │   (Context)      └─────┘                   │
               ▼                                            ▼
               ·                  ┌─────┐                   │
            GC : ◄────────◄───────┤  EP ├────────◄──────────┘
               ·                  └─────┘
      ┌───┐    │
x_hat◄─┤g_s├───┘
      └───┘
```

---

### Attention-based Models

#### Cheng2020AnchorCheckerboard

**Papers:**
- [Learned Image Compression with GMM and Attention](https://arxiv.org/abs/2001.01568) (Cheng et al., CVPR 2020)
- [Checkerboard Context Model](https://arxiv.org/abs/2103.15306) (He et al., CVPR 2021)

Key innovations:

- **Residual blocks:** Uses ResidualBlockWithStride/Upsample instead of plain convolutions
- **Sub-pixel convolution:** For efficient upsampling
- **Checkerboard context:** 2-pass spatial context allowing parallel decoding
- **CheckerboardMaskedConv2d:** Alternating anchor/non-anchor positions

#### ELIC (Elic2022Official / Elic2022Chandelier)

**Paper:** [ELIC: Efficient Learned Image Compression with Unevenly Grouped Space-Channel Contextual Adaptive Coding](https://arxiv.org/abs/2203.10886) (He et al., CVPR 2022)

State-of-the-art architecture with:

- **Uneven channel groups:** Progressive decoding with groups [16, 16, 32, 64, 192]
- **Space-channel context (SCCTX):** Combines spatial and channel context
- **Attention blocks:** Non-local attention in encoder/decoder
- **ResidualBottleneckBlock:** Efficient 1x1→3x3→1x1 bottleneck

```python
# Channel groups configuration
groups = [16, 16, 32, 64, M - 128]  # M=320 total channels
```

---

### Variable Bitrate Models

**Paper:** [Variable-Rate Learned Image Compression with Multi-Objective Optimization](https://arxiv.org/abs/2402.18930) (Kamisli et al., DCC 2024)

| Model | Base Architecture |
|-------|-------------------|
| ScaleHyperpriorVbr | bmshj2018-hyperprior |
| MeanScaleHyperpriorVbr | mbt2018-mean |
| JointAutoregressiveHierarchicalPriorsVbr | mbt2018 |

Key techniques:

1. **Learnable Gain Parameter:**
   ```python
   self.Gain = nn.Parameter(torch.tensor([0.1, 0.14, 0.19, 0.27, 0.37, 0.52, 0.72, 1.0]))
   ```

2. **Quantization-Reconstruction Offsets:**
   ```python
   # 3-layer NN predicts offset from (gain, stdev)
   self.QuantABCD = nn.Sequential(
       nn.Linear(2, 12), nn.ReLU(),
       nn.Linear(12, 12), nn.ReLU(),
       nn.Linear(12, 1)
   )
   ```

3. **Variable-rate Entropy Bottleneck:** Optional `EntropyBottleneckVbr` with adjustable quantization step

4. **Two-stage Training:**
   - Stage 1: Standard training (VBR modules frozen)
   - Stage 2: Multi-objective optimization with VBR modules active

---

## Video Compression Models

### ScaleSpaceFlow (ssf2020)

**Paper:** [Scale-Space Flow for End-to-End Optimized Video Compression](https://openaccess.thecvf.com/content_CVPR_2020/html/Agustsson_Scale-Space_Flow_for_End-to-End_Optimized_Video_Compression_CVPR_2020_paper.html) (Agustsson et al., CVPR 2020)

Components:

- **Scale-space representation:** Multi-scale Gaussian blur pyramid (5 levels)
- **Flow estimation:** Predicts optical flow + scale field
- **Motion compensation:** Warps reference frame using flow
- **Residual coding:** Encodes prediction residual with hyperprior

```python
num_levels = 5      # Scale-space levels
sigma0 = 1.5        # Base Gaussian kernel std
scale_field_shift = 1.0
```

---

## Point Cloud Compression

| Model | Description |
|-------|-------------|
| **SFUPointNet** | PointNet-based geometry compression |
| **SFUPointNet2** | Hierarchical PointNet++ features |
| **HRTZXF2022** | Hierarchical point cloud compression |

---

## Entropy Models

### EntropyBottleneck

Fully-factorized learned entropy model:

- Models each channel independently with learned density
- Uses quantized CDFs for entropy coding
- Adds uniform noise during training, rounds during inference

```python
y_hat, y_likelihoods = self.entropy_bottleneck(y)
# y_likelihoods used to compute rate: -log2(likelihoods)
```

### GaussianConditional

Conditional Gaussian entropy model:

- Requires predicted scale (and optionally mean) parameters
- Uses discretized Gaussian CDF for entropy coding

```python
y_hat, y_likelihoods = self.gaussian_conditional(y, scales, means=means)
```

### GaussianMixtureConditional

Gaussian Mixture Model for multi-modal distributions:

- Multiple Gaussian components with learned weights
- Better for complex latent distributions

### EntropyBottleneckVbr

Variable bitrate entropy bottleneck:

- Adjustable quantization step size
- Supports continuous bitrate control

---

## Latent Codecs

Modular building blocks for entropy coding architectures:

| Codec | Description |
|-------|-------------|
| **HyperpriorLatentCodec** | Complete hyperprior (y + z branches) |
| **HyperLatentCodec** | Side information branch (z only) |
| **CheckerboardLatentCodec** | 2-pass checkerboard spatial context |
| **ChannelGroupsLatentCodec** | Progressive channel-wise decoding |
| **RasterScanLatentCodec** | Sequential autoregressive decoding |
| **GainHyperpriorLatentCodec** | Variable bitrate with gain control |
| **GaussianConditionalLatentCodec** | Gaussian conditional wrapper |
| **EntropyBottleneckLatentCodec** | Entropy bottleneck wrapper |

Example composing latent codecs:

```python
self.latent_codec = HyperpriorLatentCodec(
    latent_codec={
        "y": CheckerboardLatentCodec(
            latent_codec={"y": GaussianConditionalLatentCodec(quantizer="ste")},
            context_prediction=CheckerboardMaskedConv2d(N, 2*N, 5, padding=2),
            entropy_parameters=entropy_params_net,
        ),
        "hyper": HyperLatentCodec(
            entropy_bottleneck=EntropyBottleneck(N),
            h_a=h_a, h_s=h_s,
        ),
    },
)
```

---

## Neural Network Layers

### Generalized Divisive Normalization (GDN)

**Paper:** [Density Modeling of Images Using a Generalized Normalization Transformation](https://arxiv.org/abs/1511.06281) (Ballé et al., ICLR 2016)

Adaptive normalization that decorrelates features:

```python
# Forward GDN (encoder)
GDN(num_channels)

# Inverse GDN (decoder)
GDN(num_channels, inverse=True)
```

### Masked Convolutions

For autoregressive context modeling:

```python
# Type A: masks current pixel (first layer)
MaskedConv2d(in_ch, out_ch, kernel_size=5, mask_type='A')

# Type B: includes current pixel (subsequent layers)
MaskedConv2d(in_ch, out_ch, kernel_size=5, mask_type='B')

# Checkerboard: alternating anchor/non-anchor pattern
CheckerboardMaskedConv2d(in_ch, out_ch, kernel_size=5)
```

### Spectral Convolutions

**Paper:** [Efficient Nonlinear Transforms for Lossy Image Compression](https://arxiv.org/abs/1802.00847) (Ballé, PCS 2018)

Weights stored in frequency domain for better optimization:

```python
SpectralConv2d(in_ch, out_ch, kernel_size=5)
SpectralConvTranspose2d(in_ch, out_ch, kernel_size=5)
```

### Residual Blocks

```python
# Standard residual
ResidualBlock(N, N)

# With strided downsampling
ResidualBlockWithStride(N, N, stride=2)

# With sub-pixel upsampling
ResidualBlockUpsample(N, N, upsample=2)

# Bottleneck (1x1 → 3x3 → 1x1)
ResidualBottleneckBlock(N, N)
```

### Attention Block

Non-local attention for capturing long-range dependencies:

```python
AttentionBlock(num_channels)
```

### Other Layers

| Layer | Description |
|-------|-------------|
| `conv3x3` / `conv1x1` | Convenience wrappers |
| `subpel_conv3x3` | Sub-pixel convolution for 2x upsampling |
| `QReLU` | Quantization-friendly ReLU with configurable bit-depth |

---

## Loss Functions

### RateDistortionLoss

Standard rate-distortion loss:

$$\mathcal{L} = \lambda \cdot D + R$$

Where:
- $D$ = Distortion (MSE or 1 - MS-SSIM)
- $R$ = Rate (bits per pixel from likelihoods)
- $\lambda$ = Trade-off parameter

```python
criterion = RateDistortionLoss(lmbda=0.01)
out_criterion = criterion(out_net, target)
# Returns: loss, mse_loss, bpp_loss
```

**Lambda values for quality levels:**

| Quality | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---------|---|---|---|---|---|---|---|---|
| MSE | 0.0018 | 0.0035 | 0.0067 | 0.0130 | 0.0250 | 0.0483 | 0.0932 | 0.1800 |

---

## Entropy Coding

### Asymmetric Numeral Systems (ANS)

Default entropy coder - fast and efficient:

```python
from compressai.ans import BufferedRansEncoder, RansDecoder

# Encoding
encoder = BufferedRansEncoder()
encoder.encode_with_indexes(symbols, indexes, cdf, cdf_lengths, offsets)
bitstream = encoder.flush()

# Decoding
decoder = RansDecoder()
decoder.set_stream(bitstream)
symbols = decoder.decode_stream(indexes, cdf, cdf_lengths, offsets)
```

### Entropy Coder Selection

```python
import compressai

# List available coders
print(compressai.available_entropy_coders())  # ['ans', 'rangecoder']

# Set default coder
compressai.set_entropy_coder('rangecoder')
```

---

## Key Techniques Summary

### Transform Coding Pipeline

1. **Analysis transform (g_a):** Image → Latent representation
2. **Quantization:** Continuous → Discrete (with noise/STE proxy)
3. **Entropy coding:** Discrete symbols → Bitstream
4. **Entropy decoding:** Bitstream → Discrete symbols
5. **Synthesis transform (g_s):** Latent → Reconstructed image

### Quantization Strategies

| Method | Training | Inference |
|--------|----------|-----------|
| Additive Uniform Noise | Add U(-0.5, 0.5) | Round |
| Straight-Through Estimator (STE) | Round (gradient bypass) | Round |
| Quantization Offsets (VBR) | Learned offset from NN | Learned offset |

### Context Modeling Evolution

1. **No context:** Factorized prior (independent channels)
2. **Hierarchical:** Hyperprior predicts entropy parameters
3. **Autoregressive:** MaskedConv2d for causal spatial context
4. **Checkerboard:** 2-pass for parallel decoding
5. **Channel groups:** Progressive channel-wise context
6. **Space-channel (SCCTX):** Combined spatial + channel context
