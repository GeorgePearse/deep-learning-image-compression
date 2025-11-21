# Losses

Loss functions for training compression models.

## Rate-Distortion Loss

The standard loss for learned image compression combines a distortion term with a rate term:

$$\mathcal{L} = \lambda \cdot \mathcal{D} + \mathcal{R}$$

Where:

- $\mathcal{D}$ is the distortion (e.g., MSE, MS-SSIM)
- $\mathcal{R}$ is the rate (bits per pixel)
- $\lambda$ is the trade-off parameter

## Available Losses

::: compressai.losses.RateDistortionLoss
    options:
      show_source: false
