# Entropy Models

Entropy bottleneck and hyperprior models for learned compression.

## Overview

Entropy models are a critical component of learned compression. They model the probability distribution of latent representations, enabling efficient entropy coding.

## Entropy Bottleneck

The entropy bottleneck is used to compress the latent representation of an image. It learns a flexible density model of the latent distribution.

::: compressai.entropy_models.EntropyBottleneck
    options:
      members:
        - forward
        - compress
        - decompress
        - loss
        - update

## Gaussian Conditional

::: compressai.entropy_models.GaussianConditional
    options:
      members:
        - forward
        - compress
        - decompress

## Entropy Model Base

::: compressai.entropy_models.EntropyModel
    options:
      show_source: false
