# Layers

Neural network layers for compression models.

## GDN - Generalized Divisive Normalization

The GDN layer is commonly used in learned image compression for its effectiveness at decorrelating features.

::: compressai.layers.GDN
    options:
      members:
        - forward

## Attention Modules

::: compressai.layers.AttentionBlock
    options:
      show_source: false

## Convolutional Layers

::: compressai.layers.conv3x3

::: compressai.layers.subpel_conv3x3

## Residual Blocks

::: compressai.layers.ResidualBlock
    options:
      show_source: false

::: compressai.layers.ResidualBlockUpsample
    options:
      show_source: false

::: compressai.layers.ResidualBlockWithStride
    options:
      show_source: false
