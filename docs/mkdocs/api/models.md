# Models

Compression model architectures.

## Base Classes

::: compressai.models.CompressionModel
    options:
      members:
        - forward
        - aux_loss
        - update
        - load_state_dict

## Image Compression Models

### Factorized Prior

::: compressai.models.FactorizedPrior
    options:
      show_source: false

### Scale Hyperprior

::: compressai.models.ScaleHyperprior
    options:
      show_source: false

### Mean-Scale Hyperprior

::: compressai.models.MeanScaleHyperprior
    options:
      show_source: false

### Joint Autoregressive Hierarchical Priors

::: compressai.models.JointAutoregressiveHierarchicalPriors
    options:
      show_source: false

## Attention-based Models

::: compressai.models.Cheng2020Anchor
    options:
      show_source: false

::: compressai.models.Cheng2020Attention
    options:
      show_source: false

## Utility Functions

::: compressai.models.utils.conv

::: compressai.models.utils.deconv
