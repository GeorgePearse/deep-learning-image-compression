# CompressAI

<p align="center">
  <img src="assets/logo.svg" alt="CompressAI Logo" width="400">
</p>

<p align="center">
  <a href="https://github.com/InterDigitalInc/CompressAI/blob/master/LICENSE"><img src="https://img.shields.io/github/license/InterDigitalInc/CompressAI?color=blue" alt="License"></a>
  <a href="https://pypi.org/project/compressai/"><img src="https://img.shields.io/pypi/v/compressai?color=brightgreen" alt="PyPI"></a>
  <a href="https://pypi.org/project/compressai/#files"><img src="https://pepy.tech/badge/compressai" alt="Downloads"></a>
</p>

**CompressAI** (*compress-ay*) is a PyTorch library and evaluation platform for end-to-end compression research.

## Features

CompressAI provides:

- **Custom operations, layers and models** for deep learning based data compression
- **Partial port** of the official [TensorFlow compression](https://github.com/tensorflow/compression) library
- **Pre-trained models** for learned image compression
- **Evaluation scripts** to compare learned models against classical image/video compression codecs

![PSNR performances plot on Kodak](https://raw.githubusercontent.com/InterDigitalInc/CompressAI/master/assets/kodak-psnr.png)

!!! note
    Multi-GPU support is currently experimental.

## Quick Installation

```bash
pip install compressai
```

For detailed installation instructions, see the [Installation Guide](getting-started/installation.md).

## Getting Started

### Encode/Decode Images

Use the provided pre-trained models to compress images:

```bash
python examples/codec.py --help
```

### Train Your Own Model

```bash
python examples/train.py -d /path/to/dataset/ --epochs 300 -lr 1e-4 --batch-size 16 --cuda --save
```

See the [Training Tutorial](tutorials/training.md) for more details.

### Evaluate Models

```bash
# Evaluate a trained checkpoint
python -m compressai.utils.eval_model checkpoint /path/to/images/ -a $ARCH -p $CHECKPOINT

# Evaluate pre-trained models
python -m compressai.utils.eval_model pretrained /path/to/images/ -a $ARCH -q $QUALITY
```

## Available Models

| Model | Paper | Pre-trained |
|-------|-------|:-----------:|
| `bmshj2018_factorized` | [Ballé et al. 2018](https://arxiv.org/abs/1802.01436) | Yes |
| `bmshj2018_hyperprior` | [Ballé et al. 2018](https://arxiv.org/abs/1802.01436) | Yes |
| `mbt2018_mean` | [Minnen et al. 2018](https://arxiv.org/abs/1809.02736) | Yes |
| `mbt2018` | [Minnen et al. 2018](https://arxiv.org/abs/1809.02736) | Yes |
| `cheng2020_anchor` | [Cheng et al. 2020](https://arxiv.org/abs/2001.01568) | Yes |
| `cheng2020_attn` | [Cheng et al. 2020](https://arxiv.org/abs/2001.01568) | No |

See the [Model Zoo](model-zoo/index.md) for the complete list of available models.

## Library Structure

```
compressai/
├── datasets/        # Data loading utilities
├── entropy_models/  # Entropy bottleneck and hyperprior models
├── latent_codecs/   # Latent space encoding/decoding
├── layers/          # Neural network layers (GDN, attention, etc.)
├── losses/          # Rate-distortion loss functions
├── models/          # Compression model architectures
├── ops/             # Custom operations
├── transforms/      # Data transformations
└── zoo/             # Pre-trained model zoo
```

## Citation

If you use CompressAI in your research, please cite:

```bibtex
@article{begaint2020compressai,
    title={CompressAI: a PyTorch library and evaluation platform for end-to-end compression research},
    author={B{\'e}gaint, Jean and Racap{\'e}, Fabien and Feltman, Simon and Pushparaja, Akshay},
    year={2020},
    journal={arXiv preprint arXiv:2011.03029},
}
```

## License

CompressAI is licensed under the BSD 3-Clause Clear License.

## Contributing

We welcome contributions! Please read our [Contributing Guide](https://github.com/InterDigitalInc/CompressAI/blob/master/CONTRIBUTING.md) before submitting pull requests.
