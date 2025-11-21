# Training Models

This guide covers how to train compression models using CompressAI.

## Using the Example Training Script

An example training script is provided in the `examples/` folder:

```bash
python examples/train.py -m mbt2018-mean -d /path/to/image/dataset \
    --batch-size 16 -lr 1e-4 --save --cuda
```

Run `train.py --help` to see all available options.

## Dataset Structure

CompressAI expects a custom `ImageFolder` structure:

```
dataset/
├── train/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── test/
    ├── image1.png
    ├── image2.png
    └── ...
```

## Model Update

After training, update the model's internal entropy bottleneck parameters:

```bash
python -m compressai.utils.update_model --architecture ARCH checkpoint_best_loss.pth.tar
```

This updates the learned cumulative distribution functions (CDFs) required for actual entropy coding.

Alternatively, call the `update()` method at the end of training:

```python
model.update()
torch.save(model.state_dict(), "model_updated.pth.tar")
```

## Model Evaluation

Evaluate a trained checkpoint on an image dataset:

```bash
python -m compressai.utils.eval_model checkpoint /path/to/images \
    -a ARCH -p path/to/checkpoint.pth.tar
```

Run `--help` for the complete list of options.

## Entropy Coding

By default, CompressAI uses range Asymmetric Numeral Systems (ANS) for entropy coding.

```python
# List available entropy coders
print(compressai.available_entropy_coders())

# Change the default entropy coder
compressai.set_entropy_coder("rangecoder")
```

### Compressing to Bitstream

```python
x = torch.rand(1, 3, 64, 64)
y = net.encode(x)
strings = net.entropy_bottleneck.compress(y)
```

### Decompressing from Bitstream

```python
shape = y.size()[2:]
y_hat = net.entropy_bottleneck.decompress(strings, shape)
x_hat = net.decode(y_hat)
```

## Training Hyperparameters

The pre-trained models were trained with these settings:

| Parameter | Value |
|-----------|-------|
| Dataset | Vimeo90K (256x256 patches) |
| Batch size | 16-32 |
| Initial learning rate | 1e-4 |
| Training steps | 4-5M |
| LR schedule | ReduceLROnPlateau (patience=20) |

### Loss Functions

**MSE Loss:**

$$\mathcal{L} = \lambda \cdot 255^{2} \cdot \mathcal{D} + \mathcal{R}$$

**MS-SSIM Loss:**

$$\mathcal{L} = \lambda \cdot (1 - \mathcal{D}) + \mathcal{R}$$

Where $\mathcal{D}$ is the distortion and $\mathcal{R}$ is the estimated bit-rate.

### Lambda Values

| Quality | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---------|---|---|---|---|---|---|---|---|
| MSE | 0.0018 | 0.0035 | 0.0067 | 0.0130 | 0.0250 | 0.0483 | 0.0932 | 0.1800 |
| MS-SSIM | 2.40 | 4.58 | 8.73 | 16.64 | 31.73 | 60.50 | 115.37 | 220.00 |

!!! note
    MS-SSIM optimized networks were fine-tuned from pre-trained MSE networks with learning rate 1e-5.

## Channel Configuration

The number of channels depends on the target bit-rate:

- **Low bit-rates** (<0.5 bpp): 192 channels for entropy bottleneck
- **Higher bit-rates**: 320 channels recommended

See `compressai.zoo.image.cfgs` for detailed configurations.
