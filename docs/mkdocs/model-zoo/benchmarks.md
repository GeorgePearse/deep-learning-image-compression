# Performance Benchmarks

Compression performance comparisons on standard datasets.

## Training Details

Unless specified otherwise, models were trained with:

| Parameter | Value |
|-----------|-------|
| Dataset | Vimeo90K |
| Patch size | 256x256 |
| Training steps | 4-5M |
| Batch size | 16-32 |
| Initial learning rate | 1e-4 |
| LR schedule | ReduceLROnPlateau (patience=20) |

Training typically takes 1-2 weeks depending on the model and GPU.

## Loss Functions

### MSE Optimization

$$\mathcal{L} = \lambda \cdot 255^{2} \cdot \mathcal{D} + \mathcal{R}$$

### MS-SSIM Optimization

$$\mathcal{L} = \lambda \cdot (1 - \mathcal{D}) + \mathcal{R}$$

Where $\mathcal{D}$ is distortion and $\mathcal{R}$ is estimated bit-rate.

## Lambda Values

| Quality | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---------|---|---|---|---|---|---|---|---|
| MSE | 0.0018 | 0.0035 | 0.0067 | 0.0130 | 0.0250 | 0.0483 | 0.0932 | 0.1800 |
| MS-SSIM | 2.40 | 4.58 | 8.73 | 16.64 | 31.73 | 60.50 | 115.37 | 220.00 |

!!! note
    MS-SSIM models were fine-tuned from MSE pre-trained networks with learning rate 1e-5.

## Channel Configuration

| Bit-rate | Entropy Bottleneck Channels | Recommended |
|----------|----------------------------|-------------|
| <0.5 bpp | 192 | Low bit-rates |
| >0.5 bpp | 320 | High bit-rates |

See `compressai.zoo.image.cfgs` for detailed configurations.

## Kodak Dataset Results

![PSNR on Kodak](https://raw.githubusercontent.com/InterDigitalInc/CompressAI/master/assets/kodak-psnr.png)

### PSNR Comparison

CompressAI models compared against traditional codecs (JPEG, BPG, VVC/VTM) on the [Kodak dataset](http://r0k.us/graphics/kodak/).

## Running Benchmarks

### Evaluate Pre-trained Models

```bash
python -m compressai.utils.eval_model pretrained /path/to/kodak/ \
    -a mbt2018-mean -q 1 2 3 4 5 6 7 8
```

### Compare Against Traditional Codecs

```bash
# BPG codec
python -m compressai.utils.bench bpg /path/to/images/

# VTM (VVC reference)
python -m compressai.utils.bench vtm /path/to/images/
```

### Plot Results

```bash
python -m compressai.utils.plot results.json --show
```

## References

For more comparisons and evaluations, see the [CompressAI paper](https://arxiv.org/abs/2011.03029).
