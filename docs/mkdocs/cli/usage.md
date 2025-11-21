# CLI Reference

Command-line tools for model evaluation and benchmarking.

## Model Evaluation

### Evaluate Trained Checkpoints

```bash
python -m compressai.utils.eval_model checkpoint /path/to/images/ \
    -a ARCHITECTURE \
    -p /path/to/checkpoint.pth.tar
```

### Evaluate Pre-trained Models

```bash
python -m compressai.utils.eval_model pretrained /path/to/images/ \
    -a ARCHITECTURE \
    -q QUALITY_LEVELS
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `-a, --architecture` | Model architecture name |
| `-p, --path` | Path to checkpoint file |
| `-q, --quality` | Quality level(s) to evaluate |
| `--cuda` | Use GPU acceleration |
| `--half` | Use half precision (FP16) |

**Example:**

```bash
python -m compressai.utils.eval_model pretrained /path/to/kodak/ \
    -a mbt2018-mean -q 1 2 3 4 5 6 7 8 --cuda
```

## Codec Benchmarking

### BPG Codec

```bash
python -m compressai.utils.bench bpg /path/to/images/ [OPTIONS]
```

### VTM (VVC Reference)

```bash
python -m compressai.utils.bench vtm /path/to/images/ [OPTIONS]
```

### General Options

```bash
python -m compressai.utils.bench --help
```

## Plotting Results

Generate rate-distortion plots from evaluation results:

```bash
python -m compressai.utils.plot results.json [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--show` | Display plot interactively |
| `--output` | Save plot to file |
| `--metric` | Metric to plot (psnr, ms-ssim) |

## Video Compression

### Evaluate Video Models

```bash
# Trained checkpoint
python -m compressai.utils.video.eval_model checkpoint /path/to/videos/ \
    -a ssf2020 -p /path/to/checkpoint.pth.tar

# Pre-trained model
python -m compressai.utils.video.eval_model pretrained /path/to/videos/ \
    -a ssf2020 -q QUALITY_LEVELS
```

### Video Codec Benchmarks

```bash
# x265/HEVC
python -m compressai.utils.video.bench x265 /path/to/videos/

# VTM (VVC)
python -m compressai.utils.video.bench VTM /path/to/videos/
```

### Video Plot

```bash
python -m compressai.utils.video.plot results.json --show
```

## Model Update

Update entropy bottleneck parameters after training:

```bash
python -m compressai.utils.update_model \
    --architecture ARCHITECTURE \
    checkpoint.pth.tar
```

This updates the learned CDFs required for entropy coding.

## Example Workflow

1. **Train a model:**

    ```bash
    python examples/train.py -d /path/to/dataset/ \
        -a mbt2018-mean --epochs 300 --cuda --save
    ```

2. **Update the model:**

    ```bash
    python -m compressai.utils.update_model \
        --architecture mbt2018-mean \
        checkpoint_best_loss.pth.tar
    ```

3. **Evaluate:**

    ```bash
    python -m compressai.utils.eval_model checkpoint /path/to/kodak/ \
        -a mbt2018-mean -p checkpoint_best_loss.pth.tar --cuda
    ```

4. **Plot results:**

    ```bash
    python -m compressai.utils.plot results.json --show
    ```
