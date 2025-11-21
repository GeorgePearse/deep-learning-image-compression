# Installation

CompressAI supports Python 3.8+ and PyTorch 1.7+.

## From PyPI

The simplest way to install CompressAI is via pip:

```bash
pip install compressai
```

!!! note
    Pre-built wheels are available for Linux and macOS.

## From Source

For development or to get the latest features, install from source.

### Prerequisites

- A C++17 compiler (GCC 7+, Clang 5+, or MSVC 2019+)
- pip 19.0+
- Python 3.8+

### Installation Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/InterDigitalInc/CompressAI compressai
    cd compressai
    ```

2. Install in development mode:

    ```bash
    pip install -U pip && pip install -e .
    ```

### Optional Dependencies

Install additional packages for specific use cases:

=== "Development"

    ```bash
    pip install -e '.[dev]'
    ```

    Includes: testing, linting, and documentation tools.

=== "Tutorials"

    ```bash
    pip install -e '.[tutorials]'
    ```

    Includes: Jupyter notebooks and widgets.

=== "Point Cloud"

    ```bash
    pip install -e '.[pointcloud]'
    ```

    Includes: Point cloud compression dependencies.

=== "Documentation (MkDocs)"

    ```bash
    pip install -e '.[mkdocs]'
    ```

    Includes: MkDocs and related plugins.

## Using uv

For faster dependency resolution with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install compressai
```

Or from source:

```bash
uv pip install -e .
```

## Verifying Installation

After installation, verify that CompressAI is working:

```python
import compressai
print(compressai.__version__)

# List available entropy coders
print(compressai.available_entropy_coders())

# List available models
from compressai.zoo import models
print(models.keys())
```

## Docker

Docker images are planned for future releases.

!!! warning
    Conda environments are not officially supported.

## Troubleshooting

### Build Errors

If you encounter build errors related to C++ extensions:

1. Ensure you have a C++17 compatible compiler installed
2. Update pip: `pip install -U pip`
3. Try installing with `--no-build-isolation`:

    ```bash
    pip install -e . --no-build-isolation
    ```

### CUDA Issues

For GPU support, ensure you have:

- CUDA toolkit installed (compatible with your PyTorch version)
- cuDNN installed

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```
