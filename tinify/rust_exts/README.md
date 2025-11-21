# Tinify Rust Extensions

High-performance Rust implementations of entropy coding operations for the tinify library.

## Requirements

- **Rust 1.63+** (install via [rustup](https://rustup.rs/))
- **maturin** for building Python extensions

## Building

### Development build (editable install)

```bash
cd tinify/rust_exts
maturin develop
```

### Release build

```bash
cd tinify/rust_exts
maturin build --release
pip install target/wheels/*.whl
```

Or use the Makefile from the project root:

```bash
make build-rust      # Release build
make build-rust-dev  # Development build
make clean-rust      # Clean build artifacts
```

## Components

### rANS Entropy Coding (`ans` module)

- `BufferedRansEncoder` - Buffered encoder that accumulates symbols before flushing
- `RansEncoder` - One-shot encoder
- `RansDecoder` - Decoder with streaming and one-shot modes

### Utility Operations (`_CXX` module)

- `pmf_to_quantized_cdf()` - Convert probability mass functions to quantized CDFs

## Usage

The Rust extensions are automatically used when available. If not installed, the library falls back to the C++ extensions.

```python
# These imports will use Rust if available, otherwise C++
from tinify.ans_wrapper import BufferedRansEncoder, RansDecoder
from tinify._cxx_wrapper import pmf_to_quantized_cdf

# Check which backend is active
from tinify.ans_wrapper import get_backend
print(f"Using {get_backend()} backend")  # 'rust' or 'cpp'
```

## Implementation Notes

The Rust implementation is a port of the original C++ code based on:
- Fabian Giesen's `ryg_rans` (public domain)
- InterDigital's CompressAI pybind11 bindings

The API is identical to the C++ version for drop-in compatibility.
