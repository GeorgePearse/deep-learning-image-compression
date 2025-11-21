# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
# BSD 3-Clause Clear License (see LICENSE file)

"""
C++ utilities wrapper module.

This module provides utility functions originally implemented in C++.
It attempts to import from the Rust implementation first, falling back
to the C++ implementation if Rust is not available.
"""

__all__ = ["pmf_to_quantized_cdf"]

# Try Rust implementation first, fall back to C++
try:
    from tinify_exts import pmf_to_quantized_cdf

    _backend = "rust"
except ImportError:
    try:
        from tinify._CXX import pmf_to_quantized_cdf

        _backend = "cpp"
    except ImportError as e:
        raise ImportError(
            "No _CXX backend available. Install either the Rust extension "
            "(pip install tinify[rust]) or build the C++ extension."
        ) from e


def get_backend() -> str:
    """Return the name of the active _CXX backend ('rust' or 'cpp')."""
    return _backend
