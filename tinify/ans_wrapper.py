# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
# BSD 3-Clause Clear License (see LICENSE file)

"""
ANS (Asymmetric Numeral Systems) entropy coding module.

This module provides rANS encoder/decoder classes for entropy coding.
It attempts to import from the Rust implementation first, falling back
to the C++ implementation if Rust is not available.
"""

__all__ = ["BufferedRansEncoder", "RansEncoder", "RansDecoder"]

# Try Rust implementation first, fall back to C++
try:
    from tinify_exts import BufferedRansEncoder, RansDecoder, RansEncoder

    _backend = "rust"
except ImportError:
    try:
        from tinify.ans import BufferedRansEncoder, RansDecoder, RansEncoder

        _backend = "cpp"
    except ImportError as e:
        raise ImportError(
            "No ANS backend available. Install either the Rust extension "
            "(pip install tinify[rust]) or build the C++ extension."
        ) from e


def get_backend() -> str:
    """Return the name of the active ANS backend ('rust' or 'cpp')."""
    return _backend
