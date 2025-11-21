// Copyright (c) 2021-2025, InterDigital Communications, Inc
// All rights reserved.
// BSD 3-Clause Clear License (see LICENSE file)

//! Tinify Rust Extensions
//!
//! This crate provides high-performance entropy coding operations for the tinify
//! learned compression library. It includes:
//!
//! - 64-bit range Asymmetric Numeral Systems (rANS) encoder/decoder
//! - PMF to CDF conversion utilities
//!
//! The implementation is based on Fabian Giesen's public domain rANS code.

mod decoder;
mod encoder;
mod ops;
mod rans64;

use decoder::RansDecoder;
use encoder::{BufferedRansEncoder, RansEncoder};
use ops::pmf_to_quantized_cdf;
use pyo3::prelude::*;

/// Main tinify_exts module
///
/// Exports all rANS encoder/decoder classes and utility functions.
/// These are re-exported by tinify.ans and tinify._CXX for backwards compatibility.
#[pymodule]
fn tinify_exts(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // rANS encoder/decoder classes (for tinify.ans)
    m.add_class::<BufferedRansEncoder>()?;
    m.add_class::<RansEncoder>()?;
    m.add_class::<RansDecoder>()?;

    // Utility functions (for tinify._CXX)
    m.add_function(wrap_pyfunction!(pmf_to_quantized_cdf, m)?)?;

    Ok(())
}
