// Copyright (c) 2021-2025, InterDigital Communications, Inc
// All rights reserved.
// BSD 3-Clause Clear License (see LICENSE file)

use crate::rans64::{
    rans64_enc_flush, rans64_enc_init, rans64_enc_put, rans64_enc_put_bits,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Probability range (precision in bits)
const PRECISION: u32 = 16;

/// Number of bits in bypass mode
const BYPASS_PRECISION: u32 = 4;

/// Maximum value in bypass mode
const MAX_BYPASS_VAL: i32 = (1 << BYPASS_PRECISION) - 1;

/// Symbol to be encoded
#[derive(Clone, Copy)]
struct RansSymbol {
    start: u16,
    range: u16,
    bypass: bool,
}

/// Validate CDFs in debug mode
#[cfg(debug_assertions)]
fn assert_cdfs(cdfs: &[Vec<i32>], cdfs_sizes: &[i32]) {
    for (i, cdf) in cdfs.iter().enumerate() {
        let size = cdfs_sizes[i] as usize;
        debug_assert_eq!(cdf[0], 0);
        debug_assert_eq!(cdf[size - 1], 1 << PRECISION);
        for j in 0..size - 1 {
            debug_assert!(cdf[j + 1] > cdf[j]);
        }
    }
}

#[cfg(not(debug_assertions))]
fn assert_cdfs(_cdfs: &[Vec<i32>], _cdfs_sizes: &[i32]) {}

/// Buffered rANS encoder that accumulates symbols before flushing
#[pyclass]
pub struct BufferedRansEncoder {
    syms: Vec<RansSymbol>,
}

#[pymethods]
impl BufferedRansEncoder {
    #[new]
    pub fn new() -> Self {
        BufferedRansEncoder { syms: Vec::new() }
    }

    /// Encode symbols with their corresponding CDF indexes
    #[pyo3(signature = (symbols, indexes, cdfs, cdfs_sizes, offsets))]
    pub fn encode_with_indexes(
        &mut self,
        symbols: Vec<i32>,
        indexes: Vec<i32>,
        cdfs: Vec<Vec<i32>>,
        cdfs_sizes: Vec<i32>,
        offsets: Vec<i32>,
    ) {
        debug_assert_eq!(cdfs.len(), cdfs_sizes.len());
        assert_cdfs(&cdfs, &cdfs_sizes);

        for i in 0..symbols.len() {
            let cdf_idx = indexes[i] as usize;
            debug_assert!(cdf_idx < cdfs.len());

            let cdf = &cdfs[cdf_idx];
            let max_value = cdfs_sizes[cdf_idx] - 2;
            debug_assert!(max_value >= 0);
            debug_assert!((max_value + 1) < cdf.len() as i32);

            let mut value = symbols[i] - offsets[cdf_idx];
            let mut raw_val: u32 = 0;

            if value < 0 {
                raw_val = (-2 * value - 1) as u32;
                value = max_value;
            } else if value >= max_value {
                raw_val = (2 * (value - max_value)) as u32;
                value = max_value;
            }

            debug_assert!(value >= 0);
            debug_assert!(value < cdfs_sizes[cdf_idx] - 1);

            let value_usize = value as usize;
            self.syms.push(RansSymbol {
                start: cdf[value_usize] as u16,
                range: (cdf[value_usize + 1] - cdf[value_usize]) as u16,
                bypass: false,
            });

            // Bypass coding mode (value == max_value -> sentinel flag)
            if value == max_value {
                // Determine number of bypasses needed
                let mut n_bypass: i32 = 0;
                while (raw_val >> (n_bypass as u32 * BYPASS_PRECISION)) != 0 {
                    n_bypass += 1;
                }

                // Encode number of bypasses
                let mut val = n_bypass;
                while val >= MAX_BYPASS_VAL {
                    self.syms.push(RansSymbol {
                        start: MAX_BYPASS_VAL as u16,
                        range: (MAX_BYPASS_VAL + 1) as u16,
                        bypass: true,
                    });
                    val -= MAX_BYPASS_VAL;
                }
                self.syms.push(RansSymbol {
                    start: val as u16,
                    range: (val + 1) as u16,
                    bypass: true,
                });

                // Encode raw value
                for j in 0..n_bypass {
                    let v = ((raw_val >> (j as u32 * BYPASS_PRECISION)) & MAX_BYPASS_VAL as u32) as i32;
                    self.syms.push(RansSymbol {
                        start: v as u16,
                        range: (v + 1) as u16,
                        bypass: true,
                    });
                }
            }
        }
    }

    /// Flush buffered symbols and return encoded bytes
    pub fn flush<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let mut state = rans64_enc_init();
        let mut output: Vec<u32> = Vec::with_capacity(self.syms.len());

        // Process symbols in reverse order (rANS requirement)
        while let Some(sym) = self.syms.pop() {
            if !sym.bypass {
                rans64_enc_put(
                    &mut state,
                    &mut output,
                    sym.start as u32,
                    sym.range as u32,
                    PRECISION,
                );
            } else {
                rans64_enc_put_bits(&mut state, &mut output, sym.start as u32, BYPASS_PRECISION);
            }
        }

        rans64_enc_flush(state, &mut output);

        // Reverse output (rANS writes backwards) and convert to bytes
        output.reverse();
        let bytes: Vec<u8> = output
            .iter()
            .flat_map(|&x| x.to_le_bytes())
            .collect();

        PyBytes::new(py, &bytes)
    }
}

impl Default for BufferedRansEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot rANS encoder
#[pyclass]
pub struct RansEncoder;

#[pymethods]
impl RansEncoder {
    #[new]
    pub fn new() -> Self {
        RansEncoder
    }

    /// Encode symbols with their corresponding CDF indexes and return bytes
    #[pyo3(signature = (symbols, indexes, cdfs, cdfs_sizes, offsets))]
    pub fn encode_with_indexes<'py>(
        &self,
        py: Python<'py>,
        symbols: Vec<i32>,
        indexes: Vec<i32>,
        cdfs: Vec<Vec<i32>>,
        cdfs_sizes: Vec<i32>,
        offsets: Vec<i32>,
    ) -> Bound<'py, PyBytes> {
        let mut encoder = BufferedRansEncoder::new();
        encoder.encode_with_indexes(symbols, indexes, cdfs, cdfs_sizes, offsets);
        encoder.flush(py)
    }
}

impl Default for RansEncoder {
    fn default() -> Self {
        Self::new()
    }
}
