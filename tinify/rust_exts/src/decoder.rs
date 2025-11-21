// Copyright (c) 2021-2025, InterDigital Communications, Inc
// All rights reserved.
// BSD 3-Clause Clear License (see LICENSE file)

use crate::rans64::{
    rans64_dec_advance, rans64_dec_get, rans64_dec_get_bits, rans64_dec_init, Rans64State,
};
use pyo3::prelude::*;

/// Probability range (precision in bits)
const PRECISION: u32 = 16;

/// Number of bits in bypass mode
const BYPASS_PRECISION: u32 = 4;

/// Maximum value in bypass mode
const MAX_BYPASS_VAL: i32 = (1 << BYPASS_PRECISION) - 1;

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

/// Convert bytes to u32 slice
fn bytes_to_u32_vec(data: &[u8]) -> Vec<u32> {
    data.chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Find symbol from cumulative frequency using binary search
fn find_symbol(cdf: &[i32], cdf_size: usize, cum_freq: u32) -> usize {
    // Linear search (matches C++ std::find_if behavior)
    for i in 0..cdf_size {
        if cdf[i] as u32 > cum_freq {
            return i.saturating_sub(1);
        }
    }
    cdf_size.saturating_sub(2)
}

/// Decode a single symbol from the stream
fn decode_symbol(
    state: &mut Rans64State,
    data: &[u32],
    ptr: &mut usize,
    cdf: &[i32],
    cdf_size: i32,
    offset: i32,
) -> i32 {
    let max_value = cdf_size - 2;
    debug_assert!(max_value >= 0);
    debug_assert!((max_value + 1) < cdf.len() as i32);

    let cum_freq = rans64_dec_get(*state, PRECISION);

    // Find symbol
    let s = find_symbol(cdf, cdf_size as usize, cum_freq);

    rans64_dec_advance(
        state,
        data,
        ptr,
        cdf[s] as u32,
        (cdf[s + 1] - cdf[s]) as u32,
        PRECISION,
    );

    let mut value = s as i32;

    // Bypass decoding mode
    if value == max_value {
        let mut val = rans64_dec_get_bits(state, data, ptr, BYPASS_PRECISION) as i32;
        let mut n_bypass = val;

        while val == MAX_BYPASS_VAL {
            val = rans64_dec_get_bits(state, data, ptr, BYPASS_PRECISION) as i32;
            n_bypass += val;
        }

        let mut raw_val: i32 = 0;
        for j in 0..n_bypass {
            val = rans64_dec_get_bits(state, data, ptr, BYPASS_PRECISION) as i32;
            debug_assert!(val <= MAX_BYPASS_VAL);
            raw_val |= val << (j as u32 * BYPASS_PRECISION);
        }

        value = raw_val >> 1;
        if raw_val & 1 != 0 {
            value = -value - 1;
        } else {
            value += max_value;
        }
    }

    value + offset
}

/// rANS decoder
#[pyclass]
pub struct RansDecoder {
    state: Rans64State,
    stream: Vec<u32>,
    ptr: usize,
}

#[pymethods]
impl RansDecoder {
    #[new]
    pub fn new() -> Self {
        RansDecoder {
            state: 0,
            stream: Vec::new(),
            ptr: 0,
        }
    }

    /// Set the encoded stream for streaming decode
    pub fn set_stream(&mut self, encoded: &[u8]) {
        self.stream = bytes_to_u32_vec(encoded);
        self.ptr = 0;
        self.state = rans64_dec_init(&self.stream, &mut self.ptr);
    }

    /// Decode symbols from a previously set stream
    #[pyo3(signature = (indexes, cdfs, cdfs_sizes, offsets))]
    pub fn decode_stream(
        &mut self,
        indexes: Vec<i32>,
        cdfs: Vec<Vec<i32>>,
        cdfs_sizes: Vec<i32>,
        offsets: Vec<i32>,
    ) -> Vec<i32> {
        debug_assert_eq!(cdfs.len(), cdfs_sizes.len());
        assert_cdfs(&cdfs, &cdfs_sizes);

        let mut output = Vec::with_capacity(indexes.len());

        for &cdf_idx in &indexes {
            let cdf_idx = cdf_idx as usize;
            debug_assert!(cdf_idx < cdfs.len());

            let value = decode_symbol(
                &mut self.state,
                &self.stream,
                &mut self.ptr,
                &cdfs[cdf_idx],
                cdfs_sizes[cdf_idx],
                offsets[cdf_idx],
            );

            output.push(value);
        }

        output
    }

    /// Decode symbols from encoded bytes (one-shot)
    #[pyo3(signature = (encoded, indexes, cdfs, cdfs_sizes, offsets))]
    pub fn decode_with_indexes(
        &mut self,
        encoded: &[u8],
        indexes: Vec<i32>,
        cdfs: Vec<Vec<i32>>,
        cdfs_sizes: Vec<i32>,
        offsets: Vec<i32>,
    ) -> Vec<i32> {
        debug_assert_eq!(cdfs.len(), cdfs_sizes.len());
        assert_cdfs(&cdfs, &cdfs_sizes);

        let data = bytes_to_u32_vec(encoded);
        let mut ptr = 0usize;
        let mut state = rans64_dec_init(&data, &mut ptr);

        let mut output = Vec::with_capacity(indexes.len());

        for &cdf_idx in &indexes {
            let cdf_idx = cdf_idx as usize;
            debug_assert!(cdf_idx < cdfs.len());

            let value = decode_symbol(
                &mut state,
                &data,
                &mut ptr,
                &cdfs[cdf_idx],
                cdfs_sizes[cdf_idx],
                offsets[cdf_idx],
            );

            output.push(value);
        }

        output
    }
}

impl Default for RansDecoder {
    fn default() -> Self {
        Self::new()
    }
}
