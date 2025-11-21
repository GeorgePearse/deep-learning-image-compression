// Copyright (c) 2021-2025, InterDigital Communications, Inc
// All rights reserved.
// BSD 3-Clause Clear License (see LICENSE file)

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Convert a probability mass function (PMF) to a quantized cumulative distribution function (CDF).
///
/// This is ported from `ryg_rans` public implementation. Not optimal but only runs once
/// per model after training. See TensorFlow compression for an optimized version.
///
/// # Arguments
/// * `pmf` - Probability mass function (non-negative, finite values)
/// * `precision` - Number of bits for quantization (typically 16)
///
/// # Returns
/// Quantized CDF with length pmf.len() + 1
#[pyfunction]
#[pyo3(signature = (pmf, precision))]
pub fn pmf_to_quantized_cdf(pmf: Vec<f32>, precision: i32) -> PyResult<Vec<u32>> {
    // Validate PMF values
    for &p in &pmf {
        if p < 0.0 || !p.is_finite() {
            return Err(PyValueError::new_err(format!(
                "Invalid `pmf`, non-finite or negative element found: {}",
                p
            )));
        }
    }

    let scale = (1u64 << precision) as f32;
    let mut cdf: Vec<u32> = Vec::with_capacity(pmf.len() + 1);

    // First element is always 0
    cdf.push(0);

    // Convert PMF to quantized frequencies
    for &p in &pmf {
        cdf.push((p * scale).round() as u32);
    }

    // Calculate total and check for zero
    let total: u64 = cdf.iter().map(|&x| x as u64).sum();
    if total == 0 {
        return Err(PyValueError::new_err(
            "Invalid `pmf`: at least one element must have a non-zero probability.",
        ));
    }

    // Normalize to sum to 1 << precision
    let target = 1u64 << precision;
    for c in cdf.iter_mut() {
        *c = ((target * (*c as u64)) / total) as u32;
    }

    // Convert to cumulative sum
    let mut cumsum = 0u32;
    for c in cdf.iter_mut() {
        let freq = *c;
        *c = cumsum;
        cumsum = cumsum.wrapping_add(freq);
    }

    // Ensure last element is exactly 1 << precision
    *cdf.last_mut().unwrap() = (1u32 << precision);

    // Fix zero-frequency symbols by stealing from low-frequency symbols
    for i in 0..cdf.len() - 1 {
        if cdf[i] == cdf[i + 1] {
            // Find best symbol to steal from
            let mut best_freq = u32::MAX;
            let mut best_steal: Option<usize> = None;

            for j in 0..cdf.len() - 1 {
                let freq = cdf[j + 1] - cdf[j];
                if freq > 1 && freq < best_freq {
                    best_freq = freq;
                    best_steal = Some(j);
                }
            }

            let best_steal = best_steal.expect("No symbol to steal from");

            if best_steal < i {
                for j in (best_steal + 1)..=i {
                    cdf[j] -= 1;
                }
            } else {
                debug_assert!(best_steal > i);
                for j in (i + 1)..=best_steal {
                    cdf[j] += 1;
                }
            }
        }
    }

    // Final validation in debug mode
    debug_assert_eq!(cdf[0], 0);
    debug_assert_eq!(*cdf.last().unwrap(), 1u32 << precision);
    for i in 0..cdf.len() - 1 {
        debug_assert!(cdf[i + 1] > cdf[i], "CDF not strictly increasing at {}", i);
    }

    Ok(cdf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pmf_to_cdf_uniform() {
        let pmf = vec![0.25, 0.25, 0.25, 0.25];
        let cdf = pmf_to_quantized_cdf(pmf, 16).unwrap();

        assert_eq!(cdf.len(), 5);
        assert_eq!(cdf[0], 0);
        assert_eq!(cdf[4], 1 << 16);

        // Check monotonicity
        for i in 0..cdf.len() - 1 {
            assert!(cdf[i + 1] > cdf[i]);
        }
    }

    #[test]
    fn test_pmf_to_cdf_skewed() {
        let pmf = vec![0.9, 0.05, 0.03, 0.02];
        let cdf = pmf_to_quantized_cdf(pmf, 16).unwrap();

        assert_eq!(cdf.len(), 5);
        assert_eq!(cdf[0], 0);
        assert_eq!(cdf[4], 1 << 16);

        // First symbol should have most of the probability mass
        assert!(cdf[1] > cdf[2] - cdf[1]);
    }

    #[test]
    fn test_pmf_invalid_negative() {
        let pmf = vec![0.5, -0.1, 0.6];
        let result = pmf_to_quantized_cdf(pmf, 16);
        assert!(result.is_err());
    }

    #[test]
    fn test_pmf_invalid_nan() {
        let pmf = vec![0.5, f32::NAN, 0.5];
        let result = pmf_to_quantized_cdf(pmf, 16);
        assert!(result.is_err());
    }

    #[test]
    fn test_pmf_all_zeros() {
        let pmf = vec![0.0, 0.0, 0.0];
        let result = pmf_to_quantized_cdf(pmf, 16);
        assert!(result.is_err());
    }
}
