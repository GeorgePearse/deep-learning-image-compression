// Copyright (c) 2021-2025, InterDigital Communications, Inc
// All rights reserved.
// BSD 3-Clause Clear License (see LICENSE file)
//
// 64-bit rANS encoder/decoder - ported from ryg_rans (public domain)
// Original C implementation by Fabian 'ryg' Giesen 2014

/// Lower bound of normalization interval (L in the paper)
/// Uses 63 bits (not 64) to allow exact reciprocals to fit in 64-bit uints
pub const RANS64_L: u64 = 1u64 << 31;

/// rANS state type
pub type Rans64State = u64;

/// Initialize a rANS encoder
#[inline]
pub fn rans64_enc_init() -> Rans64State {
    RANS64_L
}

/// Encode a single symbol with range start and frequency.
/// All frequencies are assumed to sum to `1 << scale_bits`.
///
/// NOTE: With rANS, symbols must be encoded in *reverse order* (end to beginning).
/// The output buffer is also written backwards.
#[inline]
pub fn rans64_enc_put(
    state: &mut Rans64State,
    output: &mut Vec<u32>,
    start: u32,
    freq: u32,
    scale_bits: u32,
) {
    debug_assert!(freq != 0);

    let mut x = *state;

    // Renormalize (never needs to loop)
    let x_max = ((RANS64_L >> scale_bits) << 32) * freq as u64;
    if x >= x_max {
        output.push(x as u32);
        x >>= 32;
        debug_assert!(x < x_max);
    }

    // x = C(s, x)
    *state = ((x / freq as u64) << scale_bits) + (x % freq as u64) + start as u64;
}

/// Encode raw bits (bypass mode) - supports up to 16 bits
#[inline]
pub fn rans64_enc_put_bits(
    state: &mut Rans64State,
    output: &mut Vec<u32>,
    val: u32,
    nbits: u32,
) {
    debug_assert!(nbits <= 16);
    debug_assert!(val < (1u32 << nbits));

    let mut x = *state;
    let freq = 1u32 << (16 - nbits);
    let x_max = ((RANS64_L >> 16) << 32) * freq as u64;

    if x >= x_max {
        output.push(x as u32);
        x >>= 32;
        debug_assert!(x < x_max);
    }

    // x = C(s, x)
    *state = (x << nbits) | val as u64;
}

/// Flush the rANS encoder - writes final state to output
#[inline]
pub fn rans64_enc_flush(state: Rans64State, output: &mut Vec<u32>) {
    let x = state;
    output.push((x >> 0) as u32);
    output.push((x >> 32) as u32);
}

/// Initialize a rANS decoder from encoded data
#[inline]
pub fn rans64_dec_init(data: &[u32], ptr: &mut usize) -> Rans64State {
    let x = (data[*ptr] as u64) | ((data[*ptr + 1] as u64) << 32);
    *ptr += 2;
    x
}

/// Get the current cumulative frequency from decoder state
#[inline]
pub fn rans64_dec_get(state: Rans64State, scale_bits: u32) -> u32 {
    (state & ((1u64 << scale_bits) - 1)) as u32
}

/// Advance the decoder by "popping" a single symbol
#[inline]
pub fn rans64_dec_advance(
    state: &mut Rans64State,
    data: &[u32],
    ptr: &mut usize,
    start: u32,
    freq: u32,
    scale_bits: u32,
) {
    let mask = (1u64 << scale_bits) - 1;

    // s, x = D(x)
    let mut x = *state;
    x = freq as u64 * (x >> scale_bits) + (x & mask) - start as u64;

    // Renormalize
    if x < RANS64_L {
        x = (x << 32) | data[*ptr] as u64;
        *ptr += 1;
        debug_assert!(x >= RANS64_L);
    }

    *state = x;
}

/// Decode raw bits (bypass mode)
#[inline]
pub fn rans64_dec_get_bits(
    state: &mut Rans64State,
    data: &[u32],
    ptr: &mut usize,
    nbits: u32,
) -> u32 {
    let mut x = *state;
    let val = (x & ((1u64 << nbits) - 1)) as u32;

    // Renormalize
    x >>= nbits;
    if x < RANS64_L {
        x = (x << 32) | data[*ptr] as u64;
        *ptr += 1;
        debug_assert!(x >= RANS64_L);
    }

    *state = x;
    val
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        // Simple test: encode and decode a symbol
        let scale_bits = 16;
        let start = 0u32;
        let freq = 1u32 << 15; // 50% probability

        let mut state = rans64_enc_init();
        let mut output = Vec::new();

        // Encode
        rans64_enc_put(&mut state, &mut output, start, freq, scale_bits);
        rans64_enc_flush(state, &mut output);

        // Reverse output (rANS writes backwards)
        output.reverse();

        // Decode
        let mut ptr = 0;
        let mut dec_state = rans64_dec_init(&output, &mut ptr);
        let cum_freq = rans64_dec_get(dec_state, scale_bits);

        assert!(cum_freq < freq);
    }
}
