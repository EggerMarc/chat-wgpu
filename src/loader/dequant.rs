//! Dequantization to f32. Pure, GPU-free, unit-tested. Weights load as f32 into
//! VRAM today; a future q4-in-VRAM dequant-matmul kernel keeps them packed for
//! the bandwidth win, but that's an optimization layered on this.
//!
//! Supported now: F32, F16, BF16 (safetensors); F32, F16, Q8_0, Q4_0 (GGUF).
//! The GGUF k-quants (Q4_K, Q6_K, …) are super-block formats — todo, with a
//! clear error until then.

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as f32;
    let mag = if exp == 0 {
        mant * 2f32.powi(-24) // subnormal
    } else if exp == 0x1f {
        if mant == 0.0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + mant / 1024.0) * 2f32.powi(exp - 15)
    };
    if sign == 1 { -mag } else { mag }
}

pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

fn read_u16(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}

pub fn dequant_f32(bytes: &[u8], n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| f32::from_le_bytes([bytes[i * 4], bytes[i * 4 + 1], bytes[i * 4 + 2], bytes[i * 4 + 3]]))
        .collect()
}

pub fn dequant_f16(bytes: &[u8], n: usize) -> Vec<f32> {
    (0..n).map(|i| f16_to_f32(read_u16(bytes, i * 2))).collect()
}

pub fn dequant_bf16(bytes: &[u8], n: usize) -> Vec<f32> {
    (0..n).map(|i| bf16_to_f32(read_u16(bytes, i * 2))).collect()
}

/// Q8_0: blocks of 32 — f16 scale `d` then 32 i8. value = d·q.
pub fn dequant_q8_0(bytes: &[u8], n: usize) -> Vec<f32> {
    const QK: usize = 32;
    const BLK: usize = 2 + QK; // 34 bytes
    let mut out = Vec::with_capacity(n);
    for blk in 0..n / QK {
        let base = blk * BLK;
        let d = f16_to_f32(read_u16(bytes, base));
        for j in 0..QK {
            out.push(d * (bytes[base + 2 + j] as i8 as f32));
        }
    }
    out
}

/// Q4_0: blocks of 32 — f16 scale `d` then 16 bytes of packed 4-bit quants;
/// low nibble → x[j], high nibble → x[j+16], value = d·(q − 8).
pub fn dequant_q4_0(bytes: &[u8], n: usize) -> Vec<f32> {
    const QK: usize = 32;
    const BLK: usize = 2 + QK / 2; // 18 bytes
    let mut out = vec![0f32; n];
    for blk in 0..n / QK {
        let base = blk * BLK;
        let d = f16_to_f32(read_u16(bytes, base));
        for j in 0..QK / 2 {
            let byte = bytes[base + 2 + j];
            let lo = (byte & 0x0f) as i32 - 8;
            let hi = (byte >> 4) as i32 - 8;
            out[blk * QK + j] = d * lo as f32;
            out[blk * QK + j + QK / 2] = d * hi as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_known() {
        assert_eq!(f16_to_f32(0x3c00), 1.0); // 1.0 in f16
        assert_eq!(f16_to_f32(0xc000), -2.0); // -2.0
        assert_eq!(f16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn bf16_known() {
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xc000), -2.0);
    }

    #[test]
    fn q8_0_block() {
        // d = 0.5 (f16 0x3800), quants 0..32 -> values 0, 0.5, 1.0, ...
        let mut b = vec![0u8, 0x38]; // 0x3800 little-endian = [0x00, 0x38]
        for j in 0..32i8 {
            b.push(j as u8);
        }
        let out = dequant_q8_0(&b, 32);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.5);
        assert_eq!(out[10], 5.0);
    }

    #[test]
    fn q4_0_block() {
        // d = 1.0 (f16 0x3c00). nibble q=8 -> value 0; q=15 -> 7; q=0 -> -8.
        let mut b = vec![0x00, 0x3c];
        for _ in 0..16 {
            b.push((8) | (15 << 4)); // lo=8 (->0), hi=15 (->7)
        }
        let out = dequant_q4_0(&b, 32);
        assert_eq!(out[0], 0.0); // lo nibble of first byte
        assert_eq!(out[16], 7.0); // hi nibble
    }
}
