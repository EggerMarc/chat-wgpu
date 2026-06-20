//! GGUF loader — a hand-written parser (no candle) over the binary container,
//! exposing weights to a model through the [`Weights`] trait. Metadata keys and
//! tensor names are GGUF's (llama.cpp) convention, which the models speak
//! natively.

use std::collections::HashMap;

use super::dequant;
use crate::context::GpuContext;
use crate::model::Weights;

/// A parsed metadata value (only the kinds models ask for are kept typed).
#[derive(Clone, Debug)]
enum Meta {
    U(u64),
    F(f64),
    Str(String),
    Other,
}

struct TensorInfo {
    dtype: u32,
    n_elems: usize,
    /// Byte offset into the data blob.
    offset: usize,
}

pub struct GgufWeights {
    meta: HashMap<String, Meta>,
    tensors: HashMap<String, TensorInfo>,
    data: Vec<u8>, // the tensor-data section
}

struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Cur<'a> {
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.b[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
    fn u64(&mut self) -> u64 {
        let v = u64::from_le_bytes(self.b[self.p..self.p + 8].try_into().unwrap());
        self.p += 8;
        v
    }
    fn str(&mut self) -> String {
        let len = self.u64() as usize;
        let s = String::from_utf8_lossy(&self.b[self.p..self.p + len]).into_owned();
        self.p += len;
        s
    }
    fn bytes(&mut self, n: usize) -> &'a [u8] {
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        s
    }
}

impl GgufWeights {
    /// Parse a whole GGUF file (already in memory — a file on native, fetched
    /// bytes in the browser).
    pub fn parse(bytes: Vec<u8>) -> Result<Self, String> {
        let mut c = Cur { b: &bytes, p: 0 };
        if c.bytes(4) != b"GGUF" {
            return Err("not a GGUF file".into());
        }
        let _version = c.u32();
        let n_tensors = c.u64() as usize;
        let n_meta = c.u64() as usize;

        let mut meta = HashMap::new();
        for _ in 0..n_meta {
            let key = c.str();
            let val = read_meta_value(&mut c);
            meta.insert(key, val);
        }

        let mut infos = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = c.str();
            let n_dims = c.u32() as usize;
            let dims: Vec<u64> = (0..n_dims).map(|_| c.u64()).collect();
            let dtype = c.u32();
            let offset = c.u64() as usize;
            let n_elems = dims.iter().product::<u64>() as usize;
            infos.push((name, TensorInfo { dtype, n_elems, offset }));
        }

        // Tensor data begins at the next `alignment` boundary after the header.
        let alignment = match meta.get("general.alignment") {
            Some(Meta::U(a)) => *a as usize,
            _ => 32,
        };
        let data_start = c.p.div_ceil(alignment) * alignment;
        let data = bytes[data_start..].to_vec();

        let tensors = infos.into_iter().collect();
        Ok(Self { meta, tensors, data })
    }

    /// Dequantize a tensor to f32 in its stored shape (no transpose).
    fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let t = self.tensors.get(name).ok_or_else(|| format!("missing tensor {name}"))?;
        let b = &self.data[t.offset..];
        let n = t.n_elems;
        Ok(match t.dtype {
            0 => dequant::dequant_f32(b, n),
            1 => dequant::dequant_f16(b, n),
            2 => dequant::dequant_q4_0(b, n),
            8 => dequant::dequant_q8_0(b, n),
            d => return Err(format!("tensor {name}: ggml dtype {d} not supported yet (k-quants todo)")),
        })
    }
}

fn read_meta_value(c: &mut Cur) -> Meta {
    let t = c.u32();
    match t {
        0 => Meta::U(c.bytes(1)[0] as u64),                          // u8
        1 => Meta::U(c.bytes(1)[0] as i8 as i64 as u64),            // i8
        2 => Meta::U(u16::from_le_bytes(c.bytes(2).try_into().unwrap()) as u64),
        3 => Meta::U(i16::from_le_bytes(c.bytes(2).try_into().unwrap()) as i64 as u64),
        4 => Meta::U(c.u32() as u64),                               // u32
        5 => Meta::U(c.u32() as i32 as i64 as u64),                 // i32
        6 => Meta::F(f32::from_le_bytes(c.bytes(4).try_into().unwrap()) as f64), // f32
        7 => Meta::U(c.bytes(1)[0] as u64),                         // bool
        8 => Meta::Str(c.str()),                                    // string
        9 => {
            // array: elem_type, count, elems — skip the values, keep nothing.
            let et = c.u32();
            let count = c.u64() as usize;
            for _ in 0..count {
                skip_scalar(c, et);
            }
            Meta::Other
        }
        10 => Meta::U(c.u64()),                                     // u64
        11 => Meta::U(c.u64()),                                     // i64
        12 => Meta::F(f64::from_le_bytes(c.bytes(8).try_into().unwrap())), // f64
        _ => Meta::Other,
    }
}

fn skip_scalar(c: &mut Cur, t: u32) {
    match t {
        0 | 1 | 7 => { c.bytes(1); }
        2 | 3 => { c.bytes(2); }
        4 | 5 | 6 => { c.bytes(4); }
        8 => { c.str(); }
        10 | 11 | 12 => { c.bytes(8); }
        _ => {}
    }
}

impl Weights for GgufWeights {
    fn meta_u32(&self, key: &str) -> u32 {
        match self.meta.get(key) {
            Some(Meta::U(v)) => *v as u32,
            Some(Meta::F(v)) => *v as u32,
            _ => 0,
        }
    }
    fn meta_f32(&self, key: &str) -> f32 {
        match self.meta.get(key) {
            Some(Meta::F(v)) => *v as f32,
            Some(Meta::U(v)) => *v as f32,
            _ => 0.0,
        }
    }
    fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
    fn matrix(&mut self, ctx: &GpuContext, name: &str, in_f: usize, out_f: usize) -> wgpu::Buffer {
        // GGUF stores [out, in] row-major; our matmul B operand is [in, out].
        let w = self.tensor_f32(name).expect("tensor");
        let mut b = vec![0f32; in_f * out_f];
        for o in 0..out_f {
            for i in 0..in_f {
                b[i * out_f + o] = w[o * in_f + i];
            }
        }
        ctx.storage(&b)
    }
    fn vector(&mut self, ctx: &GpuContext, name: &str, _len: usize) -> wgpu::Buffer {
        ctx.storage(&self.tensor_f32(name).expect("tensor"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gstr(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    #[test]
    fn parse_minimal_gguf() {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&1u64.to_le_bytes()); // n_tensors
        b.extend_from_slice(&1u64.to_le_bytes()); // n_meta

        // metadata: "qwen3.block_count" = u32 28
        gstr(&mut b, "qwen3.block_count");
        b.extend_from_slice(&4u32.to_le_bytes()); // type 4 = u32
        b.extend_from_slice(&28u32.to_le_bytes());

        // tensor info: "output_norm.weight", 1 dim [4], dtype 0 (F32), offset 0
        gstr(&mut b, "output_norm.weight");
        b.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        b.extend_from_slice(&4u64.to_le_bytes()); // dims[0]
        b.extend_from_slice(&0u32.to_le_bytes()); // dtype F32
        b.extend_from_slice(&0u64.to_le_bytes()); // offset

        // pad to 32-byte alignment, then tensor data [1,2,3,4]
        while b.len() % 32 != 0 {
            b.push(0);
        }
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            b.extend_from_slice(&v.to_le_bytes());
        }

        let w = GgufWeights::parse(b).unwrap();
        assert_eq!(w.meta_u32("qwen3.block_count"), 28);
        assert_eq!(w.tensor_f32("output_norm.weight").unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert!(w.has("output_norm.weight"));
    }
}
