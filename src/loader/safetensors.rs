//! Safetensors loader. The container is a JSON header + raw tensor data; HF
//! ships fp16/bf16 weights this way, with hyperparameters in a separate
//! `config.json`. Models speak GGUF/llama.cpp names; this loader translates them
//! to HF names (and GGUF metadata keys to config.json keys), so a model's
//! `forward` doesn't care which file format it came from.

use std::collections::HashMap;

use super::dequant;
use crate::model::Weights;

struct Entry {
    dtype: String,
    n_elems: usize,
    start: usize,
    end: usize,
}

pub struct SafetensorsWeights {
    tensors: HashMap<String, Entry>, // HF names
    bytes: Vec<u8>,
    data_base: usize, // 8 + header_len
    config: serde_json::Value,
}

impl SafetensorsWeights {
    /// Parse `.safetensors` bytes + the model's `config.json` (as JSON).
    pub fn parse(bytes: Vec<u8>, config: serde_json::Value) -> Result<Self, String> {
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let header: serde_json::Value =
            serde_json::from_slice(&bytes[8..8 + header_len]).map_err(|e| e.to_string())?;
        let obj = header.as_object().ok_or("safetensors header not an object")?;

        let mut tensors = HashMap::new();
        for (name, v) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = v["dtype"].as_str().unwrap_or("").to_string();
            let shape: Vec<usize> =
                v["shape"].as_array().map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect())
                    .unwrap_or_default();
            let off = v["data_offsets"].as_array().ok_or("missing data_offsets")?;
            tensors.insert(
                name.clone(),
                Entry {
                    dtype,
                    n_elems: shape.iter().product::<usize>().max(1),
                    start: off[0].as_u64().unwrap() as usize,
                    end: off[1].as_u64().unwrap() as usize,
                },
            );
        }
        Ok(Self { tensors, bytes, data_base: 8 + header_len, config })
    }

    fn tensor_f32(&self, hf_name: &str) -> Result<Vec<f32>, String> {
        let e = self.tensors.get(hf_name).ok_or_else(|| format!("missing tensor {hf_name}"))?;
        let b = &self.bytes[self.data_base + e.start..self.data_base + e.end];
        Ok(match e.dtype.as_str() {
            "F32" => dequant::dequant_f32(b, e.n_elems),
            "F16" => dequant::dequant_f16(b, e.n_elems),
            "BF16" => dequant::dequant_bf16(b, e.n_elems),
            d => return Err(format!("tensor {hf_name}: safetensors dtype {d} not supported")),
        })
    }
}

/// Translate a GGUF/llama.cpp tensor name to its HF safetensors name.
fn to_hf(gguf: &str) -> String {
    let (stem, suffix) = match gguf.strip_suffix(".weight") {
        Some(s) => (s, ".weight"),
        None => match gguf.strip_suffix(".bias") {
            Some(s) => (s, ".bias"),
            None => (gguf, ""),
        },
    };
    let mapped = if let Some(rest) = stem.strip_prefix("blk.") {
        // blk.{i}.{part}
        let (i, part) = rest.split_once('.').unwrap_or((rest, ""));
        let hf_part = match part {
            "attn_norm" => "input_layernorm".to_string(),
            "attn_q" => "self_attn.q_proj".to_string(),
            "attn_k" => "self_attn.k_proj".to_string(),
            "attn_v" => "self_attn.v_proj".to_string(),
            "attn_output" => "self_attn.o_proj".to_string(),
            "attn_q_norm" => "self_attn.q_norm".to_string(),
            "attn_k_norm" => "self_attn.k_norm".to_string(),
            "ffn_norm" => "post_attention_layernorm".to_string(),
            "ffn_gate" => "mlp.gate_proj".to_string(),
            "ffn_up" => "mlp.up_proj".to_string(),
            "ffn_down" => "mlp.down_proj".to_string(),
            other => other.to_string(),
        };
        format!("model.layers.{i}.{hf_part}")
    } else {
        match stem {
            "output_norm" => "model.norm".to_string(),
            "token_embd" => "model.embed_tokens".to_string(),
            "output" => "lm_head".to_string(),
            other => other.to_string(),
        }
    };
    format!("{mapped}{suffix}")
}

/// Map a GGUF metadata key (`{arch}.suffix`) to a config.json key.
fn config_key(gguf_key: &str) -> &'static str {
    let suffix = gguf_key.split_once('.').map(|(_, s)| s).unwrap_or(gguf_key);
    match suffix {
        "block_count" => "num_hidden_layers",
        "embedding_length" => "hidden_size",
        "attention.head_count" => "num_attention_heads",
        "attention.head_count_kv" => "num_key_value_heads",
        "attention.key_length" => "head_dim",
        "feed_forward_length" => "intermediate_size",
        "attention.layer_norm_rms_epsilon" => "rms_norm_eps",
        "rope.freq_base" => "rope_theta",
        _ => "",
    }
}

impl Weights for SafetensorsWeights {
    fn meta_u32(&self, key: &str) -> u32 {
        self.config[config_key(key)].as_u64().unwrap_or(0) as u32
    }
    fn meta_f32(&self, key: &str) -> f32 {
        self.config[config_key(key)].as_f64().unwrap_or(0.0) as f32
    }
    fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(&to_hf(name))
    }
    fn matrix_data(&mut self, name: &str, in_f: usize, out_f: usize) -> Vec<f32> {
        // HF stores [out, in] row-major; our matmul B operand is [in, out].
        let w = self.tensor_f32(&to_hf(name)).expect("tensor");
        let mut b = vec![0f32; in_f * out_f];
        for o in 0..out_f {
            for i in 0..in_f {
                b[i * out_f + o] = w[o * in_f + i];
            }
        }
        b
    }
    fn matrix_raw(&mut self, name: &str, _out_f: usize, _in_f: usize) -> Vec<f32> {
        // HF stores [out, in] row-major — already the native orientation.
        self.tensor_f32(&to_hf(name)).expect("tensor")
    }
    fn vector_data(&mut self, name: &str, _len: usize) -> Vec<f32> {
        self.tensor_f32(&to_hf(name)).expect("tensor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_mapping() {
        assert_eq!(to_hf("blk.0.attn_q.weight"), "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(to_hf("blk.3.attn_q_norm.weight"), "model.layers.3.self_attn.q_norm.weight");
        assert_eq!(to_hf("blk.7.ffn_gate.weight"), "model.layers.7.mlp.gate_proj.weight");
        assert_eq!(to_hf("output_norm.weight"), "model.norm.weight");
        assert_eq!(to_hf("token_embd.weight"), "model.embed_tokens.weight");
    }

    #[test]
    fn parse_and_read_f32() {
        // one F32 tensor "model.norm.weight" = [1,2,3,4]
        let vals = [1.0f32, 2.0, 3.0, 4.0];
        let mut data = Vec::new();
        for v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let header = serde_json::json!({
            "model.norm.weight": {"dtype":"F32","shape":[4],"data_offsets":[0,16]}
        })
        .to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&data);

        let w = SafetensorsWeights::parse(bytes, serde_json::json!({"num_hidden_layers": 28}))
            .unwrap();
        assert_eq!(w.tensor_f32("model.norm.weight").unwrap(), vals);
        assert_eq!(w.meta_u32("qwen3.block_count"), 28);
        assert!(w.has("output_norm.weight")); // maps to model.norm.weight
    }
}
