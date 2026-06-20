//! Weight loaders — each parses a file format and implements [`crate::model::Weights`],
//! so any model can load from any format. GGUF (quantized, llama.cpp names) and
//! Safetensors (HF fp16/bf16, translated to llama.cpp names + `config.json`
//! metadata). Both dequantize to f32 on load via the shared `dequant` module.

pub mod dequant;
pub mod gguf;
pub mod safetensors;

pub use gguf::GgufWeights;
pub use safetensors::SafetensorsWeights;
