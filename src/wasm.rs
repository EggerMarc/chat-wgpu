//! Browser entry point — the `LocalChat` wasm-bindgen API.
//!
//! Weights (GGUF) and tokenizer arrive as bytes from JS (fetched + cached there).
//! Inference runs on the **WebGPU** backend via the same `GpuContext`, kernels,
//! model, and loader as native — no separate browser engine.
//!
//! Memory caveat: weights dequantize to f32 in VRAM, and a model's embedding /
//! lm-head can exceed WebGPU's `maxStorageBufferBindingSize`. We request the
//! adapter's max limits, so a strong desktop GPU runs it; the lightweight path
//! (q4 kept packed in VRAM) is the perf milestone.

use wasm_bindgen::prelude::*;

use crate::context::GpuContext;
use crate::loader::GgufWeights;
use crate::model::qwen3::Qwen3;
use crate::model::Model;
use tokenizers::Tokenizer;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// A loaded local chat model, callable from JavaScript.
#[wasm_bindgen]
pub struct LocalChat {
    ctx: GpuContext,
    model: Qwen3,
    tokenizer: Tokenizer,
    eos: Vec<u32>,
}

#[wasm_bindgen]
impl LocalChat {
    /// Build from quantized GGUF bytes + `tokenizer.json` bytes. Async: requests
    /// the WebGPU device and dequantizes/uploads the weights.
    pub async fn create(gguf: Vec<u8>, tokenizer_json: Vec<u8>) -> Result<LocalChat, JsValue> {
        let ctx = GpuContext::new().await.map_err(err)?;
        let mut weights = GgufWeights::parse(gguf).map_err(err)?;
        let model = Qwen3::load(&ctx, &mut weights).map_err(err)?;
        let tokenizer = Tokenizer::from_bytes(&tokenizer_json).map_err(err)?;

        let eos = ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|t| tokenizer.token_to_id(t))
            .collect();

        Ok(LocalChat { ctx, model, tokenizer, eos })
    }

    /// Generate a reply, streaming each decoded text piece to `on_token`.
    /// Returns the full text. (Greedy for now; `temperature` is reserved.)
    pub async fn generate(
        &self,
        prompt: String,
        system: Option<String>,
        max_tokens: usize,
        _temperature: f32,
        on_token: js_sys::Function,
    ) -> Result<String, JsValue> {
        // Qwen3 ChatML frame.
        let mut text = String::new();
        if let Some(sys) = system {
            text.push_str(&format!("<|im_start|>system\n{sys}<|im_end|>\n"));
        }
        text.push_str(&format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"));

        let enc = self.tokenizer.encode(text, true).map_err(err)?;
        let ids = enc.get_ids();

        let ctx = &self.ctx;
        let model = &self.model;
        let vocab = model.vocab_size();
        let mut cache = model.new_cache(ctx, ids.len() + max_tokens);

        // prefill
        let mut pos = 0usize;
        let mut hidden = None;
        for &t in ids {
            let x = model.embed(ctx, t);
            hidden = Some(model.forward(ctx, &x, pos, &mut cache, &mut ()));
            pos += 1;
        }

        // decode (greedy), streaming detokenized pieces
        let this = JsValue::null();
        let mut produced: Vec<u32> = Vec::new();
        let mut shown = String::new();
        for _ in 0..max_tokens {
            let logits = model.logits(ctx, hidden.as_ref().unwrap());
            let lv = ctx.read(&logits, vocab).await;
            let next = argmax(&lv);
            if self.eos.contains(&next) {
                break;
            }
            produced.push(next);

            // incremental detokenize: decode all, emit the new suffix
            let full = self.tokenizer.decode(&produced, true).map_err(err)?;
            if full.len() > shown.len() {
                let piece = full[shown.len()..].to_string();
                let _ = on_token.call1(&this, &JsValue::from_str(&piece));
                shown = full;
            }

            let x = model.embed(ctx, next);
            hidden = Some(model.forward(ctx, &x, pos, &mut cache, &mut ()));
            pos += 1;
        }
        Ok(shown)
    }
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best as u32
}

fn err<E: std::fmt::Display>(e: E) -> JsValue {
    JsValue::from_str(&e.to_string())
}
