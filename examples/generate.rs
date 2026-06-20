//! End-to-end native generation: load a real Qwen3 GGUF + tokenizer, encode a
//! prompt, generate greedily, decode to text.
//!
//!   cargo run --release --example generate --features verify -- \
//!     <model.gguf> <tokenizer.json> "Your prompt"

use chat_wgpu::context::GpuContext;
use chat_wgpu::loader::GgufWeights;
use chat_wgpu::model::qwen3::Qwen3;
use chat_wgpu::model::{self, Model};
use tokenizers::Tokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: generate <model.gguf> <tokenizer.json> <prompt> [max_new]");
        std::process::exit(2);
    }
    let gguf_path = &args[1];
    let tok_path = &args[2];
    let prompt = &args[3];
    let max_new: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(32);

    pollster::block_on(run(gguf_path, tok_path, prompt, max_new));
}

async fn run(gguf_path: &str, tok_path: &str, prompt: &str, max_new: usize) {
    let ctx = GpuContext::new().await.expect("gpu");
    eprintln!("[info] backend: {}", ctx.backend);

    let tokenizer = Tokenizer::from_file(tok_path).expect("tokenizer");

    eprintln!("[info] reading {gguf_path}");
    let bytes = std::fs::read(gguf_path).expect("read gguf");
    eprintln!("[info] parsing GGUF ({} MB)", bytes.len() / 1_000_000);
    let mut weights = GgufWeights::parse(bytes).expect("parse gguf");

    eprintln!("[info] loading model (dequantizing weights to f32 on GPU)…");
    let t0 = std::time::Instant::now();
    let model = Qwen3::load(&ctx, &mut weights).expect("load model");
    eprintln!("[info] model ready in {:.1}s", t0.elapsed().as_secs_f64());

    // Qwen3 ChatML prompt.
    let text = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
    let enc = tokenizer.encode(text, true).expect("encode");
    let ids: Vec<u32> = enc.get_ids().to_vec();
    eprintln!("[info] prompt tokens: {}", ids.len());

    eprintln!("[info] generating {max_new} tokens…");
    let t1 = std::time::Instant::now();
    let out = model::generate(&ctx, &model, &ids, max_new, &mut ()).await;
    let dt = t1.elapsed().as_secs_f64();

    let reply = tokenizer.decode(&out, true).expect("decode");
    println!("\n=== prompt ===\n{prompt}\n=== reply ===\n{reply}\n");
    eprintln!(
        "[info] {} tokens in {:.1}s ({:.2} tok/s)",
        out.len(),
        dt,
        out.len() as f64 / dt
    );
}
